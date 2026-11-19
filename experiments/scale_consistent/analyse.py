"""Verified matched summaries and aggregate-only export for Experiment 3."""
from __future__ import annotations
import os, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from exp3_utils import (
    ROOT, VERSION, TARGET_LABELS, SEEDS, PERCENTS, CODE_FILES, inside, read_json,
    atomic_json, atomic_csv, sha256, object_hash, jobs, load_setup,
    check_external_files, receipt_folder, control_folder, assert_statistics_equal, file_lock,
)
from metrics import fold_metrics, combine_folds

ARMS = ("reset_control", "scale_consistent", "target_only_validation_selected")


def read_errors(path: Path):
    # The original collector calculates float32 errors; CSV is not a new precision model.
    frame = pd.read_csv(path, dtype={"person_id": str}, float_precision="round_trip")
    frame["error"] = frame["error"].astype(np.float32)
    return frame


def build_tables(scores):
    """Paired differences across seeds, with five-target macros computed explicitly."""
    expected = {(t, p, s, a) for t in TARGET_LABELS for p in PERCENTS for s in SEEDS for a in ARMS}
    observed = set(scores[["target", "percent", "seed", "arm"]].itertuples(index=False, name=None))
    if observed != expected or len(scores) != len(expected):
        raise ValueError("A complete report requires all 5 targets x 2 fractions x 3 seeds x 3 arms, without duplicates.")
    if not np.isfinite(scores.nrmse).all() or (scores.nrmse <= 0).any():
        raise ValueError("Invalid NRMSE values.")
    rows = []
    for (target, percent, seed), group in scores.groupby(["target", "percent", "seed"], sort=True):
        if not group.n_folds.eq(5).all():
            raise ValueError("A dataset score must combine all five original folds.")
        for col in ("n_attributes", "n_primary_errors", "n_test_people"):
            if group[col].nunique() != 1:
                raise ValueError(f"Evaluation coverage differs between reported arms: {col}")
        g = group.set_index("arm")
        rows.append({"target": target, "target_label": TARGET_LABELS[target], "percent": int(percent), "seed": int(seed),
                     "reset_nrmse": float(g.loc["reset_control", "nrmse"]),
                     "scale_consistent_nrmse": float(g.loc["scale_consistent", "nrmse"]),
                     "target_only_nrmse": float(g.loc["target_only_validation_selected", "nrmse"]), "n_targets": 1})
    paired = pd.DataFrame(rows)
    overall = paired.groupby(["percent", "seed"], as_index=False)[
        ["reset_nrmse", "scale_consistent_nrmse", "target_only_nrmse"]].mean()
    overall["target"] = "FIVE_TARGET_MACRO"
    overall["target_label"] = "Five-target macro (not seven)"
    overall["n_targets"] = 5
    def add_deltas(frame):
        frame = frame.copy()
        frame["scale_minus_reset_nrmse"] = frame.scale_consistent_nrmse - frame.reset_nrmse
        frame["normalization_error_change_pct"] = 100 * frame.scale_minus_reset_nrmse / frame.reset_nrmse
        frame["reset_gain_vs_target_only_pct"] = 100 * (frame.target_only_nrmse - frame.reset_nrmse) / frame.target_only_nrmse
        frame["scale_gain_vs_target_only_pct"] = 100 * (frame.target_only_nrmse - frame.scale_consistent_nrmse) / frame.target_only_nrmse
        return frame
    paired, overall = add_deltas(paired), add_deltas(overall)
    def summarize_seeds(frame):
        result = []
        for (target, label, percent), group in frame.groupby(["target", "target_label", "percent"], sort=True):
            if len(group) != 3 or set(group.seed) != set(SEEDS):
                raise ValueError("Exactly three specified seeds are required.")
            row = {"target": target, "target_label": label, "percent": int(percent), "n_seeds": 3,
                   "n_targets": int(group.n_targets.iloc[0])}
            for col in ("reset_nrmse", "scale_consistent_nrmse", "target_only_nrmse", "scale_minus_reset_nrmse"):
                row[col + "_mean"] = float(group[col].mean())
                row[col + "_sd"] = float(group[col].std(ddof=1))
            reset, scale, baseline = row["reset_nrmse_mean"], row["scale_consistent_nrmse_mean"], row["target_only_nrmse_mean"]
            row["normalization_error_change_pct"] = 100 * (scale - reset) / reset
            row["reset_gain_vs_target_only_pct"] = 100 * (baseline - reset) / baseline
            row["scale_gain_vs_target_only_pct"] = 100 * (baseline - scale) / baseline
            result.append(row)
        return pd.DataFrame(result)
    return paired, overall, summarize_seeds(paired), summarize_seeds(overall)


def report_text(dataset, overall, audits, invariance):
    lines = ["# Experiment 3 — scale-consistent transfer", "",
        "Complete: **150 new fine-tuning fits**, 15 reused full-run source checkpoints, no new pretraining. "
        "Five targets, 10%/100% target training, five participant folds and three seeds.", "",
        "## Primary matched normalization comparison", "",
        "The reset control is Experiment 1's **fresh all-source pretraining/fine-tuning** arm, not the historical paper results. "
        "The new arm keeps source-trained value, time and continuous-context normalization and converts the final output head to target-standardized units. "
        "Target-derived loss weights, label scaling, learning rate, stopping criterion, participants and source checkpoint are unchanged.", "",
        "NRMSE is lower-is-better. Delta = scale-consistent minus reset; a negative delta favors scale consistency. "
        "Means ± sample SD are across three seeds, not 15 independent folds. Relative changes are calculated from mean scores.", "",
        "| Target | Training | Reset NRMSE | Scale-consistent NRMSE | Paired delta | Target-only STraTS |", 
        "|---|---:|---:|---:|---:|---:|"]
    for row in pd.concat([dataset, overall], ignore_index=True).itertuples(index=False):
        def fmt(name):
            return f"{getattr(row, name + '_mean'):.4f} ± {getattr(row, name + '_sd'):.4f}"
        lines.append(f"| {row.target_label} | {row.percent}% | {fmt('reset_nrmse')} | {fmt('scale_consistent_nrmse')} | {fmt('scale_minus_reset_nrmse')} | {fmt('target_only_nrmse')} |")
    lines += ["", "## Five-target interpretation", ""]
    for row in overall.itertuples(index=False):
        lines.append(f"At {row.percent}% training, scale consistency changes mean NRMSE by **{row.scale_minus_reset_nrmse_mean:+.5f}** "
            f"({row.normalization_error_change_pct:+.2f}% relative error change). Gains over the validation-selected target-only STraTS baseline "
            f"are {row.reset_gain_vs_target_only_pct:.2f}% for reset transfer and {row.scale_gain_vs_target_only_pct:.2f}% for scale-consistent transfer.")
        lines.append("")
    lines += ["## Checks and training diagnostics", "",
        f"All {len(invariance)} initial function-preservation checks passed. Maximum absolute raw-unit prediction difference during these float32 checks: "
        f"{invariance.max_absolute_raw_difference.max():.8g}. The acceptance tolerance is scale-aware and recorded per fit.", "",
        f"Fits reaching the 1,000-epoch cap: {int(audits.hit_epoch_cap.sum())}/{len(audits)}. "
        "All new test metrics were recomputed from saved raw errors, restoring the original float32 error precision before float64 metric arithmetic. "
        "Every new fit matches the control's evaluation identity digest and primary-panel coverage. All original checkpoint/control files passed their saved SHA-256 checks.", "",
        "## Scope and caveats", "",
        "This jointly checks source input/time/context scaling plus output-coordinate correction; it does **not** isolate each component's separate effect. "
        "An affine head conversion preserves the initial function, not optimizer trajectories or AdamW weight-decay invariance. This is a sensitivity analysis of a coherent transfer initialization policy.", "",
        "The target-only comparator is Experiment 2's frozen, validation-selected STraTS baseline, not an ensemble and not the best test result. "
        "Its learning-rate selection rule was not rerun or changed here. Baseline gains are contextual; the paired causal contrast in this experiment is between the two transfer-normalization policies.", "",
        "Only the five targets with reusable Experiment 1 checkpoints are evaluated. ABCD and YFS are untested. "
        "The five-target macro must not replace or be labelled as the original seven-target benchmark. No source-excluded, pooled or single-source model was rerun. "
        "This experiment therefore does not establish normalization robustness under participant-overlap exclusion; Experiment 1 remains a separate sensitivity analysis.", "",
        "Only training availability changes at 10%; validation/test sizes stay fixed. Three-seed SDs describe initialization variability, not sampling confidence intervals or statistical significance. "
        "Do not select an arm based on test performance or hide worsened cohorts. Persistent gains support only the tested STraTS implementation and settings.", "",
        "## Files", "",
        "Start with `dataset_summary.csv`, `five_target_summary.csv`, `paired_by_seed.csv`, and `fit_audit.csv`. "
        "Attribute-level scores, initialization checks, head conversions, training histories, source hashes and provenance are also included. "
        "`results_to_share.zip` excludes participant-level errors, participant identifiers, model weights and training logs.", ""]
    return "\n".join(lines)


def summarize():
    setup, project = load_setup()
    missing = [f"job {j['job_id']:03d}/fold{f}" for j in jobs() for f in range(5)
               if not (ROOT / "output/jobs" / f"job_{j['job_id']:03d}" / f"fold{f}/done.json").is_file()]
    if missing:
        raise RuntimeError(f"Cannot issue a complete report: {len(missing)}/150 fits missing. First: {', '.join(missing[:12])}")
    with file_lock(ROOT / "summary/.summary.lock"):
        print("Verifying original checkpoint/control hashes and all new fits...", flush=True)
        check_external_files(setup)
        score_rows, attribute_rows, audit_rows, invariant_rows, conversion_rows = [], [], [], [], []
        recomputation_rows, environment_rows, histories = [], [], []
        for job in jobs():
            folder = ROOT / "output/jobs" / f"job_{job['job_id']:03d}"
            config = read_json(folder / "config.json")
            if config.get("smoke") is not False or config.get("experiment_fingerprint") != setup["experiment_fingerprint"] or config.get("job") != job:
                raise RuntimeError("Research summary encountered a smoke, foreign or incorrectly assigned job.")
            original_config = read_json(receipt_folder(job) / "config.json")
            if config["libraries"] != original_config["libraries"]:
                raise RuntimeError("Different software stacks in a matched comparison.")
            config_hash = object_hash(config)
            new_folds, reset_folds = [], []
            pre = read_json(receipt_folder(job) / "pretrain_done.json")
            for fold in range(5):
                stage = folder / f"fold{fold}"
                done = read_json(stage / "done.json")
                if done.get("config_hash") != config_hash or done.get("smoke") is not False or done.get("fold") != fold or done.get("job") != job:
                    raise RuntimeError(f"Fit receipt mismatch: {stage}")
                if done["source_checkpoint_sha256"] != pre["file_sha256"]["best_weights.pt"]:
                    raise RuntimeError("Fit did not use the specified checkpoint.")
                if done["normalization_sha256"] != sha256(ROOT / "setup/normalizers" / (job["target"] + ".json")):
                    raise RuntimeError("Fit used a different source normalizer.")
                for name, digest in done["file_sha256"].items():
                    if not (stage / name).is_file() or sha256(stage / name) != digest:
                        raise RuntimeError(f"Completed fit file changed/missing: {stage / name}")
                errors = read_errors(stage / "errors.csv.gz")
                actual = fold_metrics(errors, project, job["target"], fold, strict=True)
                stored = read_json(stage / "metrics.json")
                assert_statistics_equal(actual, stored, "raw_error_metric_recomputation")
                control = read_json(control_folder(job, fold) / "done.json")
                control_metric = read_json(control_folder(job, fold) / "metrics.json")
                if actual["identity_digest"] != control_metric["identity_digest"]:
                    raise ValueError("New and reset-control observations are not matched.")
                assert_statistics_equal(done["target_statistics"], control["target_statistics"], "target_loss_scale")
                if done["stage_seed"] != control["stage_seed"]:
                    raise ValueError("Fine-tuning initialization/dropout seed mismatch.")
                inv = read_json(stage / "invariance.json")
                if inv.get("passed") is not True:
                    raise RuntimeError("Missing initial function-preservation check.")
                if actual["n_raw_errors"] != done["n_raw_errors"]:
                    raise ValueError("Unexpected raw error count.")
                common = {"job_id": job["job_id"], "target": job["target"], "target_label": job["target_label"],
                          "percent": job["percent"], "seed": job["seed"], "fold": fold}
                audit_rows.append({**common, **done["training"],
                    "hit_epoch_cap": done["training"]["epochs_run"] >= 1000,
                    "reset_epochs_run": control["training"]["epochs_run"],
                    "reset_best_epoch": control["training"]["best_epoch"],
                    "reset_selected_validation_mse": control["training"]["selected_validation_mse"],
                    "stage_seed": done["stage_seed"], "n_train_records": done["n_train_records"],
                    "n_validation_records": done["n_val_records"], "n_test_records": done["n_test_records"],
                    "n_raw_errors": done["n_raw_errors"], "source_checkpoint_sha256": done["source_checkpoint_sha256"]})
                invariant_rows.append({**common, **{k: inv[k] for k in (
                    "passed", "n_training_records_checked", "max_absolute_raw_difference", "max_scale_adjusted_difference")},
                    "source_unobserved_target_observed": "|".join(inv["conversion"]["source_unobserved_target_observed"]),
                    "target_inactive": "|".join(inv["conversion"]["target_inactive"])})
                for col, scale, shift in zip(inv["conversion"]["attributes"], inv["conversion"]["scale"], inv["conversion"]["shift"]):
                    conversion_rows.append({**common, "attribute": col, "head_scale": scale, "head_shift": shift,
                        "initial_raw_max_abs_diff": inv["absolute_difference_by_attribute"][col]})
                recomputation_rows.append({**common, "float32_errors_restored": True, "metrics_match": True,
                    "matched_control_identities": True, "n_primary_errors": actual["n_primary_errors"]})
                environment_rows.append({**common, **done["environment"],
                    "reset_device_name": control["environment"]["device_name"],
                    "reset_python": control["environment"]["python"],
                    "reset_cpu_threads": control["environment"]["cpu_threads"],
                    "same_gpu_model_as_reset": done["environment"]["device_name"] == control["environment"]["device_name"]})
                history = pd.read_csv(stage / "history.csv", float_precision="round_trip")
                if len(history) != done["training"]["epochs_run"] or not np.isfinite(history[["train_mse", "validation_mse"]].to_numpy()).all():
                    raise ValueError("Invalid training history.")
                history = history.assign(**common)
                histories.append(history)
                new_folds.append(actual)
                reset_folds.append(control_metric)
                del errors
            for arm, folds in (("scale_consistent", new_folds), ("reset_control", reset_folds)):
                aggregate, attrs = combine_folds(folds)
                common = {"target": job["target"], "target_label": job["target_label"], "percent": job["percent"], "seed": job["seed"], "arm": arm}
                score_rows.append({**common, **aggregate})
                attribute_rows.extend({**common, **row} for row in attrs)
            print(f"Verified/recomputed job {job['job_id']:03d}: {job['target_label']}, {job['percent']}%, seed {job['seed']}", flush=True)

        scores = pd.DataFrame(score_rows)
        baseline = pd.read_csv(ROOT / "reference/exp2_dataset_scores_by_seed.csv")
        baseline = baseline[baseline.policy.eq("validation_selected") & baseline.target.isin(TARGET_LABELS)].copy()
        baseline["arm"] = "target_only_validation_selected"
        scores = pd.concat([scores, baseline.drop(columns="policy")], ignore_index=True)
        attr_baseline = pd.read_csv(ROOT / "reference/exp2_attribute_scores_by_seed.csv")
        attr_baseline = attr_baseline[attr_baseline.policy.eq("validation_selected") & attr_baseline.target.isin(TARGET_LABELS)].copy()
        attr_baseline["arm"] = "target_only_validation_selected"
        attrs = pd.concat([pd.DataFrame(attribute_rows), attr_baseline.drop(columns="policy")], ignore_index=True)
        paired, overall_seed, dataset_summary, overall_summary = build_tables(scores)
        audits, invariance = pd.DataFrame(audit_rows), pd.DataFrame(invariant_rows)
        tables = {"dataset_scores_by_seed.csv": scores, "attribute_scores_by_seed.csv": attrs,
            "paired_by_seed.csv": paired, "five_target_scores_by_seed.csv": overall_seed,
            "dataset_summary.csv": dataset_summary, "five_target_summary.csv": overall_summary,
            "fit_audit.csv": audits, "invariance_audit.csv": invariance,
            "head_mapping_audit.csv": pd.DataFrame(conversion_rows),
            "recomputation_audit.csv": pd.DataFrame(recomputation_rows),
            "environment_by_fit.csv": pd.DataFrame(environment_rows),
            "training_history.csv.gz": pd.concat(histories, ignore_index=True),
            "normalization_audit.csv": pd.read_csv(ROOT / "setup/normalization_audit.csv")}
        checkpoints = []
        for old_id, job in sorted({j["exp1_job_id"]: j for j in jobs()}.items()):
            pre = read_json(receipt_folder(job) / "pretrain_done.json")
            checkpoints.append({"exp1_job_id": old_id, "target": job["target"], "seed": job["seed"],
                "checkpoint_sha256": pre["file_sha256"]["best_weights.pt"],
                "source_training_records": pre["n_train_records"],
                "source_validation_records": pre["n_val_records"],
                "source_pretraining_best_epoch": pre["training"]["best_epoch"]})
        tables["source_checkpoints.csv"] = pd.DataFrame(checkpoints)
        for name, table in tables.items():
            atomic_csv(table, ROOT / "summary" / name)
        report_path = inside(ROOT / "summary/REPORT.md")
        report_path.write_text(report_text(dataset_summary, overall_summary, audits, invariance), encoding="utf-8")
        provenance = {"version": VERSION, "complete": True, "research_target_fits": 150,
            "reused_source_checkpoints": 15, "new_source_training_fits": 0,
            "experiment_fingerprint": setup["experiment_fingerprint"],
            "package_fingerprint": setup["package_fingerprint"], "input_sha256": setup["input_sha256"],
            "targets": list(TARGET_LABELS), "seeds": list(SEEDS), "percents": list(PERCENTS),
            "source_reference": read_json(ROOT / "reference/provenance.json"),
            "metric": "Person mean squared fold-reference-standardized error; mean persons across folds; square root per attribute; mean eligible primary attributes; equal five-target macro.",
            "delta": "scale-consistent NRMSE minus Experiment 1 all-source reset-control NRMSE; negative favors scale consistency",
            "uncertainty": "Sample SD across three seeds, not CI or significance test; folds not treated as independent replicates",
            "baseline": "Frozen Experiment 2 validation-selected target-only STraTS; no new rate selection",
            "recomputed_new_errors": True, "float32_error_restoration": True,
            "old_checkpoint_and_control_hashes_verified": True, "all_evaluation_identity_digests_match": True,
            "all_initial_prediction_checks_passed": True,
            "limitations": ["Five targets, not the original seven-target benchmark", "All-source only; does not combine normalization with source exclusion",
                            "Joint input/time/context/output-coordinate sensitivity, not component attribution",
                            "Initial raw predictions preserved; AdamW trajectories and weight decay are not parameterization-invariant",
                            "No claims about other architectures or untested cohorts"]}
        atomic_json(ROOT / "summary/provenance.json", provenance)
        output_names = sorted(list(tables) + ["REPORT.md", "provenance.json"])
        atomic_json(ROOT / "summary/file_sha256.json", {n: sha256(ROOT / "summary" / n) for n in output_names})
        output_names.append("file_sha256.json")
        archive = inside(ROOT / "results_to_share.zip")
        temporary = inside(ROOT / f"results_to_share.{os.getpid()}.tmp.zip")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name in output_names:
                z.write(ROOT / "summary" / name, "summary/" + name)
            for name in CODE_FILES + ("PROTOCOL.md",):
                z.write(ROOT / name, "package/" + name)
            for path in sorted((ROOT / "reference").iterdir()):
                if path.is_file():
                    z.write(path, "package/reference/" + path.name)
        os.replace(temporary, archive)
        print("Complete: summary/REPORT.md and results_to_share.zip. The ZIP contains no participant-level errors, identifiers or model weights.")
