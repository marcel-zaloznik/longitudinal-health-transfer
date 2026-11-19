"""Verify old and added fits, then publish one complete seven-dataset report."""
from __future__ import annotations
import os, shutil, tempfile, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from exp3_utils import (
    ROOT, SEEDS, PERCENTS, CODE_FILES, inside, read_json, atomic_json, atomic_csv,
    sha256, object_hash, jobs as legacy_jobs, receipt_folder, control_folder,
    assert_statistics_equal, file_lock, stage_seed,
)
from analyse import read_errors
from metrics import fold_metrics, combine_folds
from training import verified_done
from seven_utils import (
    VERSION, ALL_TARGETS, LEGACY_TARGETS, NEW_TARGETS, ARMS, TRAINING_ARMS,
    SETUP, OUTPUT, EXT_CODE, EXT_DOCS, LIBRARY_KEYS, extension_jobs, comparison_job,
    job_folder, stage_folder, normalizer_path, load_extension, check_legacy_inventory,
    checked_metric, validate_history,
)


def empty_bundle():
    return {key: [] for key in ("scores", "attributes", "audit", "invariance", "conversion",
        "recomputation", "environment", "histories", "checkpoints", "source_histories")}


def add_pair(bundle, job, fold, scale, reset, scale_metric, reset_metric, inv, history, reset_history=None, origin="extension"):
    if scale_metric["identity_digest"] != reset_metric["identity_digest"]:
        raise ValueError("Scale-consistent and reset-control evaluation identities differ.")
    for field in ("target_statistics", "target_train", "target_val", "target_test"):
        assert_statistics_equal(scale[field], reset[field], "paired_" + field)
    for field in ("stage_seed", "n_train_records", "n_val_records", "n_test_records", "n_raw_errors"):
        if scale[field] != reset[field]:
            raise ValueError(f"Paired fit differs in {field}")
    if inv.get("passed") is not True:
        raise RuntimeError("The initial raw-prediction invariance check did not pass.")
    common = {"job_id": job["job_id"], "target": job["target"], "target_label": job["target_label"],
              "percent": job["percent"], "seed": job["seed"], "fold": fold}
    bundle["audit"].append({**common, **scale["training"], "fit_origin": origin,
        "hit_epoch_cap": scale["training"]["epochs_run"] >= 1000,
        "reset_epochs_run": reset["training"]["epochs_run"],
        "reset_best_epoch": reset["training"]["best_epoch"],
        "reset_selected_validation_mse": reset["training"]["selected_validation_mse"],
        "reset_elapsed_seconds": reset["training"]["elapsed_seconds"],
        "reset_hit_epoch_cap": reset["training"]["epochs_run"] >= 1000,
        "stage_seed": scale["stage_seed"], "n_train_records": scale["n_train_records"],
        "n_validation_records": scale["n_val_records"], "n_test_records": scale["n_test_records"],
        "n_raw_errors": scale["n_raw_errors"], "source_checkpoint_sha256": scale["source_checkpoint_sha256"]})
    bundle["invariance"].append({**common, **{k: inv[k] for k in (
        "passed", "n_training_records_checked", "max_absolute_raw_difference", "max_scale_adjusted_difference")},
        "source_unobserved_target_observed": "|".join(inv["conversion"]["source_unobserved_target_observed"]),
        "target_inactive": "|".join(inv["conversion"]["target_inactive"])})
    for attr, a, b in zip(inv["conversion"]["attributes"], inv["conversion"]["scale"], inv["conversion"]["shift"]):
        bundle["conversion"].append({**common, "attribute": attr, "head_scale": a, "head_shift": b,
            "initial_raw_max_abs_diff": inv["absolute_difference_by_attribute"][attr]})
    bundle["recomputation"].append({**common, "scale_errors_recomputed": True,
        "reset_errors_recomputed": origin == "extension", "reset_verified_frozen_receipt": origin == "original_five",
        "float32_errors_restored": True, "metrics_match": True, "matched_control_identities": True,
        "n_primary_errors": scale_metric["n_primary_errors"]})
    bundle["environment"].append({**common, **scale["environment"],
        "reset_device_name": reset["environment"]["device_name"],
        "reset_python": reset["environment"]["python"],
        "reset_cpu_threads": reset["environment"]["cpu_threads"],
        "same_gpu_model_as_reset": scale["environment"]["device_name"] == reset["environment"]["device_name"]})
    bundle["histories"].append(history.assign(**common, arm="scale_consistent"))
    if reset_history is not None:
        bundle["histories"].append(reset_history.assign(**common, arm="reset_control"))


def add_scores(bundle, job, per_arm):
    for arm, folds in per_arm.items():
        if len(folds) != 5 or {m["fold"] for m in folds} != set(range(5)):
            raise ValueError("A dataset/seed score requires all five distinct folds.")
        aggregate, attributes = combine_folds(folds)
        common = {"target": job["target"], "target_label": job["target_label"],
                  "percent": job["percent"], "seed": job["seed"], "arm": arm}
        bundle["scores"].append({**common, **aggregate})
        bundle["attributes"].extend({**common, **row} for row in attributes)


def collect_legacy(setup, project, *, quiet=False):
    """Recompute all 150 saved scale fits; verify frozen reset sufficient statistics.

    Original external checkpoint/error files are unnecessary for an already completed
    Experiment 3: its preparation has copied and hash-pinned the required receipts.
    """
    bundle, inventory, libraries, seen_sources = empty_bundle(), {}, None, set()
    original_scores = pd.read_csv(ROOT / "reference/exp1_dataset_scores.csv", float_precision="round_trip")
    def pin(path):
        inventory[str(path.relative_to(ROOT))] = sha256(path)
    for job in legacy_jobs():
        folder = ROOT / "output/jobs" / f"job_{job['job_id']:03d}"
        config = read_json(folder / "config.json")
        if config.get("smoke") is not False or config.get("job") != job or config.get("experiment_fingerprint") != setup["experiment_fingerprint"]:
            raise ValueError("An original Experiment 3 job is incomplete, foreign or a smoke run.")
        original_config = read_json(receipt_folder(job) / "config.json")
        if config["libraries"] != original_config["libraries"]:
            raise ValueError("Original transfer pair used different libraries.")
        if libraries is None:
            libraries = config["libraries"]
        elif libraries != config["libraries"]:
            raise ValueError("Original five-target jobs used inconsistent library environments.")
        pin(folder / "config.json")
        config_hash = object_hash(config)
        pre = read_json(receipt_folder(job) / "pretrain_done.json")
        if pre["config_hash"] != object_hash(original_config):
            raise ValueError("Frozen source receipt has the wrong configuration.")
        norm_file = ROOT / "setup/normalizers" / (job["target"] + ".json")
        norm = read_json(norm_file)
        assert_statistics_equal(norm["target"], pre["target_statistics"], "original_source_output_normalization")
        per_arm = {arm: [] for arm in TRAINING_ARMS}
        for fold in range(5):
            stage = folder / f"fold{fold}"
            done = verified_done(stage, config_hash, ("errors.csv.gz", "metrics.json", "history.csv", "invariance.json"))
            if done is None:
                raise RuntimeError(f"Complete the original five-target run first: {stage}")
            if done.get("smoke") is not False or done.get("job") != job or done.get("fold") != fold or done.get("experiment_fingerprint") != setup["experiment_fingerprint"]:
                raise ValueError("Invalid original fit receipt.")
            if done["source_checkpoint_sha256"] != pre["file_sha256"]["best_weights.pt"] or done["normalization_sha256"] != sha256(norm_file):
                raise ValueError("Original fit checkpoint/normalization mismatch.")
            expected_seed = stage_seed(job["seed"], setup["origins"].index(job["target"]), fold)
            if done["stage_seed"] != expected_seed:
                raise ValueError("Original stage seed mismatch.")
            for name in ("done.json", *done["file_sha256"]):
                pin(stage / name)
            errors = read_errors(stage / "errors.csv.gz")
            metric = fold_metrics(errors, project, job["target"], fold, strict=True)
            assert_statistics_equal(metric, read_json(stage / "metrics.json"), "original_raw_errors_recomputed")
            if metric["n_raw_errors"] != done["n_raw_errors"]:
                raise ValueError("Original raw error count mismatch.")
            reset = read_json(control_folder(job, fold) / "done.json")
            reset_metric = read_json(control_folder(job, fold) / "metrics.json")
            if reset["config_hash"] != object_hash(original_config):
                raise ValueError("Frozen control receipt has the wrong configuration.")
            history = validate_history(stage / "history.csv", done["training"])
            add_pair(bundle, job, fold, done, reset, metric, reset_metric,
                     read_json(stage / "invariance.json"), history, origin="original_five")
            per_arm["scale_consistent"].append(metric)
            per_arm["reset_control"].append(reset_metric)
        add_scores(bundle, job, per_arm)
        reset_score = next(r for r in bundle["scores"][-2:] if r["arm"] == "reset_control")
        reference = original_scores[original_scores.target.eq(job["target"]) & original_scores.percent.eq(job["percent"]) &
            original_scores.seed.eq(job["seed"]) & original_scores["mode"].eq("pretrain") & original_scores.arm.eq("all_source")]
        if len(reference) != 1:
            raise ValueError("Original reset score missing/duplicated in frozen Experiment 1 report.")
        for col in ("nrmse", "nmae", "n_attributes", "n_primary_errors", "n_test_people", "n_folds"):
            if not np.isclose(reset_score[col], reference.iloc[0][col], rtol=1e-12, atol=1e-12):
                raise ValueError("Frozen reset-control receipt no longer matches its original published summary.")
        if job["exp1_job_id"] not in seen_sources:
            seen_sources.add(job["exp1_job_id"])
            bundle["checkpoints"].append({"checkpoint_id": f"experiment1/job_{job['exp1_job_id']:03d}",
                "origin": "reused_experiment1_receipt", "target": job["target"], "seed": job["seed"],
                "checkpoint_sha256": pre["file_sha256"]["best_weights.pt"],
                "source_training_records": pre["n_train_records"], "source_validation_records": pre["n_val_records"],
                "source_pretraining_best_epoch": pre["training"]["best_epoch"],
                "source_pretraining_epochs": pre["training"]["epochs_run"]})
        if not quiet:
            print(f"Verified original job {job['job_id']:03d}: {job['target_label']}, {job['percent']}%, seed {job['seed']}", flush=True)
    if len(bundle["audit"]) != 150 or len(seen_sources) != 15:
        raise ValueError("The original five-target experiment is not complete.")
    return bundle, inventory, libraries


def collect_added(manifest, project):
    bundle = empty_bundle()
    expected = read_json(SETUP / "expected_test_counts.json")
    partitions = read_json(SETUP / "target_partitions.json")
    target_stats = read_json(SETUP / "target_statistics.json")
    source_partitions = read_json(SETUP / "source_partitions.json")
    for job in extension_jobs():
        folder = job_folder(job)
        config = read_json(folder / "config.json")
        if config.get("smoke") is not False or config.get("job") != job or config.get("experiment_fingerprint") != manifest["experiment_fingerprint"]:
            raise ValueError("Research summary encountered a smoke/foreign extension job.")
        if config["libraries"] != manifest["required_libraries"]:
            raise ValueError("Extension library environment differs from the original experiment.")
        config_hash = object_hash(config)
        pre_stage = folder / "pretrain"
        pre = verified_done(pre_stage, config_hash, ("best_weights.pt", "history.csv", "normalizers.json"))
        if pre is None or pre.get("smoke") is not False or pre.get("job") != job:
            raise ValueError("Missing/full-run source checkpoint receipt.")
        if pre["stage_seed"] != stage_seed(job["seed"], manifest["origins"].index(job["target"])):
            raise ValueError("Source checkpoint used the wrong initialization seed.")
        norm = read_json(normalizer_path(job["target"]))
        assert_statistics_equal(read_json(pre_stage / "normalizers.json"), norm, "actual_source_normalization")
        for role in ("source_train", "source_val"):
            assert_statistics_equal(pre[role], source_partitions[job["target"]][role], role)
        assert_statistics_equal(pre["target_statistics"], norm["target"], "source_output_statistics")
        source_history = validate_history(pre_stage / "history.csv", pre["training"])
        bundle["source_histories"].append(source_history.assign(target=job["target"], seed=job["seed"], extension_job_id=job["extension_job_id"]))
        checkpoint_hash = pre["file_sha256"]["best_weights.pt"]
        bundle["checkpoints"].append({"checkpoint_id": f"experiment3-extension/job_{job['extension_job_id']:03d}",
            "origin": "new_experiment3_source_pretraining", "target": job["target"], "seed": job["seed"],
            "checkpoint_sha256": checkpoint_hash, "source_training_records": pre["n_train_records"],
            "source_validation_records": pre["n_val_records"], "source_pretraining_best_epoch": pre["training"]["best_epoch"],
            "source_pretraining_epochs": pre["training"]["epochs_run"]})
        for percent in PERCENTS:
            comparison = comparison_job(job, percent)
            per_arm = {arm: [] for arm in TRAINING_ARMS}
            for fold in range(5):
                records, metrics, histories = {}, {}, {}
                for arm in TRAINING_ARMS:
                    stage = stage_folder(job, arm, percent, fold)
                    required = ("errors.csv.gz", "metrics.json", "history.csv") + (("invariance.json",) if arm == "scale_consistent" else ())
                    done = verified_done(stage, config_hash, required)
                    if done is None or done.get("smoke") is not False or done.get("job") != job or done.get("fold") != fold or done.get("percent") != percent or done.get("arm") != arm:
                        raise ValueError(f"Missing/misassigned extension fit: {stage}")
                    if done.get("experiment_fingerprint") != manifest["experiment_fingerprint"]:
                        raise ValueError("Extension fit belongs to a different preparation.")
                    if done["source_checkpoint_sha256"] != checkpoint_hash or done["normalization_sha256"] != sha256(normalizer_path(job["target"])):
                        raise ValueError("Extension fit did not use the specified shared source checkpoint/normalization.")
                    seed = stage_seed(job["seed"], manifest["origins"].index(job["target"]), fold)
                    if done["stage_seed"] != seed:
                        raise ValueError("Extension fine-tuning seed mismatch.")
                    for role in ("target_train", "target_val", "target_test"):
                        assert_statistics_equal(done[role], partitions[job["target"]][str(percent)][str(fold)][role], role)
                    assert_statistics_equal(done["target_statistics"], target_stats[job["target"]][str(percent)][str(fold)], "target_statistics")
                    for role, n in (("target_train", "n_train_records"), ("target_val", "n_val_records"), ("target_test", "n_test_records")):
                        if done[n] != done[role]["n_pairs"]:
                            raise ValueError("Prepared and actual prediction-pair counts differ.")
                    errors = read_errors(stage / "errors.csv.gz")
                    metric = checked_metric(errors, project, job["target"], fold, expected)
                    assert_statistics_equal(metric, read_json(stage / "metrics.json"), "added_raw_errors_recomputed")
                    if metric["n_raw_errors"] != done["n_raw_errors"]:
                        raise ValueError("Extension raw error count mismatch.")
                    metrics[arm], records[arm] = metric, done
                    histories[arm] = validate_history(stage / "history.csv", done["training"])
                    per_arm[arm].append(metric)
                inv = read_json(stage_folder(job, "scale_consistent", percent, fold) / "invariance.json")
                add_pair(bundle, comparison, fold, records["scale_consistent"], records["reset_control"],
                    metrics["scale_consistent"], metrics["reset_control"], inv,
                    histories["scale_consistent"], histories["reset_control"])
            add_scores(bundle, comparison, per_arm)
        print(f"Verified added source job {job['extension_job_id']}: {job['target']}, seed {job['seed']}; 20 target fits", flush=True)
    return bundle


def build_tables(scores):
    expected = {(t,p,s,a) for t in ALL_TARGETS for p in PERCENTS for s in SEEDS for a in ARMS}
    observed = set(scores[["target", "percent", "seed", "arm"]].itertuples(index=False, name=None))
    if observed != expected or len(scores) != len(expected):
        raise ValueError("Complete report requires all 7 targets x 2 fractions x 3 seeds x 3 arms, without duplicates.")
    if not np.isfinite(scores.nrmse).all() or (scores.nrmse <= 0).any():
        raise ValueError("Invalid NRMSE values.")
    rows = []
    for (target, percent, seed), group in scores.groupby(["target", "percent", "seed"], sort=True):
        if not group.n_folds.eq(5).all():
            raise ValueError("Five folds are required per score.")
        for col in ("n_attributes", "n_primary_errors", "n_test_people"):
            if group[col].nunique() != 1:
                raise ValueError(f"Evaluation coverage differs between arms: {target}/{col}")
        g = group.set_index("arm")
        rows.append({"target": target, "target_label": ALL_TARGETS[target], "percent": int(percent), "seed": int(seed),
            "reset_nrmse": float(g.loc["reset_control", "nrmse"]), "scale_consistent_nrmse": float(g.loc["scale_consistent", "nrmse"]),
            "target_only_nrmse": float(g.loc["target_only_validation_selected", "nrmse"]), "n_targets": 1})
    paired = pd.DataFrame(rows)
    overall = paired.groupby(["percent", "seed"], as_index=False)[["reset_nrmse", "scale_consistent_nrmse", "target_only_nrmse"]].mean()
    overall["target"], overall["target_label"], overall["n_targets"] = "SEVEN_TARGET_MACRO", "Seven-target macro", 7
    def deltas(df):
        df = df.copy()
        df["scale_minus_reset_nrmse"] = df.scale_consistent_nrmse - df.reset_nrmse
        df["normalization_error_change_pct"] = 100 * df.scale_minus_reset_nrmse / df.reset_nrmse
        df["reset_gain_vs_target_only_pct"] = 100 * (df.target_only_nrmse - df.reset_nrmse) / df.target_only_nrmse
        df["scale_gain_vs_target_only_pct"] = 100 * (df.target_only_nrmse - df.scale_consistent_nrmse) / df.target_only_nrmse
        return df
    paired, overall = deltas(paired), deltas(overall)
    def summarize_seeds(df):
        rows = []
        for (target,label,percent), group in df.groupby(["target", "target_label", "percent"], sort=True):
            if len(group) != 3 or set(group.seed) != set(SEEDS):
                raise ValueError("Exactly the original three seeds are required.")
            row = {"target": target, "target_label": label, "percent": int(percent), "n_seeds": 3, "n_targets": int(group.n_targets.iloc[0])}
            for col in ("reset_nrmse", "scale_consistent_nrmse", "target_only_nrmse", "scale_minus_reset_nrmse"):
                row[col + "_mean"] = float(group[col].mean())
                row[col + "_sd"] = float(group[col].std(ddof=1))
            reset, scale, baseline = (row[k + "_mean"] for k in ("reset_nrmse", "scale_consistent_nrmse", "target_only_nrmse"))
            row["normalization_error_change_pct"] = 100 * (scale-reset) / reset
            row["reset_gain_vs_target_only_pct"] = 100 * (baseline-reset) / baseline
            row["scale_gain_vs_target_only_pct"] = 100 * (baseline-scale) / baseline
            rows.append(row)
        return pd.DataFrame(rows)
    return paired, overall, summarize_seeds(paired), summarize_seeds(overall)


def verify_attribute_coverage(attrs):
    if attrs.duplicated(["target", "percent", "seed", "arm", "attribute"]).any():
        raise ValueError("Duplicate attribute-level result.")
    for key, group in attrs.groupby(["target", "percent", "seed", "attribute"]):
        if set(group.arm) != set(ARMS):
            raise ValueError(f"Incomplete attribute-level arm coverage: {key}")
        for col in ("n_errors", "n_test_people", "n_folds_evaluated"):
            if group[col].nunique() != 1:
                raise ValueError(f"Attribute evaluation coverage mismatch: {key}/{col}")


def report_text(dataset, overall, audits, invariance):
    lines = ["# Experiment 3 — scale-consistent transfer: all seven datasets", "",
        "Complete: **210 scale-consistent fits and 210 matched reset-control fits** across all seven targets, "
        "10%/100% target training, five participant folds and three seeds. The target-only comparator is the "
        "frozen Experiment 2 validation-selected STraTS baseline for all seven datasets.", "",
        "The original five datasets were not retrained. The extension added **six source-pretraining fits, "
        "60 reset-control fine-tuning fits and 60 scale-consistent fine-tuning fits** for ABCD and YFS. "
        "Each new target/seed uses one shared source checkpoint for both arms and training fractions. "
        "Fifteen original source checkpoint identities are reused through their verified completed-fit receipts.", "",
        "## Primary matched comparison", "",
        "Scale consistency retains source-trained input/value, time and continuous-context normalization, "
        "and converts the output head to target-standardized coordinates. Target labels, loss weights, "
        "training participants, learning rate, stopping rule and per-stage seeds remain matched. "
        "Reset controls use the original Experiment 1 all-source training recipe, not historical paper scores.", "",
        "NRMSE is lower-is-better. Delta = scale-consistent minus reset; negative favors scale consistency. "
        "Means ± sample SD are across the three paired seeds. The seven-target macro weights datasets equally.", "",
        "| Target | Training | Reset NRMSE | Scale-consistent NRMSE | Paired delta | Target-only STraTS |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in pd.concat([dataset, overall], ignore_index=True).itertuples(index=False):
        def fmt(name):
            return f"{getattr(row,name+'_mean'):.4f} ± {getattr(row,name+'_sd'):.4f}"
        lines.append(f"| {row.target_label} | {row.percent}% | {fmt('reset_nrmse')} | {fmt('scale_consistent_nrmse')} | {fmt('scale_minus_reset_nrmse')} | {fmt('target_only_nrmse')} |")
    lines += ["", "## Seven-target interpretation", ""]
    for row in overall.itertuples(index=False):
        lines.append(f"At {row.percent}% training, scale consistency changes mean NRMSE by **{row.scale_minus_reset_nrmse_mean:+.5f}** "
            f"({row.normalization_error_change_pct:+.2f}% relative error change). Gains over validation-selected target-only STraTS "
            f"are {row.reset_gain_vs_target_only_pct:.2f}% for reset transfer and {row.scale_gain_vs_target_only_pct:.2f}% for scale-consistent transfer.")
        lines.append("")
    lines += ["## Integrity and diagnostics", "",
        f"All {len(invariance)} initial raw-prediction invariance checks passed. Maximum absolute float32 raw-unit "
        f"difference: {invariance.max_absolute_raw_difference.max():.8g}; acceptance is scale-aware.", "",
        f"Fits reaching the 1,000-epoch cap: scale-consistent {int(audits.hit_epoch_cap.sum())}/210; "
        f"reset control {int(audits.reset_hit_epoch_cap.sum())}/210.", "",
        "All 210 scale-consistent metrics and the 60 added reset metrics were recomputed from saved errors, "
        "restoring float32 error precision. The 150 original reset metrics use Experiment 3's hash-verified "
        "Experiment 1 receipt snapshots; their combined scores were checked against the frozen Experiment 1 report. "
        "The original five-target output bytes were verified unchanged since extension preparation. "
        "Every transfer pair has matching evaluation identity digests, target statistics and stage seeds. "
        "Attribute-level evaluation coverage also matches the frozen target-only baseline.", "",
        "## Scope and caveats", "",
        "This jointly tests input/time/context scaling and output-coordinate correction; it does not attribute "
        "effects to individual components. The head conversion preserves initial raw predictions, not AdamW "
        "trajectories or weight-decay invariance. No pooled, single-source or source-exclusion arm was rerun.", "",
        "The five-target analysis existed before ABCD and YFS were added. This is a coverage extension using "
        "the same fixed protocol, not a prospectively planned seven-target experiment. The target-only "
        "learning-rate selection rule was not changed or rerun. Test performance does not select an arm.", "",
        "Three-seed SDs describe initialization variability, not sampling confidence intervals or significance. "
        "Folds are not treated as independent replicates. The analysis does not resolve cross-dataset participant "
        "overlap and supports only the tested STraTS implementation and settings.", "",
        "## Files", "",
        "Start with `dataset_summary.csv`, `seven_target_summary.csv`, `paired_by_seed.csv`, and `fit_audit.csv`. "
        "All seven datasets occur together in the dataset and attribute tables. `results_to_share.zip` is the "
        "single combined export; it excludes participant-level errors/identifiers and model weights. "
        "The superseded five-target summary is retained only as a local archival backup.", ""]
    return "\n".join(lines)


def archive_previous_summary():
    archive = inside(ROOT / "_archive/before_seven_dataset_summary.zip")
    if archive.exists():
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + f".{os.getpid()}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT / "summary").glob("*")):
            if path.is_file() and not path.name.startswith("."):
                z.write(path, "summary/" + path.name)
        if (ROOT / "results_to_share.zip").is_file():
            z.write(ROOT / "results_to_share.zip", "results_to_share.zip")
    os.replace(temporary, archive)


def summarize():
    manifest, legacy, project = load_extension()
    missing = [str(stage_folder(j,a,p,f).relative_to(ROOT)) for j in extension_jobs()
               for a in TRAINING_ARMS for p in PERCENTS for f in range(5)
               if not (stage_folder(j,a,p,f) / "done.json").is_file()]
    if missing:
        raise RuntimeError(f"Seven-dataset report refused: {len(missing)}/120 added target fits missing. First: " + ", ".join(missing[:5]))
    with file_lock(ROOT / "summary/.summary.lock"):
        check_legacy_inventory(manifest)
        old, _, libraries = collect_legacy(legacy, project)
        if libraries != manifest["required_libraries"]:
            raise ValueError("Original software environment changed.")
        added = collect_added(manifest, project)
        combined = {k: old[k] + added[k] for k in old}
        scores = pd.DataFrame(combined["scores"])
        frozen_old = pd.read_csv(SETUP / "legacy_dataset_scores.csv", float_precision="round_trip")
        key = ["target", "percent", "seed", "arm"]
        old_scores = scores[scores.target.isin(LEGACY_TARGETS)].sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(old_scores, frozen_old.sort_values(key).reset_index(drop=True), check_exact=False, rtol=1e-12, atol=1e-12)
        baseline = pd.read_csv(ROOT / "reference/exp2_dataset_scores_by_seed.csv", float_precision="round_trip")
        baseline = baseline[baseline.policy.eq("validation_selected")].copy()
        baseline["arm"] = "target_only_validation_selected"
        scores = pd.concat([scores, baseline.drop(columns="policy")], ignore_index=True)
        attrs = pd.DataFrame(combined["attributes"])
        attr_baseline = pd.read_csv(ROOT / "reference/exp2_attribute_scores_by_seed.csv", float_precision="round_trip")
        attr_baseline = attr_baseline[attr_baseline.policy.eq("validation_selected")].copy()
        attr_baseline["arm"] = "target_only_validation_selected"
        attrs = pd.concat([attrs, attr_baseline.drop(columns="policy")], ignore_index=True)
        verify_attribute_coverage(attrs)
        paired, overall_seed, dataset_summary, overall_summary = build_tables(scores)
        audit, invariance = pd.DataFrame(combined["audit"]), pd.DataFrame(combined["invariance"])
        if len(audit) != 210 or len(invariance) != 210 or len(combined["checkpoints"]) != 21:
            raise ValueError("Incomplete seven-dataset audit matrix.")
        norm_audit = pd.concat([pd.read_csv(ROOT / "setup/normalization_audit.csv"),
                              pd.read_csv(SETUP / "normalization_audit.csv")], ignore_index=True)
        tables = {"dataset_scores_by_seed.csv": scores.sort_values(key),
            "attribute_scores_by_seed.csv": attrs.sort_values(key + ["attribute"]),
            "paired_by_seed.csv": paired, "seven_target_scores_by_seed.csv": overall_seed,
            "dataset_summary.csv": dataset_summary, "seven_target_summary.csv": overall_summary,
            "fit_audit.csv": audit, "invariance_audit.csv": invariance,
            "head_mapping_audit.csv": pd.DataFrame(combined["conversion"]),
            "recomputation_audit.csv": pd.DataFrame(combined["recomputation"]),
            "environment_by_fit.csv": pd.DataFrame(combined["environment"]),
            "training_history.csv.gz": pd.concat(combined["histories"], ignore_index=True),
            "added_source_training_history.csv.gz": pd.concat(combined["source_histories"], ignore_index=True),
            "source_checkpoints.csv": pd.DataFrame(combined["checkpoints"]),
            "normalization_audit.csv": norm_audit}
        provenance = {"version": VERSION, "complete": True, "targets": list(ALL_TARGETS), "n_targets": 7,
            "seeds": list(SEEDS), "percents": list(PERCENTS), "folds": 5,
            "scale_consistent_target_fits": 210, "reset_control_target_fits": 210,
            "preserved_scale_fits": 150, "reused_reset_fits": 150, "added_scale_fits": 60, "added_reset_fits": 60,
            "reused_source_checkpoints": 15, "new_source_training_fits": 6,
            "experiment_fingerprint": manifest["experiment_fingerprint"],
            "legacy_experiment_fingerprint": legacy["experiment_fingerprint"],
            "legacy_package_fingerprint": legacy["package_fingerprint"],
            "extension_code_fingerprint": manifest["extension_code_fingerprint"],
            "input_sha256": legacy["input_sha256"], "required_libraries": libraries,
            "baseline": "Frozen Experiment 2 validation-selected target-only STraTS, all seven datasets",
            "metric": "Person-balanced fold-reference-standardized error; combine person MSE across folds, sqrt per attribute, equal eligible-attribute mean, equal seven-dataset mean",
            "delta": "scale-consistent NRMSE minus reset-control NRMSE; negative favors scale consistency",
            "uncertainty": "Sample SD across three paired seeds; not independent folds, confidence intervals or significance tests",
            "original_output_bytes_unchanged": True, "original_reset_receipts_hash_verified": True,
            "original_reset_scores_match_frozen_experiment1": True, "scale_raw_metrics_recomputed": 210,
            "reset_raw_metrics_recomputed": 60, "all_initial_prediction_checks_passed": True,
            "source_reference": read_json(ROOT / "reference/provenance.json"),
            "limitations": ["ABCD/YFS were added after the five-target analysis; fixed-protocol coverage extension",
                "All-source only; not combined with participant-overlap exclusions", "Joint normalization policy, not component attribution",
                "Initial raw predictions preserved; optimizer trajectories need not be parameterization-invariant",
                "Original reset controls rely on previously verified frozen sufficient-statistic receipts",
                "Three seeds quantify initialization variation, not sampling uncertainty"]}
        with tempfile.TemporaryDirectory(prefix="seven_summary_", dir=inside(ROOT / "_runtime/tmp")) as temporary:
            staging = Path(temporary)
            for name, table in tables.items():
                atomic_csv(table, staging / name)
            (staging / "REPORT.md").write_text(report_text(dataset_summary, overall_summary, audit, invariance), encoding="utf-8")
            atomic_json(staging / "provenance.json", provenance)
            names = sorted(list(tables) + ["REPORT.md", "provenance.json"])
            atomic_json(staging / "file_sha256.json", {n: sha256(staging / n) for n in names})
            names.append("file_sha256.json")
            export = staging / "results_to_share.zip"
            with zipfile.ZipFile(export, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for name in names:
                    z.write(staging / name, "summary/" + name)
                for name in CODE_FILES + EXT_CODE + EXT_DOCS:
                    z.write(ROOT / name, "package/" + name)
                for path in sorted((ROOT / "reference").iterdir()):
                    if path.is_file():
                        z.write(path, "package/reference/" + path.name)
            archive_previous_summary()
            for name in names:
                os.replace(staging / name, inside(ROOT / "summary" / name))
            for obsolete in ("five_target_summary.csv", "five_target_scores_by_seed.csv"):
                (ROOT / "summary" / obsolete).unlink(missing_ok=True)
            os.replace(export, inside(ROOT / "results_to_share.zip"))
        print("COMPLETE: one seven-dataset summary/REPORT.md and results_to_share.zip. Original training outputs are unchanged.")
