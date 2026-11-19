"""Primary-panel metrics and matched comparisons; no patient-level export in share ZIP."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import zipfile

from exp1_utils import (
    ROOT, SEEDS, PERCENTS, ARMS, TARGET_LABELS, inside, read_json,
    atomic_json, atomic_csv, sha256, object_hash, load_setup, jobs,
)


def identity_digest(errors) -> str:
    import numpy as np
    import pandas as pd
    columns = ["dataset", "fold", "person_id", "attribute", "input_age", "output_age", "horizon", "n_input_visits"]
    identity = errors[columns].copy()
    for col in ("dataset", "person_id", "attribute"):
        identity[col] = identity[col].astype(str)
    for col in ("fold", "n_input_visits"):
        identity[col] = identity[col].astype("int64")
    for col in ("input_age", "output_age", "horizon"):
        identity[col] = identity[col].astype("float64")
    values = pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype="uint64")
    return hashlib.sha256(np.sort(values).tobytes()).hexdigest()


def fold_metrics(errors, project: Path, target: str, fold: int, strict: bool = True) -> dict:
    """Sufficient statistics for the submitted person-balanced NRMSE.

    Crucially, combine PERSON mean-squared errors across folds before taking
    the attribute square root; never average fold RMSEs.
    """
    import numpy as np
    import pandas as pd
    if errors.empty or not errors.dataset.eq(target).all() or not errors.fold.eq(fold).all():
        raise ValueError("Raw errors contain unexpected/empty target or fold metadata.")
    if not np.isfinite(errors.error.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite raw errors.")
    if errors.person_id.isna().any():
        raise ValueError("Missing participant identifier in errors.")
    stds = read_json(project / "data/target_attribute_stds_by_dataset_fold.json")[target][str(fold)]
    eligibility = read_json(project / "data/eligible_dataset_attribute_pairs.json")[target][str(fold)]
    primary = read_json(project / "data/primary_target_panel.json")
    selected = [attribute for attribute in primary if eligibility[attribute]["eligible"]]
    selected_errors = errors[errors.attribute.isin(selected)].copy()
    if selected_errors.empty:
        raise ValueError("No evaluable primary-panel errors.")
    attribute_stds = selected_errors.attribute.map(stds).to_numpy(dtype=float)
    if not (np.isfinite(attribute_stds) & (attribute_stds > 0)).all():
        raise ValueError("An eligible evaluation cell has no positive finite reference standard deviation.")
    normalized = selected_errors.error.to_numpy(dtype=float) / attribute_stds
    selected_errors["squared"] = normalized ** 2
    selected_errors["absolute"] = np.abs(normalized)
    if not np.isfinite(selected_errors["squared"]).all():
        raise FloatingPointError("Normalized squared errors overflowed.")
    if strict:
        expected = read_json(ROOT / "setup/expected_test_counts.json")[target][str(fold)]
        observed = {str(k): int(v) for k, v in selected_errors.groupby("attribute").size().items()}
        if observed != expected:
            raise ValueError(f"Incomplete/changed evaluation observations for {target}, fold {fold}: expected {expected}; observed {observed}")
    people = selected_errors.groupby(["person_id", "attribute"], sort=False).agg(
        person_mse=("squared", "mean"), person_mae=("absolute", "mean"), n_errors=("error", "size"))
    attributes = {}
    for attribute, group in people.groupby(level="attribute", sort=False):
        attributes[str(attribute)] = {
            "sum_person_normalized_mse": float(group.person_mse.sum()),
            "sum_person_normalized_mae": float(group.person_mae.sum()),
            "n_people": len(group), "n_errors": int(group.n_errors.sum()),
        }
    return {"target": target, "fold": fold, "panel": "primary_9", "horizon_group": "all",
            "identity_digest": identity_digest(errors), "n_raw_errors": len(errors),
            "n_primary_errors": len(selected_errors), "n_test_people": int(selected_errors.person_id.nunique()),
            "attributes": attributes}


def combine_folds(folds: list[dict]) -> tuple[dict, list[dict]]:
    import numpy as np
    attributes = sorted({attribute for item in folds for attribute in item["attributes"]})
    rows = []
    for attribute in attributes:
        values = [item["attributes"][attribute] for item in folds if attribute in item["attributes"]]
        n_people = sum(value["n_people"] for value in values)
        mse = sum(value["sum_person_normalized_mse"] for value in values) / n_people
        mae = sum(value["sum_person_normalized_mae"] for value in values) / n_people
        rows.append({"attribute": attribute, "nrmse": math.sqrt(mse), "nmae": mae,
                     "n_test_people": n_people, "n_errors": sum(value["n_errors"] for value in values),
                     "n_folds_evaluated": len(values)})
    if not rows:
        raise ValueError("No attribute scores to combine.")
    result = {"nrmse": float(np.mean([row["nrmse"] for row in rows])),
              "nmae": float(np.mean([row["nmae"] for row in rows])),
              "n_attributes": len(rows), "n_primary_errors": sum(item["n_primary_errors"] for item in folds),
              "n_test_people": sum(item["n_test_people"] for item in folds), "n_folds": len(folds)}
    return result, rows


def compare_reference(scores, reference):
    import pandas as pd
    base = reference.set_index(["dataset", "train_data_percent", "seed", "base_configuration_id"])
    records = []
    for (target, percent, mode, seed), group in scores.groupby(["target", "percent", "mode", "seed"], sort=True):
        if set(group.arm) != set(ARMS) or len(group) != 2:
            raise ValueError("A matched comparison is missing one arm.")
        group = group.set_index("arm")
        baseline_row = base.loc[(target, percent, seed, "strats_no_pretrain")]
        config = "strats_main" if mode == "pretrain" else "strats_pooled_all_sources"
        original_row = base.loc[(target, percent, seed, config)]
        for arm in ARMS:
            if int(group.loc[arm, "n_primary_errors"]) != int(baseline_row.n_error_rows):
                raise ValueError("New/historical evaluated error counts differ.")
            if int(group.loc[arm, "n_test_people"]) != int(baseline_row.n_test_people):
                raise ValueError("New/historical evaluated participant counts differ.")
            if int(group.loc[arm, "n_attributes"]) != int(baseline_row.n_attributes_evaluated):
                raise ValueError("New/historical evaluation panels differ.")
        baseline = float(baseline_row.main_macro_nrmse)
        original = float(original_row.main_macro_nrmse)
        control = float(group.loc["all_source", "nrmse"])
        excluded = float(group.loc["excluded", "nrmse"])
        if baseline <= 0 or control <= 0:
            raise ValueError("Cannot form relative gains from nonpositive baseline/control errors.")
        records.append({"target": target, "target_label": TARGET_LABELS[target], "percent": int(percent),
            "mode": mode, "seed": int(seed), "target_only_nrmse": baseline,
            "historical_all_source_nrmse": original, "matched_all_source_nrmse": control,
            "excluded_nrmse": excluded,
            "excluded_minus_matched_nrmse": excluded - control,
            "matched_minus_historical_nrmse": control - original,
            "historical_all_source_gain_pct": 100 * (baseline - original) / baseline,
            "matched_all_source_gain_pct": 100 * (baseline - control) / baseline,
            "excluded_gain_pct": 100 * (baseline - excluded) / baseline,
            "exclusion_error_change_pct": 100 * (excluded - control) / control})
    return pd.DataFrame(records)


def summarize_seed_rows(paired, keys):
    import pandas as pd
    output = []
    quantities = ["target_only_nrmse", "historical_all_source_nrmse", "matched_all_source_nrmse", "excluded_nrmse",
                  "excluded_minus_matched_nrmse", "matched_minus_historical_nrmse"]
    for key, group in paired.groupby(keys, sort=True):
        if len(group) != 3 or set(group.seed) != set(SEEDS):
            raise ValueError("A final comparison requires all three specified seeds.")
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["n_seeds"] = 3
        for quantity in quantities:
            row[quantity + "_mean"] = float(group[quantity].mean())
            row[quantity + "_sd"] = float(group[quantity].std(ddof=1))
        baseline = row["target_only_nrmse_mean"]
        control = row["matched_all_source_nrmse_mean"]
        excluded = row["excluded_nrmse_mean"]
        row["historical_all_source_gain_pct"] = 100 * (baseline - row["historical_all_source_nrmse_mean"]) / baseline
        row["matched_all_source_gain_pct"] = 100 * (baseline - control) / baseline
        row["excluded_gain_pct"] = 100 * (baseline - excluded) / baseline
        row["exclusion_error_change_pct"] = 100 * (excluded - control) / control
        row["exclusion_gain_change_percentage_points"] = row["excluded_gain_pct"] - row["matched_all_source_gain_pct"]
        output.append(row)
    return pd.DataFrame(output)


def summarize(recompute: bool = False) -> None:
    import numpy as np
    import pandas as pd
    setup, project = load_setup()
    scores, attribute_scores, stage_audits, environments = [], [], [], []
    missing = []
    for job in jobs():
        folder = ROOT / "output/jobs" / f"job_{job['job_id']:03d}"
        for arm in ARMS:
            for percent in PERCENTS:
                for fold in range(5):
                    if not (folder / arm / f"pct{percent}" / f"fold{fold}" / "done.json").exists():
                        missing.append(f"job {job['job_id']}, {arm}, {percent}%, fold {fold}")
    if missing:
        raise RuntimeError(f"Cannot issue a complete report: {len(missing)}/600 target fits missing. First missing: {missing[0]}. Use experiment.py status.")
    for job in jobs():
        folder = ROOT / "output/jobs" / f"job_{job['job_id']:03d}"
        config = read_json(folder / "config.json")
        if config["smoke"] or config["experiment_fingerprint"] != setup["experiment_fingerprint"] or config["job"] != job:
            raise ValueError("Research output has incompatible provenance.")
        config_hash = object_hash(config)
        environment = read_json(folder / "environment.json")
        environments.append({"job_id": job["job_id"], **environment, "num_workers": config["num_workers"]})
        for arm in ARMS:
            if job["mode"] == "pretrain":
                source_stage = folder / arm / "pretrain"
                done = read_json(source_stage / "done.json")
                if done["config_hash"] != config_hash:
                    raise ValueError("Pretraining configuration mismatch.")
                for filename, expected_hash in done["file_sha256"].items():
                    if sha256(source_stage / filename) != expected_hash:
                        raise ValueError(f"Corrupt source checkpoint: {source_stage / filename}")
            for percent in PERCENTS:
                collected = []
                for fold in range(5):
                    stage = folder / arm / f"pct{percent}" / f"fold{fold}"
                    done = read_json(stage / "done.json")
                    if done["config_hash"] != config_hash:
                        raise ValueError("Completed stage configuration mismatch.")
                    for filename, expected_hash in done["file_sha256"].items():
                        if sha256(stage / filename) != expected_hash:
                            raise ValueError(f"Corrupt completed output: {stage / filename}")
                    metrics = read_json(stage / "metrics.json")
                    if recompute:
                        errors = pd.read_csv(stage / "errors.csv.gz", dtype={"person_id": str}, float_precision="round_trip")
                        reconstructed = fold_metrics(errors, project, job["target"], fold)
                        if reconstructed["identity_digest"] != metrics["identity_digest"]:
                            raise ValueError("Raw error identities differ from their saved audit.")
                        for attribute, values in metrics["attributes"].items():
                            for name, value in values.items():
                                if not np.isclose(reconstructed["attributes"][attribute][name], value, rtol=1e-12, atol=1e-12):
                                    raise ValueError("Reconstructed error metrics differ from cached sufficient statistics.")
                        metrics = reconstructed
                    control_metrics = read_json(folder / "all_source" / f"pct{percent}" / f"fold{fold}" / "metrics.json")
                    if metrics["identity_digest"] != control_metrics["identity_digest"]:
                        raise ValueError("Matched arms have different test observations.")
                    collected.append(metrics)
                    stage_audits.append({"job_id": job["job_id"], "target": job["target"], "mode": job["mode"],
                        "seed": job["seed"], "arm": arm, "percent": percent, "fold": fold,
                        "sources": "|".join(done["sources"]), "stage_seed": done["stage_seed"],
                        "train_people": done["target_train"]["n_people"], "val_people": done["target_val"]["n_people"],
                        "test_people": done["target_test"]["n_people"], "actual_train_people": done["actual_train"]["n_people"],
                        "train_records": done["n_train_records"], "test_records": done["n_test_records"],
                        "n_raw_errors": done["n_raw_errors"], "epochs_run": done["training"]["epochs_run"],
                        "best_epoch": done["training"]["best_epoch"], "selected_validation_mse": done["training"]["selected_validation_mse"],
                        "elapsed_seconds": done["training"]["elapsed_seconds"]})
                result, attributes = combine_folds(collected)
                meta = {"job_id": job["job_id"], "target": job["target"], "target_label": job["target_label"],
                        "mode": job["mode"], "seed": job["seed"], "arm": arm, "percent": percent}
                scores.append({**meta, **result})
                attribute_scores.extend({**meta, **row} for row in attributes)
    scores = pd.DataFrame(scores)
    reference = pd.read_csv(ROOT / "reference/original_metrics.csv")
    paired = compare_reference(scores, reference)
    summary = summarize_seed_rows(paired, ["target", "target_label", "percent", "mode"])
    value_columns = ["target_only_nrmse", "historical_all_source_nrmse", "matched_all_source_nrmse", "excluded_nrmse",
                     "excluded_minus_matched_nrmse", "matched_minus_historical_nrmse"]
    # Each of the FIVE affected datasets receives equal weight. These are not seven-cohort scores.
    affected_seed = paired.groupby(["percent", "mode", "seed"], as_index=False)[value_columns].mean()
    affected_summary = summarize_seed_rows(affected_seed, ["percent", "mode"])
    affected_summary.insert(0, "scope", "five_affected_targets_only")
    out = inside(ROOT / "output/summary")
    tables = {
        "comparison_by_target.csv": summary, "comparison_by_seed.csv": paired,
        "affected_targets_summary.csv": affected_summary, "dataset_scores.csv": scores,
        "attribute_scores.csv": pd.DataFrame(attribute_scores), "stage_audit.csv": pd.DataFrame(stage_audits),
        "environments.csv": pd.DataFrame(environments),
    }
    for name, table in tables.items():
        atomic_csv(table, out / name)
    atomic_json(out / "report_manifest.json", {
        "complete": True, "experiment_fingerprint": setup["experiment_fingerprint"],
        "jobs": 30, "target_fits": 600, "source_pretraining_fits": 30,
        "primary_panel": setup["primary_panel"], "seeds": list(SEEDS), "exclusions": setup["exclusions"],
        "historical_reference": read_json(ROOT / "reference/provenance.json"),
        "metric": "Within each person and eligible attribute, mean squared fold-reference-standardized error; then average persons across folds, square root, average attributes; affected-target summary averages five datasets equally.",
        "gain_definition": "100*(mean target-only NRMSE - mean model NRMSE)/mean target-only NRMSE",
        "delta_definition": "excluded NRMSE minus newly rerun matched all-source NRMSE; positive means exclusion worsened error",
        "uncertainty": "Sample standard deviation across three seeds; descriptive, not a confidence interval or significance test.",
        "raw_errors_recomputed": recompute, "limitations": setup["limitations"],
        "files": {name: sha256(out / name) for name in tables},
    })
    note = """# Experiment 1 results

Read `comparison_by_target.csv` first. Lower NRMSE is better. Positive
`excluded_minus_matched_nrmse_mean` means that excluding suspected overlapping
sources worsened prediction. `excluded_gain_pct` compares the exclusion arm
with the submitted target-only STraTS baseline. Gain percentages use ratios
of seed-mean errors; SDs describe three initialization seeds, not population
uncertainty.

`historical_all_source_nrmse_mean` is the submitted result;
`matched_all_source_nrmse_mean` is the new, stage-seeded control. The latter
is the primary comparator for the source-exclusion effect. The existing
target-only baseline was not retrained.

`affected_targets_summary.csv` covers only the five affected targets, not
all seven datasets. Source-size and source-composition changes accompany
exclusion, so deterioration or improvement does not establish participant
leakage. The analysis is not participant-level deduplication.

The ZIP contains aggregate scores and audit metadata only. Raw errors,
participant identifiers, target-subset lists and model weights remain in
the experiment folder and are not included.
"""
    (out / "RESULTS_README.md").write_text(note, encoding="utf-8")
    share = inside(ROOT / "results_to_share.zip")
    temporary = share.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.is_file():
                archive.write(path, "summary/" + path.name)
        for name in ("source_counts.csv", "target_partitions.csv", "validation.json", "jobs.csv"):
            archive.write(ROOT / "setup" / name, "audit/" + name)
        archive.write(ROOT / "PROTOCOL.md", "PROTOCOL.md")
        archive.write(ROOT / "exclusions.json", "exclusions.json")
    temporary.replace(share)
    print(f"Complete report: {out}")
    print(f"Local results archive: {share}")
    print("The ZIP contains aggregate results, not participant-level errors.")


if __name__ == "__main__":
    raise SystemExit("Use: python experiment.py summarize")
