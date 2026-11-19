#!/usr/bin/env python3
"""Build local reference tables for the revision experiments from completed runs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    "dataset.csv", "input_output_pairs.csv", "split5k.json",
    "target_train_subsets_by_dataset_fold_percent.json",
    "target_attribute_stds_by_dataset_fold.json",
    "eligible_dataset_attribute_pairs.json", "primary_target_panel.json",
)
TARGETS = ("ABCD", "BE_KUL_Wave_1A", "BE_KUL_Wave_1B", "BE_KUL_Wave_23456",
           "SLOfit_lifelong", "SLOfit_students", "YFS")
SEEDS = (1234, 2345, 3456)
PERCENTS = (10, 100)
STRATS_CONFIGS = ("strats_main", "strats_no_pretrain", "strats_pooled_all_sources")
EXP1_FILES = ("engine.py", "experiment.py", "exp1_utils.py", "analyse.py", "exclusions.json",
              "reference/original_metrics.csv", "reference/provenance.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def input_hashes(root: Path) -> dict[str, str]:
    return {name: sha256(root / "data" / name) for name in INPUTS}


def main_metrics(root: Path) -> tuple[pd.DataFrame, Path]:
    path = root / "results/metrics/dataset_metrics.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    selected = frame.loc[frame["panel"].eq("primary_9") & frame["horizon_group"].eq("all")
                         & frame["train_data_percent"].isin(PERCENTS)].copy()
    return selected, path


def check_grid(frame: pd.DataFrame, expected: set[tuple]) -> None:
    keys = ["base_configuration_id", "dataset", "train_data_percent", "seed"]
    actual = set(frame[keys].itertuples(index=False, name=None))
    if frame.duplicated(keys).any() or actual != expected:
        raise ValueError(f"Incomplete or duplicate reference grid: {len(expected - actual)} missing, "
                         f"{len(actual - expected)} unexpected combinations.")
    scores = frame[["main_macro_nrmse", "main_macro_nmae"]].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Nonfinite or negative reference scores.")
    if not frame["n_folds_evaluated"].eq(5).all():
        raise ValueError("Every reference must cover all five folds.")
    if not np.allclose(frame["evaluation_cell_coverage"], 1.0, rtol=0, atol=1e-12):
        raise ValueError("Reference evaluation-cell coverage is incomplete.")


def prepare_main(root: Path, folder: str, output: Path, hashes: dict) -> dict:
    metrics, source = main_metrics(root)
    targets = TARGETS if folder == "learning_rate" else TARGETS[1:-1]
    selected = metrics.loc[metrics["base_configuration_id"].isin(STRATS_CONFIGS)
                           & metrics["dataset"].isin(targets) & metrics["seed"].isin(SEEDS)].copy()
    expected = {(model, target, percent, seed) for model in STRATS_CONFIGS
                for target in targets for percent in PERCENTS for seed in SEEDS}
    check_grid(selected, expected)
    columns = ["base_configuration_id", "dataset", "seed", "train_data_percent", "panel", "horizon_group",
               "main_macro_nrmse", "main_macro_nmae", "n_attributes_evaluated", "n_error_rows",
               "n_test_people", "n_folds_evaluated", "evaluation_cell_coverage"]
    if folder == "source_exclusion":
        columns = ["base_configuration_id", "method", "seed", "train_data_percent", "dataset", *columns[4:]]
    selected[columns].to_csv(output / "original_metrics.csv", index=False)
    provenance = {"input_sha256": hashes, "reference_source": "results/metrics/dataset_metrics.csv",
                  "reference_source_sha256": sha256(source), "reference_rows": len(selected),
                  "reference_subset_sha256": sha256(output / "original_metrics.csv"),
                  "original_model_source": "strats.py", "original_model_sha256": sha256(root / "strats.py")}
    if folder == "learning_rate":
        original_source = (root / "strats.py").read_text(encoding="utf-8")
        original_tree = ast.parse(original_source)
        original = {n.name: n for n in original_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        engine_tree = ast.parse((root / "experiments/learning_rate/engine.py").read_text(encoding="utf-8"))
        engine = {n.name: n for n in engine_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        fragment = ast.get_source_segment(original_source, original["train_model"]) + "\n"
        (output / "original_train_model.py.txt").write_text(fragment, encoding="utf-8")
        unchanged = {name: hashlib.sha256(ast.dump(node).encode()).hexdigest()
                     for name, node in original.items() if name in engine
                     and ast.dump(node) == ast.dump(engine[name])}
        if not unchanged:
            raise ValueError("No matching STraTS definitions were found.")
        provenance.update(original_train_model_sha256=sha256(output / "original_train_model.py.txt"),
                          unchanged_ast_sha256=unchanged)
    return provenance


def prepare_scale(root: Path, output: Path, hashes: dict) -> dict:
    first = root / "experiments/source_exclusion"
    second = root / "experiments/learning_rate"
    setup = read_json(first / "setup/manifest.json")
    if setup["input_sha256"] != hashes:
        raise ValueError("Source-exclusion inputs differ from the current inputs.")
    fingerprint = object_hash({name: sha256(first / name) for name in EXP1_FILES})
    if setup["package_fingerprint"] != fingerprint:
        raise ValueError("Source-exclusion code or references changed after preparation.")
    if read_json(second / "summary/provenance.json")["input_sha256"] != hashes:
        raise ValueError("Learning-rate inputs differ from the current inputs.")
    copies = {
        "exp1_attribute_scores.csv": first / "output/summary/attribute_scores.csv",
        "exp1_dataset_scores.csv": first / "output/summary/dataset_scores.csv",
        "exp1_report_manifest.json": first / "output/summary/report_manifest.json",
        "exp1_source_counts.csv": first / "setup/source_counts.csv",
        "exp1_stage_audit.csv": first / "output/summary/stage_audit.csv",
        "exp1_target_partitions.csv": first / "setup/target_partitions.csv",
    }
    for name in ("attribute_scores_by_seed.csv", "dataset_scores_by_seed.csv", "file_sha256.json",
                 "precision_fix.json", "provenance.json", "selected_rates.csv", "selection_provenance.json"):
        copies["exp2_" + name] = second / "summary" / name
    for name, source in copies.items():
        shutil.copyfile(source, output / name)
    if sha256(first / "engine.py") != sha256(root / "experiments/scale_consistent/engine.py"):
        raise ValueError("The source-exclusion and scale-consistent engines differ.")
    return {"input_sha256": hashes, "exp1_experiment_fingerprint": setup["experiment_fingerprint"],
            "exp1_package_fingerprint": fingerprint, "exp1_package_names": list(EXP1_FILES),
            "exp1_engine_sha256": sha256(first / "engine.py"),
            "exp2_baseline": "validation_selected; selection uses validation, not test scores",
            "reference_sha256": {name: sha256(output / name) for name in sorted(copies)}}


def prepare_grud(root: Path, output: Path, hashes: dict) -> dict:
    scale = root / "experiments/scale_consistent/summary"
    # The seven-target reporter, not the earlier five-target reporter, supplies this table.
    if not scale.is_dir():
        raise FileNotFoundError("Run scale_consistent/experiment7.py summarize first.")
    source = scale / "dataset_scores_by_seed.csv"
    reference = pd.read_csv(source, float_precision="round_trip")
    keys = ["target", "percent", "seed", "arm"]
    arms = ("reset_control", "scale_consistent", "target_only_validation_selected")
    expected = {(target, percent, seed, arm) for target in TARGETS
                for percent in PERCENTS for seed in SEEDS for arm in arms}
    if reference.duplicated(keys).any() or set(reference[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("The scale-consistent reference must contain all seven targets and three arms.")
    provenance = read_json(scale / "provenance.json")
    if provenance.get("input_sha256") != hashes:
        raise ValueError("Scale-consistent inputs differ from the current inputs.")
    shutil.copyfile(source, output / "experiment3_dataset_scores.csv")
    metrics, source_metrics = main_metrics(root)
    configs = ("gru_d_target_only", "random_forest_fixed_target_only")
    historical = metrics.loc[metrics["base_configuration_id"].isin(configs)].copy()
    expected = {(model, target, percent, seed) for model in configs for target in TARGETS
                for percent in PERCENTS for seed in (SEEDS if model == configs[0] else (1234,))}
    check_grid(historical, expected)
    columns = ["dataset", "seed", "method", "train_data_percent", "main_macro_nrmse", "main_macro_nmae",
               "n_attributes_evaluated", "n_error_rows", "n_test_people", "n_folds_evaluated"]
    historical[columns].to_csv(output / "historical_target_only_scores.csv", index=False)
    return {"input_sha256": hashes, "original_grud_sha256": sha256(root / "grud.py"),
            "references": {"historical_dataset_metrics_sha256": sha256(source_metrics),
                           "scale_consistent_dataset_scores_sha256": sha256(source)},
            "frozen_reference_files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("source_exclusion", "learning_rate", "scale_consistent", "grud_transfer"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    folder = root / "experiments" / args.experiment
    if not (folder / "experiment.py").is_file():
        raise FileNotFoundError(f"Missing experiment scripts: {folder}")
    destination = folder / "reference"
    if destination.exists() or (folder / "setup/manifest.json").exists():
        raise FileExistsError("References or setup already exist. Use a fresh experiment directory; do not mix runs.")
    hashes = input_hashes(root)
    with tempfile.TemporaryDirectory(prefix=".reference-", dir=folder) as temporary:
        output = Path(temporary)
        if args.experiment in ("source_exclusion", "learning_rate"):
            provenance = prepare_main(root, args.experiment, output, hashes)
        elif args.experiment == "scale_consistent":
            provenance = prepare_scale(root, output, hashes)
        else:
            provenance = prepare_grud(root, output, hashes)
        write_json(output / "provenance.json", provenance)
        shutil.copytree(output, destination)
    print(f"Wrote {destination}. These local references are not public release files.")


if __name__ == "__main__":
    main()
