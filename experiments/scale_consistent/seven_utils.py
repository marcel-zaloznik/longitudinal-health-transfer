"""Additive seven-dataset extension; never edits the original experiment code/setup."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from exp3_utils import (
    ROOT, SEEDS, PERCENTS, TARGET_LABELS as LEGACY_TARGETS, CODE_FILES,
    inside, read_json, atomic_json, sha256, object_hash, load_setup,
    verify_inputs, find_project_root, jobs as legacy_jobs,
    receipt_folder, control_folder, assert_statistics_equal,
)
VERSION = "1.0.0-seven-dataset-extension"
NEW_TARGETS = {"ABCD": "ABCD", "YFS": "YFS"}
ALL_TARGETS = {name: ({**LEGACY_TARGETS, **NEW_TARGETS})[name]
               for name in sorted({**LEGACY_TARGETS, **NEW_TARGETS})}
ARMS = ("reset_control", "scale_consistent", "target_only_validation_selected")
TRAINING_ARMS = ARMS[:2]
SETUP = ROOT / "setup/seven_dataset_extension"
OUTPUT = ROOT / "output/seven_dataset_extension"
SMOKE = ROOT / "smoke_output/seven_dataset_extension"
EXT_CODE = ("experiment7.py", "seven_utils.py", "seven_train.py", "seven_analyse.py", "tests7.py")
EXT_DOCS = ("PROTOCOL.md",)
LIBRARY_KEYS = ("torch", "numpy", "pandas", "cuda_build", "device_type")


def extension_jobs() -> list[dict]:
    # Alternate targets: with six groups each GPU gets one target/seed checkpoint.
    return [{"extension_job_id": i, "target": t, "target_label": NEW_TARGETS[t], "seed": s}
            for i, (s, t) in enumerate((s, t) for s in SEEDS for t in NEW_TARGETS)]


def group_jobs(groups: int, group: int) -> list[dict]:
    if not 1 <= groups <= 6 or not 0 <= group < groups:
        raise ValueError("Use --groups 1..6 and --group 0..groups-1.")
    return [j for j in extension_jobs() if j["extension_job_id"] % groups == group]


def comparison_job(job: dict, percent: int) -> dict:
    if percent not in PERCENTS:
        raise ValueError("Only the original 10% and 100% training fractions are allowed.")
    index = list(NEW_TARGETS).index(job["target"])
    number = 30 + index * 6 + list(PERCENTS).index(percent) * 3 + list(SEEDS).index(job["seed"])
    return {"job_id": number, **job, "percent": percent}


def job_folder(job: dict, smoke: bool = False) -> Path:
    return inside((SMOKE if smoke else OUTPUT) / "jobs" / f"job_{job['extension_job_id']:03d}")


def stage_folder(job: dict, arm: str, percent: int, fold: int, smoke: bool = False) -> Path:
    if arm not in TRAINING_ARMS:
        raise ValueError("Unknown transfer arm.")
    return job_folder(job, smoke) / arm / f"pct{percent}" / f"fold{fold}"


def normalizer_path(target: str) -> Path:
    if target not in NEW_TARGETS:
        raise ValueError("New source normalization is only prepared for ABCD and YFS.")
    return SETUP / "normalizers" / (target + ".json")


def extension_fingerprint() -> str:
    return object_hash({name: sha256(ROOT / name) for name in EXT_CODE})


def load_legacy(project_arg: str | None = None, check_data: bool = True):
    setup, project = load_setup(check_data=False)
    if project_arg:
        project = Path(project_arg).expanduser().resolve()
    elif not (project / "data/dataset.csv").is_file():
        project = find_project_root()
    if check_data:
        verify_inputs(project)
    if setup["targets"] != list(LEGACY_TARGETS) or setup["jobs"] != legacy_jobs():
        raise RuntimeError("This extension requires the original completed five-target Experiment 3.")
    return setup, project


def load_extension(check_data: bool = True):
    path = SETUP / "manifest.json"
    if not path.is_file():
        raise RuntimeError("First run: python -B experiment7.py prepare")
    manifest = read_json(path)
    if manifest["extension_code_fingerprint"] != extension_fingerprint():
        raise RuntimeError("Extension code changed after preparation. Restore the matching extension files; do not mix protocols.")
    if sha256(ROOT / "setup/manifest.json") != manifest["legacy_manifest_sha256"]:
        raise RuntimeError("Original Experiment 3 setup changed after extension preparation.")
    legacy, project = load_legacy(manifest["project_root"], check_data=check_data)
    if legacy["experiment_fingerprint"] != manifest["legacy_experiment_fingerprint"]:
        raise RuntimeError("Original experiment identity changed.")
    for name, digest in manifest["setup_sha256"].items():
        path = inside(ROOT / name)
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Extension preparation file changed: {name}")
    return manifest, legacy, project


def check_legacy_inventory(manifest: dict) -> None:
    inventory = read_json(SETUP / "legacy_output_sha256.json")
    for name, digest in inventory.items():
        path = inside(ROOT / name)
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"A previously completed five-target result changed: {name}")


def checked_metric(errors, project, target: str, fold: int, expected: dict | None):
    from metrics import fold_metrics
    # The frozen metric's strict loader only knows the original five targets.
    # Perform exactly the same count check against the additive seven-target panel.
    result = fold_metrics(errors, project, target, fold, strict=False)
    if expected is not None:
        observed = {a: v["n_errors"] for a, v in result["attributes"].items()}
        if observed != expected[target][str(fold)]:
            raise ValueError(f"Incomplete primary-panel coverage: {target}, fold {fold}")
    return result


def verify_baseline_coverage(target: str, counts: dict, people: int, baseline):
    selected = baseline[baseline.target.eq(target) & baseline.policy.eq("validation_selected")]
    if len(selected) != 6 or selected.duplicated(["percent", "seed"]).any():
        raise ValueError(f"Frozen validation-selected baseline incomplete: {target}")
    if set(selected[["percent", "seed"]].itertuples(index=False, name=None)) != {(p,s) for p in PERCENTS for s in SEEDS}:
        raise ValueError("Unexpected baseline training fractions/seeds.")
    errors = sum(sum(x.values()) for x in counts.values())
    attrs = len({a for x in counts.values() for a in x})
    for key, expected in (("n_primary_errors", errors), ("n_test_people", people), ("n_attributes", attrs), ("n_folds", 5)):
        if not selected[key].eq(expected).all():
            raise ValueError(f"Baseline evaluation coverage mismatch: {target}/{key}")


def validate_history(path, details):
    import numpy as np
    import pandas as pd
    history = pd.read_csv(path, float_precision="round_trip")
    if len(history) != details["epochs_run"] or not np.isfinite(history[["train_mse", "validation_mse"]].to_numpy()).all():
        raise ValueError(f"Invalid/nonfinite training history: {path}")
    if history.epoch.tolist() != list(range(1, len(history) + 1)):
        raise ValueError("Training epochs are not consecutive.")
    best, epoch = float("inf"), 0
    for row in history.itertuples(index=False):
        if row.validation_mse + 1e-5 < best:
            best, epoch = float(row.validation_mse), int(row.epoch)
    if epoch != details["best_epoch"] or not np.isclose(best, details["selected_validation_mse"], rtol=1e-12, atol=1e-12):
        raise ValueError("Selected epoch does not follow the original validation/min-delta rule.")
    if sha256(path) != details["history_sha256"]:
        raise ValueError("Training-history digest differs from its training receipt.")
    return history
