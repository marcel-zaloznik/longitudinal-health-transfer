#!/usr/bin/env python3
"""Recompute source-exclusion summaries with the original float32 error precision."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import contextlib
import math
from pathlib import Path
import traceback

import exp1_utils as utils
utils.configure_local_runtime()

import numpy as np
import pandas as pd
import analyse

FIX_VERSION = "1.0.0"
RTOL = 1e-12
ATOL = 1e-12
STAT_FIELDS = {"sum_person_normalized_mse", "sum_person_normalized_mae", "n_people", "n_errors"}
COUNT_FIELDS = {"n_people", "n_errors"}


def differences(cached: dict, rebuilt: dict) -> list[dict]:
    """Return discrepancies, preserving the original floating-point tolerance.

    Counts and metadata must match exactly, rather than approximately.
    No participant identifiers or raw errors are included in diagnostics.
    """
    issues = []
    for key in ("target", "fold", "panel", "horizon_group", "identity_digest",
                "n_raw_errors", "n_primary_errors", "n_test_people"):
        if cached.get(key) != rebuilt.get(key):
            issues.append({"field": key, "reason": "metadata/identity/count mismatch"})
    left, right = cached["attributes"], rebuilt["attributes"]
    if set(left) != set(right):
        issues.append({"field": "attributes", "reason": "evaluation-panel mismatch"})
    for attribute in sorted(set(left) & set(right)):
        if set(left[attribute]) != STAT_FIELDS or set(right[attribute]) != STAT_FIELDS:
            issues.append({"attribute": attribute, "reason": "unexpected sufficient-statistic fields"})
            continue
        for key in sorted(STAT_FIELDS):
            a, b = left[attribute][key], right[attribute][key]
            if key in COUNT_FIELDS:
                matches = type(a) is int and type(b) is int and a == b
            else:
                matches = math.isfinite(a) and math.isfinite(b) and bool(np.isclose(b, a, rtol=RTOL, atol=ATOL))
            if not matches:
                item = {"attribute": attribute, "field": key}
                if math.isfinite(a) and math.isfinite(b):
                    item.update(cached=a, reconstructed=b, absolute_difference=abs(b-a))
                else:
                    item["reason"] = "nonfinite sufficient statistic"
                issues.append(item)
    return issues


class AuditedFloat32Metrics:
    """Temporary adapter used only inside this reporting process."""
    def __init__(self, original):
        self.original = original
        self.seen = set()
        self.current_stage = None
        self.first_stage_diagnosis = None
        self.last_failure = None
        self.max_absolute_statistic_difference = 0.0
        self.max_relative_statistic_difference = 0.0
        self.expected_fits = len(utils.jobs()) * len(utils.ARMS) * len(utils.PERCENTS) * 5

    def __call__(self, errors, project: Path, target: str, fold: int, strict: bool = True) -> dict:
        # Metadata is used only for safe attribution and diagnostics. The original
        # summarizer has already checked the saved file hashes before this call.
        def one(column):
            if column not in errors or errors[column].isna().any() or errors[column].nunique() != 1:
                raise ValueError(f"Missing or inconsistent raw-error metadata: {column}")
            return errors[column].iloc[0]

        job_id, percent, arm = one("job_id"), one("train_data_percent"), one("experiment_arm")
        if job_id not in range(len(utils.jobs())) or percent not in utils.PERCENTS or arm not in utils.ARMS:
            raise ValueError("Raw errors have an invalid job, training fraction, or arm.")
        job_id, percent = int(job_id), int(percent)
        job = utils.jobs()[job_id]
        if (job["target"] != target or one("dataset") != target or one("fold") != fold
                or one("seed") != job["seed"] or one("model_mode") != job["mode"]):
            raise ValueError("Raw-error job/target/fold/seed/mode metadata mismatch.")
        stage = utils.inside(utils.ROOT / "output/jobs" / f"job_{job_id:03d}" / arm / f"pct{percent}" / f"fold{fold}")
        self.current_stage = str(stage.relative_to(utils.ROOT))
        if self.current_stage in self.seen:
            raise ValueError(f"Duplicate reconstructed stage: {self.current_stage}")
        cached = utils.read_json(stage / "metrics.json")

        # Only the residual column was float32 in engine.collect_errors. Ages,
        # horizons, participant IDs and all other columns are left untouched.
        # fold_metrics itself still accumulates normalized errors in float64.
        diagnostic = None
        if not self.seen:
            uncorrected = self.original(errors, project, target, fold, strict=strict)
            diagnostic = differences(cached, uncorrected)
        corrected = errors.copy(deep=False)
        with np.errstate(over="raise", invalid="raise"):
            corrected["error"] = errors["error"].astype(np.float32)
        rebuilt = self.original(corrected, project, target, fold, strict=strict)
        mismatches = differences(cached, rebuilt)
        if diagnostic is not None:
            self.first_stage_diagnosis = {
                "stage": self.current_stage,
                "uncorrected_float64_mismatch_count": len(diagnostic),
                "uncorrected_first_discrepancy": diagnostic[0] if diagnostic else None,
                "restored_float32_mismatch_count": len(mismatches),
            }
            print(f"First-fit check: {len(diagnostic)} discrepancies before dtype restoration; "
                  f"{len(mismatches)} after.", flush=True)
        if mismatches:
            self.last_failure = {"stage": self.current_stage, "discrepancies": mismatches}
            raise ValueError(f"Strict recomputation still disagrees at {self.current_stage}: {mismatches[0]}. "
                             "No tolerance was relaxed and no saved fit was changed.")
        for attribute, values in cached["attributes"].items():
            for key in STAT_FIELDS - COUNT_FIELDS:
                a, b = values[key], rebuilt["attributes"][attribute][key]
                difference = abs(b-a)
                self.max_absolute_statistic_difference = max(self.max_absolute_statistic_difference, difference)
                if a != 0:
                    self.max_relative_statistic_difference = max(self.max_relative_statistic_difference, difference / abs(a))
        self.seen.add(self.current_stage)
        if len(self.seen) == 1 or len(self.seen) % 20 == 0:
            print(f"Verified {len(self.seen)}/{self.expected_fits} target fits: {self.current_stage}", flush=True)
        return rebuilt

    def report(self, complete: bool) -> dict:
        return {
            "fix_version": FIX_VERSION,
            "complete": complete,
            "runner_sha256": utils.sha256(Path(__file__)),
            "original_package_fingerprint": analyse.load_setup()[0]["package_fingerprint"],
            "change": "Restore only the saved error column to float32 before the original fold_metrics calculation.",
            "accumulation_dtype": "float64, unchanged in the original fold_metrics",
            "rtol": RTOL, "atol": ATOL,
            "counts_and_identity_metadata": "exact comparison",
            "original_integrity_checks": "retained",
            "training_or_cached_fit_files_modified": False,
            "target_fits_verified": len(self.seen),
            "target_fits_expected": self.expected_fits,
            "max_absolute_cached_statistic_difference_after_correction": self.max_absolute_statistic_difference,
            "max_relative_cached_statistic_difference_after_correction_nonzero_cache": self.max_relative_statistic_difference,
            "first_stage_diagnosis": self.first_stage_diagnosis,
            "last_stage": self.current_stage,
            "failure_detail": self.last_failure,
            "reporting_environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        }


def run_summary() -> None:
    setup, _ = analyse.load_setup()
    if utils.package_fingerprint() != setup["package_fingerprint"]:
        raise RuntimeError("Experiment code changed after preparation.")
    original_metrics, original_json = analyse.fold_metrics, analyse.atomic_json
    adapter = AuditedFloat32Metrics(original_metrics)
    report_path = utils.inside(utils.ROOT / "output/summary/report_manifest.json")
    audit_path = utils.inside(utils.ROOT / "output/summary/float32_recompute_audit.json")
    failure_path = utils.inside(utils.ROOT / "output/float32_recompute_failure.json")

    def audited_json(path, value):
        if Path(path).resolve() == report_path:
            if len(adapter.seen) != adapter.expected_fits:
                raise RuntimeError("Refusing a complete report without all target fits reconstructed.")
            original_json(audit_path, adapter.report(complete=True))
            value = dict(value)
            value["files"] = {**value["files"], audit_path.name: utils.sha256(audit_path)}
            value["postprocessing_correction"] = {
                "runner": Path(__file__).name, "runner_sha256": utils.sha256(Path(__file__)),
                "audit_file": audit_path.name, "fix_version": FIX_VERSION,
                "error_column_restored_dtype": "float32", "original_package_unmodified": True,
            }
        original_json(path, value)

    print("Reporting only: no training or GPU allocation; all 600 saved fits will be checked.", flush=True)
    print("Original source, setup, raw errors, metrics and done files are not modified.", flush=True)
    try:
        analyse.fold_metrics = adapter
        analyse.atomic_json = audited_json
        # Original load_setup, data/code fingerprints, file hashes, identity,
        # sufficient-statistic, matched-arm and historical-reference checks run.
        analyse.summarize(recompute=True)
    except (Exception, KeyboardInterrupt) as exc:
        diagnostic = adapter.report(complete=False)
        diagnostic["exception_type"] = type(exc).__name__
        diagnostic["exception"] = str(exc)
        original_json(failure_path, diagnostic)
        print(f"Diagnostic saved: {failure_path}", file=sys.stderr, flush=True)
        raise
    finally:
        analyse.fold_metrics = original_metrics
        analyse.atomic_json = original_json
    if failure_path.exists():
        failure_path.unlink()  # Only this add-on's stale diagnostic, never a fit.
    print(f"Reporting fix passed: {len(adapter.seen)}/{adapter.expected_fits} fits verified. Keep results_to_share.zip local.", flush=True)


def main() -> int:
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    log_path = utils.inside(utils.ROOT / "output/summary_recompute_float32.log")
    try:
        # Serialize direct invocations of the summary correction.
        with utils.file_lock(utils.ROOT / "output/.float32_summary.lock"):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                with contextlib.redirect_stdout(utils.Tee(sys.stdout, log)), contextlib.redirect_stderr(utils.Tee(sys.stderr, log)):
                    print("\n=== Float32 reporting correction ===", flush=True)
                    try:
                        run_summary()
                    except (Exception, KeyboardInterrupt):
                        traceback.print_exc()
                        print("STOPPED. Completed fits remain untouched. Do not retrain or remove output/.", file=sys.stderr)
                        print(f"Reporting log: {log_path}", file=sys.stderr)
                        return 1
        return 0
    except (Exception, KeyboardInterrupt):
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
