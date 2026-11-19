#!/usr/bin/env python3
"""Recompute learning-rate summaries with the original float32 error precision."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import contextlib
import io
import os
from pathlib import Path
import traceback
import zipfile

import exp2_utils as U
U.configure_local_runtime()

FIX_ID = "exp2-v1.0.0-float32-recomputation-1"
PATCH_FILES = ("summarize_fixed.py", "test_summary_fix.py")
RTOL = ATOL = 1e-12


def restore_error_dtype(errors):
    """Leave identities/ages and all other columns unchanged; restore raw errors."""
    import numpy as np
    if "error" not in errors:
        raise ValueError("Missing raw-error column.")
    values = errors["error"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite raw errors; refusing precision repair.")
    with np.errstate(over="raise", invalid="raise"):
        restored = errors["error"].astype(np.float32)
    if not np.isfinite(restored.to_numpy()).all():
        raise ValueError("Raw errors cannot be represented as finite float32.")
    result = errors.copy(deep=False)
    result["error"] = restored
    return result


def precision_selftest():
    """Check CSV reconstruction in the local pandas/NumPy environment."""
    import numpy as np
    import pandas as pd
    values = np.array([0.1, -0.3, 1.234567, 123.45678, 9.876543, 0.0], dtype=np.float32)
    csv = pd.DataFrame({"error": values}).to_csv(index=False)
    loaded = pd.read_csv(io.StringIO(csv), float_precision="round_trip")
    fixed = restore_error_dtype(loaded)
    if not np.array_equal(values.view(np.uint32), fixed.error.to_numpy().view(np.uint32)):
        raise RuntimeError("Local CSV float32 reconstruction self-test failed. No fits were changed.")


def compare_metrics(rebuilt: dict, saved: dict) -> dict:
    """Original strict numerical tolerance, plus exact schema/identity/count checks."""
    import numpy as np
    if set(rebuilt) != set(saved):
        raise ValueError("Recomputed metric schema differs.")
    for field in saved:
        if field != "attributes" and rebuilt[field] != saved[field]:
            raise ValueError(f"Recomputed metadata differs: {field}.")
    if set(rebuilt["attributes"]) != set(saved["attributes"]):
        raise ValueError("Recomputed attribute panel differs.")
    max_abs = max_rel = 0.0
    compared = 0
    for attribute, values in rebuilt["attributes"].items():
        expected = saved["attributes"][attribute]
        if set(values) != set(expected):
            raise ValueError(f"Recomputed fields differ for {attribute}.")
        for field, value in values.items():
            old = expected[field]
            if field in ("n_people", "n_errors"):
                if value != old:
                    raise ValueError(f"Recomputed count differs: {attribute}/{field}: saved={old}, rebuilt={value}.")
                continue
            if not np.isfinite(value) or not np.isfinite(old) or not np.isclose(value, old, rtol=RTOL, atol=ATOL):
                raise ValueError(
                    f"Sufficient statistic still differs after restoring float32: {attribute}/{field}; "
                    f"saved={old:.17g}, rebuilt={value:.17g}, absolute_difference={abs(value-old):.6g}. "
                    "The tolerance was not relaxed. Keep the completed fits."
                )
            difference = abs(value - old)
            max_abs = max(max_abs, difference)
            if old != 0:
                max_rel = max(max_rel, difference / abs(old))
            compared += 1
    return {"n_statistics_checked": compared, "max_absolute_statistic_difference": max_abs,
            "max_relative_statistic_difference": max_rel}


class RecomputeAudit:
    """Targeted, temporary replacement of only the recomputation callback."""
    def __init__(self, original):
        self.original = original
        self.rows = []
        self.current = None
        self.expected = [(job, fold, key) for job in U.jobs() for fold in range(5) for key in U.LRS]

    def __call__(self, errors, project, target, fold, strict=True):
        import numpy as np
        index = len(self.rows)
        if index >= len(self.expected) or not strict:
            raise ValueError("Unexpected recomputation call; refusing to change analysis scope.")
        job, expected_fold, key = self.expected[index]
        self.current = {"job_id": job["job_id"], "target": target, "percent": job["percent"],
                        "seed": job["seed"], "fold": fold, "lr_key": key}
        if (target, fold) != (job["target"], expected_fold):
            raise ValueError("Unexpected recomputation order.")
        for column, expected_value in (("dataset", target), ("fold", fold), ("seed", job["seed"]),
                                       ("train_data_percent", job["percent"]), ("learning_rate", U.LRS[key])):
            if column not in errors or not errors[column].eq(expected_value).all():
                raise ValueError(f"Raw-error fit metadata differs: {column}.")
        restored = restore_error_dtype(errors)
        n_restored = int(np.count_nonzero(
            errors.error.to_numpy(dtype=np.float64) != restored.error.to_numpy(dtype=np.float64)))
        rebuilt = self.original(restored, project, target, fold, strict=True)
        stage = U.inside(U.ROOT / "output/jobs" / f"job_{job['job_id']:03d}" / f"fold{fold}" / key)
        try:
            comparison = compare_metrics(rebuilt, U.read_json(stage / "metrics.json"))
        except ValueError as exc:
            raise ValueError(f"{stage.relative_to(U.ROOT)}: {exc}") from exc
        self.rows.append({**self.current, "n_raw_errors": len(errors),
                          "n_error_values_restored": n_restored, **comparison, "passed": True})
        return rebuilt


def finalize_share(audit: RecomputeAudit):
    """Record the repair explicitly; regenerate a whitelisted, aggregate-only ZIP."""
    import numpy as np
    import pandas as pd
    folder = U.inside(U.ROOT / "summary")
    with U.file_lock(folder / ".summary.lock"):
        if len(audit.rows) != 420:
            raise ValueError("Repair audit must contain all 420 research fits.")
        report = {
            "fix_id": FIX_ID, "original_package_fingerprint": U.load_setup()[0]["package_fingerprint"],
            "patch_file_sha256": {name: U.sha256(U.inside(U.ROOT / name)) for name in PATCH_FILES},
            "change": "Restore only raw error column to original float32 before original float64 metric arithmetic.",
            "research_fits_recomputed": len(audit.rows), "all_strict_recomputation_checks_passed": True,
            "rtol": RTOL, "atol": ATOL, "tolerances_relaxed": False,
            "training_rerun": False, "original_fit_files_modified": False,
            "original_package_files_modified": False, "validation_selection_rule_changed": False,
            "n_raw_error_rows": sum(row["n_raw_errors"] for row in audit.rows),
            "n_error_values_restored": sum(row["n_error_values_restored"] for row in audit.rows),
            "max_absolute_statistic_difference": max(row["max_absolute_statistic_difference"] for row in audit.rows),
            "max_relative_statistic_difference": max(row["max_relative_statistic_difference"] for row in audit.rows),
            "analysis_environment": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        }
        U.atomic_json(folder / "precision_fix.json", report)
        U.atomic_csv(pd.DataFrame(audit.rows), folder / "recomputation_audit.csv")
        provenance = U.read_json(folder / "provenance.json")
        provenance["analysis_correction"] = {"fix_id": FIX_ID, "details_file": "precision_fix.json",
                                            "audit_file": "recomputation_audit.csv"}
        U.atomic_json(folder / "provenance.json", provenance)
        text = (folder / "REPORT.md").read_text(encoding="utf-8")
        text += (
            "\n## CSV precision correction during recomputation\n\n"
            "The original training collector produced float32 raw errors. These errors were restored to "
            "float32 after CSV loading and before the original float64 metric arithmetic. All 420 fits "
            "passed the original 1e-12 absolute/relative sufficient-statistic checks. No training, "
            "validation selection, fit files, or original package files were changed. "
            "See `precision_fix.json`, `recomputation_audit.csv`, and `package/summarize_fixed.py` "
            "in the share ZIP for the explicit analysis correction.\n"
        )
        U.atomic_text(folder / "REPORT.md", text)
        names = list(U.read_json(folder / "file_sha256.json"))
        for name in ("precision_fix.json", "recomputation_audit.csv"):
            if name not in names:
                names.append(name)
        for name in names:
            if Path(name).name != name:
                raise ValueError("Unexpected summary filename.")
        U.atomic_json(folder / "file_sha256.json", {name: U.sha256(U.inside(folder / name)) for name in names})
        package_files = U.CORE_FILES + ("PROTOCOL.md", "tests.py") + PATCH_FILES
        output = U.inside(U.ROOT / "results_to_share.zip")
        temp = U.inside(U.ROOT / f"results_to_share.zip.precision_fix.{os.getpid()}.tmp")
        try:
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for name in names + ["file_sha256.json"]:
                    z.write(U.inside(folder / name), "summary/" + name)
                for name in package_files:
                    z.write(U.inside(U.ROOT / name), "package/" + name)
            os.replace(temp, output)
        finally:
            if temp.exists():
                temp.unlink()


def run():
    import analyse
    U.load_setup()
    for name in PATCH_FILES:
        if not U.inside(U.ROOT / name).is_file():
            raise FileNotFoundError(f"Missing summary script: {name}.")
    precision_selftest()
    original = analyse.fold_metrics
    audit = RecomputeAudit(original)
    folder = U.inside(U.ROOT / "summary")
    folder.mkdir(parents=True, exist_ok=True)
    with U.file_lock(folder / ".precision_fix.lock"):
        try:
            print("Analysis-only precision repair: restoring float32 raw errors; strict checks unchanged.", flush=True)
            print("No training will run. Original code, setup and completed fit files remain untouched.", flush=True)
            analyse.fold_metrics = audit
            try:
                analyse.summarize(recompute=True)
            finally:
                analyse.fold_metrics = original
            finalize_share(audit)
        except Exception:
            U.atomic_json(folder / "precision_fix_failure.json", {
                "fix_id": FIX_ID, "research_fits_recomputed_before_failure": len(audit.rows),
                "current_fit": audit.current, "status": "failed; do not use a partial summary",
                "fit_files_modified": False,
            })
            raise
        else:
            failure = U.inside(folder / "precision_fix_failure.json")
            if failure.exists():
                failure.unlink()
    print("PRECISION FIX COMPLETE: all 420 fits recomputed with strict checks.", flush=True)
    print("Read summary/REPORT.md. Keep results_to_share.zip local; no training was rerun.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run only the CSV-precision check; no research outputs are read.")
    args = parser.parse_args()
    if args.self_test:
        precision_selftest()
        print("CSV float32 reconstruction self-test passed.")
    else:
        run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nSTOPPED: {exc}\nCompleted fits are retained. Do not retrain or delete them.", file=sys.stderr)
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exc()
        raise SystemExit(1)
