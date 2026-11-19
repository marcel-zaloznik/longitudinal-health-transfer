#!/usr/bin/env python3
"""CPU regression tests for the analysis-only repair; all data below are synthetic."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import contextlib
import copy
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import summarize_fixed as F
import exp2_utils as U
import analyse
import experiment  # Ensure import-time runtime configuration precedes fixture patching.
import numpy as np
import pandas as pd

SOURCE = U.ROOT


def errors_frame(target="ABCD", fold=0, seed=1234, percent=10, key="lr1e-4"):
    frame = pd.DataFrame({
        "person_id": [f"synthetic-{fold}-001", f"synthetic-{fold}-001", f"synthetic-{fold}-002", f"synthetic-{fold}-003"],
        "input_age": [10.0, 10.5, 12.0, 13.0], "output_age": [11.0, 11.0, 13.0, 14.0],
        "horizon": [1.0, 0.5, 1.0, 1.0], "n_input_visits": [1, 2, 1, 1],
        "error": np.array([0.1, -0.3, 1.234567, 123.45678], dtype=np.float32),
    })
    for field, value in {"dataset": target, "fold": fold, "seed": seed, "train_data_percent": percent,
                         "learning_rate": U.LRS[key], "attribute": "Height"}.items():
        frame[field] = value
    return frame


@contextlib.contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix="summary_fix_test_", dir=SOURCE / "_runtime/tmp") as name:
        root = Path(name)
        with patch.object(U, "ROOT", root), patch.object(analyse, "ROOT", root), patch.object(experiment, "ROOT", root):
            project = root / "synthetic_project"
            U.atomic_json(project / "data/primary_target_panel.json", ["Height"])
            U.atomic_json(project / "data/target_attribute_stds_by_dataset_fold.json", {
                t: {str(f): {"Height": 2.0} for f in range(5)} for t in U.TARGET_LABELS})
            U.atomic_json(project / "data/eligible_dataset_attribute_pairs.json", {
                t: {str(f): {"Height": {"eligible": True}} for f in range(5)} for t in U.TARGET_LABELS})
            U.atomic_json(root / "setup/expected_test_counts.json", {
                t: {str(f): {"Height": 4} for f in range(5)} for t in U.TARGET_LABELS})
            yield root, project


def complete_synthetic_fixture(root, project):
    files = U.CORE_FILES + ("PROTOCOL.md", "tests.py") + F.PATCH_FILES
    for name in files:
        destination = U.inside(root / name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("reference/"):
            # References are created below; no private reference files are bundled.
            if name.endswith("original_train_model.py.txt"):
                destination.write_text("# Synthetic regression fixture.\n", encoding="utf-8")
            continue
        shutil.copyfile(SOURCE / name, destination)
    models = ("strats_no_pretrain", "strats_main", "strats_pooled_all_sources")
    reference = pd.DataFrame([
        {"dataset": j["target"], "train_data_percent": j["percent"], "seed": j["seed"],
         "base_configuration_id": model, "main_macro_nrmse": 50.0 - i,
         "n_error_rows": 20, "n_test_people": 15, "n_attributes_evaluated": 1, "n_folds_evaluated": 5}
        for j in U.jobs() for i, model in enumerate(models)
    ])
    U.atomic_csv(reference, root / "reference/original_metrics.csv")
    input_hashes = {path.name: U.sha256(path) for path in (project / "data").iterdir()}
    U.atomic_json(root / "reference/provenance.json", {
        "synthetic_test_only": True, "input_sha256": input_hashes,
        "reference_subset_sha256": U.sha256(root / "reference/original_metrics.csv")})
    U.atomic_csv(pd.DataFrame([{"people_digest": "not-for-sharing", "n_people": 3}]), root / "setup/target_partitions.csv")
    fingerprint = U.package_fingerprint()
    setup = {
        "project_root": str(project), "package_fingerprint": fingerprint,
        "experiment_fingerprint": U.object_hash({"package": fingerprint, "inputs": input_hashes}),
        "input_sha256": input_hashes,
        "setup_sha256": {n: U.sha256(root / "setup" / n) for n in ("expected_test_counts.json", "target_partitions.csv")},
        "selection": "Synthetic fixture: validation only", "training_settings": {"synthetic_test_only": True},
        "seed_policy": "synthetic", "limitations": ["Synthetic regression fixture; not research results."],
    }
    U.atomic_json(root / "setup/manifest.json", setup)
    for job in U.jobs():
        job_folder = root / "output/jobs" / f"job_{job['job_id']:03d}"
        config = {"smoke": False, "job": job, "experiment_fingerprint": setup["experiment_fingerprint"],
                  "runtime_signature": {"synthetic": True}}
        U.atomic_json(job_folder / "config.json", config)
        for fold in range(5):
            for key, rate in U.LRS.items():
                stage = job_folder / f"fold{fold}" / key
                errors = errors_frame(job["target"], fold, job["seed"], job["percent"], key)
                # Deliberately make the validation-preferred rate WORSE on test errors.
                errors["error"] *= np.float32((1.8 if key == "lr5e-4" else 1.0) + (job["seed"] % 10) / 100)
                saved = analyse.fold_metrics(errors, project, job["target"], fold)
                U.atomic_csv(errors, stage / "errors.csv.gz")
                U.atomic_json(stage / "metrics.json", saved)
                val = 0.8 if key == "lr5e-4" else 1.0
                U.atomic_csv(pd.DataFrame({"epoch": range(1, 12), "train_mse": [1.0] * 11,
                                          "validation_mse": [val] * 11, "elapsed_seconds": range(11)}), stage / "history.csv")
                done = {"config_hash": U.object_hash(config), "lr_key": key, "learning_rate": rate,
                    "stage_seed": job["seed"] + fold, "initial_state_sha256": "synthetic-identical-within-pair",
                    "loader_seeds": {"train": 1, "validation": 2}, "target_statistics": {"synthetic": True},
                    "train": {"n_people": 3}, "validation": {"n_people": 3}, "test": {"n_people": 3},
                    "n_train_records": 4, "n_val_records": 4, "n_test_records": 4,
                    "test_identity_digest": saved["identity_digest"],
                    "training": {"epochs_run": 11, "best_epoch": 1, "selected_validation_mse": val,
                                 "hit_epoch_cap": False, "elapsed_seconds": 1.0},
                    "environment": {"device_name": "synthetic CPU fixture"},
                    "file_sha256": {name: U.sha256(stage / name) for name in ("errors.csv.gz", "metrics.json", "history.csv")}}
                U.atomic_json(stage / "done.json", done)
    return fingerprint


class PrecisionFixTests(unittest.TestCase):
    def test_original_failure_and_exact_metric_recovery(self):
        with fixture() as (root, project):
            original = errors_frame()
            expected = analyse.fold_metrics(original, project, "ABCD", 0)
            U.atomic_csv(original, root / "raw.csv.gz")
            loaded = pd.read_csv(root / "raw.csv.gz", dtype={"person_id": str}, float_precision="round_trip")
            wrong = analyse.fold_metrics(loaded, project, "ABCD", 0)
            with self.assertRaisesRegex(ValueError, "Sufficient statistic still differs"):
                F.compare_metrics(wrong, expected)
            restored = F.restore_error_dtype(loaded)
            self.assertEqual(restored.error.dtype, np.dtype("float32"))
            recovered = analyse.fold_metrics(restored, project, "ABCD", 0)
            self.assertEqual(recovered, expected)
            F.compare_metrics(recovered, expected)
            pd.testing.assert_frame_equal(loaded.drop(columns="error"), restored.drop(columns="error"))
            self.assertEqual(loaded.error.dtype, np.dtype("float64"))

    def test_many_float32_values_roundtrip_bit_exactly(self):
        rng = np.random.default_rng(20261112)
        values = rng.integers(0, 2**32, size=20000, dtype=np.uint32).view(np.float32)
        values = values[np.isfinite(values)]
        csv = pd.DataFrame({"error": values}).to_csv(index=False)
        loaded = pd.read_csv(io.StringIO(csv), float_precision="round_trip")
        fixed = F.restore_error_dtype(loaded)
        np.testing.assert_array_equal(values.view(np.uint32), fixed.error.to_numpy().view(np.uint32))

    def test_nonfinite_and_overflow_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), 1e100):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, FloatingPointError)):
                    F.restore_error_dtype(pd.DataFrame({"error": [value]}))

    def test_changed_counts_and_metrics_rejected(self):
        with fixture() as (_, project):
            expected = analyse.fold_metrics(errors_frame(), project, "ABCD", 0)
            for field in ("n_people", "n_errors", "sum_person_normalized_mse", "sum_person_normalized_mae"):
                changed = copy.deepcopy(expected)
                changed["attributes"]["Height"][field] += 1.0
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        F.compare_metrics(changed, expected)
            changed = copy.deepcopy(expected)
            changed["identity_digest"] = "different"
            with self.assertRaises(ValueError):
                F.compare_metrics(changed, expected)

    def test_cpu_dtype_selftest(self):
        F.precision_selftest()

    def test_full_420_fit_summary_and_repeatability_and_corruption_guard(self):
        with fixture() as (root, project):
            fingerprint = complete_synthetic_fixture(root, project)
            protected = [p for p in (root / "output").rglob("*") if p.is_file()]
            protected += [p for p in (root / "setup").rglob("*") if p.is_file()]
            protected += [root / name for name in U.CORE_FILES]
            before = {p: U.sha256(p) for p in protected}
            original_function = analyse.fold_metrics
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Recomputed sufficient statistics differ"):
                    analyse.summarize(recompute=True)
            frozen_selection = U.sha256(root / "summary/selection_provenance.json")
            # This override is TEST-ONLY: fixture reference data necessarily have a different fingerprint.
            with contextlib.nullcontext():
                with contextlib.redirect_stdout(io.StringIO()):
                    F.run()
                self.assertIs(analyse.fold_metrics, original_function)
                self.assertEqual(before, {p: U.sha256(p) for p in protected})
                self.assertEqual(frozen_selection, U.sha256(root / "summary/selection_provenance.json"))
                audit = pd.read_csv(root / "summary/recomputation_audit.csv")
                self.assertEqual(len(audit), 420)
                self.assertTrue(audit.passed.all())
                self.assertTrue((audit.max_absolute_statistic_difference == 0).all())
                choices = pd.read_csv(root / "summary/selected_rates.csv")
                self.assertEqual(len(choices), 70)
                self.assertTrue(choices.selected_lr_key.eq("lr5e-4").all())
                scores = pd.read_csv(root / "summary/overall_summary.csv")
                self.assertEqual(len(scores), 6)
                for percent in U.PERCENTS:
                    block = scores[scores.percent.eq(percent)].set_index("policy")
                    self.assertGreater(block.loc["validation_selected", "nrmse_mean"], block.loc["fixed_lr1e-4", "nrmse_mean"])
                with zipfile.ZipFile(root / "results_to_share.zip") as z:
                    self.assertIn("package/summarize_fixed.py", z.namelist())
                    self.assertIn("summary/precision_fix.json", z.namelist())
                    self.assertFalse(any("errors.csv.gz" in name or name.endswith(".pt") or name.startswith("output/") for name in z.namelist()))
                    self.assertFalse(any(b"synthetic-0-001" in z.read(name) for name in z.namelist() if name.startswith("summary/")))
                    for name, digest in U.read_json(root / "summary/file_sha256.json").items():
                        self.assertEqual(U.sha256(root / "summary" / name), digest)
                        self.assertEqual(z.read("summary/" + name), (root / "summary" / name).read_bytes())
                stable = {name: U.sha256(root / "summary" / name) for name in ("overall_summary.csv", "selected_rates.csv", "REPORT.md", "precision_fix.json")}
                with contextlib.redirect_stdout(io.StringIO()):
                    F.run()
                self.assertEqual(stable, {name: U.sha256(root / "summary" / name) for name in stable})
                self.assertEqual(before, {p: U.sha256(p) for p in protected})
                # Actual corruption must still stop before recomputation; never bypass file hashes.
                raw = root / "output/jobs/job_000/fold0/lr1e-4/errors.csv.gz"
                raw.write_bytes(raw.read_bytes() + b"corruption")
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "Missing/corrupt completed output"):
                        F.run()
                self.assertIs(analyse.fold_metrics, original_function)
                self.assertTrue((root / "summary/precision_fix_failure.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
