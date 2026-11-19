#!/usr/bin/env python3
"""Synthetic, CPU-only tests for the reporting add-on; no project data needed.

All fixtures live in a temporary subfolder of this experiment and are deleted.
Real code/data fingerprint validation is mocked ONLY for synthetic integration
fixtures; the production runner retains the original validation functions.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import summarize_float32 as fix
import analyse
import exp1_utils as utils
import numpy as np
import pandas as pd


class ReportingFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real_root = utils.ROOT
        cls.test_root = utils.inside(utils.ROOT / "_reporting_fix_tests")
        cls.test_root.mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=cls.test_root)
        cls.root = Path(cls.temporary.name)
        cls.project = cls.root / "synthetic_project"
        cls.setup = {"package_fingerprint": "synthetic-test", "experiment_fingerprint": "SYNTHETIC-NOT-RESEARCH",
                     "primary_panel": ["Height", "Weight", "BMI"],
                     "exclusions": {}, "limitations": ["Synthetic test data only."]}
        with patch.object(utils, "ROOT", cls.root), patch.object(analyse, "ROOT", cls.root):
            cls.build_fixture()
        cls.first = cls.root / "output/jobs/job_000/all_source/pct10/fold0"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def build_fixture(cls):
        panel = cls.setup["primary_panel"]
        stds = {target: {str(fold): {a: 1.37 + i + fold / 7 for i, a in enumerate(panel)}
                         for fold in range(5)} for target in utils.TARGET_LABELS}
        eligible = {target: {str(fold): {a: {"eligible": True} for a in panel}
                             for fold in range(5)} for target in utils.TARGET_LABELS}
        expected = {target: {str(fold): {a: 4 for a in panel} for fold in range(5)}
                    for target in utils.TARGET_LABELS}
        utils.atomic_json(cls.project / "data/target_attribute_stds_by_dataset_fold.json", stds)
        utils.atomic_json(cls.project / "data/eligible_dataset_attribute_pairs.json", eligible)
        utils.atomic_json(cls.project / "data/primary_target_panel.json", panel)
        utils.atomic_json(cls.root / "setup/expected_test_counts.json", expected)
        utils.atomic_json(cls.root / "setup/manifest.json", cls.setup)
        utils.atomic_json(cls.root / "setup/validation.json", {"synthetic": True})
        utils.atomic_json(cls.root / "reference/provenance.json", {"synthetic": True})
        for name in ("source_counts.csv", "target_partitions.csv", "jobs.csv"):
            utils.atomic_csv(pd.DataFrame({"synthetic": [True]}), cls.root / "setup" / name)
        (cls.root / "PROTOCOL.md").write_text("SYNTHETIC TEST ONLY\n")
        utils.atomic_json(cls.root / "exclusions.json", {})
        reference = []
        for target in utils.TARGET_LABELS:
            for percent in utils.PERCENTS:
                for seed in utils.SEEDS:
                    for config in ("strats_no_pretrain", "strats_main", "strats_pooled_all_sources"):
                        reference.append({"dataset": target, "train_data_percent": percent, "seed": seed,
                                          "base_configuration_id": config, "main_macro_nrmse": 1.,
                                          "n_error_rows": 60, "n_test_people": 10, "n_attributes_evaluated": 3})
        utils.atomic_csv(pd.DataFrame(reference), cls.root / "reference/original_metrics.csv")
        for job in utils.jobs():
            folder = cls.root / "output/jobs" / f"job_{job['job_id']:03d}"
            config = {"experiment_fingerprint": cls.setup["experiment_fingerprint"], "smoke": False,
                      "job": job, "num_workers": 0}
            config_hash = utils.object_hash(config)
            utils.atomic_json(folder / "config.json", config)
            utils.atomic_json(folder / "environment.json", {"synthetic": True})
            for arm in utils.ARMS:
                if job["mode"] == "pretrain":
                    source = folder / arm / "pretrain"
                    source.mkdir(parents=True)
                    (source / "synthetic_checkpoint.txt").write_text("NOT A MODEL\n")
                    utils.atomic_json(source / "done.json", {"config_hash": config_hash,
                                      "file_sha256": {"synthetic_checkpoint.txt": utils.sha256(source / "synthetic_checkpoint.txt")}})
                for percent in utils.PERCENTS:
                    for fold in range(5):
                        stage = folder / arm / f"pct{percent}" / f"fold{fold}"
                        rows = []
                        for person, repeats in enumerate((1, 3)):
                            for index, attribute in enumerate(panel):
                                for repeat in range(repeats):
                                    error = np.float32((.123456789 + index * .34567891 + repeat * .086753091)
                                                       * (1 + person + fold / 3) * (1.1 if arm == "excluded" else 1.))
                                    rows.append({"dataset": job["target"], "fold": fold,
                                                 "person_id": f"synthetic_{fold}_{person}", "attribute": attribute,
                                                 "error": error, "input_age": 9.1234567890123 + repeat,
                                                 "output_age": 14.2468135790123 + repeat, "horizon": 5.123356789,
                                                 "n_input_visits": repeat + 1, "job_id": job["job_id"],
                                                 "seed": job["seed"], "model_mode": job["mode"],
                                                 "experiment_arm": arm, "train_data_percent": percent})
                        errors = pd.DataFrame(rows)
                        assert errors.error.dtype == np.float32
                        if job["job_id"] == 0 and arm == "all_source" and percent == 10 and fold == 0:
                            cls.raw_first = errors.copy()
                        metrics = analyse.fold_metrics(errors, cls.project, job["target"], fold)
                        utils.atomic_csv(errors, stage / "errors.csv.gz")
                        utils.atomic_json(stage / "metrics.json", metrics)
                        utils.atomic_csv(pd.DataFrame({"epoch": [1], "synthetic": [True]}), stage / "history.csv")
                        done = {"config_hash": config_hash, "sources": ["synthetic_source"], "stage_seed": 123,
                                "n_train_records": 1, "n_test_records": 4, "n_raw_errors": len(errors),
                                "training": {"epochs_run": 1, "best_epoch": 1, "selected_validation_mse": 1., "elapsed_seconds": 0.},
                                "file_sha256": {n: utils.sha256(stage / n) for n in ("errors.csv.gz", "metrics.json", "history.csv")}}
                        for field in ("target_train", "target_val", "target_test", "actual_train"):
                            done[field] = {"n_people": 2}
                        utils.atomic_json(stage / "done.json", done)

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(utils, "ROOT", self.root))
        self.stack.enter_context(patch.object(analyse, "ROOT", self.root))
        self.stack.enter_context(patch.object(analyse, "load_setup", return_value=(self.setup, self.project)))
        self.stack.enter_context(patch.object(utils, "package_fingerprint", return_value=self.setup["package_fingerprint"]))
        self.output = io.StringIO()
        self.stack.enter_context(contextlib.redirect_stdout(self.output))
        self.stack.enter_context(contextlib.redirect_stderr(self.output))
        self.addCleanup(self.stack.close)

    @contextlib.contextmanager
    def altered(self, names):
        saved = {name: (self.first / name).read_bytes() for name in names}
        try:
            yield
        finally:
            for name, content in saved.items():
                (self.first / name).write_bytes(content)

    def reseal(self, name):
        done = utils.read_json(self.first / "done.json")
        done["file_sha256"][name] = utils.sha256(self.first / name)
        utils.atomic_json(self.first / "done.json", done)

    def test_01_original_failure_reproduced(self):
        with self.assertRaisesRegex(ValueError, "Reconstructed error metrics differ"):
            analyse.summarize(recompute=True)

    def test_02_dtype_restoration_is_exact(self):
        read = pd.read_csv(self.first / "errors.csv.gz", dtype={"person_id": str}, float_precision="round_trip")
        self.assertEqual(read.error.dtype, np.float64)
        self.assertFalse(np.array_equal(read.error.to_numpy(), self.raw_first.error.to_numpy(dtype=float)))
        read["error"] = read.error.astype(np.float32)
        np.testing.assert_array_equal(read.error.to_numpy(), self.raw_first.error.to_numpy())
        cached = utils.read_json(self.first / "metrics.json")
        rebuilt = analyse.fold_metrics(read, self.project, utils.jobs()[0]["target"], 0)
        self.assertEqual(cached, rebuilt)

    def test_03_all_600_fits_report_without_modifying_inputs(self):
        protected = [p for folder in ("output/jobs", "setup", "reference", "synthetic_project")
                     for p in (self.root / folder).rglob("*") if p.is_file()]
        before = {p: utils.sha256(p) for p in protected}
        original = analyse.fold_metrics
        fix.run_summary()
        self.assertIs(analyse.fold_metrics, original)
        self.assertEqual(before, {p: utils.sha256(p) for p in protected})
        summary = self.root / "output/summary"
        audit = utils.read_json(summary / "float32_recompute_audit.json")
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["target_fits_verified"], 600)
        self.assertGreater(audit["first_stage_diagnosis"]["uncorrected_float64_mismatch_count"], 0)
        self.assertEqual(audit["first_stage_diagnosis"]["restored_float32_mismatch_count"], 0)
        self.assertEqual(audit["max_absolute_cached_statistic_difference_after_correction"], 0.)
        manifest = utils.read_json(summary / "report_manifest.json")
        self.assertTrue(manifest["raw_errors_recomputed"])
        for name, digest in manifest["files"].items():
            self.assertEqual(digest, utils.sha256(summary / name))
        self.assertEqual(len(pd.read_csv(summary / "dataset_scores.csv")), 120)
        self.assertEqual(len(pd.read_csv(summary / "comparison_by_target.csv")), 20)
        with zipfile.ZipFile(self.root / "results_to_share.zip") as archive:
            self.assertIn("summary/float32_recompute_audit.json", archive.namelist())
            self.assertFalse(any("errors.csv" in n or "checkpoint" in n for n in archive.namelist()))

    def test_04_changed_error_file_hash_rejected(self):
        with self.altered(["errors.csv.gz"]):
            with (self.first / "errors.csv.gz").open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "Corrupt completed output"):
                fix.run_summary()

    def test_05_changed_metric_file_hash_rejected(self):
        with self.altered(["metrics.json"]):
            (self.first / "metrics.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Corrupt completed output"):
                fix.run_summary()

    def test_06_real_metric_mismatch_rejected_even_with_valid_hash(self):
        with self.altered(["metrics.json", "done.json"]):
            metrics = utils.read_json(self.first / "metrics.json")
            metrics["attributes"]["Height"]["sum_person_normalized_mse"] *= 1.01
            utils.atomic_json(self.first / "metrics.json", metrics)
            self.reseal("metrics.json")
            original = analyse.fold_metrics
            with self.assertRaisesRegex(ValueError, "Strict recomputation still disagrees.*Height"):
                fix.run_summary()
            self.assertIs(analyse.fold_metrics, original)
            failure = utils.read_json(self.root / "output/float32_recompute_failure.json")
            self.assertFalse(failure["complete"])
            self.assertEqual(failure["target_fits_verified"], 0)

    def test_07_changed_identity_rejected(self):
        with self.altered(["errors.csv.gz", "done.json"]):
            errors = self.raw_first.copy()
            errors.loc[errors.person_id.eq("synthetic_0_0"), "person_id"] = "synthetic_changed"
            utils.atomic_csv(errors, self.first / "errors.csv.gz")
            self.reseal("errors.csv.gz")
            with self.assertRaisesRegex(ValueError, "identity_digest"):
                fix.run_summary()

    def test_08_cached_count_mismatch_rejected(self):
        with self.altered(["metrics.json", "done.json"]):
            metrics = utils.read_json(self.first / "metrics.json")
            metrics["n_raw_errors"] += 1
            utils.atomic_json(self.first / "metrics.json", metrics)
            self.reseal("metrics.json")
            with self.assertRaisesRegex(ValueError, "n_raw_errors"):
                fix.run_summary()

    def test_09_incomplete_experiment_rejected(self):
        with self.altered(["done.json"]):
            (self.first / "done.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "Cannot issue a complete report"):
                fix.run_summary()

    def test_10_unknown_package_rejected(self):
        with patch.object(utils, "package_fingerprint", return_value="different"):
            with self.assertRaisesRegex(RuntimeError, "code changed after preparation"):
                fix.run_summary()

    def test_11_real_residual_change_not_hidden_by_cast(self):
        with self.altered(["errors.csv.gz", "done.json"]):
            errors = self.raw_first.copy()
            errors.loc[0, "error"] = np.float32(100.)
            utils.atomic_csv(errors, self.first / "errors.csv.gz")
            self.reseal("errors.csv.gz")
            with self.assertRaisesRegex(ValueError, "Strict recomputation still disagrees"):
                fix.run_summary()

    def test_12_attribute_set_and_integer_counts_strict(self):
        metrics = utils.read_json(self.first / "metrics.json")
        other = copy.deepcopy(metrics)
        del other["attributes"]["Height"]
        self.assertTrue(fix.differences(metrics, other))
        other = copy.deepcopy(metrics)
        other["attributes"]["Height"]["n_people"] = 2.0
        self.assertTrue(fix.differences(metrics, other))

    def test_13_random_float32_bit_roundtrip(self):
        rng = np.random.default_rng(1234)
        bits = rng.integers(0, 2**32, size=25000, dtype=np.uint32)
        values = bits.view(np.float32)
        values = values[np.isfinite(values)]
        # Include signed zero, minimum subnormal and maximum finite float32.
        values = np.r_[values, np.array([0., -0., np.nextafter(np.float32(0), np.float32(1)),
                                         np.finfo(np.float32).max], dtype=np.float32)]
        text = pd.DataFrame({"error": values}).to_csv(index=False)
        read = pd.read_csv(io.StringIO(text), float_precision="round_trip").error.to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(read.view(np.uint32), values.view(np.uint32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
