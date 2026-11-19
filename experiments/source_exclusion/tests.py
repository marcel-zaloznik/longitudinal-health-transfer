#!/usr/bin/env python3
"""CPU-only checks. Synthetic fixtures are temporary and never research output."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

import copy
import contextlib
import io
import subprocess
from unittest.mock import patch
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import types
import unittest

from exp1_utils import *
configure_local_runtime()
import numpy as np
import pandas as pd
import torch
import engine
import analyse
import experiment


def module_from_text(name, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    finally:
        sys.path.pop(0)
    return module


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "setup/manifest.json").exists():
            experiment.prepare()
        cls.setup, cls.project = load_setup()
        cls.test_root = inside(ROOT / "_tests")
        cls.test_root.mkdir(exist_ok=True)
        torch.set_num_threads(2)

    def test_matrix_and_exclusions(self):
        self.assertEqual(len(jobs()), 30)
        self.assertEqual(30 * 2 * 2 * 5, 600)
        self.assertEqual(sum(job["mode"] == "pretrain" for job in jobs()) * 2, 30)
        exclusions = read_json(ROOT / "exclusions.json")
        for target in TARGET_LABELS:
            full = source_names(engine, target, "all_source")
            selected = source_names(engine, target, "excluded")
            self.assertEqual(set(full) - set(selected), set(exclusions[target]))
            self.assertNotIn(target, selected)
            self.assertEqual(len(selected), 4 if target == "BE_KUL_Wave_23456" else 5)
        self.assertEqual(stage_seed(1234, 2), 103234)
        self.assertEqual(stage_seed(1234, 2, 0), 203234)

    def test_group_partition_and_defaults(self):
        flattened = [j["job_id"] for i in range(N_GROUPS) for j in group_jobs(i)]
        self.assertEqual(flattened, list(range(30)))
        self.assertEqual(len(set(flattened)), 30)
        for i in range(N_GROUPS):
            self.assertEqual(len(group_jobs(i)), 5)
            args = experiment.parser().parse_args(["run", "--group", str(i)])
            self.assertEqual(args.num_workers, 0)
            self.assertEqual(experiment.selected_jobs(args), group_jobs(i))
        for invalid in (-1, 6):
            with self.assertRaises(ValueError):
                group_jobs(invalid)
        for tokens in (["run", "--group", "6"], ["run", "--group", "0", "--job", "0"],
                       ["run", "--group", "0", "--num-workers", "1"],
                       ["smoke", "--num-workers", "2"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                experiment.parser().parse_args(tokens)
        self.assertEqual(experiment.parser().parse_args(["smoke"]).num_workers, 0)

    def test_group_launcher_sequential_mapping(self):
        # Mock child training, not the queue: check all six real launch loops.
        with tempfile.TemporaryDirectory(dir=self.test_root) as tmp:
            with patch.object(experiment, "ROOT", Path(tmp)), \
                 patch.object(experiment, "load_setup", return_value=(self.setup, self.project)), \
                 patch.object(experiment.subprocess, "run") as launch, \
                 contextlib.redirect_stdout(io.StringIO()):
                for i in range(N_GROUPS):
                    args = experiment.parser().parse_args(["run", "--group", str(i), "--device", "cpu"])
                    experiment.launch_jobs(args)
                self.assertEqual(launch.call_count, 30)
                ids = []
                for call in launch.call_args_list:
                    command = call.args[0]
                    ids.append(int(command[command.index("--job") + 1]))
                    self.assertEqual(command[command.index("--num-workers") + 1], "0")
                    self.assertEqual(command[command.index("--device") + 1], "cpu")
                    self.assertTrue(call.kwargs["check"])
                self.assertEqual(ids, list(range(30)))
                for i in range(N_GROUPS):
                    progress = read_json(Path(tmp) / f"output/launchers/group_{i}/progress.json")
                    self.assertEqual(progress["state"], "complete")
                    self.assertEqual(progress["job_ids"], list(range(5 * i, 5 * i + 5)))

    def test_group_stops_on_child_failure(self):
        with tempfile.TemporaryDirectory(dir=self.test_root) as tmp:
            with patch.object(experiment, "ROOT", Path(tmp)), \
                 patch.object(experiment, "load_setup", return_value=(self.setup, self.project)), \
                 patch.object(experiment.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "test")) as launch, \
                 contextlib.redirect_stdout(io.StringIO()):
                args = experiment.parser().parse_args(["run", "--group", "1", "--device", "cpu"])
                with self.assertRaises(subprocess.CalledProcessError):
                    experiment.launch_jobs(args)
                self.assertEqual(launch.call_count, 1)
                progress = read_json(Path(tmp) / "output/launchers/group_1/progress.json")
                self.assertEqual(progress["state"], "stopped")
                self.assertEqual(progress["current_job_id"], 5)

    def test_worker_guard_and_runtime_confinement(self):
        self.assertEqual(engine.NUM_WORKERS, 0)
        with patch.object(engine, "NUM_WORKERS", 1):
            with self.assertRaises(ValueError):
                engine.make_loader([], batch_size=1, shuffle=False, seed=1, device=torch.device("cpu"))
        for name in ("TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH"):
            self.assertIn(ROOT, Path(os.environ[name]).parents)

    def test_local_file_lock(self):
        with tempfile.TemporaryDirectory(dir=self.test_root) as tmp:
            path = Path(tmp) / ".lock"
            with file_lock(path):
                with self.assertRaises(RuntimeError):
                    with file_lock(path):
                        self.fail("Duplicate launcher/job lock was admitted")
            with file_lock(path):
                pass

    def test_write_guard(self):
        self.assertEqual(inside(ROOT / "_tests/check.txt").parent, self.test_root)
        with self.assertRaises(ValueError):
            inside(ROOT.parent / "must_not_be_written.txt")

    def test_historical_reference(self):
        reference = pd.read_csv(ROOT / "reference/original_metrics.csv")
        self.assertEqual(len(reference), 90)
        self.assertTrue(reference.evaluation_cell_coverage.eq(1).all())
        self.assertTrue(reference.n_folds_evaluated.eq(5).all())
        self.assertEqual(verify_inputs(self.project), self.setup["input_sha256"])

    def test_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory(dir=self.test_root) as tmp:
            folder = Path(tmp)
            (folder / "result.txt").write_text("finished")
            atomic_json(folder / "done.json", {"config_hash": "x", "file_sha256": {"result.txt": sha256(folder / "result.txt")}})
            self.assertIsNotNone(experiment.verified_done(folder, "x", ("result.txt",)))
            with self.assertRaises(RuntimeError):
                experiment.verified_done(folder, "different", ("result.txt",))
            (folder / "result.txt").write_text("corrupted")
            with self.assertRaises(RuntimeError):
                experiment.verified_done(folder, "x", ("result.txt",))

    def synthetic_errors(self):
        target = "BE_KUL_Wave_1B"
        stds = read_json(self.project / "data/target_attribute_stds_by_dataset_fold.json")
        rows, expected = [], {attribute: [] for attribute in ("Height", "Weight", "BMI")}
        for fold in range(5):
            for person in range(fold + 2):
                for number, attribute in enumerate(expected, start=1):
                    z = (fold + 1) * (person + 1) * number / 10
                    expected[attribute].append(z ** 2)
                    for repeat in range(1 if person % 2 == 0 else 7):
                        rows.append({"method": "strats_unit_test", "seed": 1234, "dataset": target,
                            "fold": fold, "person_id": f"fictional_{fold}_{person}", "attribute": attribute,
                            "error": z * stds[target][str(fold)][attribute], "input_age": 10., "output_age": 12.,
                            "horizon": 2., "n_input_visits": repeat + 1, "source_config": "unit_test",
                            "source_dataset": "", "n_pretrain_datasets": 0})
        expected_score = float(np.mean([np.sqrt(np.mean(values)) for values in expected.values()]))
        return pd.DataFrame(rows), expected_score

    def test_metric_formula_and_csv_round_trip(self):
        raw, expected = self.synthetic_errors()
        folds = [analyse.fold_metrics(raw[raw.fold.eq(f)].copy(), self.project, "BE_KUL_Wave_1B", f, strict=False) for f in range(5)]
        result, _ = analyse.combine_folds(folds)
        self.assertAlmostEqual(result["nrmse"], expected, places=12)
        with tempfile.TemporaryDirectory(dir=self.test_root) as tmp:
            path = Path(tmp) / "errors.csv.gz"
            atomic_csv(raw, path)
            reconstructed = pd.read_csv(path, float_precision="round_trip", dtype={"person_id": str})
            self.assertEqual(analyse.identity_digest(raw), analyse.identity_digest(reconstructed))

    def test_metric_matches_original_calculator(self):
        path = self.project / "evaluate.py"
        if not path.exists():
            self.skipTest("Optional original aggregation script not present on this server.")
        canonical = module_from_text("exp1_test_canonical", path)
        horizon = canonical.HorizonDefinition(mode="fixed", short_max=3., medium_max=6.)
        auxiliary = canonical.load_auxiliary_data(self.project / "data", horizon)
        raw, expected = self.synthetic_errors()
        with tempfile.TemporaryDirectory(dir=self.test_root) as tmp:
            path = Path(tmp) / "strats_test_seed1234.csv"
            atomic_csv(raw, path)
            result = canonical.process_error_file(path, auxiliary, horizon, 11, Path(tmp), False)
            row = result.dataset_metrics.query("panel == 'primary_9' and horizon_group == 'all'").iloc[0]
            self.assertAlmostEqual(float(row.main_macro_nrmse), expected, places=12)

    def test_training_matches_original(self):
        path = self.project / "strats.py"
        if not path.exists():
            self.skipTest("Optional original STraTS source not present on this server.")
        original = module_from_text("exp1_test_original_model", path)
        frame, pairs, split, subsets = load_data(self.project, engine, smoke=True)
        train, val, test = target_frames(frame, "BE_KUL_Wave_1B", 0, 10, split, subsets, smoke=True)
        current = {name: getattr(engine, name) for name in ("HIDDEN_DIM", "NUM_LAYERS", "NUM_WORKERS", "SEED", "EPOCH_PRINT_EVERY")}
        try:
            results = []
            observations = []
            engine.BATCH_TRACE_CALLBACK = lambda *event: observations.append(event)
            for module in (original, engine):
                module.HIDDEN_DIM = 16
                module.NUM_LAYERS = 1
                module.NUM_WORKERS = 0
                module.SEED = 1234
                module.EPOCH_PRINT_EVERY = 0
                records, validation, _, stats = module.prepare_finetune(train, val, test, pairs, 1234, stats_df=train)
                module.set_seed(203234)
                model = module.new_model(torch.device("cpu"))
                state = module.train_model(model, records, validation, stats, batch_size=4, lr=1e-4,
                    max_epochs=2, patience=10, device=torch.device("cpu"), seed_offset=6000, name="equivalence")
                results.append(state)
            self.assertTrue(observations)
            self.assertEqual(set(results[0]), set(results[1]))
            for name in results[0]:
                self.assertTrue(torch.equal(results[0][name], results[1][name]), name)
        finally:
            for name, value in current.items():
                setattr(engine, name, value)
            engine.TRAIN_TRACE_CALLBACK = None
            engine.BATCH_TRACE_CALLBACK = None

    def test_complete_reporting_pipeline(self):
        # Exercise all 600 metadata slots with clearly synthetic, temporary data.
        # These fixtures never enter output/ and are deleted after this test.
        import shutil
        reference = pd.read_csv(ROOT / "reference/original_metrics.csv")
        expected = read_json(ROOT / "setup/expected_test_counts.json")
        actual_root = analyse.ROOT
        actual_loader = analyse.load_setup
        with tempfile.TemporaryDirectory(dir=self.test_root) as tmp:
            fake = Path(tmp)
            for directory in ("reference", "setup"):
                (fake / directory).mkdir()
            for relative in ("reference/original_metrics.csv", "reference/provenance.json", "PROTOCOL.md", "exclusions.json",
                             "setup/source_counts.csv", "setup/target_partitions.csv", "setup/validation.json", "setup/jobs.csv"):
                shutil.copyfile(ROOT / relative, fake / relative)
            for job in jobs():
                folder = fake / "output/jobs" / f"job_{job['job_id']:03d}"
                config = {"experiment_fingerprint": self.setup["experiment_fingerprint"], "job": job,
                          "smoke": False, "num_workers": 0, "keep_models": False,
                          "libraries": {"torch": "TEST", "numpy": "TEST", "pandas": "TEST", "cuda_build": None, "device_type": "cpu"}}
                atomic_json(folder / "config.json", config)
                atomic_json(folder / "environment.json", {"test_fixture": True})
                config_hash = object_hash(config)
                for arm in ARMS:
                    if job["mode"] == "pretrain":
                        atomic_json(folder / arm / "pretrain/done.json", {"config_hash": config_hash, "file_sha256": {}})
                    for percent in PERCENTS:
                        base = reference[(reference.dataset == job["target"]) & (reference.seed == job["seed"]) &
                            (reference.train_data_percent == percent) & (reference.base_configuration_id == "strats_no_pretrain")].iloc[0]
                        desired = float(base.main_macro_nrmse) * (1.02 if arm == "all_source" else 1.03)
                        for fold in range(5):
                            stage = folder / arm / f"pct{percent}" / f"fold{fold}"
                            count = expected[job["target"]][str(fold)]
                            metrics = {"target": job["target"], "fold": fold, "panel": "primary_9", "horizon_group": "all",
                                "identity_digest": f"synthetic_same_ids_{job['target']}_{fold}", "n_raw_errors": sum(count.values()),
                                "n_primary_errors": sum(count.values()),
                                "n_test_people": int(base.n_test_people) // 5 + (fold < int(base.n_test_people) % 5),
                                "attributes": {attr: {"sum_person_normalized_mse": desired**2,
                                    "sum_person_normalized_mae": desired, "n_people": 1, "n_errors": n} for attr, n in count.items()}}
                            atomic_json(stage / "metrics.json", metrics)
                            done = {"config_hash": config_hash, "sources": source_names(engine, job["target"], arm),
                                    "stage_seed": stage_seed(job["seed"], self.setup["origins"].index(job["target"]), fold),
                                    "n_train_records": 1, "n_test_records": 1, "n_raw_errors": sum(count.values()),
                                    "training": {"epochs_run": 1, "best_epoch": 1, "selected_validation_mse": 1., "elapsed_seconds": 1.},
                                    "file_sha256": {"metrics.json": sha256(stage / "metrics.json")}}
                            for name in ("target_train", "target_val", "target_test", "actual_train"):
                                done[name] = {"n_people": 1}
                            atomic_json(stage / "done.json", done)
            try:
                analyse.ROOT = fake
                analyse.load_setup = lambda: (self.setup, self.project)
                analyse.summarize()
                summary = pd.read_csv(fake / "output/summary/comparison_by_target.csv")
                self.assertEqual(len(summary), 20)
                np.testing.assert_allclose(summary.excluded_gain_pct, -3., atol=1e-12)
                np.testing.assert_allclose(summary.matched_all_source_gain_pct, -2., atol=1e-12)
                self.assertTrue((fake / "results_to_share.zip").exists())
                import zipfile
                with zipfile.ZipFile(fake / "results_to_share.zip") as archive:
                    self.assertFalse(any("errors.csv" in name or "weights" in name for name in archive.namelist()))
            finally:
                analyse.ROOT = actual_root
                analyse.load_setup = actual_loader


if __name__ == "__main__":
    unittest.main(verbosity=2)
