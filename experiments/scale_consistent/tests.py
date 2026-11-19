#!/usr/bin/env python3
"""CPU tests with synthetic records only. No original project data are modified."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import copy, json, math, os, tempfile, unittest
from pathlib import Path
from exp3_utils import (
    ROOT, configure_local_runtime, inside, atomic_json, atomic_csv, read_json, Tee,
    jobs, grouped_jobs, verify_references, validate_source_config, assert_statistics_equal,
    SEEDS, PERCENTS, TARGET_LABELS,
)
configure_local_runtime()
import numpy as np
import pandas as pd
import torch
import engine
from normalization import PairMetadata, prepare_scale_consistent, map_and_check, map_head_in_place, validate_reconstruction
from metrics import fold_metrics, combine_folds
from analyse import read_errors, build_tables, ARMS
from training import training, save_weights, verified_done

torch.set_num_threads(2)


def fixture():
    rows, pair_rows = [], []
    specs = [("source_long", 18), ("source_short", 5), ("source_none", 4),
             ("source_validation", 5), ("target_train_1", 4), ("target_train_2", 4),
             ("target_validation", 4), ("target_test", 4)]
    for pi, (pid, visits) in enumerate(specs):
        indices = []
        for v in range(visits):
            idx = 10 + 2 * len(rows)
            indices.append(idx)
            row = {"_index": idx, "PersonID": pid,
                   "Origin": "S" if pid.startswith("source") else "T",
                   "Sex": "Male" if pi % 2 == 0 else "Female",
                   "Age": 10.0 + pi * 2.0 + v * 0.73}
            for j, col in enumerate(engine.TARGET_COLS):
                row[col] = (j + 1) * 0.07 + pi * 0.13 + v * (0.01 + j * 0.001)
            row["Height"] = 140.0 + pi * 2.0 + v * 0.81
            row["Weight"] = 40.0 + pi * 1.1 + v * 0.37
            if pid.startswith("source"):
                row["Diet_T2P"] = np.nan  # Source-absent but target-observed outcome.
            if pid == "source_none":
                for col in engine.TARGET_COLS:
                    row[col] = np.nan
            rows.append(row)
        for v in range(1, visits):
            pair_rows.append({"PersonID": pid, "input_rows": tuple(indices[:v]), "output_row": indices[v]})
            if v >= 2:
                pair_rows.append({"PersonID": pid, "input_rows": (indices[v - 1],), "output_row": indices[v]})
    frame = pd.DataFrame(rows).set_index("_index")
    frame.index.name = None
    pairs = pd.DataFrame(pair_rows)
    source = frame[frame.PersonID.isin(["source_long", "source_short", "source_none"])].copy()
    train = frame[frame.PersonID.isin(["target_train_1", "target_train_2"])].copy()
    val = frame[frame.PersonID.eq("target_validation")].copy()
    test = frame[frame.PersonID.eq("target_test")].copy()
    return frame, pairs, source, train, val, test


class Experiment3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.pairs, cls.source, cls.train, cls.val, cls.test = fixture()
        cls.meta = PairMetadata(cls.frame, cls.pairs)
        cls.norm = cls.meta.normalizers(cls.source)

    def test_01_normalization_matches_original_records(self):
        records = engine.build_records(self.source, self.pairs, self.norm["value"], self.source, engine.MAX_OBS)
        expected = {"value": engine.compute_value_stats(self.source), "time": engine.compute_time_stats(records),
                    "context": engine.compute_demo_stats(records), "target": engine.compute_target_stats(records),
                    "n_records": len(records)}
        assert_statistics_equal(self.norm, expected, "normalization")
        self.assertTrue(any(len(r["values"]) == engine.MAX_OBS for r in records))
        self.assertTrue(any(r["varis"].tolist() == [engine.NO_OBS_VAR_ID] for r in records))
        self.assertEqual(self.norm["target"]["Diet_T2P"]["n"], 0)

    def test_02_real_builder_sample_validator(self):
        audit = validate_reconstruction(self.frame, self.pairs)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["origin_count"], 2)

    def test_03_source_statistics_do_not_use_other_partitions(self):
        changed = self.frame.copy()
        outside = ~changed.PersonID.isin(set(self.source.PersonID))
        changed.loc[outside, engine.TARGET_COLS] = changed.loc[outside, engine.TARGET_COLS] * 97 + 1000
        changed.loc[outside, "Age"] = changed.loc[outside, "Age"] * 3 + 100
        new = PairMetadata(changed, self.pairs).normalizers(changed.loc[self.source.index])
        assert_statistics_equal(new, self.norm, "train_only_source")

    def test_04_target_labels_masks_and_loss_match_original(self):
        a = prepare_scale_consistent(self.train, self.val, self.test, self.pairs, self.norm)
        b = engine.prepare_finetune(self.train, self.val, self.test, self.pairs, seed=1234, stats_df=self.train)
        assert_statistics_equal(a[3], b[3], "target_statistics")
        for part_a, part_b in zip(a[:3], b[:3]):
            self.assertEqual(len(part_a), len(part_b))
            for ra, rb in zip(part_a, part_b):
                for key in ("target", "target_mask"):
                    np.testing.assert_array_equal(ra[key], rb[key])
                for key in ("person_id", "input_row", "output_row"):
                    self.assertEqual(ra[key], rb[key])
        torch.testing.assert_close(engine.loss_weights(a[3], torch.device("cpu")), engine.loss_weights(b[3], torch.device("cpu")), rtol=0, atol=0)
        changed = self.val.copy()
        changed.loc[:, engine.TARGET_COLS] = changed[engine.TARGET_COLS] + 100
        c = prepare_scale_consistent(self.train, changed, self.test, self.pairs, self.norm)
        assert_statistics_equal(a[3], c[3], "no_target_validation_statistics")

    def test_05_input_coordinates_are_source_coordinates(self):
        train, _, _, _ = prepare_scale_consistent(self.train, self.val, self.test, self.pairs, self.norm)
        original = engine.build_records(self.train, self.pairs, self.norm["value"], self.train, engine.MAX_OBS)
        engine.apply_time_stats(original, self.norm["time"])
        engine.apply_demo_stats(original, {int(k): v for k, v in self.norm["context"].items()})
        for actual, expected in zip(train, original):
            for key in ("values", "times", "varis", "demo"):
                np.testing.assert_array_equal(actual[key], expected[key])

    def test_06_head_mapping_preserves_predictions_and_rng(self):
        train, _, _, stats = prepare_scale_consistent(self.train, self.val, self.test, self.pairs, self.norm)
        engine.set_seed(1234)
        model = engine.new_model(torch.device("cpu"))
        original = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        audit = map_and_check(model, self.norm["target"], stats, train, torch.device("cpu"))
        self.assertTrue(audit["passed"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertIn("Diet_T2P", audit["conversion"]["source_unobserved_target_observed"])
        for key, value in model.state_dict().items():
            if not key.startswith("head."):
                self.assertTrue(torch.equal(value, original[key]), key)

    def test_07_identity_head_conversion(self):
        model = engine.new_model(torch.device("cpu"))
        w, b = model.head.weight.detach().clone(), model.head.bias.detach().clone()
        map_head_in_place(model, self.norm["target"], self.norm["target"])
        self.assertTrue(torch.equal(w, model.head.weight))
        self.assertTrue(torch.equal(b, model.head.bias))

    def test_08_invalid_head_scaling_fails(self):
        model = engine.new_model(torch.device("cpu"))
        invalid = copy.deepcopy(self.norm["target"])
        invalid["Height"]["std"] = 0.0
        with self.assertRaises(ValueError):
            map_head_in_place(model, self.norm["target"], invalid)

    def test_09_metric_aggregation_is_person_balanced_across_folds(self):
        def fold(total, people, errors):
            return {"attributes": {"Height": {"sum_person_normalized_mse": total,
                "sum_person_normalized_mae": total, "n_people": people, "n_errors": errors}},
                "n_primary_errors": errors, "n_test_people": people}
        aggregate, _ = combine_folds([fold(1.0, 1, 10), fold(13.0, 2, 3)])
        self.assertAlmostEqual(aggregate["nrmse"], math.sqrt(14 / 3))
        self.assertNotAlmostEqual(aggregate["nrmse"], (1 + math.sqrt(13 / 2)) / 2)

    @unittest.skipUnless((ROOT / "reference/exp1_stage_audit.csv").is_file(), "Requires local source-exclusion references")
    def test_10_groups_cover_all_fits_without_overlap(self):
        self.assertEqual(len(jobs()), 30)
        self.assertEqual(len({j["exp1_job_id"] for j in jobs()}), 15)
        for n in (1, 2, 3, 6, 8, 30):
            groups = grouped_jobs(n)
            ids = [j["job_id"] for group in groups for j in group]
            self.assertEqual(sorted(ids), list(range(30)))
            self.assertEqual(groups, grouped_jobs(n))

    @unittest.skipUnless((ROOT / "reference/provenance.json").is_file(), "Requires local reference provenance")
    def test_11_references_and_smoke_rejection(self):
        ref = verify_references()
        job = jobs()[0]
        config = {"job": {"job_id": 0, "target": job["target"], "seed": job["seed"], "mode": "pretrain"},
                  "smoke": False, "experiment_fingerprint": ref["exp1_experiment_fingerprint"], "num_workers": 0}
        validate_source_config(config, 0, job["target"], job["seed"], ref)
        config["smoke"] = True
        with self.assertRaises(RuntimeError):
            validate_source_config(config, 0, job["target"], job["seed"], ref)

    def test_12_output_containment_and_runtime(self):
        with self.assertRaises(ValueError):
            inside(ROOT / "../outside.txt")
        for key in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "TRITON_CACHE_DIR"):
            inside(Path(os.environ[key]))
        with tempfile.TemporaryDirectory(dir=inside(ROOT / "_runtime/tmp")) as tmp:
            link = Path(tmp) / "escape"
            try:
                link.symlink_to(ROOT.parent, target_is_directory=True)
            except OSError:
                return
            with self.assertRaises(ValueError):
                inside(link / "outside.txt")

    def test_13_end_to_end_cpu_training_save_and_float32_metric_roundtrip(self):
        train, val, test, stats = prepare_scale_consistent(self.train, self.val, self.test, self.pairs, self.norm)
        engine.set_seed(4321)
        model = engine.new_model(torch.device("cpu"))
        map_and_check(model, self.norm["target"], stats, train, torch.device("cpu"))
        with tempfile.TemporaryDirectory(dir=inside(ROOT / "_runtime/tmp")) as tmp:
            folder = Path(tmp)
            best, detail = training(engine, model, train, val, stats, device=torch.device("cpu"), batch=4,
                lr=1e-4, epochs=1, offset=123, name="CPU integration test", folder=folder)
            self.assertEqual(detail["epochs_run"], 1)
            save_weights(torch, best, folder / "weights.pt")
            restored = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
            self.assertEqual(set(best), set(restored))
            errors = engine.collect_errors(model, test, stats, "T", 0, torch.device("cpu"),
                method_name="unit_test", source_config="synthetic", source_dataset="S", n_pretrain_datasets=1, seed=1234)
            data = folder / "data"
            data.mkdir()
            for name, value in (("primary_target_panel.json", ["Height", "Weight"]),
                ("target_attribute_stds_by_dataset_fold.json", {"T": {"0": {"Height": 2., "Weight": 3.}}}),
                ("eligible_dataset_attribute_pairs.json", {"T": {"0": {"Height": {"eligible": True}, "Weight": {"eligible": True}}}})):
                atomic_json(data / name, value)
            metric = fold_metrics(errors, folder, "T", 0, strict=False)
            atomic_csv(errors, folder / "errors.csv.gz")
            reloaded = read_errors(folder / "errors.csv.gz")
            self.assertEqual(str(reloaded.error.dtype), "float32")
            assert_statistics_equal(fold_metrics(reloaded, folder, "T", 0, strict=False), metric, "roundtrip")
            from exp3_utils import sha256
            atomic_json(folder / "done.json", {"config_hash": "test", "file_sha256": {"errors.csv.gz": sha256(folder / "errors.csv.gz")}})
            self.assertIsNotNone(verified_done(folder, "test", ("errors.csv.gz",)))
            with self.assertRaises(RuntimeError):
                verified_done(folder, "changed", ("errors.csv.gz",))

    def test_14_summary_pairing_and_five_target_macro(self):
        rows = []
        for ti, target in enumerate(TARGET_LABELS):
            for percent in PERCENTS:
                for si, seed in enumerate(SEEDS):
                    for arm, offset in zip(ARMS, (0.0, -0.1, 0.2)):
                        rows.append({"target": target, "percent": percent, "seed": seed, "arm": arm,
                            "nrmse": 1 + ti * 0.1 + si * 0.01 + offset,
                            "n_folds": 5, "n_attributes": 4, "n_primary_errors": 1000 + ti, "n_test_people": 10})
        scores = pd.DataFrame(rows)
        paired, macro, ds, summary = build_tables(scores)
        self.assertEqual((len(paired), len(macro), len(ds), len(summary)), (30, 6, 10, 2))
        np.testing.assert_allclose(paired.scale_minus_reset_nrmse, -0.1)
        np.testing.assert_allclose(summary.scale_minus_reset_nrmse_mean, -0.1)
        self.assertTrue(summary.n_targets.eq(5).all())
        with self.assertRaises(ValueError):
            build_tables(scores.iloc[:-1])
        with self.assertRaises(ValueError):
            build_tables(pd.concat([scores, scores.iloc[:1]]))


if __name__ == "__main__":
    with inside(ROOT / "UNIT_TEST_RESULTS.txt").open("w", encoding="utf-8") as handle:
        stream = Tee(sys.stdout, handle)
        stream.write(f"Synthetic CPU tests; torch={torch.__version__}, numpy={np.__version__}, pandas={pd.__version__}\n")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Experiment3Tests)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
