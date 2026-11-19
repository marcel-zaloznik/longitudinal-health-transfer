#!/usr/bin/env python3
"""CPU regression tests for the additive seven-dataset extension and original core."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import copy, tempfile, unittest
from pathlib import Path
from exp3_utils import ROOT, configure_local_runtime, read_json, atomic_json, inside, package_fingerprint, assert_statistics_equal
configure_local_runtime()
import numpy as np
import pandas as pd
import torch
import engine
import tests as original_tests
from seven_utils import (
    ALL_TARGETS, LEGACY_TARGETS, NEW_TARGETS, ARMS, SEEDS, PERCENTS,
    extension_jobs, group_jobs, comparison_job, checked_metric, job_folder,
    verify_baseline_coverage, validate_history,
)
from seven_train import normalizers_from_records, check_source_receipt
from seven_analyse import build_tables, verify_attribute_coverage, report_text
from normalization import PairMetadata
from training import verified_done


def score_fixture():
    rows = []
    for i,target in enumerate(ALL_TARGETS):
        for p in PERCENTS:
            for s in SEEDS:
                for a in ARMS:
                    value = 0.5 + i * 0.05 + (s-1234)*1e-6 + (0.01 if a=="scale_consistent" else 0.0)
                    rows.append({"target":target,"target_label":ALL_TARGETS[target],"percent":p,"seed":s,"arm":a,
                        "nrmse":value,"nmae":value/2,"n_attributes":1,"n_primary_errors":100,"n_test_people":10,"n_folds":5})
    return pd.DataFrame(rows)


class ExtensionTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "setup/manifest.json").is_file(), "Requires prepared scale-consistent experiment")
    def test_original_package_fingerprint_unchanged(self):
        manifest = read_json(ROOT/"setup/manifest.json")
        self.assertEqual(package_fingerprint(),manifest["package_fingerprint"])

    def test_six_new_source_jobs_only(self):
        jobs = extension_jobs()
        self.assertEqual(len(jobs),6)
        self.assertEqual({(j["target"],j["seed"]) for j in jobs},{(t,s) for t in NEW_TARGETS for s in SEEDS})
        self.assertTrue(all(j["target"] not in LEGACY_TARGETS for j in jobs))

    def test_gpu_groups_cover_jobs_without_overlap(self):
        for groups in range(1,7):
            combined = [j["extension_job_id"] for g in range(groups) for j in group_jobs(groups,g)]
            self.assertEqual(sorted(combined),list(range(6)))
        for groups,group in ((0,0),(7,0),(6,6),(2,-1)):
            with self.assertRaises(ValueError): group_jobs(groups,group)

    def test_comparison_ids_do_not_collide_with_old_jobs(self):
        ids = [comparison_job(j,p)["job_id"] for j in extension_jobs() for p in PERCENTS]
        self.assertEqual(sorted(ids),list(range(30,42)))
        self.assertEqual(len(set(ids)),12)
        for j in extension_jobs():
            self.assertTrue(str(job_folder(j)).startswith(str(ROOT/"output/seven_dataset_extension")))

    def test_complete_seven_dataset_tables(self):
        scores = score_fixture()
        paired,overall,dataset,summary = build_tables(scores)
        self.assertEqual(len(scores),126)
        self.assertEqual(len(paired),42)
        self.assertEqual(len(dataset),14)
        self.assertEqual(len(overall),6)
        self.assertEqual(len(summary),2)
        self.assertTrue(overall.n_targets.eq(7).all())
        self.assertTrue(summary.n_targets.eq(7).all())
        for r in overall.itertuples(index=False):
            expected = paired[(paired.percent==r.percent)&(paired.seed==r.seed)].reset_nrmse.mean()
            self.assertAlmostEqual(r.reset_nrmse,expected)
            self.assertAlmostEqual(r.scale_minus_reset_nrmse,0.01)

    def test_original_five_dataset_numbers_do_not_change(self):
        from analyse import build_tables as old_tables
        scores = score_fixture()
        old = old_tables(scores[scores.target.isin(LEGACY_TARGETS)])[2]
        new = build_tables(scores)[2]
        new = new[new.target.isin(LEGACY_TARGETS)]
        key = ["target","percent"]
        pd.testing.assert_frame_equal(old.sort_values(key).reset_index(drop=True),new.sort_values(key).reset_index(drop=True))

    def test_incomplete_or_duplicate_reports_fail(self):
        scores = score_fixture()
        for bad in (scores[scores.target!="YFS"],scores.iloc[:-1],pd.concat([scores,scores.iloc[[0]]])):
            with self.assertRaises(ValueError): build_tables(bad)

    def test_changed_coverage_or_folds_fail(self):
        for col in ("n_primary_errors","n_test_people","n_attributes","n_folds"):
            bad = score_fixture()
            bad.loc[0,col] += 1
            with self.assertRaises(ValueError): build_tables(bad)

    def test_nonfinite_results_fail(self):
        bad = score_fixture()
        bad.loc[0,"nrmse"] = float("nan")
        with self.assertRaises(ValueError): build_tables(bad)

    def test_actual_source_records_match_fast_reconstruction(self):
        frame,pairs,source,train,val,test = original_tests.fixture()
        source_val = frame[frame.PersonID.eq("source_validation")].copy()
        tr,vr,stats = engine.prepare_train_val(source,source_val,pairs,seed=1234,max_train_records=None)
        actual = normalizers_from_records(engine,source,tr)
        expected = PairMetadata(frame,pairs).normalizers(source)
        assert_statistics_equal(actual,expected,"actual_source_normalizers")
        assert_statistics_equal(actual["target"],stats,"actual_source_label_scale")

    def test_extended_strict_metric_checks_new_target(self):
        with tempfile.TemporaryDirectory() as d:
            project = Path(d)
            atomic_json(project/"data/primary_target_panel.json",["Height"])
            atomic_json(project/"data/eligible_dataset_attribute_pairs.json",{"ABCD":{"0":{"Height":{"eligible":True}}}})
            atomic_json(project/"data/target_attribute_stds_by_dataset_fold.json",{"ABCD":{"0":{"Height":2.0}}})
            errors = pd.DataFrame({"dataset":["ABCD"]*2,"fold":[0]*2,"person_id":["a","b"],"attribute":["Height"]*2,
                "error":np.array([1.,-2.],dtype=np.float32),"input_age":[10.,11.],"output_age":[11.,12.],"horizon":[1.,1.],"n_input_visits":[1,1]})
            result = checked_metric(errors,project,"ABCD",0,{"ABCD":{"0":{"Height":2}}})
            self.assertEqual(result["n_primary_errors"],2)
            with self.assertRaises(ValueError): checked_metric(errors,project,"ABCD",0,{"ABCD":{"0":{"Height":3}}})

    def test_attribute_coverage_is_matched(self):
        attrs = pd.DataFrame([{"target":"ABCD","percent":10,"seed":1234,"arm":a,"attribute":"Height",
            "n_errors":100,"n_test_people":10,"n_folds_evaluated":5} for a in ARMS])
        verify_attribute_coverage(attrs)
        attrs.loc[0,"n_errors"] = 99
        with self.assertRaises(ValueError): verify_attribute_coverage(attrs)

    def test_frozen_baseline_coverage_validation(self):
        baseline = score_fixture()
        baseline = baseline[baseline.arm.eq("target_only_validation_selected")].drop(columns="arm")
        baseline["policy"] = "validation_selected"
        expected = {str(f):{"Height":20} for f in range(5)}
        verify_baseline_coverage("ABCD",expected,10,baseline)
        with self.assertRaises(ValueError): verify_baseline_coverage("ABCD",expected,11,baseline)
        with self.assertRaises(ValueError): verify_baseline_coverage("ABCD",expected,10,baseline.iloc[1:])

    def test_smoke_source_receipt_cannot_be_research(self):
        job = extension_jobs()[0]
        with self.assertRaises(RuntimeError):
            check_source_receipt({"job":job,"smoke":True,"experiment_fingerprint":"x"},job,
                {"experiment_fingerprint":"x"},{},{},False)

    def test_corrupt_completed_artifact_is_not_skipped(self):
        from exp3_utils import sha256
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/"best_weights.pt").write_bytes(b"test artifact")
            atomic_json(p/"done.json",{"config_hash":"example","file_sha256":{"best_weights.pt":sha256(p/"best_weights.pt")}})
            self.assertIsNotNone(verified_done(p,"example",("best_weights.pt",)))
            (p/"best_weights.pt").write_bytes(b"corruption")
            with self.assertRaises(RuntimeError): verified_done(p,"example",("best_weights.pt",))

    def test_report_is_one_seven_target_result(self):
        _,_,dataset,overall = build_tables(score_fixture())
        audit = pd.DataFrame({"hit_epoch_cap":[False]*210,"reset_hit_epoch_cap":[False]*210})
        inv = pd.DataFrame({"max_absolute_raw_difference":[1e-6]*210})
        report = report_text(dataset,overall,audit,inv)
        self.assertIn("Seven-target macro",report)
        self.assertIn("ABCD",report)
        self.assertIn("YFS",report)
        self.assertNotIn("ABCD and YFS are untested",report)
        self.assertNotIn("five_target_summary.csv",report)

    def test_combined_export_and_archive_with_mocked_complete_fits(self):
        # Exercise report assembly/publication with clearly synthetic test scores.
        # Actual saved-fit verification is tested separately against the uploaded data.
        import contextlib, io, zipfile
        from unittest.mock import patch
        import seven_analyse as analysis
        scores = score_fixture()
        transfer = scores[scores.arm.ne("target_only_validation_selected")].copy()
        baseline = scores[scores.arm.eq("target_only_validation_selected")].drop(columns="arm").copy()
        baseline["policy"] = "validation_selected"
        attrs = scores.drop(columns=["n_attributes","n_primary_errors","n_folds"]).copy()
        attrs["attribute"],attrs["n_errors"],attrs["n_folds_evaluated"] = "Height",100,5
        attr_baseline = attrs[attrs.arm.eq("target_only_validation_selected")].drop(columns="arm").copy()
        attr_baseline["policy"] = "validation_selected"
        def make_bundle(targets):
            b = analysis.empty_bundle()
            b["scores"] = transfer[transfer.target.isin(targets)].to_dict("records")
            b["attributes"] = attrs[attrs.target.isin(targets)&attrs.arm.ne("target_only_validation_selected")].to_dict("records")
            count = len(targets)*30
            b["audit"] = [{"hit_epoch_cap":False,"reset_hit_epoch_cap":False} for _ in range(count)]
            b["invariance"] = [{"max_absolute_raw_difference":1e-6,"passed":True} for _ in range(count)]
            b["conversion"],b["recomputation"],b["environment"] = [{"test_fixture":True}],[{"test_fixture":True}],[{"test_fixture":True}]
            b["histories"] = [pd.DataFrame({"epoch":[1],"validation_mse":[1.0],"arm":["scale_consistent"]})]
            b["checkpoints"] = [{"target":t,"seed":s,"origin":"SYNTHETIC_UNIT_TEST_ONLY"} for t in targets for s in SEEDS]
            b["source_histories"] = [pd.DataFrame({"epoch":[1],"validation_mse":[1.0]})] if "ABCD" in targets else []
            return b
        old,added = make_bundle(LEGACY_TARGETS),make_bundle(NEW_TARGETS)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            setup = root/"setup7"
            (root/"summary").mkdir()
            (root/"reference").mkdir()
            (root/"setup").mkdir()
            setup.mkdir()
            (root/"_runtime/tmp").mkdir(parents=True)
            (root/"present").mkdir()
            atomic_json(root/"present/done.json",{"fixture":True})
            for name in ("five_target_summary.csv","five_target_scores_by_seed.csv"):
                (root/"summary"/name).write_text("old,test\n1,2\n")
            (root/"summary/REPORT.md").write_text("Previous five-target fixture report")
            (root/"results_to_share.zip").write_bytes(b"Previous test-only archive placeholder")
            pd.DataFrame(old["scores"]).to_csv(setup/"legacy_dataset_scores.csv",index=False)
            baseline.to_csv(root/"reference/exp2_dataset_scores_by_seed.csv",index=False)
            attr_baseline.to_csv(root/"reference/exp2_attribute_scores_by_seed.csv",index=False)
            atomic_json(root/"reference/provenance.json",{"fixture":True})
            pd.DataFrame({"target":list(LEGACY_TARGETS),"attribute":"Height"}).to_csv(root/"setup/normalization_audit.csv",index=False)
            pd.DataFrame({"target":list(NEW_TARGETS),"attribute":"Height"}).to_csv(setup/"normalization_audit.csv",index=False)
            manifest = {"experiment_fingerprint":"synthetic","extension_code_fingerprint":"synthetic","required_libraries":{}}
            legacy = {"experiment_fingerprint":"old-synthetic","package_fingerprint":"old-synthetic","input_sha256":{}}
            with patch.multiple(analysis,ROOT=root,SETUP=setup,CODE_FILES=(),EXT_CODE=(),EXT_DOCS=()), \
                 patch.object(analysis,"load_extension",return_value=(manifest,legacy,root)), \
                 patch.object(analysis,"check_legacy_inventory"), \
                 patch.object(analysis,"collect_legacy",return_value=(old,{},{})), \
                 patch.object(analysis,"collect_added",return_value=added), \
                 patch.object(analysis,"stage_folder",return_value=root/"present"), \
                 contextlib.redirect_stdout(io.StringIO()):
                analysis.summarize()
            self.assertFalse((root/"summary/five_target_summary.csv").exists())
            self.assertTrue((root/"_archive/before_seven_dataset_summary.zip").is_file())
            self.assertEqual(read_json(root/"summary/provenance.json")["n_targets"],7)
            result = pd.read_csv(root/"summary/dataset_scores_by_seed.csv")
            self.assertEqual(len(result),126)
            self.assertEqual(set(result.target),set(ALL_TARGETS))
            self.assertEqual(len(pd.read_csv(root/"summary/dataset_summary.csv")),14)
            with zipfile.ZipFile(root/"results_to_share.zip") as z:
                self.assertIn("summary/seven_target_summary.csv",z.namelist())
                self.assertNotIn("summary/five_target_summary.csv",z.namelist())
                self.assertFalse(any("errors.csv" in n or n.endswith(".pt") for n in z.namelist()))

    def test_outputs_cannot_escape_folder(self):
        with self.assertRaises(ValueError): inside(ROOT.parent/"outside_test_output")


if __name__ == "__main__":
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(original_tests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(ExtensionTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
