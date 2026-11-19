#!/usr/bin/env python3
"""Experiment 3: scale-consistent transfer using Experiment 1 source checkpoints."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, contextlib, gc, os, subprocess, time, traceback
from pathlib import Path
from exp3_utils import (
    ROOT, VERSION, TARGET_LABELS, SEEDS, PERCENTS, inside, configure_local_runtime,
    sha256, object_hash, read_json, atomic_json, atomic_csv, package_fingerprint,
    find_project_root, verify_inputs, verify_references, jobs, grouped_jobs,
    load_data, target_frames, frame_counts, stage_seed, file_lock, Tee, environment,
    assert_statistics_equal, load_setup, check_external_files, receipt_folder,
    control_folder, validate_source_config,
)
configure_local_runtime()


def find_experiment1(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not (candidate / "setup/manifest.json").is_file():
            raise FileNotFoundError("--experiment1 must name the Experiment 1 folder, not its output folder.")
        return candidate
    candidates = []
    for path in ROOT.parent.iterdir():
        manifest = path / "setup/manifest.json"
        if path != ROOT and manifest.is_file():
            try:
                if read_json(manifest).get("experiment_fingerprint") == verify_references()["exp1_experiment_fingerprint"]:
                    candidates.append(path.resolve())
            except (ValueError, OSError):
                pass
    if len(candidates) != 1:
        raise RuntimeError("Cannot uniquely locate sibling Experiment 1. Add --experiment1 '/absolute/path/to/source_exclusion'.")
    return candidates[0]


def prepare(project_arg: str | None = None, exp1_arg: str | None = None) -> None:
    import numpy as np
    import pandas as pd
    import engine
    from metrics import combine_folds
    from normalization import PairMetadata, validate_reconstruction

    with file_lock(ROOT / "setup/.prepare.lock"):
        if (ROOT / "setup/manifest.json").exists():
            setup, project = load_setup()
            if project_arg and Path(project_arg).expanduser().resolve() != project:
                raise RuntimeError("Prepared project path changed. Use a fresh Experiment 3 folder.")
            if exp1_arg and Path(exp1_arg).expanduser().resolve() != Path(setup["experiment1_root"]):
                raise RuntimeError("Prepared Experiment 1 path changed. Use a fresh Experiment 3 folder.")
            check_external_files(setup)
            print("Preparation already complete; inputs and receipts verified. No files in Experiment 1 were modified.")
            return
        project = find_project_root(project_arg)
        exp1 = find_experiment1(exp1_arg)
        if exp1 == ROOT or ROOT in exp1.parents:
            raise RuntimeError("Experiment 1 must be a separate, read-only input folder.")
        ref = verify_references()
        hashes = verify_inputs(project)
        if read_json(ROOT / "reference/exp2_provenance.json")["input_sha256"] != hashes:
            raise RuntimeError("Frozen Experiment 2 baseline used different inputs.")
        old_setup = read_json(exp1 / "setup/manifest.json")
        actual_code_hash = object_hash({n: sha256(exp1 / n) for n in ref["exp1_package_names"]})
        if actual_code_hash != ref["exp1_package_fingerprint"] or old_setup["experiment_fingerprint"] != ref["exp1_experiment_fingerprint"]:
            raise RuntimeError("Experiment 1 code/protocol differs from the completed source-exclusion experiment.")
        if old_setup["input_sha256"] != hashes:
            raise RuntimeError("Experiment 1 used different data or splits.")
        required_jobs = {j["exp1_job_id"]: j for j in jobs()}
        missing = []
        for old_id in required_jobs:
            folder = exp1 / "output/jobs" / f"job_{old_id:03d}"
            for suffix in ("config.json", "all_source/pretrain/best_weights.pt", "all_source/pretrain/done.json"):
                if not (folder / suffix).is_file():
                    missing.append(str((folder / suffix).relative_to(exp1)))
        if missing:
            raise FileNotFoundError("Full Experiment 1 outputs are required (never smoke_output). Missing:\n  " + "\n  ".join(missing[:20]) +
                                    (f"\n  ... {len(missing)} missing files in total." if len(missing) > 20 else ""))
        print("Checking original checkpoints and full-run control receipts...", flush=True)
        import torch
        torch.set_num_threads(2)
        shape_check_model = engine.new_model(torch.device("cpu"))
        inventory, copied = [], []
        def record(path, old_id, expected=None):
            if not path.is_file():
                raise FileNotFoundError(f"Required Experiment 1 input missing: {path}")
            digest = sha256(path)
            if expected is not None and digest != expected:
                raise RuntimeError(f"Experiment 1 hash mismatch: {path}")
            inventory.append({"exp1_job_id": old_id, "relative_path": str(path.relative_to(exp1)), "sha256": digest})
            return digest
        def snapshot(path, data):
            atomic_json(path, data)
            copied.append(path)
        original = {}
        for old_id, job in sorted(required_jobs.items()):
            folder = exp1 / "output/jobs" / f"job_{old_id:03d}"
            config = read_json(folder / "config.json")
            validate_source_config(config, old_id, job["target"], job["seed"], ref)
            config_hash = object_hash(config)
            record(folder / "config.json", old_id)
            snapshot(receipt_folder(job) / "config.json", config)
            pre = folder / "all_source/pretrain"
            pre_done = read_json(pre / "done.json")
            if pre_done["config_hash"] != config_hash:
                raise RuntimeError("Source checkpoint's done receipt does not match its job configuration.")
            record(pre / "done.json", old_id)
            for name in ("best_weights.pt", "history.csv"):
                record(pre / name, old_id, pre_done["file_sha256"][name])
            weights = torch.load(pre / "best_weights.pt", map_location="cpu", weights_only=True)
            shape_check_model.load_state_dict(weights, strict=True)
            if any(not torch.isfinite(tensor).all().item() for tensor in weights.values()):
                raise FloatingPointError("A saved source checkpoint contains nonfinite weights.")
            del weights
            snapshot(receipt_folder(job) / "pretrain_done.json", pre_done)
            controls = {}
            for percent in PERCENTS:
                for fold in range(5):
                    stage = folder / "all_source" / f"pct{percent}" / f"fold{fold}"
                    done = read_json(stage / "done.json")
                    if done["config_hash"] != config_hash:
                        raise RuntimeError("Reset-normalization control belongs to another job configuration.")
                    record(stage / "done.json", old_id)
                    for name in ("metrics.json", "history.csv", "errors.csv.gz"):
                        record(stage / name, old_id, done["file_sha256"][name])
                    metrics = read_json(stage / "metrics.json")
                    cf = receipt_folder(job) / f"pct{percent}" / f"fold{fold}"
                    snapshot(cf / "done.json", done)
                    snapshot(cf / "metrics.json", metrics)
                    controls[(percent, fold)] = (done, metrics)
            original[old_id] = (config, pre_done, controls)
            print(f"  Verified full all-source checkpoint/control job {old_id:03d}: {job['target_label']}, seed {job['seed']}", flush=True)

        print("Reconstructing source-training normalization (no source model training)...", flush=True)
        frame, pairs, split, subsets = load_data(project, engine)
        origins = sorted(frame.Origin.unique().tolist())
        if origins != old_setup["origins"] or len(split) != 5:
            raise ValueError("Dataset origins/folds differ from Experiment 1.")
        if frame.groupby("PersonID").Origin.nunique().max() != 1:
            raise ValueError("Participant IDs are not unique across datasets.")
        if set(frame.PersonID) != set.union(*split) or any(split[i] & split[j] for i in range(5) for j in range(i)):
            raise ValueError("Fold partition coverage/disjointness failed.")
        pair_counts = pairs.groupby("PersonID").size()
        meta = PairMetadata(frame, pairs)
        reconstruction_check = validate_reconstruction(frame, pairs)
        expected, partition_rows, source_rows, normalization_rows = {}, [], [], []
        panel = read_json(project / "data/primary_target_panel.json")
        eligibility = read_json(project / "data/eligible_dataset_attribute_pairs.json")
        comparison = pd.read_csv(ROOT / "reference/exp1_dataset_scores.csv")
        comparison = comparison[comparison["mode"].eq("pretrain") & comparison.arm.eq("all_source")]
        for target, label in TARGET_LABELS.items():
            origin_idx = origins.index(target)
            source_names = list(engine.ORIGIN_MAP[target])
            if set(source_names) != set(origins) - {target}:
                raise ValueError("All-source ordering differs from the reviewed model.")
            source = frame[frame.Origin.isin(source_names)].copy()
            source_train, source_val = engine.make_pretrain_split(source, split, origin_idx)
            if source_train.empty or source_val.empty:
                raise ValueError("Empty source partition; no fallback is silently accepted.")
            norm = meta.normalizers(source_train)
            source_train_counts = frame_counts(source_train, pair_counts)
            source_val_counts = frame_counts(source_val, pair_counts)
            norm["provenance"] = {"target": target, "source_datasets": source_names,
                                  "source_training": source_train_counts,
                                  "source_validation": source_val_counts,
                                  "policy": "Fit only on original source training IDs; source validation/test/target data excluded."}
            norm_path = ROOT / "setup/normalizers" / (target + ".json")
            snapshot(norm_path, norm)
            source_rows.append({"target": target, "target_label": label, "sources": "|".join(source_names),
                                **source_train_counts, "normalization_sha256": sha256(norm_path)})
            for col in engine.TARGET_COLS:
                normalization_rows.append({"target": target, "attribute": col,
                                           "source_input_mean": norm["value"][col]["mean"],
                                           "source_input_std": norm["value"][col]["std"],
                                           "source_input_count": norm["value"][col]["n"],
                                           "source_output_mean": norm["target"][col]["mean"],
                                           "source_output_std": norm["target"][col]["std"],
                                           "source_output_count": norm["target"][col]["n"],
                                           "source_horizon_scale": norm["time"]["horizon_scale"]})
            expected[target] = {}
            target_frame = frame[frame.Origin.eq(target)]
            for fold in range(5):
                test_pairs = pairs[pairs.PersonID.isin(set(target_frame.PersonID) & split[fold])]
                outputs = frame.loc[test_pairs.output_row, panel]
                expected[target][str(fold)] = {a: int(np.isfinite(outputs[a].to_numpy(dtype=float)).sum())
                    for a in panel if eligibility[target][str(fold)][a]["eligible"]}
                if any(n <= 0 for n in expected[target][str(fold)].values()):
                    raise ValueError("An eligible primary outcome has no test observations.")
            for job in [j for j in jobs() if j["target"] == target]:
                config, pre_done, controls = original[job["exp1_job_id"]]
                if pre_done["sources"] != source_names or pre_done["stage_seed"] != stage_seed(job["seed"], origin_idx):
                    raise ValueError("Source composition/seeding mismatch.")
                assert_statistics_equal(source_train_counts, pre_done["source_train"], "source_train_partition")
                assert_statistics_equal(source_val_counts, pre_done["source_val"], "source_val_partition")
                assert_statistics_equal(norm["target"], pre_done["target_statistics"], "source_output_statistics")
                if norm["n_records"] != pre_done["n_train_records"] or source_val_counts["n_pairs"] != pre_done["n_val_records"]:
                    raise ValueError("Source training/validation record counts differ from the saved checkpoint receipt.")
                scores = []
                for fold in range(5):
                    train, val, test = target_frames(frame, target, fold, job["percent"], split, subsets)
                    done, metric = controls[(job["percent"], fold)]
                    if done["sources"] != source_names or done["stage_seed"] != stage_seed(job["seed"], origin_idx, fold):
                        raise ValueError("Control composition/seeding mismatch.")
                    for role, part in (("target_train", train), ("target_val", val), ("target_test", test)):
                        counts = frame_counts(part, pair_counts)
                        assert_statistics_equal(counts, done[role], role)
                        if job["seed"] == SEEDS[0]:
                            partition_rows.append({"target": target, "percent": job["percent"], "fold": fold, "role": role, **counts})
                    assert_statistics_equal(done["actual_train"], done["target_train"], "target_only_finetuning")
                    assert_statistics_equal(meta.target_statistics(train), done["target_statistics"], "target_output_statistics")
                    if metric["target"] != target or metric["fold"] != fold:
                        raise ValueError("Control metric has the wrong target/fold.")
                    observed = {a: s["n_errors"] for a, s in metric["attributes"].items()}
                    if observed != expected[target][str(fold)]:
                        raise ValueError("Control primary-panel coverage differs from the original frozen eligibility panel.")
                    scores.append(metric)
                aggregate, _ = combine_folds(scores)
                old_row = comparison[comparison.target.eq(target) & comparison.percent.eq(job["percent"]) & comparison.seed.eq(job["seed"])]
                if len(old_row) != 1:
                    raise ValueError("Frozen control score key is missing/duplicated.")
                old_row = old_row.iloc[0]
                for quantity in ("nrmse", "nmae", "n_attributes", "n_primary_errors", "n_test_people", "n_folds"):
                    if not np.isclose(aggregate[quantity], old_row[quantity], rtol=1e-12, atol=1e-12):
                        raise ValueError(f"Control metrics differ from the completed Experiment 1 report: {target}/{quantity}")
            print(f"  Reconstructed and verified: {label}; {norm['n_records']} source-training records", flush=True)
        if expected != read_json(exp1 / "setup/expected_test_counts.json"):
            raise ValueError("Primary-panel expected counts differ from Experiment 1 preparation.")
        # The frozen baseline is already validation-selected in Experiment 2.
        baseline = pd.read_csv(ROOT / "reference/exp2_dataset_scores_by_seed.csv")
        baseline = baseline[baseline.policy.eq("validation_selected") & baseline.target.isin(TARGET_LABELS)]
        if len(baseline) != 30 or baseline.duplicated(["target", "percent", "seed"]).any():
            raise ValueError("Frozen Experiment 2 baseline is incomplete.")
        for row in baseline.itertuples(index=False):
            match = comparison[comparison.target.eq(row.target) & comparison.percent.eq(row.percent) & comparison.seed.eq(row.seed)].iloc[0]
            for col in ("n_attributes", "n_primary_errors", "n_test_people", "n_folds"):
                if getattr(row, col) != match[col]:
                    raise ValueError("Baseline/control evaluation coverage mismatch.")
        snapshot(ROOT / "setup/expected_test_counts.json", expected)
        snapshot(ROOT / "setup/validation.json", {"passed": True, "source_checkpoints": 15, "controls_verified": 150,
            "source_reconstruction": reconstruction_check, "folds_disjoint": True,
            "pair_ownership_and_future_outputs_checked": True, "original_summary_scores_match": True,
            "source_output_statistics_match_checkpoints": True, "target_output_statistics_match_controls": True,
            "baseline_coverage_matches": True, "n_people": int(frame.PersonID.nunique()),
            "n_visits": len(frame), "n_prediction_pairs": len(pairs)})
        for name, rows in (("jobs.csv", jobs()), ("source_counts.csv", source_rows),
                           ("target_partitions.csv", partition_rows), ("normalization_audit.csv", normalization_rows)):
            path = ROOT / "setup" / name
            atomic_csv(pd.DataFrame(rows), path)
            copied.append(path)
        manifest = {"version": VERSION, "project_root": str(project), "experiment1_root": str(exp1),
            "package_fingerprint": package_fingerprint(), "input_sha256": hashes,
            "origins": origins, "targets": list(TARGET_LABELS), "seeds": list(SEEDS), "percents": list(PERCENTS),
            "jobs": jobs(), "n_jobs": 30, "n_new_finetuning_fits": 150, "n_new_pretraining_fits": 0,
            "exp1_experiment_fingerprint": ref["exp1_experiment_fingerprint"],
            "external_inventory": inventory,
            "setup_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(copied))}}
        manifest["experiment_fingerprint"] = object_hash({
            "code": manifest["package_fingerprint"], "inputs": hashes,
            "receipts_and_normalization": manifest["setup_sha256"], "original_inputs": inventory})
        atomic_json(ROOT / "setup/manifest.json", manifest)
        print("Prepared: 15 reused source checkpoints; 30 jobs x 5 folds = 150 NEW fine-tuning fits; zero pretraining fits.")
        print("Next: python experiment.py smoke --device cuda:0")


def configure_engine(job, smoke):
    import engine
    engine.SEED = job["seed"]
    engine.MODE = "topk_pretrain"
    engine.TRAIN_DATA_PERCENT = job["percent"]
    engine.NUM_WORKERS = 0
    engine.EPOCH_PRINT_EVERY = 1
    engine.FINETUNE_BATCH_SIZE = 4 if smoke else 256
    engine.MAX_FINETUNE_EPOCHS = 1 if smoke else 1000
    engine.FINAL_ERROR_PATH = inside(ROOT / "output/unused_legacy_errors.csv")
    if (engine.HIDDEN_DIM, engine.MAX_OBS, engine.PATIENCE, engine.FINETUNE_LR,
        engine.MIN_DELTA, engine.WEIGHT_DECAY) != (256, 256, 10, 1e-4, 1e-5, 1e-5):
        raise RuntimeError("Unexpected model/optimization settings.")
    return engine


def run_job(job: dict, args, smoke: bool = False) -> None:
    import numpy as np
    import torch
    from normalization import prepare_scale_consistent, map_and_check
    from metrics import fold_metrics
    from training import choose_device, save_weights, training, verified_done
    setup, project = load_setup()
    engine = configure_engine(job, smoke)
    device = choose_device(args.device)
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    check_external_files(setup, job["exp1_job_id"], include_errors=False)
    current_env = environment(torch, device)
    original_config = read_json(receipt_folder(job) / "config.json")
    library_keys = ("torch", "numpy", "pandas", "cuda_build", "device_type")
    current_libraries = {k: current_env[k] for k in library_keys}
    if not smoke and current_libraries != original_config["libraries"]:
        raise RuntimeError("Use the SAME Python package/CUDA environment as Experiment 1.\n"
            f"Expected: {original_config['libraries']}\nObserved: {current_libraries}\n"
            "The package does not silently compare different software stacks.")
    if not smoke:
        smoke_marker = ROOT / "smoke_output/smoke_passed.json"
        if not smoke_marker.exists() or read_json(smoke_marker).get("experiment_fingerprint") != setup["experiment_fingerprint"]:
            raise RuntimeError("Run the smoke test successfully before research training.")
    folder = inside(ROOT / ("smoke_output" if smoke else "output") / "jobs" / f"job_{job['job_id']:03d}")
    folder.mkdir(parents=True, exist_ok=True)
    config = {"experiment_fingerprint": setup["experiment_fingerprint"], "job": job, "smoke": smoke,
              "libraries": current_libraries, "keep_models": bool(args.keep_models),
              "num_workers": 0, "cpu_threads": current_env["cpu_threads"],
              "normalization": "source inputs/time/context; target labels/loss; affine-converted output head"}
    config_hash = object_hash(config)
    with file_lock(folder / ".job.lock"), (folder / "training.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            if (folder / "config.json").exists() and read_json(folder / "config.json") != config:
                raise RuntimeError("Job configuration/library/thread settings changed. Resume with the identical command/environment.")
            atomic_json(folder / "config.json", config)
            atomic_json(folder / "environment.json", current_env)
            print(f"\nJOB {job['job_id']:03d}: {job['target_label']}, {job['percent']}%, seed {job['seed']}; {device}; smoke={smoke}", flush=True)
            norm = read_json(ROOT / "setup/normalizers" / (job["target"] + ".json"))
            pre_done = read_json(receipt_folder(job) / "pretrain_done.json")
            checkpoint = Path(setup["experiment1_root"]) / "output/jobs" / f"job_{job['exp1_job_id']:03d}" / "all_source/pretrain/best_weights.pt"
            if sha256(checkpoint) != pre_done["file_sha256"]["best_weights.pt"]:
                raise RuntimeError("Source checkpoint changed since preparation.")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            frame, pairs, split, subsets = load_data(project, engine, target=job["target"])
            if smoke:
                pairs = pairs.groupby("PersonID", sort=False).head(3).copy()
            pair_counts = pairs.groupby("PersonID").size()
            origin_idx = setup["origins"].index(job["target"])
            folds = [0] if smoke else range(5)
            for fold in folds:
                stage = folder / f"fold{fold}"
                if verified_done(stage, config_hash, ("errors.csv.gz", "metrics.json", "history.csv", "invariance.json")):
                    print(f"Skip completed fold {fold}", flush=True)
                    continue
                train, val, test = target_frames(frame, job["target"], fold, job["percent"], split, subsets, smoke=smoke)
                print(f"fold {fold}: building target records with frozen source normalization...", flush=True)
                train_r, val_r, test_r, stats = prepare_scale_consistent(train, val, test, pairs, norm)
                if not smoke:
                    control = read_json(control_folder(job, fold) / "done.json")
                    assert_statistics_equal(stats, control["target_statistics"], "target_loss_statistics")
                    for role, part in (("target_train", train), ("target_val", val), ("target_test", test)):
                        assert_statistics_equal(frame_counts(part, pair_counts), control[role], role)
                    for name, records in (("n_train_records", train_r), ("n_val_records", val_r), ("n_test_records", test_r)):
                        if len(records) != control[name]:
                            raise ValueError(f"Target record membership differs from the reset control: {name}")
                seed = stage_seed(job["seed"], origin_idx, fold)
                engine.set_seed(seed)
                model = engine.new_model(device)  # Same RNG consumption as the control.
                model.load_state_dict(state, strict=True)
                invariance = map_and_check(model, norm["target"], stats, train_r, device)
                atomic_json(stage / "invariance.json", invariance)
                print(f"fold {fold}: initial prediction invariance PASSED; pairs train={len(train_r)}, val={len(val_r)}, test={len(test_r)}", flush=True)
                best, details = training(engine, model, train_r, val_r, stats,
                    device=device, batch=engine.FINETUNE_BATCH_SIZE, lr=engine.FINETUNE_LR,
                    epochs=engine.MAX_FINETUNE_EPOCHS, offset=2000 * (origin_idx + 1) + fold,
                    name=f"scale-consistent {job['target']} {job['percent']}% fold {fold}", folder=stage)
                errors = engine.collect_errors(model, test_r, stats, job["target"], fold, device,
                    method_name="strats_exp3_scale_consistent", source_config="exp1_all_source",
                    source_dataset="|".join(pre_done["sources"]), n_pretrain_datasets=6, seed=job["seed"])
                usable = np.array([stats[c]["n"] > 0 for c in engine.TARGET_COLS])
                expected_rows = sum(int((np.isfinite(r["target"]) & usable).sum()) for r in test_r)
                if len(errors) != expected_rows or not np.isfinite(errors.error.to_numpy()).all():
                    raise FloatingPointError("Missing/nonfinite test errors.")
                errors["train_data_percent"] = job["percent"]
                errors["experiment_arm"] = "scale_consistent"
                errors["job_id"] = job["job_id"]
                metric = fold_metrics(errors, project, job["target"], fold, strict=not smoke)
                if not smoke:
                    old_metric = read_json(control_folder(job, fold) / "metrics.json")
                    if metric["identity_digest"] != old_metric["identity_digest"]:
                        raise ValueError("New/control evaluation observations differ; refusing an unmatched result.")
                atomic_csv(errors, stage / "errors.csv.gz")
                atomic_json(stage / "metrics.json", metric)
                files = ["errors.csv.gz", "metrics.json", "history.csv", "invariance.json"]
                if args.keep_models:
                    save_weights(torch, best, stage / "best_weights.pt")
                    files.append("best_weights.pt")
                done = {"config_hash": config_hash, "experiment_fingerprint": setup["experiment_fingerprint"],
                    "smoke": smoke, "stage_seed": seed, "job": job, "fold": fold,
                    "source_checkpoint_sha256": pre_done["file_sha256"]["best_weights.pt"],
                    "normalization_sha256": sha256(ROOT / "setup/normalizers" / (job["target"] + ".json")),
                    "target_statistics": stats, "n_train_records": len(train_r), "n_val_records": len(val_r),
                    "n_test_records": len(test_r), "n_raw_errors": len(errors),
                    "target_train": frame_counts(train, pair_counts), "target_val": frame_counts(val, pair_counts),
                    "target_test": frame_counts(test, pair_counts), "training": details, "environment": current_env,
                    "file_sha256": {n: sha256(stage / n) for n in files}}
                atomic_json(stage / "done.json", done)
                print(f"Saved completed fold {fold}: {stage.relative_to(ROOT)}", flush=True)
                del model, best, train_r, val_r, test_r, stats, errors, train, val, test
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            atomic_json(folder / "job_done.json", {"config_hash": config_hash, "smoke": smoke, "target_fits": 1 if smoke else 5})
            print(f"COMPLETED job {job['job_id']:03d}", flush=True)


def launch(args):
    selected = jobs() if args.all else grouped_jobs(args.groups)[args.group]
    label = "all" if args.all else f"groups{args.groups}_group{args.group}"
    print(f"{label}: {len(selected)} sequential jobs, IDs {[j['job_id'] for j in selected]}, device={args.device}; one fit at a time.", flush=True)
    for job in selected:
        print(f"  job {job['job_id']:03d}: {job['target_label']}, {job['percent']}%, seed {job['seed']}", flush=True)
    if args.dry_run:
        return
    load_setup()
    folder = ROOT / "output/launchers" / label
    with file_lock(folder / ".launcher.lock"):
        for position, job in enumerate(selected, 1):
            atomic_json(folder / "progress.json", {"state": "running", "position": position, "total": len(selected),
                "job_id": job["job_id"], "updated_unix_time": time.time()})
            command = [sys.executable, "-B", "-u", str(ROOT / "experiment.py"), "run", "--job", str(job["job_id"]), "--device", args.device]
            if args.keep_models:
                command.append("--keep-models")
            if args.cpu_threads is not None:
                command += ["--cpu-threads", str(args.cpu_threads)]
            try:
                subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
            except BaseException:
                atomic_json(folder / "progress.json", {"state": "stopped", "job_id": job["job_id"], "updated_unix_time": time.time()})
                raise
        atomic_json(folder / "progress.json", {"state": "complete", "job_ids": [j["job_id"] for j in selected]})
        print(f"COMPLETED {label}.", flush=True)


def status():
    complete = 0
    for job in jobs():
        folder = ROOT / "output/jobs" / f"job_{job['job_id']:03d}"
        count = sum((folder / f"fold{fold}" / "done.json").exists() for fold in range(5))
        complete += count
        print(f"job {job['job_id']:03d}  {job['target_label']:17s} {job['percent']:3d}%  seed {job['seed']}  {count}/5 fits")
    print(f"Total: {complete}/150 completed fits (presence check only; summarize performs integrity checks).")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare", help="Read-only input/checkpoint checks and source-normalization reconstruction.")
    q.add_argument("--project-root")
    q.add_argument("--experiment1")
    for command in ("smoke", "run"):
        q = sub.add_parser(command)
        q.add_argument("--device", default="cuda:0")
        q.add_argument("--cpu-threads", type=int, default=2, help="PyTorch CPU threads per GPU process; keep unchanged on resume.")
        q.add_argument("--keep-models", action="store_true", help="Optionally retain fine-tuned weights; not needed for the report.")
        if command == "run":
            choose = q.add_mutually_exclusive_group(required=True)
            choose.add_argument("--all", action="store_true")
            choose.add_argument("--group", type=int)
            choose.add_argument("--job", type=int, choices=range(30))
            q.add_argument("--groups", type=int, default=6)
            q.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    sub.add_parser("summarize", help="Verify all 150 fits, recompute new metrics, build an aggregate-only share ZIP.")
    return p


def main():
    args = parser().parse_args()
    if getattr(args, "cpu_threads", 2) < 1:
        raise ValueError("cpu-threads must be positive.")
    if args.command == "prepare":
        prepare(args.project_root, args.experiment1)
    elif args.command == "status":
        status()
    elif args.command == "summarize":
        from analyse import summarize
        summarize()
    elif args.command == "smoke":
        # Two tiny one-epoch tests; BOTH use real full-run source checkpoints.
        # They test a small cohort and a long-prefix cohort, outside research output.
        for job_id in (6, 27):
            run_job(jobs()[job_id], args, smoke=True)
        setup, _ = load_setup(check_data=False)
        atomic_json(ROOT / "smoke_output/smoke_passed.json", {"passed": True, "research_results": False,
            "experiment_fingerprint": setup["experiment_fingerprint"], "device": args.device,
            "target_fits": 2, "source_fits": 0})
        print("Smoke test PASSED. No smoke result enters the research summary.")
    elif args.command == "run":
        if not 1 <= args.groups <= 30:
            raise ValueError("groups must be between 1 and 30.")
        if args.group is not None and not 0 <= args.group < args.groups:
            raise ValueError("group must be in 0..groups-1.")
        if args.job is not None:
            if args.dry_run:
                print(jobs()[args.job])
            else:
                run_job(jobs()[args.job], args)
        else:
            launch(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nStopped: {exc}\nNo original files were modified. Completed fits remain available; use the identical command to resume.", file=sys.stderr)
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exc()
        raise SystemExit(1)
