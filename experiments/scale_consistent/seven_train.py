"""Six added source jobs; each runs both fractions, five folds and both arms."""
from __future__ import annotations
import contextlib, gc, sys
from pathlib import Path
from exp3_utils import (
    ROOT, PERCENTS, inside, read_json, atomic_json, atomic_csv, object_hash, sha256,
    load_data, target_frames, frame_counts, stage_seed, environment, file_lock, Tee,
    assert_statistics_equal,
)
from seven_utils import (
    SETUP, OUTPUT, SMOKE, LIBRARY_KEYS, TRAINING_ARMS, load_extension,
    job_folder, stage_folder, normalizer_path, checked_metric,
)


def configure_engine(job, smoke):
    import engine
    engine.SEED = job["seed"]
    engine.MODE = "topk_pretrain"
    engine.N_ORIGINS = 6
    engine.NUM_WORKERS = 0
    engine.EPOCH_PRINT_EVERY = 1
    engine.PRETRAIN_BATCH_SIZE = engine.FINETUNE_BATCH_SIZE = 4 if smoke else 256
    engine.MAX_PRETRAIN_EPOCHS = engine.MAX_FINETUNE_EPOCHS = 1 if smoke else 1000
    engine.FINAL_ERROR_PATH = inside(OUTPUT / "unused_legacy_errors.csv")
    if (engine.HIDDEN_DIM, engine.MAX_OBS, engine.PATIENCE, engine.PRETRAIN_LR,
        engine.FINETUNE_LR, engine.MIN_DELTA, engine.WEIGHT_DECAY) != (256, 256, 10, 5e-4, 1e-4, 1e-5, 1e-5):
        raise RuntimeError("Unexpected model/optimization configuration.")
    return engine


def small_source(source, split):
    keep = set()
    for _, group in source.groupby("Origin", sort=False):
        people = set(group.PersonID)
        keep.update(sorted(people & split[0])[:2])
        keep.update(sorted(people - split[0])[:2])
    return source[source.PersonID.isin(keep)].copy()


def normalizers_from_records(engine, training_frame, records):
    return {"value": engine.compute_value_stats(training_frame),
        "time": engine.compute_time_stats(records),
        "context": {str(k): v for k,v in engine.compute_demo_stats(records).items()},
        "target": engine.compute_target_stats(records), "n_records": len(records)}


def load_weights(torch, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if any(not torch.isfinite(v).all().item() for v in state.values()):
        raise FloatingPointError("Nonfinite source checkpoint weights.")
    return state


def check_source_receipt(done, job, manifest, source_partitions, norm, smoke):
    if done.get("job") != job or done.get("smoke") is not smoke or done.get("experiment_fingerprint") != manifest["experiment_fingerprint"]:
        raise RuntimeError("Source checkpoint receipt belongs to another job/protocol.")
    if done["stage_seed"] != stage_seed(job["seed"], manifest["origins"].index(job["target"])):
        raise ValueError("Source checkpoint stage seed differs.")
    assert_statistics_equal(done["target_statistics"], norm["target"], "source_training_output_statistics")
    if done["n_train_records"] != norm["n_records"]:
        raise ValueError("Source training record count differs from the normalizer.")
    if not smoke:
        for role in ("source_train", "source_val"):
            assert_statistics_equal(done[role], source_partitions[job["target"]][role], role)
        if done["n_val_records"] != source_partitions[job["target"]]["source_val"]["n_pairs"]:
            raise ValueError("Source validation prediction-pair count mismatch.")


def run_job(job, args, *, smoke=False):
    import numpy as np
    import torch
    from normalization import prepare_scale_consistent, map_and_check
    from training import choose_device, save_weights, training, verified_done
    manifest, legacy, project = load_extension()
    engine = configure_engine(job, smoke)
    device = choose_device(args.device)
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive.")
    torch.set_num_threads(args.cpu_threads)
    current_env = environment(torch, device)
    libraries = {k: current_env[k] for k in LIBRARY_KEYS}
    if not smoke and libraries != manifest["required_libraries"]:
        raise RuntimeError("Use the SAME CUDA/Python package environment as the completed Experiment 3.\n"
            f"Expected: {manifest['required_libraries']}\nObserved: {libraries}")
    if not smoke:
        marker = SMOKE / "smoke_passed.json"
        if not marker.is_file():
            raise RuntimeError("Run: python -B experiment7.py smoke --device cuda:0")
        passed = read_json(marker)
        if passed.get("passed") is not True or passed.get("experiment_fingerprint") != manifest["experiment_fingerprint"] or passed.get("libraries") != libraries:
            raise RuntimeError("Repeat the extension smoke test in the current research environment.")
    folder = job_folder(job, smoke)
    folder.mkdir(parents=True, exist_ok=True)
    config = {"experiment_fingerprint": manifest["experiment_fingerprint"], "job": job,
        "smoke": smoke, "libraries": libraries, "num_workers": 0,
        "cpu_threads": args.cpu_threads, "keep_models": bool(args.keep_models),
        "arms": list(TRAINING_ARMS), "percents": list(PERCENTS)}
    config_hash = object_hash(config)
    source_partitions = read_json(SETUP / "source_partitions.json")
    partitions = read_json(SETUP / "target_partitions.json")
    expected = read_json(SETUP / "expected_test_counts.json")
    target_statistics = read_json(SETUP / "target_statistics.json")
    folds = [0] if smoke else list(range(5))
    with file_lock(folder / ".job.lock"), (folder / "training.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            if (folder / "config.json").is_file() and read_json(folder / "config.json") != config:
                raise RuntimeError("Job settings/environment changed. Resume with identical threads, keep-models and library settings.")
            atomic_json(folder / "config.json", config)
            atomic_json(folder / "environment.json", current_env)
            print(f"\nADDED JOB {job['extension_job_id']}: {job['target']}, seed {job['seed']}; {device}; smoke={smoke}", flush=True)
            pre_stage = folder / "pretrain"
            pre_done = verified_done(pre_stage, config_hash, ("best_weights.pt", "history.csv", "normalizers.json"))
            finished = pre_done is not None
            for percent in PERCENTS:
                for fold in folds:
                    for arm in TRAINING_ARMS:
                        required = ("errors.csv.gz", "metrics.json", "history.csv") + (("invariance.json",) if arm == "scale_consistent" else ())
                        completed = verified_done(stage_folder(job,arm,percent,fold,smoke), config_hash, required)
                        finished = finished and completed is not None
            if finished:
                # Verify the checkpoint/normalizer too; do not rebuild any source records.
                norm = read_json(pre_stage / "normalizers.json")
                if not smoke:
                    assert_statistics_equal(norm, read_json(normalizer_path(job["target"])), "resumed_source_normalization")
                check_source_receipt(pre_done,job,manifest,source_partitions,norm,smoke)
                print("All source/target stages already complete and hash-verified. No training repeated.", flush=True)
                return
            print("Loading the frozen data and prediction pairs...", flush=True)
            frame, pairs, split, subsets = load_data(project, engine)
            if smoke:
                pairs = pairs.groupby("PersonID", sort=False).head(3).copy()
            counts = pairs.groupby("PersonID").size()
            target, origin_idx = job["target"], manifest["origins"].index(job["target"])
            sources = list(engine.ORIGIN_MAP[target])
            if set(sources) != set(manifest["origins"]) - {target}:
                raise ValueError("Source composition is not the original six-dataset all-source arm.")
            if pre_done is None:
                source = frame[frame.Origin.isin(sources)].copy()
                if smoke:
                    source = small_source(source, split)
                train_source, val_source = engine.make_pretrain_split(source, split, origin_idx)
                if train_source.empty or val_source.empty:
                    raise ValueError("Empty source training/validation partition.")
                if set(train_source.PersonID) & set(val_source.PersonID) or (set(source.PersonID) & set(frame.loc[frame.Origin.eq(target),"PersonID"])):
                    raise ValueError("Source partition or target identifier overlap.")
                if not smoke:
                    for role, part in (("source_train",train_source),("source_val",val_source)):
                        assert_statistics_equal(frame_counts(part, counts), source_partitions[target][role], role)
                seed = stage_seed(job["seed"], origin_idx)
                engine.set_seed(seed)
                model = engine.new_model(device)
                print("Building source-pretraining records (same original Experiment 1 recipe)...", flush=True)
                train_r, val_r, stats = engine.prepare_train_val(train_source, val_source, pairs,
                    seed=job["seed"] + 10000 + origin_idx, max_train_records=None)
                norm = normalizers_from_records(engine, train_source, train_r)
                assert_statistics_equal(stats, norm["target"], "source_labels")
                if not smoke:
                    assert_statistics_equal(norm, read_json(normalizer_path(target)), "source_reconstruction_vs_actual_training_records")
                atomic_json(pre_stage / "normalizers.json", norm)
                state, details = training(engine, model, train_r, val_r, stats, device=device,
                    batch=engine.PRETRAIN_BATCH_SIZE, lr=engine.PRETRAIN_LR,
                    epochs=engine.MAX_PRETRAIN_EPOCHS, offset=1000 * (origin_idx + 1),
                    name=f"all-source pretrain {target}", folder=pre_stage)
                save_weights(torch, state, pre_stage / "best_weights.pt")
                pre_done = {"config_hash": config_hash, "experiment_fingerprint": manifest["experiment_fingerprint"],
                    "job": job, "smoke": smoke, "stage_seed": seed, "sources": sources,
                    "source_train": frame_counts(train_source,counts), "source_val": frame_counts(val_source,counts),
                    "target_statistics": stats, "n_train_records": len(train_r), "n_val_records": len(val_r),
                    "training": details, "environment": current_env,
                    "file_sha256": {n:sha256(pre_stage/n) for n in ("best_weights.pt","history.csv","normalizers.json")}}
                atomic_json(pre_stage / "done.json", pre_done)
                del model, state, train_r, val_r, stats, source, train_source, val_source
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                print("Reusing the verified added source checkpoint; source training is not repeated.", flush=True)
                norm = read_json(pre_stage / "normalizers.json")
                if not smoke:
                    assert_statistics_equal(norm, read_json(normalizer_path(target)), "resumed_source_normalization")
            check_source_receipt(pre_done, job, manifest, source_partitions, norm, smoke)
            state = load_weights(torch, pre_stage / "best_weights.pt")
            checkpoint_hash = pre_done["file_sha256"]["best_weights.pt"]
            norm_hash = sha256(pre_stage / "normalizers.json") if smoke else sha256(normalizer_path(target))
            frame = frame[frame.Origin.eq(target)].copy()
            pairs = pairs[pairs.PersonID.isin(set(frame.PersonID))].copy()
            counts = pairs.groupby("PersonID").size()
            gc.collect()
            for percent in PERCENTS:
                engine.TRAIN_DATA_PERCENT = percent
                for fold in folds:
                    for arm in TRAINING_ARMS:
                        stage = stage_folder(job, arm, percent, fold, smoke)
                        required = ("errors.csv.gz", "metrics.json", "history.csv") + (("invariance.json",) if arm == "scale_consistent" else ())
                        done = verified_done(stage, config_hash, required)
                        if done is not None:
                            if done["source_checkpoint_sha256"] != checkpoint_hash or done["normalization_sha256"] != norm_hash:
                                raise ValueError("A completed target fit used another source checkpoint/normalizer.")
                            print(f"Skip completed {arm}, {percent}%, fold {fold}", flush=True)
                            continue
                        train, val, test = target_frames(frame,target,fold,percent,split,subsets,smoke=smoke)
                        if not smoke:
                            for role,part in (("target_train",train),("target_val",val),("target_test",test)):
                                assert_statistics_equal(frame_counts(part,counts),partitions[target][str(percent)][str(fold)][role],role)
                        print(f"Building {arm}, {percent}%, fold {fold} records...", flush=True)
                        if arm == "reset_control":
                            train_r,val_r,test_r,stats = engine.prepare_finetune(train,val,test,pairs,
                                seed=job["seed"]+20000+10*origin_idx+fold, max_train_records=None, stats_df=train)
                        else:
                            train_r,val_r,test_r,stats = prepare_scale_consistent(train,val,test,pairs,norm)
                        if not smoke:
                            assert_statistics_equal(stats,target_statistics[target][str(percent)][str(fold)],"target_training_statistics")
                            for part,records in ((train,train_r),(val,val_r),(test,test_r)):
                                if len(records) != frame_counts(part,counts)["n_pairs"]:
                                    raise ValueError("Record construction dropped an expected prediction pair.")
                        seed = stage_seed(job["seed"], origin_idx, fold)
                        engine.set_seed(seed)
                        model = engine.new_model(device)
                        model.load_state_dict(state, strict=True)
                        if arm == "scale_consistent":
                            inv = map_and_check(model,norm["target"],stats,train_r,device)
                            atomic_json(stage / "invariance.json", inv)
                            print("Initial raw-prediction invariance PASSED.", flush=True)
                        best,details = training(engine,model,train_r,val_r,stats,device=device,
                            batch=engine.FINETUNE_BATCH_SIZE,lr=engine.FINETUNE_LR,
                            epochs=engine.MAX_FINETUNE_EPOCHS,offset=2000*(origin_idx+1)+fold,
                            name=f"{arm} {target} {percent}% fold {fold}",folder=stage)
                        errors = engine.collect_errors(model,test_r,stats,target,fold,device,
                            method_name=f"strats_exp3_{arm}",source_config="exp3_added_all_source",
                            source_dataset="|".join(sources),n_pretrain_datasets=6,seed=job["seed"])
                        usable = np.array([stats[c]["n"] > 0 for c in engine.TARGET_COLS])
                        expected_rows = sum(int((np.isfinite(r["target"]) & usable).sum()) for r in test_r)
                        if len(errors) != expected_rows or not np.isfinite(errors.error.to_numpy()).all():
                            raise FloatingPointError("Missing/nonfinite target predictions.")
                        errors["train_data_percent"] = percent
                        errors["experiment_arm"] = arm
                        errors["extension_job_id"] = job["extension_job_id"]
                        metric = checked_metric(errors,project,target,fold,None if smoke else expected)
                        if arm == "scale_consistent":
                            other = stage_folder(job,"reset_control",percent,fold,smoke)
                            control,control_metric = read_json(other/"done.json"),read_json(other/"metrics.json")
                            if metric["identity_digest"] != control_metric["identity_digest"] or control["stage_seed"] != seed or control["source_checkpoint_sha256"] != checkpoint_hash:
                                raise ValueError("Matched arms differ in test identities, seed or source checkpoint.")
                            assert_statistics_equal(stats,control["target_statistics"],"paired_target_loss_statistics")
                        atomic_csv(errors,stage/"errors.csv.gz")
                        atomic_json(stage/"metrics.json",metric)
                        files = list(required)
                        if args.keep_models:
                            save_weights(torch,best,stage/"best_weights.pt")
                            files.append("best_weights.pt")
                        done = {"config_hash":config_hash,"experiment_fingerprint":manifest["experiment_fingerprint"],
                            "job":job,"smoke":smoke,"arm":arm,"percent":percent,"fold":fold,"stage_seed":seed,
                            "sources":sources,"source_checkpoint_sha256":checkpoint_hash,"normalization_sha256":norm_hash,
                            "target_statistics":stats,"target_train":frame_counts(train,counts),
                            "target_val":frame_counts(val,counts),"target_test":frame_counts(test,counts),
                            "n_train_records":len(train_r),"n_val_records":len(val_r),"n_test_records":len(test_r),
                            "n_raw_errors":len(errors),"training":details,"environment":current_env,
                            "file_sha256":{n:sha256(stage/n) for n in files}}
                        atomic_json(stage/"done.json",done)
                        print(f"Saved {arm}, {percent}%, fold {fold}",flush=True)
                        del model,best,train_r,val_r,test_r,stats,errors,train,val,test
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
            atomic_json(folder/"job_done.json",{"config_hash":config_hash,"smoke":smoke,
                "source_fits":1,"target_fits":4 if smoke else 20})
            print(f"COMPLETED added job {job['extension_job_id']}.",flush=True)
