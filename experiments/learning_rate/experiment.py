#!/usr/bin/env python3
"""Experiment 2: paired target-only STraTS learning-rate sensitivity.

Use the existing CUDA-enabled STraTS environment. All program-controlled writes,
including raw errors, logs, temporary files and caches, stay beside this script.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import contextlib
import csv
import gc
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from exp2_utils import (
    ROOT, VERSION, SEEDS, PERCENTS, LRS, TARGET_LABELS, DEFAULT_GROUPS, Tee,
    configure_local_runtime, inside, read_json, atomic_json, atomic_csv,
    sha256, object_hash, package_fingerprint, find_project_root, verify_inputs,
    jobs, group_jobs, load_setup, load_data, target_frames, frame_counts,
    stage_seed, state_digest, file_lock, environment, verified_done,
)
configure_local_runtime()


def prepare(project_arg=None):
    import numpy as np
    import pandas as pd
    import engine
    project = find_project_root(project_arg)
    hashes = verify_inputs(project)
    fingerprint = package_fingerprint()
    with file_lock(ROOT / "setup/.prepare.lock"):
        old = ROOT / "setup/manifest.json"
        if old.exists() and read_json(old)["package_fingerprint"] != fingerprint:
            if any((ROOT / "output").glob("jobs/job_*/**/done.json")):
                raise RuntimeError("Code changed with existing research results; use a fresh folder.")
        print("Checking original folds, fixed subsets and forecasting-pair ownership...", flush=True)
        frame, pairs, split, subsets = load_data(project, engine)
        origins = sorted(frame.Origin.unique().tolist())
        if origins != sorted(TARGET_LABELS):
            raise ValueError("Expected exactly the original seven datasets.")
        if frame.PersonID.isna().any() or frame.groupby("PersonID").Origin.nunique().max() != 1:
            raise ValueError("Missing or cross-dataset participant IDs.")
        if len(split) != 5 or any(split[i] & split[j] for i in range(5) for j in range(i)):
            raise ValueError("The original five folds are not disjoint.")
        if set(frame.PersonID) != set.union(*split):
            raise ValueError("Original folds do not cover participants exactly.")
        owners, ages = frame.PersonID.to_numpy(), frame.Age.to_numpy(dtype=float)
        for start in range(0, len(pairs), 20_000):
            block = pairs.iloc[start:start + 20_000]
            output = block.output_row.to_numpy(dtype=int)
            inputs = np.concatenate(block.input_rows.to_numpy()).astype(int)
            lengths = block.input_rows.map(len).to_numpy(dtype=int)
            if min(int(output.min()), int(inputs.min())) < 0 or max(int(output.max()), int(inputs.max())) >= len(frame):
                raise ValueError("Prediction pair points outside the harmonized dataset.")
            if not np.array_equal(owners[output], block.PersonID.to_numpy()):
                raise ValueError("Output belongs to another participant.")
            if not np.array_equal(owners[inputs], np.repeat(block.PersonID.to_numpy(), lengths)):
                raise ValueError("Input prefix crosses participant boundaries.")
            if not np.isfinite(ages[inputs]).all() or not np.isfinite(ages[output]).all():
                raise ValueError("Prediction pair has nonfinite ages.")
            if (ages[inputs] >= np.repeat(ages[output], lengths)).any():
                raise ValueError("Input prefix includes its future target or a later visit.")
        pair_counts = pairs.groupby("PersonID").size()
        primary = read_json(project / "data/primary_target_panel.json")
        eligible = read_json(project / "data/eligible_dataset_attribute_pairs.json")
        expected, partitions = {}, []
        for target in TARGET_LABELS:
            expected[target] = {}
            for fold in range(5):
                ids = split[fold] & set(frame.loc[frame.Origin.eq(target), "PersonID"])
                target_pairs = pairs[pairs.PersonID.isin(ids)]
                outputs = frame.loc[target_pairs.output_row, primary]
                expected[target][str(fold)] = {}
                for attr in primary:
                    if eligible[target][str(fold)][attr]["eligible"]:
                        count = int(np.isfinite(outputs[attr].to_numpy(dtype=float)).sum())
                        if not count:
                            raise ValueError("Eligible primary outcome has no test observations.")
                        expected[target][str(fold)][attr] = count
                training_ids = {}
                for percent in PERCENTS:
                    train, val, test = target_frames(frame, target, fold, percent, split, subsets)
                    training_ids[percent] = set(train.PersonID)
                    for role, part in (("train", train), ("validation", val), ("test", test)):
                        partitions.append({"target": target, "target_label": TARGET_LABELS[target],
                                           "fold": fold, "percent": percent, "role": role,
                                           **frame_counts(part, pair_counts)})
                if not training_ids[10].issubset(training_ids[100]):
                    raise ValueError("10% training participants are not nested in 100%.")
        reference = pd.read_csv(ROOT / "reference/original_metrics.csv")
        if len(reference) != 126 or reference.duplicated(["dataset", "seed", "train_data_percent", "base_configuration_id"]).any():
            raise ValueError("Unexpected frozen historical table.")
        for target in TARGET_LABELS:
            n = sum(sum(v.values()) for v in expected[target].values())
            if not reference.loc[reference.dataset.eq(target), "n_error_rows"].eq(n).all():
                raise ValueError(f"Historical primary-panel coverage changed: {target}")
        atomic_json(ROOT / "setup/expected_test_counts.json", expected)
        atomic_csv(pd.DataFrame(partitions), ROOT / "setup/target_partitions.csv")
        atomic_csv(pd.DataFrame([{**j, "default_group": j["job_id"] % DEFAULT_GROUPS} for j in jobs()]), ROOT / "setup/jobs.csv")
        atomic_json(ROOT / "setup/validation.json", {
            "passed": True, "n_people": int(frame.PersonID.nunique()), "n_visits": len(frame),
            "n_pairs": len(pairs), "folds_disjoint": True, "pair_ownership_verified": True,
            "strictly_future_targets": True, "nested_fixed_training_subsets": True,
            "historical_primary_error_counts_match": True,
        })
        files = ("expected_test_counts.json", "target_partitions.csv", "jobs.csv", "validation.json")
        manifest = {
            "version": VERSION, "project_root": str(project), "package_fingerprint": fingerprint,
            "input_sha256": hashes,
            "experiment_fingerprint": object_hash({"code": fingerprint, "inputs": hashes}),
            "setup_sha256": {n: sha256(ROOT / "setup" / n) for n in files},
            "targets": list(TARGET_LABELS), "origins": origins, "seeds": list(SEEDS),
            "percents": list(PERCENTS), "learning_rates": LRS, "primary_panel": primary,
            "jobs": jobs(), "n_jobs": 42, "n_research_fits": 420, "n_source_fits": 0,
            "default_groups": {str(i): [j["job_id"] for j in group_jobs(i)] for i in range(DEFAULT_GROUPS)},
            "training_settings": {"batch_size": 256, "max_epochs": 1000, "patience": 10,
                "min_delta": 1e-5, "weight_decay": 1e-5, "gradient_clip": 1.0,
                "hidden_dim": 256, "num_layers": 2, "num_heads": 4, "max_obs": 256,
                "dropout": 0.20, "attention_dropout": 0.20, "variable_dropout": 0.15,
                "layer_norm": False, "data_loader_workers": 0, "scheduler": None},
            "seed_policy": "Initialization/dropout seed = seed+200000+1000*sorted_origin_index+fold. DataLoader uses seed+2000*(sorted_origin_index+1)+fold+1/+2. LR and percent do not enter RNG seeds.",
            "selection": "For each dataset/fold/percent, minimize the mean of the three seeds' restored-checkpoint validation MSEs. Exact tie: 1e-4. All available trained outcomes enter validation, as in the original early stopping. No test metric enters selection.",
            "limitations": ["A two-rate sensitivity check, not full hyperparameter optimization.",
                "No new pooled/source-pretrained fits; historical scores are references only.",
                "Fresh RNG resetting is paired across rates, not a bitwise replay of historical initialization.",
                "10% reduces training availability only; validation/test folds remain fixed.",
                "CUDA numerical nondeterminism is not forcibly disabled; matching seeds do not guarantee cross-hardware bitwise identity."]}
        atomic_json(ROOT / "setup/manifest.json", manifest)
    print("Prepared: 42 jobs x 10 fits = 420 research fits.")
    print("Default: six disjoint groups, seven sequential jobs (70 fits) per group. Inputs are read-only.")


def choose_device(name):
    import torch
    device = torch.device(name)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Use cuda:0 or cpu.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Activate the existing CUDA-enabled STraTS environment. No silent CPU fallback.")
        torch.cuda.set_device(device)
    return device


def training(engine, model, train, val, stats, *, device, lr, offset, folder, name):
    folder = inside(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not train or not val:
        raise ValueError("Empty training/validation records.")
    start, history, last_report = time.monotonic(), [], [0.0, None]
    def batch_callback(phase, epoch, completed, total):
        now = time.monotonic()
        key = (phase, epoch)
        if key != last_report[1] or completed == total or now - last_report[0] >= 30:
            atomic_json(folder / "progress.json", {"phase": phase, "epoch": epoch,
                "batches_completed": completed, "batches_total": total,
                "elapsed_seconds": round(now-start, 1), "updated_unix_time": time.time()})
            print(f"{name} epoch {epoch} {phase}: {completed}/{total} batches", flush=True)
            last_report[:] = [now, key]
    with inside(folder / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "validation_mse", "elapsed_seconds"])
        def callback(epoch, train_loss, val_loss):
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}; fit not marked complete.")
            history.append((int(epoch), float(train_loss), float(val_loss)))
            writer.writerow([epoch, train_loss, val_loss, round(time.monotonic()-start, 3)])
            f.flush()
        engine.TRAIN_TRACE_CALLBACK, engine.BATCH_TRACE_CALLBACK = callback, batch_callback
        try:
            state = engine.train_model(model, train, val, stats,
                batch_size=engine.FINETUNE_BATCH_SIZE, lr=lr,
                max_epochs=engine.MAX_FINETUNE_EPOCHS, patience=engine.PATIENCE,
                device=device, seed_offset=offset, name=name)
        finally:
            engine.TRAIN_TRACE_CALLBACK = engine.BATCH_TRACE_CALLBACK = None
    best_epoch, best_val = 0, float("inf")
    for epoch, _, value in history:
        if value + engine.MIN_DELTA < best_val:
            best_epoch, best_val = epoch, value
    if not best_epoch:
        raise RuntimeError("No finite validation checkpoint.")
    return state, {"epochs_run": len(history), "best_epoch": best_epoch,
                   "selected_validation_mse": best_val,
                   "hit_epoch_cap": len(history) == engine.MAX_FINETUNE_EPOCHS,
                   "elapsed_seconds": time.monotonic()-start}


def pair_audit(a, b):
    for key in ("stage_seed", "initial_state_sha256", "loader_seeds", "target_statistics",
                "train", "validation", "test", "n_train_records", "n_val_records", "n_test_records"):
        if a[key] != b[key]:
            raise ValueError(f"Learning-rate pair is not matched: {key}")
    if a["test_identity_digest"] != b["test_identity_digest"]:
        raise ValueError("Learning-rate pair evaluated different observations.")


def run_job(job, args, smoke=False):
    import numpy as np
    import torch
    import engine
    from analyse import fold_metrics
    setup, project = load_setup()
    device = choose_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    engine.SEED, engine.MODE = job["seed"], "target_only"
    engine.NUM_WORKERS, engine.EPOCH_PRINT_EVERY = 0, 1
    engine.FINETUNE_BATCH_SIZE = 4 if smoke else 256
    engine.MAX_FINETUNE_EPOCHS = 1 if smoke else 1000
    engine.TRAIN_DATA_PERCENT = job["percent"]
    assert engine.HIDDEN_DIM == 256 and engine.MAX_OBS == 256 and engine.PATIENCE == 10
    assert engine.MIN_DELTA == 1e-5 and engine.WEIGHT_DECAY == 1e-5 and engine.GRAD_CLIP == 1.0
    folder = inside(ROOT / ("smoke_output" if smoke else "output") / "jobs" / f"job_{job['job_id']:03d}")
    folder.mkdir(parents=True, exist_ok=True)
    env = environment(torch, device)
    # GPU model may differ across allocated workers; software/numerical settings may not.
    signature = {k:v for k,v in env.items() if k != "device_name"}
    config = {"experiment_fingerprint": setup["experiment_fingerprint"], "job": job,
              "smoke": smoke, "runtime_signature": signature, "keep_models": bool(args.keep_models)}
    config_hash = object_hash(config)
    with file_lock(folder / ".job.lock"), inside(folder / "training.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f"\nJOB {job['job_id']}: {job['target_label']}, {job['percent']}%, seed {job['seed']}; {env['device_name']}; smoke={smoke}", flush=True)
            if (folder / "config.json").exists() and read_json(folder / "config.json") != config:
                raise RuntimeError("Job configuration/software changed. Use identical settings to resume, or a fresh folder.")
            atomic_json(folder / "config.json", config)
            atomic_json(folder / "environment.json", env)
            folds = [0] if smoke else list(range(5))
            done_by_fold = {}
            for fold in folds:
                done_by_fold[fold] = {key: verified_done(folder / f"fold{fold}" / key, config_hash) for key in LRS}
                if all(done_by_fold[fold].values()):
                    pair_audit(*done_by_fold[fold].values())
            if all(all(v.values()) for v in done_by_fold.values()):
                print("All fits verified; no retraining needed.", flush=True)
                atomic_json(folder / "job_done.json", {"config_hash": config_hash, "smoke": smoke, "fits": 2*len(folds)})
                return
            print("Loading read-only target data and fixed prediction pairs...", flush=True)
            frame, pairs, split, subsets = load_data(project, engine, target=job["target"], smoke=smoke)
            pair_counts = pairs.groupby("PersonID").size()
            origin_idx = setup["origins"].index(job["target"])
            for fold in folds:
                completed = done_by_fold[fold]
                if all(completed.values()):
                    print(f"Skip verified fold {fold} (both rates).", flush=True)
                    continue
                train_df, val_df, test_df = target_frames(frame, job["target"], fold, job["percent"], split, subsets, smoke=smoke)
                print(f"Fold {fold}: building records once for the two-rate pair...", flush=True)
                train, val, test, stats = engine.prepare_finetune(train_df, val_df, test_df, pairs,
                    seed=job["seed"]+20000+10*origin_idx+fold, max_train_records=None,
                    stats_df=train_df)
                counts = {"train": frame_counts(train_df, pair_counts),
                          "validation": frame_counts(val_df, pair_counts), "test": frame_counts(test_df, pair_counts)}
                if any(not r for r in (train, val, test)):
                    raise ValueError("Empty forecasting records; refusing an untrained result.")
                for name, records in (("train",train),("validation",val),("test",test)):
                    if len(records) != counts[name]["n_pairs"]:
                        raise ValueError(f"Record construction lost {name} prediction pairs.")
                print(f"Records: train={len(train)}, validation={len(val)}, test={len(test)}", flush=True)
                seed = stage_seed(job["seed"], origin_idx, fold)
                offset = 2000*(origin_idx+1)+fold
                for key, lr in LRS.items():
                    stage = inside(folder / f"fold{fold}" / key)
                    if completed[key]:
                        print(f"Skip verified {key}, fold {fold}.", flush=True)
                        continue
                    stage.mkdir(parents=True, exist_ok=True)
                    engine.set_seed(seed)  # Reset before EVERY fit, including resume.
                    model = engine.new_model(device)
                    initial_hash = state_digest(model.state_dict())
                    print(f"Training {key}, fold {fold}; paired seed={seed}", flush=True)
                    state, details = training(engine, model, train, val, stats, device=device,
                        lr=lr, offset=offset, folder=stage, name=f"job{job['job_id']} fold{fold} {key}")
                    print("Evaluating restored validation checkpoint on fixed test fold...", flush=True)
                    errors = engine.collect_errors(model, test, stats, job["target"], fold, device,
                        method_name=f"strats_exp2_{key}", source_config="target_only",
                        source_dataset="", n_pretrain_datasets=0, seed=job["seed"])
                    usable = np.array([stats[c]["n"]>0 for c in engine.TARGET_COLS])
                    expected_rows = sum(int((np.isfinite(r["target"]) & usable).sum()) for r in test)
                    if len(errors) != expected_rows or not np.isfinite(errors.error.to_numpy()).all():
                        raise FloatingPointError("Missing/nonfinite test predictions.")
                    errors["train_data_percent"] = job["percent"]
                    errors["learning_rate"] = lr
                    scores = fold_metrics(errors, project, job["target"], fold, strict=not smoke)
                    atomic_csv(errors, stage / "errors.csv.gz")
                    atomic_json(stage / "metrics.json", scores)
                    files = ["errors.csv.gz", "metrics.json", "history.csv"]
                    if args.keep_models:
                        path = inside(stage / "best_weights.pt")
                        temporary = inside(stage / f"best_weights.pt.{os.getpid()}.tmp")
                        torch.save(state, temporary); os.replace(temporary, path)
                        files.append("best_weights.pt")
                    done = {"config_hash": config_hash, "lr_key": key, "learning_rate": lr,
                            "stage_seed": seed, "initial_state_sha256": initial_hash,
                            "loader_seeds": {"train": job["seed"]+offset+1, "validation": job["seed"]+offset+2},
                            "target_statistics": stats, **counts,
                            "n_train_records": len(train), "n_val_records": len(val), "n_test_records": len(test),
                            "test_identity_digest": scores["identity_digest"], "training": details,
                            "environment": env, "file_sha256": {n:sha256(stage/n) for n in files}}
                    # Compare before marking the second member of a pair complete.
                    for other in completed.values():
                        if other is not None:
                            pair_audit(other, done)
                    atomic_json(stage / "done.json", done)
                    completed[key] = done
                    atomic_json(stage / "progress.json", {"phase": "complete", "epochs_run": details["epochs_run"],
                        "best_epoch": details["best_epoch"], "updated_unix_time": time.time()})
                    print(f"Saved {len(errors)} errors; best epoch={details['best_epoch']}; validation MSE={details['selected_validation_mse']:.8g}", flush=True)
                    del model, state, errors
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                pair_audit(*completed.values())
                del train, val, test, stats, train_df, val_df, test_df
                gc.collect()
            atomic_json(folder / "job_done.json", {"config_hash": config_hash, "smoke": smoke, "fits": 2*len(folds)})
            print(f"Completed job {job['job_id']}.", flush=True)


def selected_jobs(args):
    if args.group is not None:
        return group_jobs(args.group, args.groups)
    if args.all:
        return jobs()
    return [jobs()[args.job]]


def launch_jobs(args):
    selected = selected_jobs(args)
    label = f"group_{args.group}_of_{args.groups}" if args.group is not None else "all_jobs"
    if args.dry_run:
        print(f"{label}: {len(selected)} sequential jobs, {10*len(selected)} fits, device={args.device}")
        for job in selected:
            print(f"  {job['job_id']:2d}: {job['target_label']}, {job['percent']}%, seed {job['seed']}")
        return
    load_setup()
    folder = inside(ROOT / "output/launchers" / label)
    with file_lock(folder / ".launcher.lock"):
        print(f"{label}: {len(selected)} sequential jobs; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}", flush=True)
        for position, job in enumerate(selected, start=1):
            atomic_json(folder / "progress.json", {"state":"running", "current_job_id":job["job_id"],
                "position":position, "n_jobs":len(selected), "updated_unix_time":time.time()})
            print(f"\n[{label}] job {position}/{len(selected)} (ID {job['job_id']})", flush=True)
            command = [sys.executable, "-B", "-u", str(ROOT/"experiment.py"), "run", "--job", str(job["job_id"]),
                       "--device", args.device, "--cpu-threads", str(args.cpu_threads)]
            if args.keep_models:
                command.append("--keep-models")
            try:
                subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)
            except BaseException:
                atomic_json(folder / "progress.json", {"state":"stopped", "current_job_id":job["job_id"],
                    "position":position, "n_jobs":len(selected), "updated_unix_time":time.time()})
                raise
        atomic_json(folder / "progress.json", {"state":"complete", "job_ids":[j["job_id"] for j in selected],
            "updated_unix_time":time.time()})
        print(f"COMPLETED {label}", flush=True)


def status():
    total = complete = 0
    for job in jobs():
        folder = ROOT / "output/jobs" / f"job_{job['job_id']:03d}"
        count = sum((folder/f"fold{fold}"/key/"done.json").exists() for fold in range(5) for key in LRS)
        total += count; complete += count == 10
        print(f"default group {job['job_id']%6}  job {job['job_id']:2d}  {job['target_label']:17s} {job['percent']:3d}% seed {job['seed']}: {count:2d}/10 fits")
    print(f"Total: {complete}/42 jobs; {total}/420 research fits. Smoke results excluded.")
    print("Status counts completion markers; summarize performs full integrity checks.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--project-root")
    for command in ("run", "smoke"):
        q = sub.add_parser(command)
        q.add_argument("--device", default="cuda:0")
        q.add_argument("--cpu-threads", type=int, default=2)
        q.add_argument("--keep-models", action="store_true", help="Optional: also keep best model weights locally; preserve this setting on resume.")
        if command == "run":
            group = q.add_mutually_exclusive_group(required=True)
            group.add_argument("--all", action="store_true")
            group.add_argument("--group", type=int)
            group.add_argument("--job", type=int, choices=range(42))
            q.add_argument("--groups", type=int, default=6, help="Number of disjoint groups; use one group per GPU.")
            q.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    q = sub.add_parser("summarize")
    q.add_argument("--recompute", action="store_true", help="Accepted for compatibility; summaries always verify the raw errors.")
    return p


def main():
    args = parser().parse_args()
    if getattr(args,"cpu_threads",2) < 1:
        raise ValueError("cpu-threads must be positive.")
    if args.command == "prepare":
        prepare(args.project_root)
    elif args.command == "status":
        status()
    elif args.command == "summarize":
        from summarize_fixed import run
        run()
    elif args.command == "smoke":
        job = next(j for j in jobs() if j["target"]=="BE_KUL_Wave_1B" and j["percent"]==10 and j["seed"]==1234)
        run_job(job,args,smoke=True)
        atomic_json(ROOT/"smoke_output/smoke_passed.json", {"passed":True,"device":args.device,
            "research_results":False,"fits":2,"epochs_per_fit":1})
        print("Smoke test passed. Two tiny one-epoch fits are excluded from research summaries.")
    elif args.all or args.group is not None or args.dry_run:
        launch_jobs(args)
    else:
        run_job(jobs()[args.job],args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nStopped: {exc}\nCompleted fits are retained. Rerun the identical command to resume.", file=sys.stderr)
        if not isinstance(exc,KeyboardInterrupt):
            traceback.print_exc()
        raise SystemExit(1)
