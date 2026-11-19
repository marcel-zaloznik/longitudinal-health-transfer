#!/usr/bin/env python3
"""Isolated, resumable participant-overlap sensitivity experiment.

Use the CUDA-enabled environment that already runs strats.py.
All writes are rooted beside this file, regardless of the working directory.
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

from exp1_utils import (
    ROOT, VERSION, TARGET_LABELS, SEEDS, PERCENTS, ARMS, MODES, Tee, N_GROUPS, JOBS_PER_GROUP,
    configure_local_runtime, inside, read_json, atomic_json, atomic_csv,
    sha256, object_hash, package_fingerprint, find_project_root, verify_inputs,
    jobs, group_jobs, load_setup, load_data, target_frames, frame_counts, source_names,
    stage_seed, file_lock, environment,
)
configure_local_runtime()


def prepare(project_arg: str | None = None) -> None:
    import numpy as np
    import pandas as pd
    import engine

    project = find_project_root(project_arg)
    hashes = verify_inputs(project)
    fingerprint = package_fingerprint()
    setup_path = ROOT / "setup/manifest.json"
    if setup_path.exists():
        previous = read_json(setup_path)
        if previous["package_fingerprint"] != fingerprint:
            if any((ROOT / "output").glob("jobs/job_*/**/done.json")):
                raise RuntimeError("Code changed with existing results. Use a new experiment folder.")
    frame, pairs, split, subsets = load_data(project, engine)
    origins = sorted(frame.Origin.unique().tolist())
    if set(origins) != set(engine.ORIGIN_MAP):
        raise ValueError("Observed datasets do not match the original seven datasets.")
    if frame.PersonID.isna().any() or frame.groupby("PersonID").Origin.nunique().max() != 1:
        raise ValueError("PersonID is missing or used in more than one dataset.")
    if len(split) != 5:
        raise ValueError("Expected the original five folds.")
    for i in range(5):
        for j in range(i):
            if split[i] & split[j]:
                raise ValueError("A participant ID occurs in multiple folds.")
    if set(frame.PersonID) != set.union(*split):
        raise ValueError("Fold membership does not cover the supplied participants exactly.")
    for target, order in engine.ORIGIN_MAP.items():
        if len(order) != 6 or set(order) != set(origins) - {target}:
            raise ValueError("Invalid original source ordering.")

    # Validate positional pair indices without changing the original row index.
    owners = frame.PersonID.to_numpy()
    ages = frame.Age.to_numpy(dtype=float)
    for start in range(0, len(pairs), 20_000):
        block = pairs.iloc[start:start + 20_000]
        output = block.output_row.to_numpy(dtype=int)
        inputs = np.concatenate(block.input_rows.to_numpy()).astype(int)
        lengths = block.input_rows.map(len).to_numpy(dtype=int)
        if min(int(output.min()), int(inputs.min())) < 0 or max(int(output.max()), int(inputs.max())) >= len(frame):
            raise ValueError("A prediction pair points outside the harmonized dataset.")
        if not np.array_equal(owners[output], block.PersonID.to_numpy()):
            raise ValueError("An output row belongs to the wrong participant.")
        if not np.array_equal(owners[inputs], np.repeat(block.PersonID.to_numpy(), lengths)):
            raise ValueError("An input prefix crosses participant boundaries.")
        if not np.isfinite(ages[inputs]).all() or not np.isfinite(ages[output]).all():
            raise ValueError("Nonfinite ages occur in a prediction pair.")
        if (ages[inputs] >= np.repeat(ages[output], lengths)).any():
            raise ValueError("A prediction prefix includes the requested future visit or a later age.")

    pair_counts = pairs.groupby("PersonID").size()
    exclusions = read_json(ROOT / "exclusions.json")
    audit = []
    partitions = []
    expected = {}
    primary = read_json(project / "data/primary_target_panel.json")
    eligible = read_json(project / "data/eligible_dataset_attribute_pairs.json")
    for target in TARGET_LABELS:
        expected[target] = {}
        origin_idx = origins.index(target)
        for arm in ARMS:
            sources = source_names(engine, target, arm)
            source = frame[frame.Origin.isin(sources)].copy()
            pre_train, pre_val = engine.make_pretrain_split(source, split, origin_idx)
            if pre_train.empty or pre_val.empty:
                raise ValueError("Empty source pretraining partition.")
            for role, part in (("source_all", source), ("pretrain_train", pre_train), ("pretrain_val", pre_val)):
                audit.append({"target": target, "target_label": TARGET_LABELS[target], "arm": arm,
                              "role": role, "sources": "|".join(sources),
                              "excluded_sources": "|".join(exclusions[target]) if arm == "excluded" else "",
                              **frame_counts(part, pair_counts)})
        for fold in range(5):
            expected[target][str(fold)] = {}
            test_ids = split[fold]
            test_pairs = pairs[pairs.PersonID.isin(test_ids & set(frame.loc[frame.Origin.eq(target), "PersonID"]))]
            outputs = frame.loc[test_pairs.output_row, primary]
            for attribute in primary:
                if eligible[target][str(fold)][attribute]["eligible"]:
                    count = int(np.isfinite(outputs[attribute].to_numpy(dtype=float)).sum())
                    if count == 0:
                        raise ValueError("An eligible primary outcome has no test observations.")
                    expected[target][str(fold)][attribute] = count
            for percent in PERCENTS:
                train, val, test = target_frames(frame, target, fold, percent, split, subsets)
                for role, part in (("target_train", train), ("target_val", val), ("target_test", test)):
                    partitions.append({"target": target, "fold": fold, "percent": percent,
                                       "role": role, **frame_counts(part, pair_counts)})

    reference = pd.read_csv(ROOT / "reference/original_metrics.csv")
    for target in TARGET_LABELS:
        expected_rows = sum(sum(x.values()) for x in expected[target].values())
        if not reference.loc[reference.dataset.eq(target), "n_error_rows"].eq(expected_rows).all():
            raise ValueError(f"Historical primary-panel coverage differs from reconstructed test pairs: {target}")
    manifest = {
        "version": VERSION, "project_root": str(project), "package_fingerprint": fingerprint,
        "input_sha256": hashes, "experiment_fingerprint": object_hash({"code": fingerprint, "inputs": hashes}),
        "targets": list(TARGET_LABELS), "origins": origins, "seeds": list(SEEDS),
        "percents": list(PERCENTS), "arms": list(ARMS), "modes": list(MODES),
        "exclusions": exclusions, "jobs": jobs(), "n_jobs": 30,
        "job_groups": {str(i): [job["job_id"] for job in group_jobs(i)] for i in range(N_GROUPS)},
        "data_loading_workers": 0,
        "n_target_fits": 600, "n_source_pretraining_fits": 30,
        "seed_policy": "pretrain=seed+100000+1000*sorted_origin_index; target=seed+200000+1000*sorted_origin_index+fold. Original DataLoader seed offsets retained. Arms and percentages share stage seeds.",
        "primary_panel": primary,
        "limitations": ["Exclusion removes whole source cohorts, not linked individuals.",
                        "Source size and composition change together.",
                        "Target-only and historical all-source scores come from the submitted experiment; fresh controls use stage-wise seeding.",
                        "Results describe the five affected targets, not a new seven-target benchmark."]
    }
    with file_lock(ROOT / "setup/.prepare.lock"):
        atomic_json(setup_path, manifest)
        atomic_json(ROOT / "setup/expected_test_counts.json", expected)
        atomic_csv(pd.DataFrame([{**job, "group_id": job["job_id"] // JOBS_PER_GROUP}
                                 for job in jobs()]), ROOT / "setup/jobs.csv")
        atomic_csv(pd.DataFrame(audit), ROOT / "setup/source_counts.csv")
        atomic_csv(pd.DataFrame(partitions), ROOT / "setup/target_partitions.csv")
        atomic_json(ROOT / "setup/validation.json", {
            "passed": True, "n_people": int(frame.PersonID.nunique()), "n_visits": len(frame),
            "n_pairs": len(pairs), "folds_disjoint": True, "pair_ownership_verified": True,
            "strictly_future_targets": True, "historical_primary_error_counts_match": True,
            "warning": "Distinct IDs across datasets do not prove distinct real-world people."
        })
    print("Prepared: 30 jobs, 600 target fits and 30 source-pretraining fits.")
    print("Six groups: --group 0 = jobs 0-4, ... --group 5 = jobs 25-29; five sequential jobs per GPU.")
    print("DataLoader workers are disabled (0). The target-only baseline is reused. Original inputs are unchanged.")


def choose_device(name: str):
    import torch
    device = torch.device(name)
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Use cuda:0 (GPU) or cpu.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment. Activate your existing CUDA-enabled PyTorch environment; no silent CPU fallback is used.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def save_weights(torch, state, path: Path) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def training(engine, model, train, val, stats, *, device, batch, lr, epochs, offset, name, folder):
    folder = inside(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not train or not val:
        raise ValueError(f"{name}: empty training or validation records; refusing an untrained result.")
    history_path = folder / "history.csv"
    start = time.monotonic()
    history = []
    last_report = [0.0, None]
    def batch_progress(phase, epoch, completed, total):
        now = time.monotonic()
        key = (phase, epoch)
        if key != last_report[1] or completed == total or now - last_report[0] >= 30:
            atomic_json(folder / "progress.json", {
                "stage": name, "phase": phase, "epoch": epoch,
                "batches_completed": completed, "batches_total": total,
                "elapsed_seconds": round(now - start, 1), "updated_unix_time": time.time(),
            })
            print(f"{name} epoch {epoch} {phase}: {completed}/{total} batches "
                  f"(stage elapsed {now - start:.0f}s)", flush=True)
            last_report[:] = [now, key]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_mse", "validation_mse", "elapsed_seconds"])
        def callback(epoch, train_loss, val_loss):
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise FloatingPointError(f"{name}: nonfinite loss at epoch {epoch}; no successful checkpoint recorded.")
            history.append((int(epoch), float(train_loss), float(val_loss)))
            writer.writerow([epoch, train_loss, val_loss, round(time.monotonic() - start, 3)])
            handle.flush()
        engine.TRAIN_TRACE_CALLBACK = callback
        engine.BATCH_TRACE_CALLBACK = batch_progress
        try:
            state = engine.train_model(model=model, train_records=train, val_records=val,
                target_stats=stats, batch_size=batch, lr=lr, max_epochs=epochs,
                patience=engine.PATIENCE, device=device, seed_offset=offset, name=name)
        finally:
            engine.TRAIN_TRACE_CALLBACK = None
            engine.BATCH_TRACE_CALLBACK = None
    if not history:
        raise RuntimeError(f"{name}: training completed without any epochs.")
    # Apply the original min-delta rule when identifying the selected epoch.
    best_epoch, best_val = 0, float("inf")
    for epoch, _, loss in history:
        if loss + engine.MIN_DELTA < best_val:
            best_epoch, best_val = epoch, loss
    return state, {"epochs_run": len(history), "best_epoch": best_epoch,
                   "selected_validation_mse": best_val, "elapsed_seconds": time.monotonic() - start,
                   "history_sha256": sha256(history_path)}


def verified_done(folder: Path, config_hash: str, required: tuple[str, ...]) -> dict | None:
    done_path = folder / "done.json"
    if not done_path.exists():
        return None
    done = read_json(done_path)
    if done.get("config_hash") != config_hash:
        raise RuntimeError(f"Stored configuration mismatch: {folder}")
    for name in required:
        path = folder / name
        if not path.exists() or sha256(path) != done["file_sha256"].get(name):
            raise RuntimeError(f"Completed output is missing or corrupted: {path}. Restore it or remove this stage directory before rerunning.")
    return done


def small_source(source, split):
    # Smoke tests use real rows and original partitions but are NOT research results.
    keep = set()
    for _, group in source.groupby("Origin", sort=False):
        people = set(group.PersonID)
        keep.update(sorted(people & split[0])[:2])
        keep.update(sorted(people - split[0])[:2])
    return source[source.PersonID.isin(keep)].copy()


def run_job(job: dict, args, smoke: bool = False) -> None:
    import numpy as np
    import pandas as pd
    import torch
    import engine
    from analyse import fold_metrics

    setup, project = load_setup()
    device = choose_device(args.device)
    if smoke:
        torch.set_num_threads(2)
    engine.SEED = job["seed"]
    engine.MODE = "topk_pretrain" if job["mode"] == "pretrain" else "pooled"
    if args.num_workers != 0:
        raise ValueError("This package requires --num-workers 0 to avoid long-path worker-socket failures.")
    engine.NUM_WORKERS = 0
    engine.EPOCH_PRINT_EVERY = 1
    engine.MAX_PRETRAIN_EPOCHS = 1 if smoke else 1000
    engine.MAX_FINETUNE_EPOCHS = 1 if smoke else 1000
    engine.PRETRAIN_BATCH_SIZE = 4 if smoke else 256
    engine.FINETUNE_BATCH_SIZE = 4 if smoke else 256
    # All original model/optimization defaults are retained in research runs.
    assert engine.HIDDEN_DIM == 256 and engine.MAX_OBS == 256 and engine.PATIENCE == 10
    assert engine.PRETRAIN_LR == 5e-4 and engine.FINETUNE_LR == 1e-4
    output = ROOT / ("smoke_output" if smoke else "output")
    folder = inside(output / "jobs" / f"job_{job['job_id']:03d}")
    folder.mkdir(parents=True, exist_ok=True)
    current_env = environment(torch, device)
    config = {"experiment_fingerprint": setup["experiment_fingerprint"], "job": job, "smoke": smoke,
              "num_workers": engine.NUM_WORKERS, "keep_models": bool(args.keep_models),
              "libraries": {key: current_env[key] for key in ("torch", "numpy", "pandas", "cuda_build", "device_type")}}
    config_hash = object_hash(config)
    with file_lock(folder / ".job.lock"), (folder / "training.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f"\nJOB {job['job_id']}: {job['target_label']}, {job['mode']}, seed {job['seed']}; device={device}; smoke={smoke}", flush=True)
            if (folder / "config.json").exists() and read_json(folder / "config.json") != config:
                raise RuntimeError("This job was started with different code, library versions, workers or keep-models settings. Use identical settings or a fresh experiment folder.")
            atomic_json(folder / "config.json", config)
            atomic_json(folder / "environment.json", current_env)
            print(f"GPU: {current_env['device_name']}; DataLoader workers: 0. Loading read-only inputs...", flush=True)
            frame, pairs, split, subsets = load_data(project, engine, smoke=smoke)
            pair_counts = pairs.groupby("PersonID").size()
            target = job["target"]
            origin_idx = setup["origins"].index(target)  # Do not renumber after filtering targets.
            folds = [0] if smoke else list(range(5))
            for arm in ARMS:
                sources = source_names(engine, target, arm)
                source = frame[frame.Origin.isin(sources)].copy()
                if smoke:
                    source = small_source(source, split)
                if set(source.PersonID) & set(frame.loc[frame.Origin.eq(target), "PersonID"]):
                    raise ValueError("Source and target identifiers intersect.")
                pretrain_state = None
                if job["mode"] == "pretrain":
                    stage = folder / arm / "pretrain"
                    completed = verified_done(stage, config_hash, ("best_weights.pt", "history.csv"))
                    if completed:
                        print(f"Reuse source checkpoint: {arm}", flush=True)
                        pretrain_state = torch.load(stage / "best_weights.pt", map_location="cpu", weights_only=True)
                    else:
                        pre_train, pre_val = engine.make_pretrain_split(source, split, origin_idx)
                        seed = stage_seed(job["seed"], origin_idx)
                        engine.set_seed(seed)
                        model = engine.new_model(device)
                        print(f"{arm}: building source-pretraining records...", flush=True)
                        train_records, val_records, stats = engine.prepare_train_val(
                            pre_train, pre_val, pairs, seed=job["seed"] + 10000 + origin_idx,
                            max_train_records=None)
                        print(f"{arm}: pretraining pairs train={len(train_records)}, validation={len(val_records)}", flush=True)
                        pretrain_state, details = training(engine, model, train_records, val_records, stats,
                            device=device, batch=engine.PRETRAIN_BATCH_SIZE, lr=engine.PRETRAIN_LR,
                            epochs=engine.MAX_PRETRAIN_EPOCHS, offset=1000 * (origin_idx + 1),
                            name=f"{arm} pretrain {target}", folder=stage)
                        save_weights(torch, pretrain_state, stage / "best_weights.pt")
                        atomic_json(stage / "done.json", {"config_hash": config_hash, "stage_seed": seed,
                            "sources": sources, "source_train": frame_counts(pre_train, pair_counts),
                            "source_val": frame_counts(pre_val, pair_counts), "target_statistics": stats,
                            "n_train_records": len(train_records), "n_val_records": len(val_records),
                            "training": details, "environment": current_env,
                            "file_sha256": {name: sha256(stage / name) for name in ("best_weights.pt", "history.csv")}})
                        del model, train_records, val_records, stats, pre_train, pre_val
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                for percent in PERCENTS:
                    engine.TRAIN_DATA_PERCENT = percent
                    for fold in folds:
                        stage = folder / arm / f"pct{percent}" / f"fold{fold}"
                        if verified_done(stage, config_hash, ("errors.csv.gz", "metrics.json", "history.csv")):
                            print(f"Skip completed: {arm}, {percent}%, fold {fold}", flush=True)
                            continue
                        train_target, val, test = target_frames(frame, target, fold, percent, split, subsets, smoke=smoke)
                        train = pd.concat([train_target, source], axis=0).copy() if job["mode"] == "pooled" else train_target
                        print(f"{arm}, {percent}%, fold {fold}: building target-fit records...", flush=True)
                        train_records, val_records, test_records, stats = engine.prepare_finetune(
                            train, val, test, pairs, seed=job["seed"] + 20000 + 10 * origin_idx + fold,
                            max_train_records=None, stats_df=train_target)
                        if not test_records:
                            raise ValueError("No target test records.")
                        seed = stage_seed(job["seed"], origin_idx, fold)
                        engine.set_seed(seed)
                        model = engine.new_model(device)
                        if pretrain_state is not None:
                            model.load_state_dict(pretrain_state)
                        print(f"{arm}, {percent}%, fold {fold}: pairs train={len(train_records)}, val={len(val_records)}, test={len(test_records)}", flush=True)
                        state, details = training(engine, model, train_records, val_records, stats,
                            device=device, batch=engine.FINETUNE_BATCH_SIZE, lr=engine.FINETUNE_LR,
                            epochs=engine.MAX_FINETUNE_EPOCHS, offset=2000 * (origin_idx + 1) + fold,
                            name=f"{arm} {percent}% {target} fold {fold}", folder=stage)
                        errors = engine.collect_errors(model, test_records, stats, target, fold, device,
                            method_name=f"strats_exp1_{job['mode']}_{arm}", source_config=f"exp1_{arm}",
                            source_dataset="|".join(sources), n_pretrain_datasets=len(sources), seed=job["seed"])
                        usable = np.array([stats[col]["n"] > 0 for col in engine.TARGET_COLS])
                        expected_rows = sum(int((np.isfinite(record["target"]) & usable).sum()) for record in test_records)
                        if len(errors) != expected_rows or not np.isfinite(errors.error.to_numpy()).all():
                            raise FloatingPointError("Missing/nonfinite predictions: refusing incomplete errors.")
                        errors["train_data_percent"] = percent
                        errors["experiment_arm"] = arm
                        errors["model_mode"] = job["mode"]
                        errors["job_id"] = job["job_id"]
                        scores = fold_metrics(errors, project, target, fold, strict=not smoke)
                        # A matched pair must evaluate exactly the same error identities.
                        other = folder / "all_source" / f"pct{percent}" / f"fold{fold}" / "metrics.json"
                        if arm == "excluded" and other.exists():
                            if scores["identity_digest"] != read_json(other)["identity_digest"]:
                                raise ValueError("All-source and excluded arms do not evaluate the same observations.")
                        atomic_csv(errors, stage / "errors.csv.gz")
                        atomic_json(stage / "metrics.json", scores)
                        if args.keep_models:
                            save_weights(torch, state, stage / "best_weights.pt")
                        files = ["errors.csv.gz", "metrics.json", "history.csv"]
                        if args.keep_models:
                            files.append("best_weights.pt")
                        atomic_json(stage / "done.json", {
                            "config_hash": config_hash, "stage_seed": seed, "sources": sources,
                            "target_train": frame_counts(train_target, pair_counts),
                            "target_val": frame_counts(val, pair_counts), "target_test": frame_counts(test, pair_counts),
                            "actual_train": frame_counts(train, pair_counts), "target_statistics": stats,
                            "n_train_records": len(train_records), "n_val_records": len(val_records),
                            "n_test_records": len(test_records), "n_raw_errors": len(errors),
                            "training": details, "environment": current_env,
                            "file_sha256": {name: sha256(stage / name) for name in files}})
                        print(f"Saved {len(errors)} errors: {stage.relative_to(ROOT)}", flush=True)
                        del model, state, train_records, val_records, test_records, stats, errors
                        del train, train_target, val, test
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                del pretrain_state, source
                gc.collect()
            atomic_json(folder / "job_done.json", {"config_hash": config_hash, "smoke": smoke,
                        "target_fits": 2 * 2 * len(folds), "source_fits": 2 if job["mode"] == "pretrain" else 0})
            print(f"Completed job {job['job_id']}.", flush=True)


def selected_jobs(args) -> list[dict]:
    if args.group is not None:
        return group_jobs(args.group)
    if args.all:
        return jobs()
    return [jobs()[args.job]]


def launch_jobs(args) -> None:
    """One sequential queue per GPU; a fresh Python process releases each job's memory."""
    selected = selected_jobs(args)
    label = f"group_{args.group}" if args.group is not None else "all_jobs"
    if args.dry_run:
        print(f"{label}: job IDs {[job['job_id'] for job in selected]}; device={args.device}; workers=0")
        for job in selected:
            print(f"  {job['job_id']:2d}: {job['target_label']}, {job['mode']}, seed {job['seed']}")
        return
    load_setup()
    folder = inside(ROOT / "output/launchers" / label)
    with file_lock(folder / ".launcher.lock"):
        print(f"{label}: running {len(selected)} jobs SEQUENTIALLY on {args.device} "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}).", flush=True)
        for position, job in enumerate(selected, start=1):
            print(f"\n[{label}] Job {position}/{len(selected)}: ID {job['job_id']} "
                  f"({job['target_label']}, {job['mode']}, seed {job['seed']})", flush=True)
            atomic_json(folder / "progress.json", {
                "group_id": args.group, "job_ids": [j["job_id"] for j in selected],
                "state": "running", "current_job_id": job["job_id"],
                "position": position, "total_jobs": len(selected),
                "updated_unix_time": time.time(),
            })
            command = [sys.executable, "-u", str(ROOT / "experiment.py"), "run", "--job", str(job["job_id"]),
                       "--device", args.device, "--num-workers", "0"]
            if args.keep_models:
                command.append("--keep-models")
            try:
                # No torch import or CUDA initialization in this launcher.
                subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())
            except BaseException:
                atomic_json(folder / "progress.json", {
                    "group_id": args.group, "state": "stopped", "current_job_id": job["job_id"],
                    "position": position, "total_jobs": len(selected), "updated_unix_time": time.time(),
                })
                raise
        atomic_json(folder / "progress.json", {
            "group_id": args.group, "state": "complete",
            "job_ids": [j["job_id"] for j in selected], "total_jobs": len(selected),
            "updated_unix_time": time.time(),
        })
        print(f"\nCOMPLETED {label}: all {len(selected)} jobs finished.", flush=True)


def status() -> None:
    complete = 0
    fits = 0
    for job in jobs():
        path = ROOT / "output/jobs" / f"job_{job['job_id']:03d}"
        count = sum((path / arm / f"pct{percent}" / f"fold{fold}" / "done.json").exists()
                    for arm in ARMS for percent in PERCENTS for fold in range(5))
        complete += count == 20
        fits += count
        print(f"group {job['job_id'] // JOBS_PER_GROUP}  job {job['job_id']:2d}  {job['target_label']:17s} {job['mode']:8s} seed={job['seed']}  {count:2d}/20 target fits")
    print(f"Total: {complete}/30 jobs; {fits}/600 target fits. Run summarize for full integrity checks.")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    prepare_p = sub.add_parser("prepare", help="Check input hashes, partitions, pair indices and historical coverage.")
    prepare_p.add_argument("--project-root", default=None)
    for command in ("run", "smoke"):
        q = sub.add_parser(command)
        q.add_argument("--device", default="cuda:0", help="No automatic CPU fallback. Use cpu only deliberately.")
        q.add_argument("--num-workers", type=int, default=0, choices=(0,),
                       help="Must be 0: no DataLoader subprocesses; safe for long experiment paths.")
        q.add_argument("--keep-models", action="store_true", help="Also retain target-model weights; otherwise keep errors and source checkpoints only.")
        if command == "run":
            group = q.add_mutually_exclusive_group(required=True)
            group.add_argument("--all", action="store_true", help="All 30 jobs, sequentially on one device.")
            group.add_argument("--group", type=int, choices=range(N_GROUPS), metavar="0..5",
                               help="Five sequential jobs: group 0 = jobs 0-4, ..., group 5 = jobs 25-29.")
            group.add_argument("--job", type=int, choices=range(30), metavar="0..29")
            q.add_argument("--dry-run", action="store_true", help="Print selected job IDs without loading data or training.")
    sub.add_parser("status")
    q = sub.add_parser("summarize")
    q.add_argument("--recompute", action="store_true", help="Accepted for compatibility; summaries always verify the raw errors.")
    return p


def main() -> int:
    args = parser().parse_args()
    if getattr(args, "num_workers", 0) < 0:
        raise ValueError("num-workers must be nonnegative.")
    if args.command == "prepare":
        prepare(args.project_root)
    elif args.command == "status":
        status()
    elif args.command == "summarize":
        from summarize_float32 import run_summary
        run_summary()
    elif args.command == "smoke":
        for mode in MODES:
            job = next(j for j in jobs() if j["target"] == "BE_KUL_Wave_1B" and j["mode"] == mode and j["seed"] == 1234)
            run_job(job, args, smoke=True)
        atomic_json(ROOT / "smoke_output/smoke_passed.json", {"passed": True, "device": args.device,
                    "research_results": False, "target_fits": 8, "source_fits": 2})
        print("Smoke test passed. Its tiny one-epoch results are excluded from research summaries.")
    elif args.command == "run" and (args.all or args.group is not None or args.dry_run):
        launch_jobs(args)
    else:
        run_job(jobs()[args.job], args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nStopped: {exc}\nCompleted stages remain available. Rerun the same command to resume.", file=sys.stderr)
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exc()
        raise SystemExit(1)
