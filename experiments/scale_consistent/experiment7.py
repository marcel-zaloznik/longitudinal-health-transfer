#!/usr/bin/env python3
"""Extend the completed Experiment 3 to ABCD and YFS; publish one seven-target report."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, os, subprocess, time, traceback
from pathlib import Path
from exp3_utils import (
    ROOT, SEEDS, PERCENTS, configure_local_runtime, inside, read_json,
    atomic_json, atomic_csv, sha256, object_hash, file_lock, load_data,
    target_frames, frame_counts, environment,
)
configure_local_runtime()
from seven_utils import (
    VERSION, NEW_TARGETS, ALL_TARGETS, SETUP, OUTPUT, SMOKE, LIBRARY_KEYS,
    extension_jobs, group_jobs, normalizer_path, extension_fingerprint,
    load_legacy, load_extension, check_legacy_inventory, verify_baseline_coverage,
    job_folder, stage_folder, TRAINING_ARMS,
)


def prepare(project_arg=None):
    import numpy as np
    import pandas as pd
    import engine
    from normalization import PairMetadata, validate_reconstruction
    from seven_analyse import collect_legacy
    with file_lock(SETUP / ".prepare.lock"):
        if (SETUP / "manifest.json").is_file():
            manifest, _, project = load_extension()
            if project_arg and Path(project_arg).expanduser().resolve() != project:
                raise RuntimeError("Prepared project path differs. Keep the same read-only project path.")
            check_legacy_inventory(manifest)
            print("Seven-dataset preparation already complete; original five-target outputs are unchanged.")
            return
        legacy, project = load_legacy(project_arg)
        print("Checking all 150 existing scale-consistent fits and frozen reset controls; no original file is edited...", flush=True)
        bundle, inventory, libraries = collect_legacy(legacy, project)
        print("Loading the frozen data and reconstructing ABCD/YFS training-only partitions and normalizers...", flush=True)
        frame, pairs, split, subsets = load_data(project, engine)
        origins = sorted(frame.Origin.unique().tolist())
        if origins != legacy["origins"] or origins != list(ALL_TARGETS) or len(split) != 5:
            raise ValueError("Original seven origins/five folds differ.")
        if frame.groupby("PersonID").Origin.nunique().max() != 1:
            raise ValueError("Participant identifiers occur in multiple origins.")
        if set(frame.PersonID) != set.union(*split) or any(split[i] & split[j] for i in range(5) for j in range(i)):
            raise ValueError("Original folds are not a disjoint complete participant partition.")
        meta = PairMetadata(frame,pairs)
        reconstruction = validate_reconstruction(frame,pairs)
        pair_counts = pairs.groupby("PersonID").size()
        expected = read_json(ROOT / "setup/expected_test_counts.json")
        panel = read_json(project / "data/primary_target_panel.json")
        eligibility = read_json(project / "data/eligible_dataset_attribute_pairs.json")
        baseline = pd.read_csv(ROOT / "reference/exp2_dataset_scores_by_seed.csv", float_precision="round_trip")
        source_partitions, partitions, target_statistics = {}, {}, {}
        partition_rows, normalization_rows, source_rows, assets = [], [], [], []
        def save(name, value):
            path = SETUP / name
            atomic_json(path,value)
            assets.append(path)
        for target in NEW_TARGETS:
            target_frame = frame[frame.Origin.eq(target)]
            sources = list(engine.ORIGIN_MAP[target])
            if set(sources) != set(origins)-{target} or len(sources) != 6:
                raise ValueError("All-source dataset composition differs from the original engine.")
            source = frame[frame.Origin.isin(sources)].copy()
            source_train,source_val = engine.make_pretrain_split(source,split,origins.index(target))
            if source_train.empty or source_val.empty or set(source.PersonID) & set(target_frame.PersonID):
                raise ValueError("Empty source partition or source/target identifier overlap.")
            if not set(source_val.PersonID).issubset(split[0]) or set(source_train.PersonID) & split[0]:
                raise ValueError("Unexpected source split fallback; original source-fold-0 policy is required.")
            source_partitions[target] = {"sources":sources,"source_train":frame_counts(source_train,pair_counts),
                                         "source_val":frame_counts(source_val,pair_counts)}
            norm = meta.normalizers(source_train)
            save("normalizers/"+target+".json",norm)
            for role in ("source_train","source_val"):
                source_rows.append({"target":target,"role":role,"sources":"|".join(sources),**source_partitions[target][role]})
            for col in engine.TARGET_COLS:
                normalization_rows.append({"target":target,"attribute":col,
                    "source_input_mean":norm["value"][col]["mean"],"source_input_std":norm["value"][col]["std"],
                    "source_input_count":norm["value"][col]["n"],"source_output_mean":norm["target"][col]["mean"],
                    "source_output_std":norm["target"][col]["std"],"source_output_count":norm["target"][col]["n"],
                    "source_horizon_scale":norm["time"]["horizon_scale"]})
            expected[target],partitions[target],target_statistics[target] = {},{},{}
            n_test_people = 0
            for fold in range(5):
                test_pairs = pairs[pairs.PersonID.isin(set(target_frame.PersonID)&split[fold])]
                attrs = [a for a in panel if eligibility[target][str(fold)][a]["eligible"]]
                values = frame.loc[test_pairs.output_row,attrs].to_numpy(dtype=float)
                expected[target][str(fold)] = {a:int(np.isfinite(values[:,j]).sum()) for j,a in enumerate(attrs)}
                if not attrs or any(n <= 0 for n in expected[target][str(fold)].values()):
                    raise ValueError("An eligible primary test outcome has no observations.")
                n_test_people += int(test_pairs.loc[np.isfinite(values).any(axis=1),"PersonID"].nunique())
            verify_baseline_coverage(target,expected[target],n_test_people,baseline)
            for percent in PERCENTS:
                partitions[target][str(percent)],target_statistics[target][str(percent)] = {},{}
                for fold in range(5):
                    train,val,test = target_frames(frame,target,fold,percent,split,subsets)
                    row = {role:frame_counts(part,pair_counts) for role,part in
                           (("target_train",train),("target_val",val),("target_test",test))}
                    partitions[target][str(percent)][str(fold)] = row
                    target_statistics[target][str(percent)][str(fold)] = meta.target_statistics(train)
                    for role,counts in row.items():
                        partition_rows.append({"target":target,"percent":percent,"fold":fold,"role":role,**counts})
            print(f"Prepared {target}: source train {norm['n_records']:,} pairs; baseline coverage matched; no target validation/test statistics used.",flush=True)
        save("expected_test_counts.json",expected)
        save("source_partitions.json",source_partitions)
        save("target_partitions.json",partitions)
        save("target_statistics.json",target_statistics)
        save("legacy_output_sha256.json",inventory)
        save("validation.json",{"passed":True,"original_scale_fits_verified":150,"original_reset_controls_verified":150,
            "source_reconstruction":reconstruction,"baseline_coverage_matches":True,"n_people":int(frame.PersonID.nunique()),
            "n_visits":len(frame),"n_prediction_pairs":len(pairs),"folds_disjoint":True,
            "pair_ownership_and_future_outputs_verified":True,"warning":"Distinct IDs across cohorts do not prove distinct real-world people."})
        for name,rows in (("jobs.csv",extension_jobs()),("target_partitions.csv",partition_rows),
            ("source_counts.csv",source_rows),("normalization_audit.csv",normalization_rows),
            ("legacy_dataset_scores.csv",bundle["scores"])):
            path = SETUP/name
            atomic_csv(pd.DataFrame(rows),path)
            assets.append(path)
        manifest = {"version":VERSION,"project_root":str(project),"targets":list(ALL_TARGETS),
            "added_targets":list(NEW_TARGETS),"origins":origins,"seeds":list(SEEDS),"percents":list(PERCENTS),
            "jobs":extension_jobs(),"n_added_source_fits":6,"n_added_reset_fits":60,"n_added_scale_fits":60,
            "total_scale_fits":210,"total_reset_fits":210,"required_libraries":libraries,
            "legacy_manifest_sha256":sha256(ROOT/"setup/manifest.json"),
            "legacy_experiment_fingerprint":legacy["experiment_fingerprint"],
            "extension_code_fingerprint":extension_fingerprint(),"input_sha256":legacy["input_sha256"],
            "setup_sha256":{str(p.relative_to(ROOT)):sha256(p) for p in sorted(assets)}}
        manifest["experiment_fingerprint"] = object_hash(manifest)
        atomic_json(SETUP/"manifest.json",manifest)
        print("Prepared: six new source jobs, 60 reset fits, 60 scale-consistent fits. Existing 150 scale fits remain untouched.")
        print("Next: python -B experiment7.py smoke --device cuda:0")


def launch(args):
    selected = extension_jobs() if args.all else group_jobs(args.groups,args.group)
    label = "all" if args.all else f"groups{args.groups}_group{args.group}"
    for job in selected:
        print(f"Added source job {job['extension_job_id']}: {job['target']}, seed {job['seed']}; one source fit + 20 target fits; device={args.device}",flush=True)
    if args.dry_run:
        return
    load_extension()
    folder = OUTPUT/"launchers"/label
    with file_lock(folder/".launcher.lock"):
        for job in selected:
            atomic_json(folder/"progress.json",{"state":"running","extension_job_id":job["extension_job_id"],"updated_unix_time":time.time()})
            command = [sys.executable,"-B","-u",str(ROOT/"experiment7.py"),"run","--job",str(job["extension_job_id"]),
                "--device",args.device,"--cpu-threads",str(args.cpu_threads)]
            if args.keep_models:
                command.append("--keep-models")
            try:
                subprocess.run(command,cwd=ROOT,env=os.environ.copy(),check=True)
            except BaseException:
                atomic_json(folder/"progress.json",{"state":"stopped","extension_job_id":job["extension_job_id"]})
                raise
        atomic_json(folder/"progress.json",{"state":"complete","extension_job_ids":[j["extension_job_id"] for j in selected]})
        print(f"COMPLETED extension {label}.",flush=True)


def status():
    from exp3_utils import jobs as legacy_jobs
    old = sum((ROOT/"output/jobs"/f"job_{j['job_id']:03d}"/f"fold{f}/done.json").is_file() for j in legacy_jobs() for f in range(5))
    source,reset,scale = 0,0,0
    for job in extension_jobs():
        pre = int((job_folder(job)/"pretrain/done.json").is_file())
        counts = {a:sum((stage_folder(job,a,p,f)/"done.json").is_file() for p in PERCENTS for f in range(5)) for a in TRAINING_ARMS}
        source += pre
        reset += counts["reset_control"]
        scale += counts["scale_consistent"]
        print(f"Added job {job['extension_job_id']}: {job['target']:4s} seed {job['seed']}: source {pre}/1; reset {counts['reset_control']}/10; scale {counts['scale_consistent']}/10")
    print(f"Existing scale fits: {old}/150. Added sources: {source}/6. Added reset fits: {reset}/60. Added scale fits: {scale}/60.")
    print(f"Combined scale fits: {old+scale}/210. Presence counts only; summarize performs full integrity and matching checks.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    q = sub.add_parser("prepare",help="Verify completed work and prepare ABCD/YFS without changing old files.")
    q.add_argument("--project-root",help="Optional original main folder; data/splits must match frozen hashes.")
    for command in ("smoke","run"):
        q = sub.add_parser(command)
        q.add_argument("--device",default="cuda:0")
        q.add_argument("--cpu-threads",type=int,default=2)
        q.add_argument("--keep-models",action="store_true",help="Optionally save target weights. Source checkpoints are always saved.")
        if command == "run":
            sel = q.add_mutually_exclusive_group(required=True)
            sel.add_argument("--all",action="store_true")
            sel.add_argument("--group",type=int)
            sel.add_argument("--job",type=int,choices=range(6))
            q.add_argument("--groups",type=int,default=6)
            q.add_argument("--dry-run",action="store_true")
    sub.add_parser("status")
    sub.add_parser("summarize",help="Replace the current report/export with one complete seven-target result.")
    return p


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args.project_root)
    elif args.command == "status":
        status()
    elif args.command == "summarize":
        from seven_analyse import summarize
        summarize()
    elif args.command == "smoke":
        import torch
        from seven_train import run_job
        from training import choose_device
        for job in extension_jobs()[:2]:
            run_job(job,args,smoke=True)
        manifest,_,_ = load_extension(check_data=False)
        env = environment(torch,choose_device(args.device))
        atomic_json(SMOKE/"smoke_passed.json",{"passed":True,"research_results":False,
            "experiment_fingerprint":manifest["experiment_fingerprint"],"device":args.device,
            "libraries":{k:env[k] for k in LIBRARY_KEYS},"source_fits":2,"target_fits":8})
        print("Extension smoke test PASSED for ABCD and YFS, both arms and both fractions. Smoke results never enter the report.")
    elif args.command == "run":
        if args.job is not None:
            job = extension_jobs()[args.job]
            if args.dry_run:
                print(job)
            else:
                from seven_train import run_job
                run_job(job,args)
        else:
            launch(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as exc:
        print(f"\nStopped: {exc}\nOriginal five-target outputs are unchanged. Resume using the identical extension command.",file=sys.stderr)
        if not isinstance(exc,KeyboardInterrupt):
            traceback.print_exc()
        raise SystemExit(1)
