#!/usr/bin/env python3
"""Experiment 4: fixed-slot GRU-D source pretraining and full fine-tuning."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from runtime import *
configure()
import argparse, contextlib
from pathlib import Path


def load_manifest(check_cache=True):
    path=ROOT/'setup/manifest.json'
    if not path.is_file(): raise RuntimeError('First run: python -B experiment.py prepare')
    m=read_json(path)
    if m['package_fingerprint']!=package_fingerprint():
        raise RuntimeError('Code/reference files changed after preparation. Restore the original files; do not mix experiments.')
    if check_cache:
        for name,wanted in m['setup_sha256'].items():
            p=inside(ROOT/name)
            if not p.is_file() or sha256(p)!=wanted: raise RuntimeError(f'Prepared input/cache changed: {name}')
    return m


def find_project(explicit=None):
    candidates=[Path(explicit).expanduser().resolve()] if explicit else ROOT.parents
    for p in candidates:
        if (p/'data/dataset.csv').is_file(): return p
    raise FileNotFoundError('Cannot find the project data folder. Use prepare --project-root "path/to/main".')


def prepare(args):
    import pandas as pd
    from data import build_cache,Store
    with lock(ROOT/'setup/.prepare.lock'):
        if (ROOT/'setup/manifest.json').is_file():
            m=load_manifest(); print('Preparation already complete and verified. No inputs were changed.'); return
        project=find_project(args.project_root)
        prov=read_json(ROOT/'reference/provenance.json')
        for name,wanted in prov['input_sha256'].items():
            p=project/'data'/name
            if not p.is_file() or sha256(p)!=wanted:
                raise RuntimeError(f'Missing/changed reviewed input: {name}. Use the unchanged original inputs.')
        print('Verified read-only original inputs. Building local compact cache...',flush=True)
        overview=build_cache(project); store=Store()
        partitions,expected,normalization=store.audit()
        atomic_csv(ROOT/'setup/partition_counts.csv',pd.DataFrame(partitions))
        atomic_csv(ROOT/'setup/availability_audit.csv',pd.DataFrame(normalization))
        atomic_json(ROOT/'setup/expected_test_counts.json',expected)
        atomic_csv(ROOT/'setup/jobs.csv',pd.DataFrame([dict(j,sources='|'.join(sources(j['target'])),
                    excluded_sources='|'.join(EXCLUSIONS.get(j['target'],[]))) for j in jobs()]))
        from tests import verify_real_data
        parity=verify_real_data(store)
        from analyse import verify_reference_coverage
        reference_check=verify_reference_coverage(store)
        validation=dict(passed=True,real_data_sequence_parity=parity,reference_coverage=reference_check,
                        original_folds_and_saved_10_and_100_percent_subsets=True,
                        future_measurements_hidden=True,source_validation_fold=0)
        atomic_json(ROOT/'setup/validation.json',validation)
        setup_hashes={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'setup').rglob('*'))
                      if p.is_file() and not p.name.startswith('.') and p.name!='manifest.json'}
        fp=package_fingerprint()
        m=dict(version=VERSION,package_fingerprint=fp,project_root_at_preparation=str(project),
               original_input_sha256=prov['input_sha256'],setup_sha256=setup_hashes,overview=overview,
               hyperparameters=HYPER,seeds=list(SEEDS),fractions=list(PERCENTS),origins=list(ORIGINS),
               exclusions=EXCLUSIONS,jobs=jobs(),n_source_pretrainings=21,n_target_only_fits=210,
               n_finetuning_fits=210,total_fits=441,
               experiment_fingerprint=digest(dict(package=fp,inputs=prov['input_sha256'],setup=setup_hashes)))
        atomic_json(ROOT/'setup/manifest.json',m)
    print(f'Prepared {overview["n_pairs"]:,} pairs: 21 jobs, 441 research fits. All validation checks passed.')


def select_jobs(args):
    if args.all: return jobs()
    if args.job is not None:
        if args.job not in range(21): raise ValueError('--job must be 0..20.')
        return [jobs()[args.job]]
    if args.groups is None or args.group is None: raise ValueError('Choose --all, --job N, or --groups N --group K.')
    if not 1<=args.groups<=21 or not 0<=args.group<args.groups:
        raise ValueError('Use --groups 1..21 and --group 0..groups-1.')
    return [j for j in jobs() if j['job_id']%args.groups==args.group]


def configure_device(args):
    import torch
    if args.cpu_threads<1: raise ValueError('--cpu-threads must be positive.')
    torch.set_num_threads(args.cpu_threads)
    torch.set_default_dtype(torch.float32)
    device=torch.device(args.device)
    if device.type not in ('cpu','cuda'): raise ValueError('Only CPU and CUDA are supported.')
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in this Python environment. No silent CPU fallback.')
        if device.index is None: device=torch.device('cuda:0')
        if device.index>=torch.cuda.device_count(): raise ValueError('GPU index is outside CUDA_VISIBLE_DEVICES.')
        torch.cuda.set_device(device)
    # No AMP, compile or TF32 shortcut: preserve the original float32 GRU-D arithmetic policy.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    return device,environment(device,args.cpu_threads)


def run_job(job,store,manifest,device,env,smoke=False):
    from training import stage,verify_receipt
    base=inside(ROOT/('smoke_output' if smoke else 'output')/'jobs'/f'job_{job["job_id"]:03d}')
    base.mkdir(parents=True,exist_ok=True)
    with lock(base/'.job.lock'):
        with inside(base/'run.log').open('a',encoding='utf-8',buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
                print(f'Job {job["job_id"]}: {job["target"]}, seed {job["seed"]}; device {device}',flush=True)
                stage(base/'pretrain',store,job,'pretrain',None,None,device,env,manifest,smoke=smoke)
                for pct in PERCENTS:
                    for fold in ([0] if smoke else range(5)):
                        for arm in ARMS:
                            stage(base/arm/f'pct{pct}'/f'fold{fold}',store,job,arm,fold,pct,
                                  device,env,manifest,source_folder=base/'pretrain' if arm=='transfer' else None,smoke=smoke)
                atomic_json(base/'complete.json',dict(job=job,smoke=smoke,
                            target_fits=4 if smoke else 20,source_fits=1,
                            experiment_fingerprint=manifest['experiment_fingerprint']))
                print(f'Job {job["job_id"]} complete.',flush=True)


def run(args,smoke=False):
    m=load_manifest()
    selected=[jobs()[args.job]] if smoke else select_jobs(args)
    if not smoke and args.dry_run:
        for j in selected: print(f'job {j["job_id"]:02d}: {j["target"]}, seed {j["seed"]}, sources={",".join(sources(j["target"]))}')
        print(f'{len(selected)} jobs; {len(selected)*21} research fits. No training performed.'); return
    device,env=configure_device(args)
    base=ROOT/('smoke_output' if smoke else 'output')
    with lock(base/'.environment.lock'):
        p=base/'environment.json'
        if p.is_file() and environment_signature(read_json(p))!=environment_signature(env):
            raise RuntimeError('A different Python/library/numerical environment already started this output folder. Do not mix environments.')
        if not p.is_file(): atomic_json(p,env)
    from data import Store
    store=Store()
    for j in selected: run_job(j,store,m,device,env,smoke)
    if smoke:
        from analyse import verify_smoke
        result=verify_smoke(store,selected[0]); atomic_json(ROOT/'smoke_output/validation.json',result)
        print('SMOKE PASSED: one source fit, four target fits, recomputed metrics and invariance checks. Not research results.')


def status(args):
    load_manifest(check_cache=args.verify)
    from training import verify_receipt
    counts={'pretrain':0,'target_only':0,'transfer':0}; total={'pretrain':21,'target_only':210,'transfer':210}
    records=[]
    for j in jobs():
        base=ROOT/'output/jobs'/f'job_{j["job_id"]:03d}'
        n={'pretrain':0,'target_only':0,'transfer':0}
        for arm in n:
            paths=[base/'pretrain'] if arm=='pretrain' else [base/arm/f'pct{p}'/f'fold{k}' for p in PERCENTS for k in range(5)]
            for path in paths:
                good=verify_receipt(path) is not None if args.verify else (path/'done.json').is_file()
                n[arm]+=int(good); counts[arm]+=int(good)
        records.append(dict(j,**n))
        print(f'job {j["job_id"]:02d} {j["target"]:22s} seed={j["seed"]}: source {n["pretrain"]}/1, target-only {n["target_only"]}/10, transfer {n["transfer"]}/10')
    print(' | '.join(f'{k}: {counts[k]}/{total[k]}' for k in counts))
    if not args.verify: print('Completion markers only. Use status --verify for file-hash validation; summarize always validates.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--project-root')
    for name in ('run','smoke'):
        p=sub.add_parser(name); p.add_argument('--device',default='cuda:0'); p.add_argument('--cpu-threads',type=int,default=2)
        if name=='run':
            choice=p.add_mutually_exclusive_group(required=True)
            choice.add_argument('--all',action='store_true'); choice.add_argument('--job',type=int)
            choice.add_argument('--groups',type=int)
            p.add_argument('--group',type=int); p.add_argument('--dry-run',action='store_true')
        else: p.add_argument('--job',type=int,default=0,choices=range(21))
    p=sub.add_parser('status'); p.add_argument('--verify',action='store_true')
    sub.add_parser('summarize')
    args=parser.parse_args()
    if args.command=='prepare': prepare(args)
    elif args.command=='run': run(args)
    elif args.command=='smoke': run(args,smoke=True)
    elif args.command=='status': status(args)
    else:
        m=load_manifest()
        from analyse import summarize
        summarize(m)

if __name__=='__main__':
    try: main()
    except (ValueError,RuntimeError,FileNotFoundError,FloatingPointError) as e:
        print(f'ERROR: {e}',file=sys.stderr,flush=True); sys.exit(1)
