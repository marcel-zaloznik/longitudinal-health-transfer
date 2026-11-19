"""Audited aggregate reporting and privacy-limited results export."""
from __future__ import annotations
import contextlib, math, os, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from runtime import (ROOT, ATTRS, LABELS, ORIGINS, SEEDS, PERCENTS, ARMS, HYPER, CODE_FILES,
                     inside, read_json, atomic_json, atomic_csv, atomic_text, sha256, digest,
                     jobs, lock, sources, stage_seed, loader_seed, environment_signature)
from metrics import fold_metrics,combine_folds,assert_close
from training import verify_receipt


def references():
    exp3=pd.read_csv(ROOT/'reference/experiment3_dataset_scores.csv',float_precision='round_trip')
    selected=exp3[exp3.arm.isin(['scale_consistent','target_only_validation_selected'])].copy()
    selected['model']=selected.arm.map({'scale_consistent':'STraTS_exp3_scale_consistent_all_sources',
                                      'target_only_validation_selected':'STraTS_exp2_target_only_validation_selected'})
    selected['context']='Context only; different architecture/training; transfer uses all sources without GRU-D exclusions.'
    old=pd.read_csv(ROOT/'reference/historical_target_only_scores.csv',float_precision='round_trip')
    old=old.rename(columns={'dataset':'target','train_data_percent':'percent','main_macro_nrmse':'nrmse',
        'main_macro_nmae':'nmae','n_attributes_evaluated':'n_attributes','n_error_rows':'n_primary_errors',
        'n_folds_evaluated':'n_folds'})
    old['model']=old.method.map({'random_forest_fixed':'RF_historical_target_only_single_seed',
                               'gru_d':'GRUD_historical_target_only_variable_slots'})
    old['context']='Historical, not rerun; RF has one seed; historical GRU-D drops unavailable slots and has different RNG orchestration.'
    cols=['target','percent','seed','model','nrmse','nmae','n_attributes','n_primary_errors','n_test_people','n_folds','context']
    return pd.concat([selected[cols],old[cols]],ignore_index=True)


def verify_reference_coverage(store):
    refs=references()
    for target in ORIGINS:
        counts=[]; people=0; attrs=set()
        for k in range(5):
            ids=store.pair_indices(store.target_people(target,k,100)[2])
            obs=np.isfinite(store.raw[store.outputs[ids]])
            selected=[a for a in store.primary if store.eligibility[target][str(k)][a]['eligible']]
            keep=obs[:,[ATTRS.index(a) for a in selected]]
            counts.append(int(keep.sum())); attrs.update(selected)
            people+=int(np.unique(store.pair_person[ids][keep.any(axis=1)]).size)
        part=refs[refs.target.eq(target)]
        for name,value in [('n_primary_errors',sum(counts)),('n_test_people',people),('n_attributes',len(attrs)),('n_folds',5)]:
            if not part[name].eq(value).all(): raise ValueError(f'Frozen comparator coverage differs: {target}/{name}')
    return dict(passed=True,n_reference_rows=len(refs),includes='Experiment 2 selected STraTS, Experiment 3 scale-consistent STraTS, historical RF and historical GRU-D')


def verify_history(folder,receipt,smoke=False):
    history=pd.read_csv(folder/'history.csv',float_precision='round_trip')
    details=receipt['details']
    if history.epoch.tolist()!=list(range(1,len(history)+1)) or len(history)!=details['epochs_run']:
        raise ValueError('Invalid epoch history.')
    if not np.isfinite(history[['train_mse','validation_mse']].to_numpy()).all(): raise ValueError('Nonfinite training history.')
    best=float('inf'); best_epoch=0; wait=0
    for row in history.itertuples(index=False):
        if wait>=HYPER['patience']: raise ValueError('Training continued past early stopping.')
        improved=row.validation_mse+HYPER['min_delta']<best
        if bool(row.improved)!=improved: raise ValueError('History improvement marker disagrees with stopping rule.')
        if improved: best=float(row.validation_mse);best_epoch=int(row.epoch);wait=0
        else: wait+=1
    if best_epoch!=details['best_epoch'] or not np.isclose(best,details['selected_validation_mse'],rtol=1e-12,atol=1e-12):
        raise ValueError('Selected checkpoint does not follow the original min-delta rule.')
    cap=2 if smoke else HYPER['max_epochs']
    if len(history)!=cap and wait<HYPER['patience']: raise ValueError('Incomplete training history.')
    return history


def recompute(folder,store,job,fold,percent,smoke=False):
    with np.load(folder/'predictions.npz',allow_pickle=False) as saved:
        ids=saved['pair_ids']
        expected=store.pair_indices(store.target_people(job['target'],fold,percent,smoke)[2])
        if smoke: expected=expected[:128]
        if not np.array_equal(ids,expected): raise ValueError('Predictions do not cover the exact expected held-out pairs.')
        out=store.outputs[ids]; last=store.inputs[ids,store.lengths[ids]-1]
        metadata=dict(person_id=np.asarray([store.people[int(i)] for i in store.pair_person[ids]]),
                      input_row=last,output_row=out,input_age=store.age[last],output_age=store.age[out],
                      horizon=store.age[out]-store.age[last],n_input_visits=store.lengths[ids])
        for name,values in metadata.items():
            if not np.array_equal(saved[name],values): raise ValueError(f'Prediction metadata changed: {name}')
        result=fold_metrics(store,job['target'],fold,ids,saved['prediction'],saved['truth'],saved['observed'],strict=not smoke)
    assert_close(result,read_json(folder/'metrics.json'))
    return result


def verify_smoke(store,job):
    base=ROOT/'smoke_output/jobs'/f'job_{job["job_id"]:03d}'
    pre=verify_receipt(base/'pretrain')
    if pre is None: raise ValueError('Missing smoke source checkpoint.')
    verify_history(base/'pretrain',pre,True)
    for pct in PERCENTS:
        scores=[]
        for arm in ARMS:
            path=base/arm/f'pct{pct}'/'fold0';r=verify_receipt(path)
            if r is None: raise ValueError('Missing smoke stage.')
            verify_history(path,r,True);scores.append(recompute(path,store,job,0,pct,True))
        if scores[0]['identity_digest']!=scores[1]['identity_digest']: raise ValueError('Smoke test pairs differ.')
        if not read_json(base/'transfer'/f'pct{pct}'/'fold0/head_mapping.json')['passed']: raise ValueError('Invariance failed.')
    return dict(passed=True,job=job,source_fits=1,target_fits=4,research_results=False,
                checks=['completed receipts','original early-stopping rule','metric recomputation',
                        'matching test identities','raw-prediction invariance'])


def summarize_frames(scores):
    result=[]
    for (target,pct,arm),part in scores.groupby(['target','percent','arm'],sort=False):
        if set(part.seed)!=set(SEEDS) or len(part)!=3: raise ValueError('Missing/repeated seed in summary cell.')
        result.append(dict(target=target,target_label=LABELS.get(target,'Seven-dataset macro'),percent=int(pct),arm=arm,
                           nrmse_mean=float(part.nrmse.mean()),nrmse_sd=float(part.nrmse.std(ddof=1)),
                           nmae_mean=float(part.nmae.mean()),nmae_sd=float(part.nmae.std(ddof=1)),n_seeds=3))
    return pd.DataFrame(result)


def paired_tables(scores):
    pivot=scores.pivot(index=['target','percent','seed'],columns='arm',values='nrmse').reset_index()
    if pivot[list(ARMS)].isna().any().any(): raise ValueError('Unpaired arms.')
    pivot['delta_nrmse']=pivot.transfer-pivot.target_only
    pivot['gain_percent']=100*(pivot.target_only-pivot.transfer)/pivot.target_only
    rows=[]
    for (target,pct),part in pivot.groupby(['target','percent'],sort=False):
        b=float(part.target_only.mean()); t=float(part.transfer.mean())
        rows.append(dict(target=target,target_label=LABELS.get(target,'Seven-dataset macro'),percent=int(pct),
                         target_only_mean=b,target_only_sd=float(part.target_only.std(ddof=1)),
                         transfer_mean=t,transfer_sd=float(part.transfer.std(ddof=1)),
                         delta_mean=float(part.delta_nrmse.mean()),delta_sd=float(part.delta_nrmse.std(ddof=1)),
                         gain_percent=100*(b-t)/b,n_seeds=3))
    return pivot,pd.DataFrame(rows)


def context_tables(scores):
    refs=references()
    ours=scores[scores.target.ne('Overall')].copy()
    ours['model']=ours.arm.map({'target_only':'GRUD_new_target_only','transfer':'GRUD_new_transfer_excluded_scale_consistent'})
    ours['context']='New experiment. Primary controlled comparison is new GRU-D transfer vs new GRU-D target-only.'
    cols=['target','percent','seed','model','nrmse','nmae','n_attributes','n_primary_errors','n_test_people','n_folds','context']
    combined=pd.concat([ours[cols],refs[cols]],ignore_index=True)
    overall=[]
    for (model,pct,seed),part in combined.groupby(['model','percent','seed'],sort=False):
        if set(part.target)!=set(ORIGINS) or len(part)!=7: raise ValueError('Incomplete contextual seven-dataset comparator.')
        overall.append(dict(target='Overall',percent=int(pct),seed=int(seed),model=model,
                            nrmse=float(part.nrmse.mean()),nmae=float(part.nmae.mean()),context=part.context.iloc[0]))
    combined=pd.concat([combined,pd.DataFrame(overall)],ignore_index=True)
    rows=[]
    for (target,pct,model),part in combined.groupby(['target','percent','model'],sort=False):
        rows.append(dict(target=target,percent=int(pct),model=model,nrmse_mean=float(part.nrmse.mean()),
                         nrmse_sd=float(part.nrmse.std(ddof=1)) if len(part)>1 else None,n_seeds=len(part),
                         context=part.context.iloc[0]))
    summary=pd.DataFrame(rows)
    contrasts=[]
    for (target,pct),group in summary.groupby(['target','percent'],sort=False):
        transfer=float(group[group.model.eq('GRUD_new_transfer_excluded_scale_consistent')].nrmse_mean.iloc[0])
        for row in group.itertuples(index=False):
            if row.model=='GRUD_new_transfer_excluded_scale_consistent': continue
            contrasts.append(dict(target=target,percent=int(pct),comparator=row.model,comparator_n_seeds=row.n_seeds,
                comparator_nrmse=row.nrmse_mean,grud_transfer_nrmse=transfer,
                gain_percent=100*(row.nrmse_mean-transfer)/row.nrmse_mean,
                controlled_comparison=row.model=='GRUD_new_target_only'))
    return combined,summary,pd.DataFrame(contrasts)


def write_report(paired,fit_audit,context,reference_check):
    lines=['# Experiment 4 — GRU-D transfer across seven datasets','',
           'Complete: **21 source pretrainings, 210 target-only fits and 210 fine-tuning fits (441 total)**.',
           'Primary comparison: newly trained fixed-slot GRU-D target-only vs source-pretrained/full-fine-tuned GRU-D. ',
           'Both use hidden size 64, batch size 512, Adam 0.001, the original masked objective and stopping rule. ',
           'Source exclusions follow Experiment 1; inputs and decay mean remain source-normalized; the output head is converted before fine-tuning.','',
           '## Primary paired comparison','',
           'NRMSE is lower-is-better. Means ± sample SD use three seeds. Delta is transfer minus target-only; negative favors transfer. ',
           'Gains use the ratio of seed-mean scores, not the mean of seed-wise percentages.','',
           '| Dataset | Training | Target-only NRMSE | Transfer NRMSE | Paired delta | Gain |',
           '|---|---:|---:|---:|---:|---:|']
    for r in paired.itertuples(index=False):
        lines.append(f'| {r.target_label} | {r.percent}% | {r.target_only_mean:.4f} ± {r.target_only_sd:.4f} | {r.transfer_mean:.4f} ± {r.transfer_sd:.4f} | {r.delta_mean:+.4f} ± {r.delta_sd:.4f} | {r.gain_percent:+.2f}% |')
    lines+=['','## Diagnostics and interpretation','']
    for pct in PERCENTS:
        r=paired[paired.target.eq('Overall')&paired.percent.eq(pct)].iloc[0]
        sub=paired[paired.target.ne('Overall')&paired.percent.eq(pct)]
        lines.append(f'At {pct}% training, the observed seven-dataset gain is **{r.gain_percent:+.2f}%**. '
                     f'{int((sub.gain_percent>0).sum())}/7 datasets improve; {int((sub.gain_percent<0).sum())}/7 worsen. '
                     'This is descriptive, not a significance test or a guarantee of generalization.')
    lines+=['',f'Fits reaching the 1,000-epoch cap: {int(fit_audit.epoch_cap_hit.sum())}/441. '
               'All raw predictions were reloaded and their metrics recomputed; all 210 head-invariance checks passed.',
           '', '## Contextual comparisons','',
           'See `contextual_summary.csv` and `contextual_gains.csv`. These are **not a controlled architecture ranking**. ',
           'Experiment 2 supplies validation-selected target-only STraTS; Experiment 3 supplies all-source scale-consistent STraTS ',
           '(which does not apply this experiment’s source exclusions). Random forest is a historical single-seed baseline, so its SD is unavailable, not zero. ',
           'Historical GRU-D used variable-dimensional inputs and a different random-number stream; baseline drift is not a transfer effect.',
           '', '| Training | Model | Seven-dataset NRMSE | Seeds |', '|---|---|---:|---:|']
    for r in context[context.target.eq('Overall')].itertuples(index=False):
        variability=f' ± {r.nrmse_sd:.4f}' if pd.notna(r.nrmse_sd) else ' (SD unavailable)'
        lines.append(f'| {r.percent}% | {r.model} | {r.nrmse_mean:.4f}{variability} | {r.n_seeds} |')
    lines+=['','## Files and caveats','',
        '`dataset_summary.csv`, `seven_dataset_summary.csv`, `paired_by_seed.csv` and `paired_summary.csv` contain the primary results. ',
        '`raw_units_and_calibration_by_seed.csv` gives per-outcome person-balanced raw RMSE, MAE, bias and descriptive calibration. ',
        'Calibration fits observed = intercept + slope × predicted, with equal person weights within each attribute; it is not a recalibration applied to test predictions. ',
        '`fit_audit.csv`, `training_history.csv.gz`, `head_mapping_audit.csv`, `normalization_audit.csv`, and `recomputation_audit.csv` support verification.',
        '', 'The primary panel and eligibility are frozen, and vary across datasets. Folds are combined at the person-MSE level before the attribute square root. ',
        'The seven-dataset macro weights datasets equally. Three seed SDs are not confidence intervals; the five folds are not 15 independent experimental replicates. ',
        '10% means reduced training availability with unchanged validation/test partitions. Source-cohort exclusion is a conservative sensitivity policy, not verified person linkage. ',
        'No additional learning-rate search, pooled GRU-D, single-source sweep, intermediate fractions or truncation experiment was run.',
        '', 'Only aggregate tables, histories, hashes, code and non-identifying protocol metadata enter `results_to_share.zip`. ',
        'Participant-level predictions, original identifiers, local caches and model weights remain in this experiment folder. ',
        'Do not share `setup/cache/`, `output/`, or `_runtime/` as part of the results bundle.']
    return '\n'.join(lines)+'\n'


def summarize(manifest):
    from data import Store
    summary=inside(ROOT/'summary'); summary.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        stack.enter_context(lock(ROOT/'summary/.summary.lock'))
        for job in jobs(): stack.enter_context(lock(ROOT/'output/jobs'/f'job_{job["job_id"]:03d}'/'.job.lock'))
        store=Store();reference_check=verify_reference_coverage(store)
        dataset_rows=[];attribute_rows=[];raw_rows=[];audit=[];histories=[];mapping_rows=[];norm_rows=[];recomp=[];env_rows=[]
        missing=[]
        for job in jobs():
            base=ROOT/'output/jobs'/f'job_{job["job_id"]:03d}'
            expected_paths=[base/'pretrain']+[base/arm/f'pct{p}'/f'fold{k}' for arm in ARMS for p in PERCENTS for k in range(5)]
            missing += [str(p.relative_to(ROOT)) for p in expected_paths if not (p/'done.json').is_file()]
        if missing: raise RuntimeError(f'Incomplete matrix: {len(missing)} stages missing. Run status. First missing: {missing[0]}')
        print('Verifying all 441 receipts and recomputing 420 test-prediction files...',flush=True)
        expected_env=read_json(ROOT/'output/environment.json')
        for job in jobs():
            base=ROOT/'output/jobs'/f'job_{job["job_id"]:03d}'
            source=verify_receipt(base/'pretrain')
            source_norm=read_json(base/'pretrain/normalization.json')['input']
            assert_close(source_norm,store.normalizer(store.source_people(job['target'])[0]),'source_normalization')
            folds_by_cell={}
            random_init={}
            all_stages=[('pretrain',None,None,base/'pretrain')]+[(a,p,k,base/a/f'pct{p}'/f'fold{k}') for p in PERCENTS for k in range(5) for a in ARMS]
            for arm,pct,fold,path in all_stages:
                receipt=verify_receipt(path);c=receipt['config']
                if c['experiment_fingerprint']!=manifest['experiment_fingerprint'] or c['smoke'] or c['job']!=job or c['arm']!=arm or c['fold']!=fold or c['percent']!=pct:
                    raise ValueError('Mixed stage identity or smoke results.')
                if c['hyperparameters']!=HYPER or receipt['config_hash']!=digest(c): raise ValueError('Stage protocol mismatch.')
                if c['environment']!=environment_signature(expected_env): raise ValueError('Mixed numeric environments.')
                if receipt['stage_seed']!=stage_seed(job['seed'],job['target'],fold) or receipt['loader_seed']!=loader_seed(job['seed'],job['target'],fold):
                    raise ValueError('Stage seeds changed.')
                history=verify_history(path,receipt)
                ids=dict(job_id=job['job_id'],target=job['target'],target_label=LABELS[job['target']],seed=job['seed'],arm=arm,percent=pct,fold=fold)
                details=receipt['details']
                audit.append(dict(**ids,**{k:v for k,v in details.items() if k!='training_target_counts'},
                                  initial_random_parameter_sha256=receipt['initial_random_parameter_sha256'],
                                  model_sha256=receipt['files']['best_weights.pt'],source_datasets='|'.join(sources(job['target'])) if arm!='target_only' else ''))
                histories.append(history.assign(**ids)); env_rows.append(dict(**ids,**read_json(path/'environment.json')))
                norm=read_json(path/'normalization.json')
                for coordinate in ('input','output'):
                    for a,m,s,n in zip(ATTRS,norm[coordinate]['mean'],norm[coordinate]['std'],norm[coordinate]['n']):
                        norm_rows.append(dict(**ids,coordinate=coordinate,attribute=a,mean=m,std=s,n_observed_training_visits=n))
                if arm=='pretrain': continue
                people=store.target_people(job['target'],fold,pct)
                expected_norm=store.normalizer(people[0])
                assert_close(norm['output'],expected_norm,'target_output_normalization')
                assert_close(norm['input'],source_norm if arm=='transfer' else expected_norm,'target_input_normalization')
                for role,persons in zip(('train','validation','test'),people):
                    if receipt['partitions'][role]!=store.counts(persons): raise ValueError('Stage participant membership changed.')
                expected_counts=np.isfinite(store.raw[store.outputs[store.pair_indices(people[0])]]).sum(axis=0).tolist()
                if details['training_target_counts']!=expected_counts: raise ValueError('Training label masks changed.')
                metrics=recompute(path,store,job,fold,pct)
                folds_by_cell.setdefault((arm,pct),[]).append(metrics)
                recomp.append(dict(**ids,passed=True,n_primary_errors=metrics['n_primary_errors'],identity_digest=metrics['identity_digest']))
                key=(pct,fold)
                random_init.setdefault(key,receipt['initial_random_parameter_sha256'])
                if random_init[key]!=receipt['initial_random_parameter_sha256']: raise ValueError('Unpaired fresh initialization seeds.')
                if arm=='transfer':
                    if c['source_checkpoint_sha256']!=source['files']['best_weights.pt'] or c['source_receipt_sha256']!=sha256(base/'pretrain/done.json'):
                        raise ValueError('A fold used a different source checkpoint.')
                    mapping=read_json(path/'head_mapping.json')
                    if not mapping['passed']: raise ValueError('Failed head mapping.')
                    mapping_rows.append(dict(**ids,passed=True,max_absolute_raw_difference=mapping['max_absolute_raw_difference'],
                        n_training_records_checked=mapping['n_training_records_checked'],
                        source_unobserved_target_observed='|'.join(mapping['source_unobserved_target_observed'])))
            for pct in PERCENTS:
                left=sorted(folds_by_cell[('target_only',pct)],key=lambda x:x['fold'])
                right=sorted(folds_by_cell[('transfer',pct)],key=lambda x:x['fold'])
                if any(a['identity_digest']!=b['identity_digest'] for a,b in zip(left,right)): raise ValueError('Arms evaluated different test observations.')
                for arm in ARMS:
                    score,attributes,raw=combine_folds(folds_by_cell[(arm,pct)])
                    ids=dict(target=job['target'],target_label=LABELS[job['target']],percent=pct,seed=job['seed'],arm=arm)
                    dataset_rows.append(dict(**ids,**score))
                    attribute_rows.extend(dict(**ids,**r) for r in attributes)
                    raw_rows.extend(dict(**ids,**r) for r in raw)
            print(f'  Verified job {job["job_id"]:02d}: {job["target"]}, seed {job["seed"]}',flush=True)
        scores=pd.DataFrame(dataset_rows)
        if len(scores)!=84: raise ValueError('Incomplete dataset/seed/fraction/arm matrix.')
        overall=[]
        for (pct,seed,arm),group in scores.groupby(['percent','seed','arm'],sort=False):
            if set(group.target)!=set(ORIGINS): raise ValueError('Incomplete seven-dataset aggregation.')
            overall.append(dict(target='Overall',target_label='Seven-dataset macro',percent=pct,seed=seed,arm=arm,
                                nrmse=float(group.nrmse.mean()),nmae=float(group.nmae.mean())))
        overall=pd.DataFrame(overall); combined=pd.concat([scores,overall],ignore_index=True)
        paired,paired_summary=paired_tables(combined)
        contextual,contextual_summary,contextual_gains=context_tables(combined)
        tables={'dataset_scores_by_seed.csv':scores,'dataset_summary.csv':summarize_frames(scores),
            'seven_dataset_scores_by_seed.csv':overall,'seven_dataset_summary.csv':summarize_frames(overall),
            'attribute_scores_by_seed.csv':pd.DataFrame(attribute_rows),
            'raw_units_and_calibration_by_seed.csv':pd.DataFrame(raw_rows),
            'paired_by_seed.csv':paired,'paired_summary.csv':paired_summary,
            'contextual_scores_by_seed.csv':contextual,'contextual_summary.csv':contextual_summary,
            'contextual_gains.csv':contextual_gains,'fit_audit.csv':pd.DataFrame(audit),
            'environment_by_fit.csv':pd.DataFrame(env_rows),'head_mapping_audit.csv':pd.DataFrame(mapping_rows),
            'normalization_audit.csv':pd.DataFrame(norm_rows),'recomputation_audit.csv':pd.DataFrame(recomp),
            'training_history.csv.gz':pd.concat(histories,ignore_index=True)}
        for name,frame in tables.items(): atomic_csv(summary/name,frame)
        atomic_text(summary/'REPORT.md',write_report(paired_summary,tables['fit_audit.csv'],contextual_summary,reference_check))
        atomic_json(summary/'provenance.json',dict(version=manifest['version'],experiment_fingerprint=manifest['experiment_fingerprint'],
            package_fingerprint=manifest['package_fingerprint'],original_input_sha256=manifest['original_input_sha256'],
            complete_source_fits=21,complete_target_fits=420,reference_coverage=reference_check,
            excludes_person_level_predictions=True,includes_original_identifiers=False,
            all_210_head_invariance_checks_passed=True,metrics_recomputed_from_binary_float64_predictions=True))
        allowed=list(tables)+['REPORT.md','provenance.json']
        atomic_json(summary/'file_sha256.json',{name:sha256(summary/name) for name in allowed})
        allowed.append('file_sha256.json')
        final=inside(ROOT/'results_to_share.zip');tmp=inside(ROOT/'results_to_share.zip.tmp')
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in allowed: z.write(summary/name,'summary/'+name)
            for name in ('partition_counts.csv','availability_audit.csv','jobs.csv','validation.json'):
                z.write(ROOT/'setup'/name,'audit/'+name)
            for name in CODE_FILES: z.write(ROOT/name,'package/'+name)
            for p in sorted((ROOT/'reference').glob('*')):
                if p.is_file(): z.write(p,'package/reference/'+p.name)
            for name in ('PROTOCOL.md',):
                if (ROOT/name).is_file(): z.write(ROOT/name,'package/'+name)
        os.replace(tmp,final)
        print('Wrote summary/REPORT.md and results_to_share.zip. Keep generated archives local; redistribution requires approval.')
