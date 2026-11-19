"""Frozen primary panel; person -> attribute -> dataset balancing, without fold-RMSE averaging."""
from __future__ import annotations
import hashlib, math
import numpy as np
from runtime import ROOT, ATTRS, read_json


def identity_digest(indices,observed) -> str:
    h=hashlib.sha256(); h.update(np.asarray(indices,dtype='<i8').tobytes())
    h.update(np.asarray(observed,dtype=np.uint8).tobytes()); return h.hexdigest()


def person_sums(person_ids, **vectors):
    people,inverse=np.unique(person_ids,return_inverse=True)
    counts=np.bincount(inverse).astype(np.float64)
    result={name:float((np.bincount(inverse,weights=np.asarray(value,dtype=np.float64))/counts).sum())
            for name,value in vectors.items()}
    result.update(n_people=len(people),n_errors=len(person_ids))
    return result


def fold_metrics(store,target,fold,indices,prediction,truth,observed,strict=True):
    indices=np.asarray(indices,dtype=np.int64)
    expected_truth=np.asarray(store.raw[store.outputs[indices]],dtype=np.float64)
    if not np.array_equal(truth,expected_truth,equal_nan=True) or not np.array_equal(observed,np.isfinite(truth)):
        raise ValueError('Prediction truth/mask differs from the frozen raw data.')
    if prediction.shape!=truth.shape or not np.isfinite(prediction).all(): raise ValueError('Invalid predictions.')
    if len(np.unique(indices))!=len(indices): raise ValueError('Duplicate evaluated pairs.')
    primary=[a for a in store.primary if store.eligibility[target][str(fold)][a]['eligible']]
    person=store.pair_person[indices]
    error=prediction-truth
    if not np.isfinite(error[observed]).all(): raise FloatingPointError('Nonfinite raw errors.')
    attributes={}; raw_attributes={}; has_primary=np.zeros(len(indices),dtype=bool)
    for j,a in enumerate(ATTRS):
        use=observed[:,j]
        if not use.any(): continue
        e=error[use,j]; p=prediction[use,j]; y=truth[use,j]
        raw_attributes[a]=person_sums(person[use],sum_person_raw_mse=e*e,sum_person_raw_mae=np.abs(e),
            sum_person_raw_bias=e,sum_person_prediction=p,sum_person_truth=y,
            sum_person_prediction_squared=p*p,sum_person_prediction_truth=p*y)
        if a in primary:
            std=store.eval_stds[target][str(fold)][a]
            if std is None or not np.isfinite(std) or std<=0: raise ValueError('Invalid evaluation standard deviation.')
            z=e/float(std)
            attributes[a]=person_sums(person[use],sum_person_normalized_mse=z*z,sum_person_normalized_mae=np.abs(z))
            has_primary|=use
    if strict:
        expected=read_json(ROOT/'setup/expected_test_counts.json')[target][str(fold)]
        actual={a:v['n_errors'] for a,v in attributes.items()}
        if actual!=expected: raise ValueError(f'Incomplete primary coverage: {target}/{fold}: {actual} != {expected}')
    if not attributes: raise ValueError('No evaluable primary outcomes in this test slice.')
    return dict(target=target,fold=fold,panel='primary_9',horizon_group='all',
                identity_digest=identity_digest(indices,observed),n_pairs=len(indices),
                n_raw_errors=int(observed.sum()),n_primary_errors=sum(v['n_errors'] for v in attributes.values()),
                n_test_people=int(np.unique(person[has_primary]).size),attributes=attributes,
                raw_attributes=raw_attributes,raw_truth_policy='original float64 CSV values')


def combine_folds(folds, require_five=True):
    if require_five and (len(folds)!=5 or sorted(x['fold'] for x in folds)!=list(range(5))):
        raise ValueError('Need each of the five distinct held-out folds exactly once.')
    rows=[]
    for a in sorted({a for f in folds for a in f['attributes']}):
        stats=[f['attributes'][a] for f in folds if a in f['attributes']]
        people=sum(v['n_people'] for v in stats)
        rows.append(dict(attribute=a,nrmse=math.sqrt(sum(v['sum_person_normalized_mse'] for v in stats)/people),
                         nmae=sum(v['sum_person_normalized_mae'] for v in stats)/people,
                         n_test_people=people,n_errors=sum(v['n_errors'] for v in stats),n_folds_evaluated=len(stats)))
    if not rows: raise ValueError('No fold scores.')
    score=dict(nrmse=float(np.mean([r['nrmse'] for r in rows])),nmae=float(np.mean([r['nmae'] for r in rows])),
               n_attributes=len(rows),n_primary_errors=sum(f['n_primary_errors'] for f in folds),
               n_test_people=sum(f['n_test_people'] for f in folds),n_folds=len(folds))
    raw=[]
    for a in sorted({a for f in folds for a in f['raw_attributes']}):
        stats=[f['raw_attributes'][a] for f in folds if a in f['raw_attributes']]
        n=sum(s['n_people'] for s in stats)
        mean=lambda name: sum(s[name] for s in stats)/n
        mp=mean('sum_person_prediction'); my=mean('sum_person_truth')
        var=mean('sum_person_prediction_squared')-mp*mp
        cov=mean('sum_person_prediction_truth')-mp*my
        slope=cov/var if var>1e-12*max(1.0,abs(mean('sum_person_prediction_squared'))) else None
        raw.append(dict(attribute=a,person_balanced_rmse=math.sqrt(mean('sum_person_raw_mse')),
            person_balanced_mae=mean('sum_person_raw_mae'),person_balanced_bias=mean('sum_person_raw_bias'),
            calibration_slope=slope,calibration_intercept=my-slope*mp if slope is not None else None,
            n_test_people=n,n_errors=sum(s['n_errors'] for s in stats),n_folds_evaluated=len(stats),
            in_primary_panel=any(a in f['attributes'] for f in folds)))
    return score,rows,raw


def assert_close(actual,expected,path='metrics'):
    if isinstance(expected,dict):
        if set(actual)!=set(expected): raise ValueError(f'Different keys in {path}')
        for k in expected: assert_close(actual[k],expected[k],path+'.'+k)
    elif isinstance(expected,(float,np.floating)):
        if not np.isclose(actual,expected,rtol=1e-12,atol=1e-12): raise ValueError(f'Metric recomputation mismatch at {path}: {actual} != {expected}')
    elif actual!=expected: raise ValueError(f'Metric recomputation mismatch at {path}')
