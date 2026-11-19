#!/usr/bin/env python3
"""Unit tests plus an original-versus-vectorized real-data sequence check."""
from __future__ import annotations
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from runtime import *
configure()
import copy, tempfile, types, unittest
import numpy as np
import pandas as pd
import torch
from data import Batcher
from model import GRUD,target_balanced_masked_mse
from training import map_head,new_model,verify_receipt
from metrics import person_sums,combine_folds,identity_digest,assert_close


def original_functions():
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "grud.py"
    spec = importlib.util.spec_from_file_location("original_grud", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return vars(module)


def norm_for(store):
    means=pd.DataFrame(store.raw).mean().fillna(0.0).to_numpy()
    stds=pd.DataFrame(store.raw).std(ddof=1).to_numpy()
    stds=np.where(np.isfinite(stds)&(stds>0),stds,1.0)
    age_mean=float(np.mean(store.age));age_std=float(np.std(store.age)) or 1.0
    x_mean=np.r_[store.sex.mean(axis=0),0.0,pd.DataFrame((store.raw-means)/stds).mean().fillna(0.0).to_numpy()]
    return dict(attributes=list(ATTRS),mean=means.tolist(),std=stds.tolist(),
                age_mean=age_mean,age_std=age_std,X_mean=x_mean.tolist(),
                n=np.isfinite(store.raw).sum(axis=0).tolist())


def synthetic_store():
    raw=np.full((7,20),np.nan,dtype=np.float64)
    raw[:,:3]=np.array([[100,20,np.nan],[np.nan,22,50],[108,np.nan,54],[111,25,55],
                      [90,np.nan,45],[95,18,np.nan],[98,19,48]],dtype=float)
    raw[:,3]=[14,15,16,17,13,14,15]
    people=np.array([0,0,0,0,1,1,1],dtype=np.int32)
    s=types.SimpleNamespace(raw=raw,age=np.array([5,6.5,10,12,4,8,11.0]),
        sex=np.array([[1,0]]*4+[[0,1]]*3,dtype=float),row_person=people,
        people=['p0','p1'],inputs=np.array([[0,-1,-1],[0,1,-1],[0,1,2],[4,-1,-1],[4,5,-1]]),
        lengths=np.array([1,2,3,1,2]),outputs=np.array([1,2,3,5,6]),pair_person=np.array([0,0,0,1,1]))
    s.raw_last=pd.DataFrame(raw).groupby(people,sort=False).ffill().to_numpy()
    return s


def compare_original(store,indices,input_norm,output_norm=None):
    """Exact input/mask/elapsed-time parity with the public original helper."""
    output_norm=input_norm if output_norm is None else output_norm
    indices=np.asarray(indices)
    rows=np.flatnonzero(np.isin(store.row_person,np.unique(store.pair_person[indices])))
    z=(store.raw[rows]-np.asarray(input_norm['mean']))/np.asarray(input_norm['std'])
    frame=pd.DataFrame(z,columns=ATTRS,index=rows)
    frame['PersonID']=[store.people[int(i)] for i in store.row_person[rows]]
    frame['Age']=store.age[rows];frame['Sex_Male']=store.sex[rows,0];frame['Sex_Female']=store.sex[rows,1]
    frame['Age_z']=(store.age[rows]-input_norm['age_mean'])/input_norm['age_std']
    pairs=pd.DataFrame({'PersonID':[store.people[int(i)] for i in store.pair_person[indices]],
        'input_rows':[tuple(map(int,store.inputs[i,:store.lengths[i]])) for i in indices],
        'output_row':store.outputs[indices]})
    old=original_functions()
    seq,y,ym,meta=old['create_sequences'](frame,pairs,['Sex_Male','Sex_Female','Age_z'],list(ATTRS))
    ds=old['SequenceDataset'](seq,y,ym)
    ref=old['grud_collate_fn']([ds[i] for i in range(len(ds))])
    batch=Batcher(store,input_norm,output_norm)(indices.tolist())
    for k,(a,b) in enumerate(zip(ref[:5],batch[:5])):
        if not torch.equal(a,b):
            # Pandas/NumPy reductions may differ below float32 precision; report a tight tolerance only if needed.
            torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7,msg=f'Original sequence mismatch, component {k}')
    if output_norm==input_norm:
        torch.testing.assert_close(ref[5],batch[5],rtol=0,atol=0)
    torch.testing.assert_close(ref[6],batch[6],rtol=0,atol=0)
    return len(indices)


def verify_real_data(store):
    indices=[]
    for t in ORIGINS:
        ids=np.flatnonzero(store.pair_origin==ORIGINS.index(t))
        # First, middle, last and a longest-prefix pair from every cohort.
        indices.extend([int(ids[0]),int(ids[len(ids)//2]),int(ids[-1]),int(ids[np.argmax(store.lengths[ids])])])
    indices=np.unique(indices)
    source=store.source_people('ABCD')[0]
    norm=store.normalizer(source)
    count=compare_original(store,indices,norm)
    return dict(passed=True,n_pairs_compared=count,n_cohorts=7,
                compared=['observed inputs','forward-filled last observations','missingness masks',
                          'variable-specific raw-age elapsed times','future masking','padding masks','standardized targets'])


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_sequence_parity(self):
        s=synthetic_store(); self.assertEqual(compare_original(s,np.arange(5),norm_for(s)),5)

    def test_original_model_and_loss_parity(self):
        s=synthetic_store();n=norm_for(s);b=Batcher(s,n,n)([0,1,2,3,4]);old=original_functions()
        set_seeds(1234);a=GRUD(23,64,n['X_mean'],20)
        set_seeds(1234);c=old['GRUD'](23,64,n['X_mean'],20)
        self.assertEqual(state_digest(a.state_dict()),state_digest(c.state_dict()))
        torch.testing.assert_close(a(*b[:5]),c(*b[:5]),rtol=0,atol=0)
        w=torch.arange(1,21).float();p=a(*b[:5])
        torch.testing.assert_close(target_balanced_masked_mse(p,b[5],b[6],w),old['target_balanced_masked_mse'](p,b[5],b[6],w),rtol=0,atol=0)

    def test_future_measurements_hidden(self):
        s=synthetic_store();n=norm_for(s);before=Batcher(s,n,n)([2])
        after_store=copy.deepcopy(s);after_store.raw[3]=np.arange(20)*1000+7000
        after_store.raw_last=pd.DataFrame(after_store.raw).groupby(after_store.row_person).ffill().to_numpy()
        after=Batcher(after_store,n,n)([2])
        for a,b in zip(before[:5],after[:5]): torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertFalse(torch.equal(before[5],after[5]))
        self.assertTrue(torch.all(before[2][0,-1,3:]==0))

    def test_missing_slot_and_padding(self):
        s=synthetic_store();n=norm_for(s);b=Batcher(s,n,n)([0,2]);m=new_model(n,torch.device('cpu'))
        self.assertEqual(m.input_size,23);self.assertEqual(m.output_layer.out_features,20)
        self.assertTrue(torch.isfinite(m(*b[:5])).all())
        solo=Batcher(s,n,n)([0]);torch.testing.assert_close(m(*solo[:5])[0],m(*b[:5])[0],rtol=1e-5,atol=1e-6)
        self.assertEqual(n['mean'][19],0);self.assertEqual(n['std'][19],1)

    def test_elapsed_time_recurrence(self):
        s=synthetic_store();b=Batcher(s,norm_for(s),norm_for(s))([2])
        # Height observed at age 5, missing at 6.5, observed again at 10, forecast age 12.
        np.testing.assert_array_equal(b[3][0,:,3].numpy(),[0,1.5,5,2])
        self.assertTrue(torch.all(b[3][0,:,:3]==0))

    def test_head_invariance_and_mean_buffer(self):
        s=synthetic_store();source=norm_for(s);target=copy.deepcopy(source)
        target['mean']=(np.asarray(source['mean'])+np.linspace(-7,11,20)).tolist()
        target['std']=(np.asarray(source['std'])*np.linspace(.2,4,20)).tolist()
        batch=Batcher(s,source,target)([0,1,2,3,4]);m=new_model(source,torch.device('cpu'))
        before=m.X_mean.clone();result=map_head(m,source,target,batch,torch.device('cpu'))
        self.assertTrue(result['passed']);self.assertTrue(torch.equal(before,m.X_mean))

    def test_loss_masks_ignore_unobserved_labels(self):
        pred=torch.zeros((2,20),requires_grad=True);truth=torch.ones((2,20));mask=torch.zeros((2,20));mask[:,0]=1
        truth[:,1:]=1e15;loss=target_balanced_masked_mse(pred,truth,mask,torch.ones(20));loss.backward()
        self.assertAlmostEqual(float(loss.detach()),1.0);self.assertTrue(torch.all(pred.grad[:,1:]==0))

    def test_person_balanced_metric_not_pair_balanced(self):
        result=person_sums(np.array([0,0,0,1]),sum_person_normalized_mse=np.array([1,1,1,9]),
                           sum_person_normalized_mae=np.array([1,1,1,3]))
        self.assertEqual(result['sum_person_normalized_mse'],10)
        self.assertAlmostEqual(result['sum_person_normalized_mse']/result['n_people'],5)

    def test_combine_before_square_root(self):
        def metric(k,mse,n):
            v=dict(sum_person_normalized_mse=mse*n,sum_person_normalized_mae=1.0*n,n_people=n,n_errors=n)
            return dict(fold=k,attributes={'Height':v},raw_attributes={},n_primary_errors=n,n_test_people=n)
        score,rows,_=combine_folds([metric(0,1,1),metric(1,9,3)],require_five=False)
        self.assertAlmostEqual(score['nrmse'],np.sqrt(7))
        self.assertNotAlmostEqual(score['nrmse'],2.0)

    def test_output_paths_cannot_escape(self):
        with self.assertRaises(ValueError): inside(ROOT.parent/'escape.txt')
        with self.assertRaises(ValueError): inside('../escape.txt')
        with tempfile.TemporaryDirectory(dir=ROOT/'_runtime/tmp') as d:
            link=Path(d)/'outside'
            try: link.symlink_to(ROOT.parent,target_is_directory=True)
            except (OSError,NotImplementedError): return
            with self.assertRaises(ValueError): inside(link/'escape.txt')

    def test_hash_receipt_rejects_corruption(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'_runtime/tmp') as d:
            folder=Path(d);atomic_text(folder/'artifact.txt','original')
            config={'test':True};atomic_json(folder/'done.json',{'config_hash':digest(config),'files':{'artifact.txt':sha256(folder/'artifact.txt')}})
            self.assertIsNotNone(verify_receipt(folder,config))
            atomic_text(folder/'artifact.txt','changed')
            with self.assertRaises(RuntimeError): verify_receipt(folder,config)

    def test_binary_float_round_trip(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'_runtime/tmp') as d:
            values=np.array([np.nextafter(0.1,1.0),1/3,1e-12,123.4567890123456],dtype=np.float64)
            atomic_npz(Path(d)/'p.npz',prediction=values)
            with np.load(Path(d)/'p.npz',allow_pickle=False) as x: np.testing.assert_array_equal(x['prediction'],values)

    def test_all_groups_cover_jobs_once(self):
        for n in range(1,22):
            ids=[j['job_id'] for k in range(n) for j in jobs() if j['job_id']%n==k]
            self.assertEqual(sorted(ids),list(range(21)));self.assertEqual(len(set(ids)),21)

    def test_stage_seeds_do_not_depend_on_arm_or_fraction(self):
        for j in jobs():
            a=stage_seed(j['seed'],j['target'],0);self.assertEqual(a,stage_seed(j['seed'],j['target'],0))
            self.assertNotEqual(a,stage_seed(j['seed'],j['target'],1))
            self.assertNotEqual(a,stage_seed(j['seed'],j['target']))

    def test_source_exclusions_and_target_exclusion(self):
        for target in ORIGINS:
            selected=set(sources(target));self.assertNotIn(target,selected)
            self.assertFalse(selected&set(EXCLUSIONS.get(target,[])))
            self.assertEqual(selected,set(ORIGINS)-{target}-set(EXCLUSIONS.get(target,[])))

    def test_identity_digest_includes_mask(self):
        ids=np.array([1,2]);a=np.ones((2,20),dtype=bool);b=a.copy();b[0,0]=False
        self.assertNotEqual(identity_digest(ids,a),identity_digest(ids,b))

    def test_saved_metrics_close_guard(self):
        assert_close({'x':1.0},{'x':1.0+1e-14})
        with self.assertRaises(ValueError): assert_close({'x':1.0},{'x':1.1})


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--real-data',action='store_true');args=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests))
    if not result.wasSuccessful(): sys.exit(1)
    if args.real_data:
        from experiment import load_manifest
        from data import Store
        load_manifest();print(verify_real_data(Store()))

if __name__=='__main__': main()
