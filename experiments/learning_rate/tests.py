#!/usr/bin/env python3
"""CPU regression checks; these are diagnostics, not research fits."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import ast
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import exp2_utils as U
U.configure_local_runtime()
import numpy as np
import pandas as pd
import torch
import engine
import analyse
import experiment

torch.set_num_threads(2)


def toy_records():
    records=[]
    rng=np.random.default_rng(2026)
    for i in range(8):
        target=np.full(20,np.nan,dtype=np.float32)
        target[:2]=rng.normal(size=2)
        records.append({"values":rng.normal(size=3).astype(np.float32),
            "times":np.array([-2,-1,-1],dtype=np.float32),"varis":np.array([0,0,1]),
            "demo":rng.normal(size=7).astype(np.float32),"target":target,
            "target_mask":np.isfinite(target).astype(np.float32)})
    stats={c:{"mean":0.,"std":1.,"n":8 if i<2 else 0} for i,c in enumerate(engine.TARGET_COLS)}
    return records,stats


class ProtocolTests(unittest.TestCase):
    def test_matrix_and_groups(self):
        jobs=U.jobs()
        self.assertEqual(len(jobs),42)
        fits={(j['target'],j['percent'],j['seed'],fold,key) for j in jobs for fold in range(5) for key in U.LRS}
        self.assertEqual(len(fits),420)
        for n in (1,2,3,4,6,7,42):
            ids=[j['job_id'] for g in range(n) for j in U.group_jobs(g,n)]
            self.assertEqual(sorted(ids),list(range(42)))
        for g in range(6):
            jobs=U.group_jobs(g)
            self.assertEqual(len(jobs),7)
            self.assertEqual({j['target'] for j in jobs},set(U.TARGET_LABELS))
        with self.assertRaises(ValueError): U.group_jobs(6,6)

    def test_original_model_and_preprocessing_ast(self):
        manifest=U.read_json(U.ROOT/'reference/provenance.json')
        tree=ast.parse((U.ROOT/'engine.py').read_text())
        nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for name,expected in manifest['unchanged_ast_sha256'].items():
            actual=hashlib.sha256(ast.dump(nodes[name]).encode()).hexdigest()
            self.assertEqual(actual,expected,name)
        self.assertEqual(U.sha256(U.ROOT/'reference/original_train_model.py.txt'),manifest['original_train_model_sha256'])
        self.assertEqual(engine.NUM_WORKERS,0)
        self.assertEqual((engine.HIDDEN_DIM,engine.NUM_LAYERS,engine.NUM_HEADS,engine.MAX_OBS),(256,2,4,256))
        self.assertEqual((engine.PATIENCE,engine.MIN_DELTA,engine.WEIGHT_DECAY),(10,1e-5,1e-5))

    def test_original_training_numerical_parity(self):
        records,stats=toy_records()
        namespace=dict(vars(engine))
        exec((U.ROOT/'reference/original_train_model.py.txt').read_text(),namespace)
        original=namespace['train_model']
        epoch_calls,batch_calls=[],[]
        kwargs=dict(train_records=records,val_records=records,target_stats=stats,
                    batch_size=4,lr=1e-4,max_epochs=2,patience=10,device=torch.device('cpu'),
                    seed_offset=33,name='test')
        with patch.object(engine,'SEED',1234), patch.object(engine,'EPOCH_PRINT_EVERY',0):
            engine.set_seed(987)
            old_model=engine.new_model(torch.device('cpu'))
            old_state=original(model=old_model,**kwargs)
            engine.set_seed(987)
            new_model=engine.new_model(torch.device('cpu'))
            with patch.object(engine,'TRAIN_TRACE_CALLBACK',lambda *v:epoch_calls.append(v),create=True), \
                 patch.object(engine,'BATCH_TRACE_CALLBACK',lambda *v:batch_calls.append(v),create=True):
                new_state=engine.train_model(model=new_model,**kwargs)
            self.assertEqual(set(old_state),set(new_state))
            for key in old_state:
                self.assertTrue(torch.equal(old_state[key],new_state[key]),key)
            self.assertEqual(len(epoch_calls),2)
            self.assertTrue(batch_calls)

    def validation_frame(self,first=1.,second=0.8):
        return pd.DataFrame([{'target':'ABCD','percent':10,'fold':0,'seed':s,'lr_key':k,
                              'validation_mse':first if k=='lr1e-4' else second}
                             for k in U.LRS for s in U.SEEDS])

    def test_validation_selection_and_tie(self):
        self.assertEqual(analyse.select_rates(self.validation_frame()).selected_lr_key.iloc[0],'lr5e-4')
        self.assertEqual(analyse.select_rates(self.validation_frame(1.,1.)).selected_lr_key.iloc[0],'lr1e-4')
        v=self.validation_frame()
        # Different seed winners; select the mean, not the majority vote or a best seed.
        v.loc[v.lr_key.eq('lr1e-4'),'validation_mse']=[0.,0.,9.]
        v.loc[v.lr_key.eq('lr5e-4'),'validation_mse']=[1.,1.,1.]
        self.assertEqual(analyse.select_rates(v).selected_lr_key.iloc[0],'lr5e-4')

    def test_selection_rejects_test_scores_and_incomplete_seeds(self):
        v=self.validation_frame()
        with self.assertRaises(ValueError): analyse.select_rates(v.assign(test_nrmse=0.1))
        with self.assertRaises(ValueError): analyse.select_rates(v.iloc[:-1])
        v.loc[0,'validation_mse']=float('nan')
        with self.assertRaises(ValueError): analyse.select_rates(v)

    def test_person_balance_and_fold_pooling(self):
        with tempfile.TemporaryDirectory(dir=U.ROOT/'_runtime/tmp') as d:
            root=Path(d); (root/'data').mkdir()
            U.atomic_json(root/'data/primary_target_panel.json',['Height'])
            U.atomic_json(root/'data/target_attribute_stds_by_dataset_fold.json',{'ABCD':{'0':{'Height':2.},'1':{'Height':2.}}})
            U.atomic_json(root/'data/eligible_dataset_attribute_pairs.json',{'ABCD':{'0':{'Height':{'eligible':True}},'1':{'Height':{'eligible':True}}}})
            def rows(pid,errors,fold):
                return [{'dataset':'ABCD','fold':fold,'person_id':pid,'attribute':'Height','error':e,
                         'input_age':10.+i,'output_age':20.,'horizon':10.-i,'n_input_visits':i+1} for i,e in enumerate(errors)]
            a=pd.DataFrame(rows('a',[2.],0))
            b=pd.DataFrame(rows('b',[6.,6.,6.],1)+rows('c',[10.],1))
            f0=analyse.fold_metrics(a,root,'ABCD',0,strict=False)
            f1=analyse.fold_metrics(b,root,'ABCD',1,strict=False)
            score,attrs=analyse.combine_folds([f0,f1])
            expected=math.sqrt((1.+9.+25.)/3)
            self.assertAlmostEqual(score['nrmse'],expected)
            self.assertAlmostEqual(score['nmae'],3.)
            self.assertNotAlmostEqual(score['nrmse'],(1.+math.sqrt(17.))/2)
            self.assertNotAlmostEqual(score['nrmse'],math.sqrt((1.+9.*3+25.)/5))
            self.assertEqual(attrs[0]['n_test_people'],3)
            self.assertEqual(analyse.identity_digest(b),analyse.identity_digest(b.iloc[::-1]))
            b.loc[0,'error']=np.inf
            with self.assertRaises(ValueError): analyse.fold_metrics(b,root,'ABCD',1,strict=False)

    def test_output_containment(self):
        with self.assertRaises(ValueError): U.inside(U.ROOT.parent/'escape.txt')
        with tempfile.TemporaryDirectory(dir=U.ROOT/'_runtime/tmp') as d:
            link=Path(d)/'escape'
            try: link.symlink_to(U.ROOT.parent,target_is_directory=True)
            except (OSError,NotImplementedError): return
            with self.assertRaises(ValueError): U.inside(link/'outside.txt')

    def test_lock_and_completion_integrity(self):
        with tempfile.TemporaryDirectory(dir=U.ROOT/'_runtime/tmp') as d:
            d=Path(d)
            with U.file_lock(d/'.lock'):
                with self.assertRaises(RuntimeError):
                    with U.file_lock(d/'.lock'): pass
            for n in ('errors.csv.gz','metrics.json','history.csv'):
                U.atomic_text(d/n,'test')
            done={'config_hash':'x','file_sha256':{n:U.sha256(d/n) for n in ('errors.csv.gz','metrics.json','history.csv')}}
            U.atomic_json(d/'done.json',done)
            self.assertIsNotNone(U.verified_done(d,'x'))
            with self.assertRaises(RuntimeError): U.verified_done(d,'changed')
            U.atomic_text(d/'metrics.json','corrupt')
            with self.assertRaises(RuntimeError): U.verified_done(d,'x')

    def test_pair_audit(self):
        a={k:1 for k in ('stage_seed','initial_state_sha256','loader_seeds','target_statistics','train','validation','test',
                         'n_train_records','n_val_records','n_test_records','test_identity_digest')}
        experiment.pair_audit(a,dict(a))
        b=dict(a); b['initial_state_sha256']=2
        with self.assertRaises(ValueError): experiment.pair_audit(a,b)

    def test_history_uses_min_delta_checkpoint_not_absolute_minimum(self):
        with tempfile.TemporaryDirectory(dir=U.ROOT/'_runtime/tmp') as d:
            d=Path(d)
            h=pd.DataFrame({'epoch':list(range(1,12)), 'train_mse':[1.]*11,
                            'validation_mse':[1.]+[0.999995]*10,'elapsed_seconds':list(range(11))})
            U.atomic_csv(h,d/'history.csv')
            done={'training':{'epochs_run':11,'best_epoch':1,'selected_validation_mse':1.,'hit_epoch_cap':False}}
            analyse.verify_history(d,done)
            done['training']['selected_validation_mse']=0.999995
            with self.assertRaises(ValueError): analyse.verify_history(d,done)

    def test_training_loader_replay(self):
        records,_=toy_records()
        def batches():
            loader=engine.make_loader(records,3,True,123,torch.device('cpu'))
            return [batch[0].numpy().copy() for batch in loader]
        a,b=batches(),batches()
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y)


if __name__=='__main__':
    unittest.main(verbosity=2)
