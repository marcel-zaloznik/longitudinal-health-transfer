"""Compact visit cache and vectorized batches; no duplicated full-sequence storage."""
from __future__ import annotations
import ast, hashlib, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from runtime import (ROOT, ATTRS, ORIGINS, PERCENTS, INPUT_NAMES, inside, read_json,
                     atomic_json, atomic_csv, digest, sha256, sources)
CACHE_NAMES=('raw','age','sex','row_person','row_origin','row_fold','raw_last',
             'inputs','lengths','outputs','pair_person','person_origin','person_fold')

def save_array(path: Path, array) -> None:
    path=inside(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=inside(path.with_name(path.name+f'.{os.getpid()}.tmp'))
    with tmp.open('wb') as f: np.save(f,array,allow_pickle=False)
    os.replace(tmp,path)

def build_cache(project: Path) -> dict:
    folder=inside(ROOT/'setup/cache'); folder.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(project/'data/dataset.csv',dtype={'PersonID':str},low_memory=False)
    if set(frame.Origin.unique())!=set(ORIGINS): raise ValueError('Unexpected dataset origins.')
    for col,mapping in [('Smoking',{'No':0,'Quit/strike':1,'Yes':2}),('Diabetes',{'No':0,'Yes':1})]:
        if not frame[col].dropna().isin(mapping).all(): raise ValueError(f'Unexpected {col} values.')
        frame[col]=frame[col].map(mapping).astype(float)
    raw=frame[list(ATTRS)].to_numpy(dtype=np.float64)
    age=frame.Age.to_numpy(dtype=np.float64)
    if np.isinf(raw).any() or not np.isfinite(age).all(): raise ValueError('Infinite values or nonfinite ages.')
    if frame.PersonID.isna().any() or not frame.Sex.isin(['Male','Female']).all():
        raise ValueError('Missing person IDs or unexpected sex coding.')
    if frame.groupby('PersonID').Origin.nunique().max()!=1: raise ValueError('An ID crosses cohorts.')
    if frame.groupby('PersonID').Sex.nunique().max()!=1: raise ValueError('Inconsistent sex within person.')
    people=sorted(frame.PersonID.unique().tolist()); lookup={p:i for i,p in enumerate(people)}
    row_person=frame.PersonID.map(lookup).to_numpy(dtype=np.int32)
    row_origin=frame.Origin.map({x:i for i,x in enumerate(ORIGINS)}).to_numpy(dtype=np.int8)
    split=read_json(project/'data/split5k.json')
    if len(split)!=5: raise ValueError('Exactly five participant folds are required.')
    flat=[str(x) for group in split for x in group]
    if len(set(flat))!=len(flat) or set(flat)!=set(people): raise ValueError('Folds overlap or fail to partition all people.')
    fold_of={str(p):i for i,g in enumerate(split) for p in g}
    row_fold=frame.PersonID.map(fold_of).to_numpy(dtype=np.int8)
    first=frame.drop_duplicates('PersonID').set_index('PersonID').loc[people]
    person_origin=first.Origin.map({x:i for i,x in enumerate(ORIGINS)}).to_numpy(dtype=np.int8)
    person_fold=np.array([fold_of[x] for x in people],dtype=np.int8)
    sex=np.column_stack([frame.Sex.eq('Male'),frame.Sex.eq('Female')]).astype(np.float64)
    ordered=frame.sort_values(['PersonID','Age'],kind='mergesort')
    raw_last=ordered.groupby('PersonID',sort=False)[list(ATTRS)].ffill().reindex(frame.index).to_numpy(dtype=np.float64)
    pairs=pd.read_csv(project/'data/input_output_pairs.csv',dtype={'PersonID':str})
    outputs=pairs.output_row.to_numpy(dtype=np.int64)
    if not np.array_equal(outputs,pd.to_numeric(pairs.output_row).to_numpy()) or (outputs<0).any() or (outputs>=len(frame)).any():
        raise ValueError('Invalid output row references.')
    parsed=[ast.literal_eval(x) for x in pairs.input_rows]
    if any(not isinstance(x,(tuple,list)) or not x or any(not isinstance(v,int) for v in x) for x in parsed):
        raise ValueError('Input prefixes must be nonempty integer sequences.')
    lengths=np.array([len(x) for x in parsed],dtype=np.int32)
    inputs=np.full((len(pairs),int(lengths.max())),-1,dtype=np.int32)
    for i,rows in enumerate(parsed): inputs[i,:len(rows)]=rows
    del parsed
    valid=inputs>=0
    if (inputs[valid]>=len(frame)).any(): raise ValueError('Invalid input row reference.')
    if any(not isinstance(x,str) or x not in lookup for x in pairs.PersonID): raise ValueError('Unknown pair participant.')
    pair_person=pairs.PersonID.map(lookup).to_numpy(dtype=np.int32)
    if not np.array_equal(row_person[outputs],pair_person): raise ValueError('Output belongs to another participant.')
    safe=inputs.clip(min=0)
    if ((row_person[safe]!=pair_person[:,None])&valid).any(): raise ValueError('Input belongs to another participant.')
    if ((age[safe]>=age[outputs,None])&valid).any(): raise ValueError('Prediction input includes future or target visit.')
    if ((np.diff(age[safe],axis=1)<=0)&valid[:,1:]).any(): raise ValueError('Prefix ages are not strictly increasing.')
    if pairs.duplicated(['PersonID','input_rows','output_row']).any(): raise ValueError('Duplicate prediction pairs.')
    arrays=dict(raw=raw,age=age,sex=sex,row_person=row_person,row_origin=row_origin,row_fold=row_fold,
                raw_last=raw_last,inputs=inputs,lengths=lengths,outputs=outputs,pair_person=pair_person,
                person_origin=person_origin,person_fold=person_fold)
    for key,value in arrays.items(): save_array(folder/(key+'.npy'),value)
    atomic_json(folder/'people.json',people)
    subsets=read_json(project/'data/target_train_subsets_by_dataset_fold_percent.json')
    coded={}
    for target in ORIGINS:
        coded[target]={}
        domain=person_origin==ORIGINS.index(target)
        for k in range(5):
            available=set(np.flatnonzero(domain&(person_fold!=k)&(person_fold!=(k-1)%5)).tolist())
            coded[target][str(k)]={}
            for pct in PERCENTS:
                ids=list(map(str,subsets[target][str(k)][str(pct)]))
                if len(ids)!=len(set(ids)) or any(p not in lookup for p in ids): raise ValueError('Invalid saved training subset.')
                selected=sorted(lookup[p] for p in ids)
                if not selected or not set(selected)<=available: raise ValueError('Training subset leaks validation/test participants.')
                if pct==100 and set(selected)!=available: raise ValueError('Saved 100% subset differs from all training-fold participants.')
                coded[target][str(k)][str(pct)]=selected
            if not set(coded[target][str(k)]['10'])<=set(coded[target][str(k)]['100']): raise ValueError('Non-nested subsets.')
    atomic_json(folder/'subsets.json',coded)
    for name in INPUT_NAMES[-3:]: atomic_json(ROOT/'setup'/name,read_json(project/'data'/name))
    return dict(n_people=len(people),n_visits=len(frame),n_pairs=len(pairs),
                longest_input_prefix=int(lengths.max()),origins=list(ORIGINS))

class Store:
    def __init__(self):
        folder=ROOT/'setup/cache'
        for name in CACHE_NAMES: setattr(self,name,np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False))
        self.people=read_json(folder/'people.json')
        self.subsets=read_json(folder/'subsets.json')
        self.primary=read_json(ROOT/'setup/primary_target_panel.json')
        self.eligibility=read_json(ROOT/'setup/eligible_dataset_attribute_pairs.json')
        self.eval_stds=read_json(ROOT/'setup/target_attribute_stds_by_dataset_fold.json')
        self.pair_origin=self.person_origin[self.pair_person]
        self.pair_fold=self.person_fold[self.pair_person]

    def pair_indices(self, people) -> np.ndarray:
        return np.flatnonzero(np.isin(self.pair_person,np.asarray(people,dtype=np.int32)))

    def row_indices(self, people) -> np.ndarray:
        return np.flatnonzero(np.isin(self.row_person,np.asarray(people,dtype=np.int32)))

    def target_people(self,target: str,fold: int,percent: int,smoke=False):
        train=np.array(self.subsets[target][str(fold)][str(percent)],dtype=np.int32)
        domain=self.person_origin==ORIGINS.index(target)
        val=np.flatnonzero(domain&(self.person_fold==(fold-1)%5))
        test=np.flatnonzero(domain&(self.person_fold==fold))
        if smoke:
            # Keep more than one outcome/sequence when possible, without changing the full-run partitions.
            train,val,test=(x[:6] for x in (train,val,test))
        if any(len(x)==0 for x in (train,val,test)): raise ValueError('Empty target partition.')
        return train,val,test

    def source_people(self,target: str,smoke=False):
        domain=np.isin(self.person_origin,[ORIGINS.index(x) for x in sources(target)])
        train=np.flatnonzero(domain&(self.person_fold!=0)); val=np.flatnonzero(domain&(self.person_fold==0))
        if smoke:
            def small(ids):
                return np.concatenate([ids[self.person_origin[ids]==ORIGINS.index(t)][:3] for t in sources(target)])
            train,val=small(train),small(val)
        if len(train)==0 or len(val)==0: raise ValueError('Empty source split; no fallback/resampling is allowed.')
        return train,val

    def counts(self,people):
        ids=np.asarray(people); rows=self.row_indices(ids); pairs=self.pair_indices(ids)
        return dict(n_people=len(ids),n_visits=len(rows),n_pairs=len(pairs),
                    people_digest=digest(sorted(self.people[int(i)] for i in ids)))

    def normalizer(self,people) -> dict:
        rows=self.row_indices(people)
        values=self.raw[rows]
        counts=np.isfinite(values).sum(axis=0)
        # Match original GRU-D: pandas sample std (ddof=1), all training visits, not pair-weighted.
        means=pd.DataFrame(values).mean().to_numpy(dtype=np.float64)
        stds=pd.DataFrame(values).std(ddof=1).to_numpy(dtype=np.float64)
        means=np.where(np.isfinite(means),means,0.0)
        stds=np.where(np.isfinite(stds)&(stds>0),stds,1.0)
        age_mean=float(np.mean(self.age[rows])); age_std=float(np.std(self.age[rows],ddof=0))
        if not np.isfinite(age_std) or age_std==0: age_std=1.0
        z=(values-means)/stds
        dynamic_mean=pd.DataFrame(z).mean().fillna(0.0).to_numpy(dtype=np.float64)
        xmean=np.r_[self.sex[rows].mean(axis=0),((self.age[rows]-age_mean)/age_std).mean(),dynamic_mean]
        return dict(attributes=list(ATTRS),mean=means.tolist(),std=stds.tolist(),n=counts.tolist(),
                    age_mean=age_mean,age_std=age_std,X_mean=xmean.tolist(),
                    fitted_on=self.counts(people),dynamic_ddof=1,age_ddof=0,absent_fallback='mean=0,std=1,X_mean=0')

    def audit(self) -> tuple[list,dict,list]:
        partitions=[]; expected={}; normalization=[]
        for target in ORIGINS:
            expected[target]={}
            tr,va=self.source_people(target)
            for role,people in [('source_train',tr),('source_validation',va)]:
                partitions.append(dict(target=target,fold=-1,percent=-1,role=role,sources='|'.join(sources(target)),**self.counts(people)))
            sn=self.normalizer(tr)
            for a,n in zip(ATTRS,sn['n']): normalization.append(dict(target=target,fold=-1,percent=-1,kind='source',attribute=a,n_training_visits_observed=n))
            for fold in range(5):
                test=self.target_people(target,fold,100)[2]; indices=self.pair_indices(test)
                observed=np.isfinite(self.raw[self.outputs[indices]])
                primary=[a for a in self.primary if self.eligibility[target][str(fold)][a]['eligible']]
                counts={a:int(observed[:,ATTRS.index(a)].sum()) for a in primary}
                if not counts or any(n==0 for n in counts.values()): raise ValueError('Empty eligible test cell.')
                expected[target][str(fold)]=counts
                for pct in PERCENTS:
                    tr,va,te=self.target_people(target,fold,pct)
                    for role,people in [('target_train',tr),('target_validation',va),('target_test',te)]:
                        partitions.append(dict(target=target,fold=fold,percent=pct,role=role,sources='',**self.counts(people)))
                    train_out=self.raw[self.outputs[self.pair_indices(tr)]]
                    n=np.isfinite(train_out).sum(axis=0)
                    if any(n[ATTRS.index(a)]==0 for a in primary): raise ValueError('An eligible primary outcome has no training labels.')
                    for a,c in zip(ATTRS,n): normalization.append(dict(target=target,fold=fold,percent=pct,kind='target',attribute=a,n_training_pairs_observed=int(c)))
        return partitions,expected,normalization

class PairDataset(Dataset):
    def __init__(self,indices): self.indices=np.asarray(indices,dtype=np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self,index): return int(self.indices[index])

class Batcher:
    """Source-normalized inputs and independently target-normalized labels."""
    def __init__(self,store: Store,input_norm: dict,output_norm: dict):
        self.store=store
        mean=np.asarray(input_norm['mean']); std=np.asarray(input_norm['std'])
        self.x=np.column_stack([store.sex,(store.age-input_norm['age_mean'])/input_norm['age_std'],
                              np.nan_to_num((store.raw-mean)/std,nan=0.0)])
        self.mask=np.column_stack([np.ones((len(store.age),3)),np.isfinite(store.raw)]).astype(np.float64)
        self.last=np.column_stack([self.x[:,:3],np.nan_to_num((store.raw_last-mean)/std,nan=0.0)])
        self.y=np.nan_to_num((store.raw-np.asarray(output_norm['mean']))/np.asarray(output_norm['std']),nan=0.0)
        self.y_mask=np.isfinite(store.raw)

    def __call__(self,batch):
        ids=np.asarray(batch,dtype=np.int64); s=self.store
        lengths=s.lengths[ids].astype(np.int64)+1; n=len(ids); steps=int(lengths.max())
        pos=np.zeros((n,steps),dtype=np.int64)
        prefix=s.inputs[ids,:steps-1]
        pos[:,:steps-1]=np.maximum(prefix,0)
        end=lengths-1; rows=np.arange(n)
        pos[rows,end]=s.outputs[ids]
        seq_mask=np.arange(steps)[None,:]<lengths[:,None]
        x=self.x[pos].copy(); last=self.last[pos].copy(); mask=self.mask[pos].copy()
        # The future visit supplies sex and target age ONLY, never future dynamic values.
        x[rows,end,3:]=0.0; mask[rows,end,3:]=0.0
        last[rows,end,3:]=self.last[s.inputs[ids,end-1],3:]
        times=s.age[pos]; delta=np.zeros_like(x)
        for t in range(1,steps):
            elapsed=(times[:,t]-times[:,t-1])[:,None]
            delta[:,t,3:]=elapsed+(1.0-mask[:,t-1,3:])*delta[:,t-1,3:]
        for array in (x,last,mask,delta): array[~seq_mask]=0.0
        y=self.y[s.outputs[ids]]; ym=self.y_mask[s.outputs[ids]]
        return tuple(torch.from_numpy(np.asarray(v,dtype=np.float32)) for v in (x,last,mask,delta,seq_mask,y,ym))

    def loader(self,indices,shuffle: bool,seed: int,device,batch_size=512):
        return DataLoader(PairDataset(indices),batch_size=batch_size,shuffle=shuffle,num_workers=0,
            pin_memory=device.type=='cuda',collate_fn=self,generator=torch.Generator().manual_seed(seed))
