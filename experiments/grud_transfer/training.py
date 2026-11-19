"""Original GRU-D objective/stopping policy, stage-isolated seeds and local receipts."""
from __future__ import annotations
import copy, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import GRUD, target_balanced_masked_mse
from data import Batcher
from runtime import (ROOT, ATTRS, HYPER, inside, atomic_json, atomic_csv, atomic_npz,
                     atomic_torch, read_json, sha256, digest, set_seeds, stage_seed,
                     loader_seed, state_digest, environment_signature)

def new_model(norm,device):
    return GRUD(23,HYPER['hidden_size'],norm['X_mean'],20).to(device)

def move(batch,device): return tuple(t.to(device,non_blocking=True) for t in batch)

def map_head(model,source_norm,target_norm,batch,device) -> dict:
    """Preserve raw predictions, including the source X_mean buffer and hidden weights."""
    x=move(batch,device)[:5]
    sm=np.asarray(source_norm['mean']); ss=np.asarray(source_norm['std'])
    tm=np.asarray(target_norm['mean']); ts=np.asarray(target_norm['std'])
    scale=ss/ts; shift=(sm-tm)/ts
    previous=model.training; model.eval()
    hidden_before=state_digest({k:v for k,v in model.state_dict().items() if not k.startswith('output_layer.')})
    with torch.inference_mode():
        before=model(*x).double()*torch.as_tensor(ss,device=device)+torch.as_tensor(sm,device=device)
        a=torch.as_tensor(scale,device=device,dtype=torch.float64)
        b=torch.as_tensor(shift,device=device,dtype=torch.float64)
        model.output_layer.weight.copy_((model.output_layer.weight.double()*a[:,None]).float())
        model.output_layer.bias.copy_((model.output_layer.bias.double()*a+b).float())
        after=model(*x).double()*torch.as_tensor(ts,device=device)+torch.as_tensor(tm,device=device)
        delta=(before-after).abs()
        tolerance=2e-5*(before.abs()+torch.as_tensor(ss+ts+1.0,device=device))
        if not torch.isfinite(before).all() or not torch.isfinite(after).all() or (delta>tolerance).any():
            raise FloatingPointError('Raw prediction invariance failed before fine-tuning.')
    if hidden_before!=state_digest({k:v for k,v in model.state_dict().items() if not k.startswith('output_layer.')}):
        raise RuntimeError('Head conversion changed a hidden parameter or X_mean buffer.')
    model.train(previous)
    return dict(passed=True,n_training_records_checked=len(batch[0]),
        max_absolute_raw_difference=float(delta.max().cpu()),
        absolute_difference_by_attribute=dict(zip(ATTRS,delta.max(dim=0).values.cpu().tolist())),
        scale=scale.tolist(),shift=shift.tolist(),unchanged_non_head_state_sha256=hidden_before,
        source_unobserved_target_observed=[a for a,s,t in zip(ATTRS,source_norm['n'],target_norm['n']) if s==0 and t>0],
        tolerance='abs(diff) <= 2e-5*(abs(raw_before)+source_std+target_std+1)')

def verify_receipt(folder: Path,config: dict | None=None) -> dict | None:
    path=folder/'done.json'
    if not path.is_file(): return None
    receipt=read_json(path)
    if config is not None and receipt['config_hash']!=digest(config):
        raise RuntimeError(f'Completed stage has a different protocol/environment: {folder}')
    for name,expected in receipt['files'].items():
        p=inside(folder/name)
        if not p.is_file() or sha256(p)!=expected:
            raise RuntimeError(f'Completed stage is corrupt or edited: {p}')
    return receipt

def train(model,batcher,train_indices,val_indices,device,seed,folder: Path,max_epochs: int) -> dict:
    counts=np.isfinite(batcher.store.raw[batcher.store.outputs[train_indices]]).sum(axis=0)
    weights=np.divide(1.0,counts,out=np.zeros(20,dtype=np.float64),where=counts>0)
    target_weights=torch.tensor(weights,dtype=torch.float32,device=device)
    if counts.sum()==0: raise ValueError('No observed training targets.')
    val_counts=np.isfinite(batcher.store.raw[batcher.store.outputs[val_indices]]).sum(axis=0)
    if not (val_counts[counts>0].sum()>0): raise ValueError('No supported validation targets.')
    train_loader=batcher.loader(train_indices,True,seed,device,HYPER['batch_size'])
    val_loader=batcher.loader(val_indices,False,seed+1,device,HYPER['batch_size'])
    optimizer=torch.optim.Adam(model.parameters(),lr=HYPER['learning_rate'])
    best=float('inf'); best_epoch=0; wait=0; history=[]; start=time.monotonic()
    best_state=None
    for epoch in range(1,max_epochs+1):
        model.train(); train_sse=0.0; train_count=0.0
        for batch in train_loader:
            x,last,mask,delta,seq,y,ym=move(batch,device)
            optimizer.zero_grad(set_to_none=True)
            pred=model(x,last,mask,delta,seq)
            loss=target_balanced_masked_mse(pred,y,ym,target_weights)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite train loss, epoch {epoch}.')
            loss.backward(); optimizer.step()
            train_sse+=float((((pred.detach()-y)*ym)**2).sum().cpu()); train_count+=float(ym.sum().cpu())
        model.eval(); val_total=0.0; val_batches=0
        with torch.inference_mode():
            for batch in val_loader:
                x,last,mask,delta,seq,y,ym=move(batch,device)
                pred=model(x,last,mask,delta,seq)
                loss=target_balanced_masked_mse(pred,y,ym,target_weights)
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite validation loss, epoch {epoch}.')
                val_total+=float(loss.cpu()); val_batches+=1
        val_loss=val_total/max(val_batches,1)
        improved=val_loss+HYPER['min_delta']<best
        if improved:
            best=val_loss; best_epoch=epoch; wait=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: wait+=1
        history.append(dict(epoch=epoch,train_mse=train_sse/max(train_count,1),validation_mse=val_loss,
                            improved=improved,elapsed_seconds=time.monotonic()-start))
        atomic_csv(folder/'history.csv',pd.DataFrame(history))
        atomic_json(folder/'progress.json',dict(epoch=epoch,best_epoch=best_epoch,best_validation_mse=best,
                    no_improvement_epochs=wait,elapsed_seconds=time.monotonic()-start))
        if epoch==1 or epoch%10==0 or wait>=HYPER['patience']:
            print(f'    epoch {epoch}: validation={val_loss:.7g}; best={best:.7g} at {best_epoch}',flush=True)
        if wait>=HYPER['patience']: break
    if best_state is None: raise RuntimeError('No finite model checkpoint.')
    model.load_state_dict(best_state)
    return dict(best_epoch=best_epoch,epochs_run=epoch,selected_validation_mse=best,
                epoch_cap_hit=epoch==max_epochs,early_stopped=wait>=HYPER['patience'],
                elapsed_seconds=time.monotonic()-start,training_target_counts=counts.tolist(),
                train_pairs=len(train_indices),validation_pairs=len(val_indices),
                optimizer_reset=True,validation_aggregation='arithmetic mean of original weighted batch losses')

def predict(model,batcher,indices,output_norm,device,folder):
    model.eval(); predictions=[]
    with torch.inference_mode():
        for batch in batcher.loader(indices,False,0,device,HYPER['batch_size']):
            pred=model(*move(batch,device)[:5])
            if not torch.isfinite(pred).all(): raise FloatingPointError('Nonfinite test predictions.')
            predictions.append(pred.cpu().numpy())
    standardized=np.concatenate(predictions).astype(np.float64)
    prediction=standardized*np.asarray(output_norm['std'])+np.asarray(output_norm['mean'])
    s=batcher.store; rows=s.outputs[indices]; lens=s.lengths[indices]
    last=s.inputs[indices,lens-1]
    # Retain original raw truth; never reconstruct it through a float32 standardization round trip.
    truth=np.array(s.raw[rows],dtype=np.float64); observed=np.isfinite(truth)
    atomic_npz(folder/'predictions.npz',pair_ids=np.asarray(indices,dtype=np.int64),
        prediction=prediction,truth=truth,observed=observed,
        person_id=np.asarray([s.people[int(i)] for i in s.pair_person[indices]]),
        input_row=last.astype(np.int64),output_row=np.asarray(rows,dtype=np.int64),
        input_age=np.asarray(s.age[last]),output_age=np.asarray(s.age[rows]),
        horizon=np.asarray(s.age[rows]-s.age[last]),n_input_visits=np.asarray(lens))
    return prediction,truth,observed

def stage(folder: Path,store,job: dict,arm: str,fold: int | None,percent: int | None,
          device,env: dict,manifest: dict,source_folder: Path | None=None,smoke=False) -> dict:
    folder=inside(folder); folder.mkdir(parents=True,exist_ok=True)
    is_source=arm=='pretrain'; target=job['target']; seed=job['seed']
    config=dict(experiment_fingerprint=manifest['experiment_fingerprint'],job=job,arm=arm,fold=fold,
                percent=percent,smoke=smoke,hyperparameters=HYPER,
                environment=environment_signature(env))
    source_receipt=None
    if arm=='transfer':
        source_receipt=verify_receipt(source_folder)
        if source_receipt is None: raise RuntimeError('Transfer requires a completed source checkpoint.')
        config['source_checkpoint_sha256']=source_receipt['files']['best_weights.pt']
        config['source_receipt_sha256']=sha256(source_folder/'done.json')
    old=verify_receipt(folder,config)
    if old is not None:
        print(f'  Verified and skipped: {folder.relative_to(ROOT)}',flush=True); return old
    if is_source:
        tr_people,va_people=store.source_people(target,smoke)
        te_people=None; input_norm=output_norm=store.normalizer(tr_people)
    else:
        tr_people,va_people,te_people=store.target_people(target,fold,percent,smoke)
        output_norm=store.normalizer(tr_people)
        input_norm=read_json(source_folder/'normalization.json')['output'] if arm=='transfer' else output_norm
    train_indices=store.pair_indices(tr_people); val_indices=store.pair_indices(va_people)
    if smoke:
        train_indices=train_indices[:256]; val_indices=val_indices[:128]
    if not len(train_indices) or not len(val_indices): raise ValueError('Empty stage data.')
    batcher=Batcher(store,input_norm,output_norm)
    stage_rng=stage_seed(seed,target,fold); order_rng=loader_seed(seed,target,fold)
    set_seeds(stage_rng); model=new_model(input_norm,device)
    initial_random_parameter_sha256=state_digest(dict(model.named_parameters()))
    mapping=None
    if arm=='transfer':
        state=torch.load(source_folder/'best_weights.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state)
        expected=torch.tensor(input_norm['X_mean'],device=device,dtype=torch.float32)
        if not torch.equal(model.X_mean,expected): raise RuntimeError('Source X_mean buffer changed during loading.')
        probe=train_indices[np.unique(np.linspace(0,len(train_indices)-1,min(32,len(train_indices)),dtype=int))]
        mapping=map_head(model,input_norm,output_norm,batcher(probe.tolist()),device)
        atomic_json(folder/'head_mapping.json',mapping)
    atomic_json(folder/'config.json',config)
    atomic_json(folder/'environment.json',env)
    atomic_json(folder/'normalization.json',dict(input=input_norm,output=output_norm,
                  mean_buffer_policy='retained source X_mean' if arm=='transfer' else 'training-visit mean'))
    print(f'  {target}, seed {seed}, {arm}, fold {fold}, {percent}%: {len(train_indices)} train pairs',flush=True)
    details=train(model,batcher,train_indices,val_indices,device,order_rng,folder,2 if smoke else HYPER['max_epochs'])
    # These small 64-hidden-unit models are retained for every stage for later diagnostics.
    atomic_torch(folder/'best_weights.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
    if te_people is not None:
        from metrics import fold_metrics
        indices=store.pair_indices(te_people)
        if smoke: indices=indices[:128]
        pred,truth,observed=predict(model,batcher,indices,output_norm,device,folder)
        metrics=fold_metrics(store,target,fold,indices,pred,truth,observed,strict=not smoke)
        atomic_json(folder/'metrics.json',metrics)
    files=['config.json','environment.json','normalization.json','history.csv','best_weights.pt']
    if te_people is not None: files+=['metrics.json','predictions.npz']
    if mapping is not None: files+=['head_mapping.json']
    receipt=dict(config_hash=digest(config),config=config,stage_seed=stage_rng,loader_seed=order_rng,
        initial_random_parameter_sha256=initial_random_parameter_sha256,
        initial_transfer_checkpoint_sha256=config.get('source_checkpoint_sha256'),
        restored_state_sha256=state_digest(model.state_dict()),details=details,
        partitions=dict(train=store.counts(tr_people),validation=store.counts(va_people)),
        files={name:sha256(folder/name) for name in files})
    if te_people is not None: receipt['partitions']['test']=store.counts(te_people)
    atomic_json(folder/'done.json',receipt)
    del model,batcher
    if device.type=='cuda': torch.cuda.empty_cache()
    return receipt
