"""Grouped supervised training on detached cached activations only."""
import copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from evaluation.independent_dev import atomic
from internal_signals.activation_probe import ActivationProbe,pad_batch,binary_metrics


def load_values(root,records,layer):
    values=[];contract=None
    for r in records:
        item=torch.load(Path(root)/r['feature_file'],map_location='cpu',weights_only=True)
        if item['case_id']!=r['case_id']:raise ValueError('Feature identity mismatch')
        if contract is None:contract=item['contract']
        if item['contract']!=contract:raise ValueError('Mixed activation contracts')
        values.append(item['features'][str(layer)].float())
    return values,contract


def eligible(records,target):
    return [r for r in records if r.get(target) is not None]


def predict(model,values):
    model.eval();scores=[]
    with torch.inference_mode():
        for i in range(0,len(values),8):
            h,m=pad_batch(values[i:i+8]);scores.extend(model(h,m).squeeze(-1).sigmoid().tolist())
    return scores


def train_head(root,out,records,target,layers,seeds=(0,1,2)):
    torch.set_num_threads(4)
    train=eligible([r for r in records if r['role']=='probe_train'],target)
    val=eligible([r for r in records if r['role']=='probe_validation'],target)
    cal=eligible([r for r in records if r['role']=='gate_calibration'],target)
    counts=lambda rs:dict(n=len(rs),positive=sum(int(r[target]) for r in rs),tasks=len({r['task_id'] for r in rs}))
    sufficient=lambda rs,p,n:sum(r[target] for r in rs)>=p and len(rs)-sum(r[target] for r in rs)>=n
    if len({r['task_id'] for r in train})<6 or not sufficient(train,5,5) or not sufficient(val,2,2) or len(cal)<8:
        result=dict(status='insufficient_labels',target=target,train=counts(train),validation=counts(val),calibration=counts(cal),
            reason='Need>=6 training tasks,5 positive/negative Train,2 positive/negative Validation,8 Calibration. No synthetic labels or fake head.')
        atomic(Path(out)/(target+'_training.json'),result);return result
    trials=[];started=time.perf_counter()
    for layer in layers:
        tv,contract=load_values(root,train,layer);vv,val_contract=load_values(root,val,layer)
        if val_contract!=contract:raise ValueError('Train/Validation activation contract mismatch')
        total=sum(len(x) for x in tv);mu=sum(x.sum(0) for x in tv)/total
        sd=(sum(((x-mu)**2).sum(0) for x in tv)/total).sqrt().clamp_min(1e-3)
        y=torch.tensor([r[target] for r in train],dtype=torch.float32)
        for kind in ['linear','mlp','ema']:
            seed_results=[]
            for seed in seeds:
                torch.manual_seed(seed);model=ActivationProbe(len(mu),kind)
                model.mean.copy_(mu);model.std.copy_(sd)
                opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
                weight=torch.tensor((len(y)-y.sum().item())/y.sum().item())
                losses=[]
                for epoch in range(3):
                    model.train();order=torch.randperm(len(train));loss_total=0.
                    for indices in order.split(8):
                        h,m=pad_batch([tv[i] for i in indices.tolist()]);logits=model(h,m).squeeze(-1)
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[indices],pos_weight=weight)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite probe training loss')
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();loss_total+=loss.item()*len(indices)
                    losses.append(loss_total/len(train))
                scores=predict(model,vv);metrics=binary_metrics([r[target] for r in val],scores)
                file=Path(out)/(f'{target}_layer{layer}_{kind}_seed{seed}.pt')
                torch.save(dict(schema='activation-probe-v1',state_dict=model.state_dict(),kind=kind,dim=len(mu),layer=layer,target=target,
                    contract=contract,seed=seed,train_task_ids=sorted({r['task_id'] for r in train}),validation_task_ids=sorted({r['task_id'] for r in val}),
                    trained_parameters=sum(p.numel() for p in model.parameters()),training_losses=losses),file)
                seed_results.append(dict(seed=seed,metrics=metrics,checkpoint=str(file.relative_to(root)),losses=losses))
            trials.append(dict(layer=layer,kind=kind,seeds=seed_results,mean_validation_ap=float(np.mean([s['metrics']['average_precision'] for s in seed_results]))))
    best=max(trials,key=lambda x:x['mean_validation_ap']);ensemble=Ensemble(root,[s['checkpoint'] for s in best['seeds']])
    cv,cal_contract=load_values(root,cal,best['layer'])
    if cal_contract!=ensemble.contract:raise ValueError('Calibration activation contract mismatch')
    cal_scores=[ensemble.score_tensor(v) for v in cv]
    val_values,_=load_values(root,val,best['layer']);val_scores=[ensemble.score_tensor(v) for v in val_values]
    result=dict(status='trained',target=target,train=counts(train),validation=counts(val),calibration=counts(cal),
        selected_layer=best['layer'],selected_architecture=best['kind'],checkpoints=[s['checkpoint'] for s in best['seeds']],
        selection='highest mean Validation AP across3 seeds; ensemble all3, not best-seed-only',
        validation_metrics=binary_metrics([r[target] for r in val],val_scores),threshold=float(np.quantile(cal_scores,.75)),
        threshold_conflict=float(np.quantile(cal_scores,.65)),calibration_quantile=.75,calibrated_probability=False,trials=trials,
        seconds=time.perf_counter()-started,no_actor_parameter_updates=True)
    atomic(Path(out)/(target+'_training.json'),result);return result


class Ensemble:
    def __init__(self,root,files):
        self.models=[];self.contract=None
        for f in files:
            data=torch.load(Path(root)/f,map_location='cpu',weights_only=True)
            if self.contract is None:self.contract=data['contract'];self.layer=data['layer'];self.target=data['target']
            if data['contract']!=self.contract or data['layer']!=self.layer:raise ValueError('Incompatible ensemble')
            model=ActivationProbe(data['dim'],data['kind']);model.load_state_dict(data['state_dict'],strict=True);model.eval();self.models.append(model)
    @torch.inference_mode()
    def score_tensor(self,value):
        h,m=pad_batch([value]);return sum(x(h,m).squeeze().sigmoid().item() for x in self.models)/len(self.models)
    def score(self,values,contract):
        if contract!=self.contract:raise ValueError('Probe/Actor/capture contract mismatch')
        return self.score_tensor(values[str(self.layer)])
