"""Versioned linear-only utility training; original exploratory trainer preserved."""
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


def _fit_head(root,out,records,target,layers,seeds=(0,1,2),profile=None):
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
        for kind in ['linear']:
            seed_results=[]
            for seed in seeds:
                torch.manual_seed(seed);model=ActivationProbe(len(mu),kind)
                model.mean.copy_(mu);model.std.copy_(sd)
                opt=torch.optim.AdamW(model.parameters(),lr=profile['learning_rate'],weight_decay=profile['weight_decay'])
                weight=torch.tensor((len(y)-y.sum().item())/y.sum().item())
                losses=[]
                for epoch in range(profile['epochs']):
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


def readiness(records, target, profile):
    """Data targets are predeclared feasibility safeguards, not power guarantees."""
    groups={};counts={};seen=set();reasons=[]
    for role in ['probe_train','probe_validation','gate_calibration']:
        rs=eligible([r for r in records if r['role']==role],target)
        for r in rs:
            if r['case_id'] in seen:raise ValueError('Duplicate labelled state')
            seen.add(r['case_id'])
            if r[target] not in (0,1,False,True):raise ValueError('Nonbinary known label')
        groups[role]={r['goal_group'] for r in rs}
        counts[role]=dict(states=len(rs),positive=sum(int(r[target]) for r in rs),
            negative=sum(not bool(r[target]) for r in rs),tasks=len(groups[role]),
            positive_tasks=len({r['goal_group'] for r in rs if r[target]}))
        for name,minimum in profile['minimum_counts'][role].items():
            if counts[role][name]<minimum:reasons.append(f'{role}.{name}<{minimum}')
    roles=list(groups)
    for i,a in enumerate(roles):
        for b in roles[i+1:]:
            if groups[a]&groups[b]:raise ValueError('Task-group role leakage')
    return dict(ready=not reasons,counts=counts,unmet=reasons,
                scientific_effectiveness_validated=False)


def train_head(root,out,records,target,layers,profile):
    import hashlib
    if target!='utility_help':raise ValueError('First approved study trains Help only')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    gate=readiness(records,target,profile)
    if not gate['ready']:
        result=dict(status='insufficient_labels',target=target,readiness=gate,
            no_actor_parameter_updates=True,deployment_allowed=False,
            reason='Do not replace the approved data target with the old5-positive engineering floor')
        atomic(out/(target+'_training.json'),result);return result
    result=_fit_head(root,out,records,target,layers,seeds=tuple(profile['seeds']),profile=profile)
    if result['status']!='trained':return result
    goals=sorted({r['goal_group'] for r in records if r['role']=='probe_train'},
        key=lambda g:hashlib.sha256(('linear-utility-v1/'+g).encode()).hexdigest())
    curves=[]
    for fraction in profile['learning_curve_fractions']:
        n=max(1,math.ceil(len(goals)*fraction));chosen=set(goals[:n])
        subset=[r for r in records if r['role']!='probe_train' or r['goal_group'] in chosen]
        dest=out/('curve_'+str(fraction));dest.mkdir()
        trial=_fit_head(root,dest,subset,target,layers,seeds=tuple(profile['seeds']),profile=profile)
        curves.append(dict(fraction=fraction,training_goal_count=n,result=trial))
    result.update(readiness=gate,learning_curves=curves,profile=profile,
        deployment_allowed=False,scientific_effectiveness_validated=False,
        architecture='mean-pooled regularized linear, three-seed ensemble',
        threshold_scope='Separate sampled calibration goals; not natural-traffic probability or online validation')
    atomic(out/(target+'_training.json'),result)
    return result
