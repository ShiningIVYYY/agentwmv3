"""Real9B post-token alignment/repeat/causality and native Call/Skip checks."""
import argparse,copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic
from evaluation.activation_runtime import Runtime
from internal_signals.activation_capture import feature_record


def main():
    import torch
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run).absolute()
    if run.exists():raise ValueError('Fresh smoke run required')
    cfg=json.loads((ROOT/'configs/evaluation/activation_v1.json').read_text());split=json.loads((ROOT/cfg['split']).read_text())
    rt=Runtime(ROOT,run/'collect/worker0',cfg,time.time()+1800);row=split['partitions']['environment_validation'][0];env=rt.new_env(row)
    try:
        parsed,text,meta=rt.draft(env,[],row,scored=True)
        if not parsed['valid']:raise ValueError('Real Actor failed new protocol')
        features,cost=rt.features(meta);again,_=rt.features(meta)
        repeat=max((features[k].float()-again[k].float()).abs().max().item() for k in features)
        modified=feature_record(meta);suffix=rt.actor.tokenizer.encode('\nExtra',add_special_tokens=False)
        modified['response_ids']=modified['response_ids']+suffix
        modified['response']=rt.actor.tokenizer.decode(modified['response_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        future,fcost=rt.capture.extract(modified)
        from evaluation.independent_dev import append
        append(rt.out/'compute.jsonl',dict(kind='causality_test_forward',cost=fcost))
        relative={k:((features[k].float()-future[k][:len(features[k])].float()).square().mean().sqrt()/(features[k].float().square().mean().sqrt()+1e-8)).item() for k in features}
        if repeat!=0 or max(relative.values())>.03:raise ValueError('Activation repeat/causal check failed: '+str((repeat,relative)))
        p2,t2,m2=rt.draft(env,[],row,scored=True)
        if meta['generation_record']['response_ids']!=m2['generation_record']['response_ids'] or meta['generation_record']['token_nll']!=m2['generation_record']['token_nll']:
            raise ValueError('Feature forward altered generation/NLL path')
        sim=rt.simulate(env,parsed['action'])
        if not sim['prediction_available']:raise ValueError('Real WM unavailable')
        review,rm=rt.review(env,[],row,parsed['action'],sim)
        actual=rt.step(env,review['final_action'])
        if review['final_action']==parsed['action'] and actual!=sim['candidate_transition']:raise ValueError('Real KEEP mismatch')
        result=dict(status='pass',scope='real engineering smoke, not benchmark effectiveness',contract=rt.contract,
            shapes={k:list(v.shape) for k,v in features.items()},repeat_max_abs=repeat,prefix_future_relative_rms=relative,
            actor_raw=text,actor_repeat_token_ids_equal=True,nll_repeat_exact=True,wm=sim,review=review,actual=actual,
            actor_trainable_parameters=sum(p.numel() for p in rt.actor.model.parameters() if p.requires_grad),ledger=rt.ledger.state)
        atomic(run/'complete.json',result);print(json.dumps({k:result[k] for k in ['status','shapes','repeat_max_abs','prefix_future_relative_rms','actor_raw','actor_trainable_parameters']},indent=2))
    finally:env.close()


if __name__=='__main__':main()
