"""Rebuild the exact failed native state and replay the real frozen9B generation."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.aligned_runtime import AlignedRuntime
from evaluation.independent_dev import atomic
from internal_signals.actual_token_alignment import field_masks
from internal_signals.actual_token_nll import ActualTokenNLLProvider
from scripts.run_utility_sampling import read,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();out=Path(a.run).absolute()
    if out.exists():raise ValueError('Fresh smoke only')
    cfg=read(ROOT/'configs/evaluation/recovery_v4.json');split=read(ROOT/cfg['split'])
    task='alf-f9496664f205c827';row=next(r for r in split['partitions']['probe_train'] if r['task_id']==task)
    rt=AlignedRuntime(ROOT,out/'smoke/worker0',cfg,time.time()+1800);env=rt.new_env(row);past=[]
    old=read(ROOT/'docs/recovery_v4/failing_generation_record.json')
    source=[c for c in rows(ROOT/'outputs/utility_expansion_v3/train_pool/worker0/cases.jsonl') if c['task_id']==task and c['source_policy']=='never']
    try:
        for c in source:
            if rt.step(env,c['executed_action'])!=c['actual']:raise ValueError('Failed-state real prefix differs')
            past.append(dict(observation=c['observation'],action=c['executed_action']))
        parsed,text,meta=rt.draft(env,past,row,seed=1,scored=True)
        if not parsed['valid'] or meta['generation_record']['response_ids']!=old['response_ids']:raise ValueError('Real failed generation not reproduced')
        if meta['internal_signals']['status']!='ok':raise ValueError('Live actual-ID score invalid')
        repaired=ActualTokenNLLProvider(rt.actor.tokenizer,rt.actor.config).extract(old).to_dict()
        score_delta={k:abs(repaired[k]-meta['internal_signals'][k]) for k in ['local_uncertainty','global_uncertainty','premise_uncertainty','consequence_uncertainty']}
        if max(score_delta.values())>1e-6:raise ValueError('Raw-generation NLL drift')
        masks=field_masks(rt.actor,meta);values,cost=rt.features(meta)
        p2,t2,m2=rt.draft(env,past,row,seed=1,scored=True)
        if m2['generation_record']['response_ids']!=old['response_ids'] or m2['generation_record']['token_nll']!=meta['generation_record']['token_nll']:raise ValueError('Forward altered generation')
        sim=rt.simulate(env,parsed['action'])
        if not sim['prediction_available']:raise ValueError('Real WM reconstruction failed')
        review,_=rt.review(env,past,row,parsed['action'],sim);actual=rt.step(env,review['final_action'])
        if review['final_action']==parsed['action'] and actual!=sim['candidate_transition']:raise ValueError('Native KEEP mismatch')
        proof=dict(status='pass',scope='Real failed-state engineering regression, not benchmark benefit',formal_state_step_before=31,
            original_actual_ids_reproduced=True,response_tokens=len(old['response_ids']),raw_nll_delta=score_delta,repeat_ids_and_nll_exact=True,
            masks=masks,shapes={k:list(v.shape) for k,v in values.items()},capture_cost=cost,internal_signals=meta['internal_signals'],
            actor_frozen=not any(p.requires_grad for p in rt.actor.model.parameters()),wm=sim,review=review,actual=actual,ledger=rt.ledger.state)
        atomic(out/'complete.json',proof);print(json.dumps({k:proof[k] for k in ['status','original_actual_ids_reproduced','response_tokens','raw_nll_delta','shapes','actor_frozen']},indent=2))
    finally:env.close()


if __name__=='__main__':main()
