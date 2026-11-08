#!/usr/bin/env python3
"""One persistent9B; bounded offline regression, strict one-step branches,15 dev episodes."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,append,read_rows,Ledger,run_episode,request_seed,verify_model_assets
from experience.splits import file_hash,validate_manifest,goal_key
from experience.memory import ExperienceMemory
from internal_signals.qwen35_actor import Qwen35Actor
from reflection.replay_review_v2 import ReviewerProtocol,VERSION
from reflection.replay_review import review_messages
from scripts.prepare_reviewer_repair import source_case
from scripts.audit_independent_dev import recorded_seed
from worldmodel.ground_truth_environment import ReplayScienceEnvironment,state_fingerprint


def freeze_hashes(profile):
    paths=set(profile['method_hashes'])|set(profile['source_hashes'])|{'evaluation/independent_dev.py',
        'reflection/replay_review_v2.py','scripts/run_reviewer_repair.py','scripts/prepare_reviewer_repair.py',
        'configs/evaluation/phase510_reviewer_repair_v1.json','configs/reviewers/compact_json_conservative_v2.json'}
    return {name:file_hash(ROOT/name) for name in sorted(paths)}


def verify(manifest):
    for name,sha in manifest['file_hashes'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Frozen source/config/data changed '+name)


def branch(ref,action,ledger,out,branch_name,profile):
    t,past,prefix=source_case(ref);row=next(r for r in profile['_dev_rows'] if r['task_id']==t['task_id'])
    spec=dict(taskName=row['taskName'],variationIdx=row['variationIdx'],simplifications=profile['simplifications'],
        seed=profile['simulator_seed'],environment_step_limit=profile['environment_step_limit'],simulator_profile=profile['simulator_profile'])
    env=None;begin=time.perf_counter();ledger.begin_transition(len(prefix)+2,dict(kind='offline_controlled_branch',case=ref['prompt_id'],branch=branch_name))
    actual_queries=0
    try:
        env=ReplayScienceEnvironment(spec);actual_queries+=1
        if goal_key(env.goal)!=row['goal_group']:raise ValueError('Goal drift')
        for a in prefix:env.step(a);actual_queries+=1
        before=state_fingerprint(env.capture());expected=t['simulation']['checked_state_sha256']
        if before!=expected:raise ValueError('Historical same-state fingerprint mismatch; no utility value')
        actual=env.step(action);actual_queries+=1;after=state_fingerprint(env.capture())
        if branch_name=='draft':
            target=t['simulation']['candidate_transition']
            assert all(actual[k]==target[k] for k in ('observation','reward','done','score','moves'))
            assert after==t['simulation']['candidate_state_sha256']
        if branch_name=='original':
            assert all(actual[k]==t['actual'][k] for k in ('observation','reward','done','score','moves'))
            assert after==t['formal_after_state_sha256']
        result=dict(branch=branch_name,action=action,before_sha256=before,after_sha256=after,actual=actual,
            pid=env.pid,transition_queries=actual_queries,capture_bundles=env.capture_count)
        ledger.finish_transition(actual_queries);ledger.captures(env.capture_count)
    except BaseException as error:
        if ledger.state['pending_transition'] is not None:
            ledger.finish_transition(actual_queries)
            if env is not None:ledger.captures(env.capture_count)
        append(out/'branch_errors.jsonl',dict(case=ref,branch=branch_name,error=str(error),known_transition_queries=actual_queries))
        raise
    finally:
        if env is not None:env.close()
    result['total_seconds_including_cleanup']=time.perf_counter()-begin;append(out/'controlled_branches.jsonl',dict(case=ref,**result));return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--model-dir',required=True);ap.add_argument('--offline-only',action='store_true');a=ap.parse_args()
    run=Path(a.run).resolve()
    if run.exists():raise ValueError('Fresh run directory required; no silent retries of partial runs')
    run.mkdir(parents=True);profile=json.loads((ROOT/'configs/evaluation/phase510_reviewer_repair_v1.json').read_text())
    for name,sha in profile['method_hashes'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Frozen Actor/Gate/Memory/WM changed '+name)
    started=time.perf_counter();manifest=dict(source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        profile=profile,file_hashes=freeze_hashes(profile),started_at_ns=time.time_ns(),deadline_unix=time.time()+profile['max_wall_seconds'],
        pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),offline_only=a.offline_only)
    atomic(run/'manifest.json',manifest)
    ledger=Ledger(run/'ledger.json',run/'STOP',manifest['deadline_unix'],profile['limits'],profile['batch_limits'])
    def stop(signum,frame):raise RuntimeError('Owned run stop signal')
    signal.signal(signal.SIGTERM,stop)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    try:
        atomic(run/'assets_verified.json',verify_model_assets(ROOT,a.model_dir))
        before=time.perf_counter();actor=Qwen35Actor(ROOT/profile['actor_config'],a.model_dir,'cuda:0')
        atomic(run/'model.json',dict(load_seconds=time.perf_counter()-before,loading_info=actor.loading_info,
            revision=actor.config['model_revision'],config_sha256=actor.fingerprint,shared_actor_reviewer_object=True,
            gpu=__import__('torch').cuda.get_device_name(0),backend=actor.config['backend'],dtype=actor.config['dtype']))
        split=json.loads((ROOT/'docs/phase3/split_manifest.json').read_text());validate_manifest(split,require_native=True)
        profile['_dev_rows']=split['partitions']['development']
        memory=ExperienceMemory(json.loads((ROOT/profile['memory_snapshot']).read_text()),split)
        gate=json.loads((ROOT/profile['gate_config']).read_text());assert actor.fingerprint==gate['internal_signature']['config_sha256']
        original=ReviewerProtocol('original');fixed=ReviewerProtocol(VERSION)
        reviewer_cfg=json.loads((ROOT/profile['reviewer_config']).read_text())
        assert reviewer_cfg['version']==fixed.version and reviewer_cfg['max_new_tokens']==fixed.max_new_tokens
        cases=profile['old_debug_panel']+profile['format_regression_panel']+profile['quality_panel'];generated={}
        def generate(protocol,prompt,seed,identity):
            ledger.begin_model(min(actor.count(prompt),actor.config['max_prompt_tokens'])+protocol.max_new_tokens,identity)
            raw,meta=protocol.generate(actor,prompt,seed);append(run/'generation_calls.jsonl',dict(identity=identity,metadata=meta));ledger.finish_model(meta);return raw,meta
        # Paired512/768 diagnostics on two old length failures, not an online variant.
        limits=[r for r in profile['format_regression_panel'] if source_case(r)[0]['review_metadata']['finish_reason']=='length'][:2]
        for ref in limits:
            t,past,_=source_case(ref);prompt=review_messages(t['goal'],t['observation'],past,t['draft_action'],t['simulation']);seed=recorded_seed(t['review_metadata'])
            raw,meta=generate(original,prompt,seed,dict(kind='original_512_parity',case=ref['prompt_id']))
            assert meta['generation_record']['response_ids']==t['review_metadata']['generation_record']['response_ids'],'Original token parity changed'
            ext=ReviewerProtocol('original');ext.max_new_tokens=768
            text,more=generate(ext,prompt,seed,dict(kind='bounded_768_length_diagnostic',case=ref['prompt_id']))
            assert more['generation_record']['response_ids'][:512]==meta['generation_record']['response_ids']
            append(run/'length_diagnostics.jsonl',dict(ref=ref,original_metadata=meta,extended_metadata=more,
                original_parse=original.parse(raw,t['draft_action']),extended_parse=original.parse(text,t['draft_action']),extended_raw=text))
        ledger.begin_batch()
        for i,ref in enumerate(cases):
            if i and i%20==0:ledger.begin_batch()
            verify(manifest);atomic(run/'progress.json',dict(stage='offline_reviews',completed=i,total=len(cases)))
            t,past,_=source_case(ref);prompt=fixed.messages(t['goal'],t['observation'],past,t['draft_action'],t['simulation'])
            raw,meta=generate(fixed,prompt,recorded_seed(t['review_metadata']),dict(kind='fixed_offline_review',case=ref['prompt_id'],panel=ref['panel']))
            result=dict(ref=ref,raw=raw,parsed=fixed.parse(raw,t['draft_action']),metadata=meta,generated_at_ns=time.time_ns())
            append(run/'offline_reviews.jsonl',result);generated[ref['prompt_id']]=result
        olddebug=[x for x in read_rows(run/'offline_reviews.jsonl') if x['ref']['panel']=='old_debug']
        if any(x['parsed']['decision']=='INVALID' for x in olddebug):raise ValueError('Fixed Reviewer has old-debug format failures; preserve outputs, no holdout execution')
        atomic(run/'fixed_reviewer_lock.json',dict(status='format_regression_engineering_pass',reviewer=VERSION,
            sha256=file_hash(ROOT/'reflection/replay_review_v2.py'),gate_memory_thresholds_changed=False,
            not_a_task_benefit_acceptance=True,offline_requests=len(cases),locked_at_ns=time.time_ns()))
        # Feedback used here is POSTHOC evaluation only; all Reviewer outputs already sealed above.
        for i,ref in enumerate(profile['old_debug_panel']+profile['quality_panel']):
            if i%5==0:ledger.begin_batch()
            atomic(run/'progress.json',dict(stage='controlled_one_step',completed=i,total=32))
            t,_,_=source_case(ref);res=generated[ref['prompt_id']]
            branches=[branch(ref,t['draft_action'],ledger,run,'draft',profile),branch(ref,t['final_action'],ledger,run,'original',profile),branch(ref,res['parsed']['final_action'],ledger,run,'fixed',profile)]
            assert len({b['before_sha256'] for b in branches})==1
            append(run/'one_step_comparisons.jsonl',dict(ref=ref,branches=branches,original_parse=t['review']['parsed'],fixed_parse=res['parsed'],
                fixed_minus_original=branches[2]['actual']['score']-branches[1]['actual']['score'],
                original_minus_skip=branches[1]['actual']['score']-branches[0]['actual']['score'],fixed_minus_skip=branches[2]['actual']['score']-branches[0]['actual']['score'],
                full_hidden_state_equivalence_claimed=False,downstream_causal_return_measured=False))
        if not a.offline_only:
            for i,row in enumerate(profile['tasks']):
                ledger.begin_batch();verify(manifest)
                variants=profile['episode_variants'];variants=[variants[0]]+list(reversed(variants[1:])) if i%2 else variants
                for v in variants:
                    protocol=original if v['reviewer']=='original' else fixed
                    out=run/'episodes'/row['task_id']/(v['policy']+'-'+v['reviewer'])
                    atomic(run/'progress.json',dict(stage='episodes',task=row['task_id'],policy=v['policy'],reviewer=v['reviewer']))
                    e=run_episode(actor,memory,row,v['policy'],0,profile,gate,out,ledger,reviewer=protocol)
                    append(run/'episodes.jsonl',e)
        verify(manifest);assert ledger.state['pending_model'] is None and ledger.state['pending_transition'] is None
        atomic(run/'complete.json',dict(stage='complete',episodes=len(read_rows(run/'episodes.jsonl')) if (run/'episodes.jsonl').exists() else 0,
            seconds=time.perf_counter()-started,finished_at_ns=time.time_ns(),ledger=ledger.state))
        atomic(run/'progress.json',dict(stage='complete'))
    except BaseException as error:
        atomic(run/'error.json',dict(type=type(error).__name__,message=str(error),ledger=ledger.state,at_ns=time.time_ns()));raise


if __name__=='__main__':main()
