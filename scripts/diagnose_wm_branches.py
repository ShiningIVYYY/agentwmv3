#!/usr/bin/env python3
"""Five live-heap matched branch pairs, one initial WM intervention then Never.

This is a posthoc development diagnostic, not the production Agent/Gate loop.
No simulated future transition is exposed before choosing either branch action.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_audited_paper_eval as runner
from gates.generation import GateActorAdapter
from gates.worldmodel_setup import ExistingWorldModelSetup
from evaluation.observer import observe_wm
from evaluation.seeds import request_seed
from evaluation.profile import native_menu_actions
from internal_signals.decision import action_from_text
from experience.splits import file_hash,goal_key


def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser()
    for arg in ('model-dir','wm-checkpoint','science-python','output'):p.add_argument('--'+arg,required=True)
    p.add_argument('--science-url',default='http://127.0.0.1:36055');p.add_argument('--wm-url',default='http://127.0.0.1:18212')
    a=p.parse_args();run_started=time.perf_counter();out=Path(a.output).resolve()
    if out.exists():raise ValueError('New branch run ID required')
    out.mkdir(parents=True);plan=load(ROOT/'outputs/phase55_setup/plan.json')
    if not load(ROOT/'outputs/phase55_setup/checkpoint_probe.json')['passed']:raise ValueError('No validated checkpoint')
    if not load(ROOT/'outputs/phase55_setup/cohort_finished.json')['complete']:raise ValueError('Finish cohort first; do not share Actor GPU')
    cohort_steps=sum(sum(e['metrics']['environment_steps'] for e in load(ROOT/('outputs/phase55_'+variant+'_v1/episodes.json'))) for variant in plan['variants'])
    branch_max=sum(2*(21-s['source_step'])+(s['source_step']-1) for s in plan['branch_states'])
    if cohort_steps+branch_max+20>600:raise ValueError('Frozen overall600-action diagnostic cap would be exceeded')
    sources=[Path(__file__),ROOT/'diagnostics/reviewer.py',ROOT/'scripts/serve_scienceworld_diagnostics.py',ROOT/'worldmodel/HeapSnapshot.java']
    save(out/'manifest.json',dict(plan=plan,git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),
        sources={str(s.relative_to(ROOT)):file_hash(s) for s in sources},continuation_policy='Never WM',
        no_memory_or_gate_selection=True,strictness='same live JVM heap incl RNG + Python state; same actual pre-treatment Actor history and per-turn seeds'))
    for source in sources:
        dest=out/'executed_source'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    config=load(ROOT/'configs/worldmodel/scienceworld.json');config.update(checkpoint=str(Path(a.wm_checkpoint).resolve()),source_dir=str(ROOT/'external/Word2World'),url=a.wm_url)
    save(out/'wm_config.json',config)
    args=SimpleNamespace(internal_signals_config=str(ROOT/'configs/internal_signals/scienceworld_token_nll_v1.json'),
        tokenizer_path=a.model_dir,internal_device='cuda:0',science_url=a.science_url,max_steps=20,max_tokens=512,
        disable_thinking=True,wm_backend='word2world',wm_config=str(out/'wm_config.json'),
        reflection_review_protocol='json',reflection_max_rounds=3,science_context=None,science_demo=None)
    actor=runner.Agent(args,out);actor.evaluation_profile=load(ROOT/'configs/evaluation/paper_aligned_v1/scienceworld.json');actor.internal_actor.evaluation_left_truncation=True
    actor_ready=time.perf_counter()
    http=lambda method,path,**kw:runner.http(method,a.science_url+path,**kw)
    pairs=[]
    for state in plan['branch_states']:
        row=state['row'];source=ROOT/state['source_trace']
        if file_hash(source)!=state['source_trace_sha256']:raise ValueError('Source history changed')
        original=[json.loads(s) for s in source.read_text().splitlines()];start=state['source_step'];prefix=original[:start-1]
        directory=out/row['task_id'];directory.mkdir();state_started=time.perf_counter();idx=http('POST','/create')['id']
        try:
            initial=http('POST','/reset',json=dict(id=idx,data_idx=row['index']))
            if goal_key(initial['task_description'])!=row['goal_group']:raise ValueError('Wrong native goal')
            current=initial;replay=[]
            for t in prefix:
                if current['observation']!=t['observation']:raise ValueError('Prefix observation mismatch; no strict branch')
                current=http('POST','/step',json=dict(id=idx,action=t['action']))
                if any(current[k]!=t['native_receipt']['response'][k] for k in ('observation','score','reward','done')):
                    raise ValueError('Historical prefix replay mismatch; do not claim historical hidden-state equivalence')
                replay.append(current)
            if current['observation']!=original[start-1]['observation'] or current['done']:raise ValueError('Wrong/nonlive branch state')
            snapshot=http('POST','/checkpoint',json=dict(id=idx));save(directory/'checkpoint.json',dict(snapshot=snapshot,prefix_receipts=replay))
            state_setup_seconds=time.perf_counter()-state_started
            pair=[];first_ids=[]
            for branch in ('skip','call'):
                ep=directory/branch;ep.mkdir();actor.out=ep;actor.total_calls=0
                restore_started=time.perf_counter();restored=http('POST','/restore',json=dict(id=idx,snapshot_id=snapshot['snapshot_id']));restore_seconds=time.perf_counter()-restore_started
                if not restored['restore_provenance']['heap_matches'] or restored['restore_provenance']['actor_steps']!=start-1:raise ValueError('Restore failed')
                history=copy.deepcopy(prefix);adapter=GateActorAdapter(actor,False)
                adapter.original=runner.Agent.generate.__get__(actor,runner.Agent);subcalls={}
                def seeds(task,step,role):
                    key=(task,step);sub=subcalls.get(key,0)
                    if role=='review':subcalls[key]=sub+1
                    return dict(request_seed=request_seed(task,step-1,0,role=role,subcall=sub),role=role,turn_index=step-1,subcall=sub if role=='review' else 0)
                adapter.seed_provider=seeds;actor.generate=adapter.generate
                setup=ExistingWorldModelSetup(a.science_python,ROOT,ep);actor.wm_context_provider=setup.context;actor.wm_token_counter=setup.count
                controller=None;traces=[];native=restored;began=time.perf_counter();active=[None]
                with observe_wm(a.wm_url,ep/'wm_http_private.jsonl',lambda:active[0]) as requests:
                    try:
                        for step in range(start,21):
                            if native['done']:break
                            observation=native['observation'];messages,_=runner.bounded_science(actor,None,initial['task_description'],observation,history)
                            output,metadata=actor.generate(messages,row['task_id'],step);draft=action_from_text(output);action=draft;reflection={}
                            if step==start:first_ids.append(metadata['generation_record']['response_ids'])
                            active[0]=row['task_id']+'/'+branch+'/'+str(step)
                            before=time.time_ns()
                            if branch=='call' and step==start:
                                controller=runner.wm_runtime.make_controller(args,'scienceworld',row,initial['task_description'],runner.Science.transition(initial),None,actor,ROOT)
                                for prior in prefix:controller.wm.commit(prior['action'],runner.Science.transition(prior['native_receipt']['response']))
                                actions=http('GET','/admissible_actions',params=dict(id=idx))['admissible_actions']
                                actions=list(dict.fromkeys(list(actions)+native_menu_actions(observation)))
                                adapter.reviewing=True
                                try:action,effective,reflection=runner.wm_runtime.select(controller,draft,output,observation,history,actions,step)
                                finally:adapter.reviewing=False
                            started=time.time_ns();native=http('POST','/step',json=dict(id=idx,action=action));available=time.time_ns()
                            if controller is not None:
                                if step==start:runner.wm_runtime.commit(controller,action,runner.Science.transition(native),reflection)
                                else:controller.wm.commit(action,runner.Science.transition(native))
                            record=dict(task_id=row['task_id'],branch=branch,step_index=step,observation=observation,draft_action=draft,action=action,
                                output=output,next_observation=native['observation'],score=native['score'],native=native,
                                decision_started_at_ns=before,environment_step_started_at_ns=started,feedback_available_at_ns=available,
                                treatment_called=branch=='call' and step==start,**metadata,**reflection)
                            traces.append(record);history.append(record)
                            with (ep/'steps.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                    finally:
                        if controller is not None:controller.wm.close()
                usage=[c['usage'] for c in adapter.calls]+[q['usage'] for q in requests]
                summary=dict(task_id=row['task_id'],branch=branch,source_step=start,remaining_actor_budget=21-start,
                    initial_score=restored['score'],initial_simulator_moves=restored['simulator_moves'],
                    final_score=native['score'],native_final=native,continuation_actions=len(traces),
                    actor_calls=len(adapter.calls),wm_requests=len(requests),review_calls=sum(c['role']=='review' for c in adapter.calls),
                    total_tokens=sum(c['total_tokens'] for c in usage),input_tokens=sum(c['prompt_tokens'] for c in usage),output_tokens=sum(c['completion_tokens'] for c in usage),
                    seconds=time.perf_counter()-began,restore_provenance=restored['restore_provenance'])
                summary.update(restore_seconds=restore_seconds,wm_context_setup=setup.costs)
                save(ep/'summary.json',summary);save(ep/'actor_calls.json',adapter.calls);pair.append(summary)
            if first_ids[0]!=first_ids[1] or first_ids[0]!=original[start-1]['generation_record']['response_ids']:
                raise ValueError('Pre-treatment Actor draft differs; reject strict matching')
            pairs.append(dict(task_id=row['task_id'],source_step=start,matched_first_draft_ids=True,matched_live_heap=True,
                branches=pair,score_utility_call_minus_skip=pair[1]['final_score']-pair[0]['final_score'],
                shared_state_reset_prefix_checkpoint_seconds=state_setup_seconds,
                causal_scope='single first intervention followed by identical Never policy and task-turn RNG, one seed; no broad utility claim'))
            save(out/'pairs.json',pairs);save(out/'progress.json',dict(completed_pairs=len(pairs),total_pairs=5))
        finally:http('POST','/close',json=dict(id=idx))
    save(out/'summary.json',dict(completed_pairs=len(pairs),actor_load_hash_seconds=actor_ready-run_started,whole_run_seconds=time.perf_counter()-run_started,positive=sum(p['score_utility_call_minus_skip']>0 for p in pairs),negative=sum(p['score_utility_call_minus_skip']<0 for p in pairs),zero=sum(p['score_utility_call_minus_skip']==0 for p in pairs),pairs=pairs))
    print(json.dumps(load(out/'summary.json'),indent=2))


if __name__=='__main__':main()
