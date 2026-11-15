"""Strict complete-cohort audit/export; no outcome-based filtering or tuning."""
import argparse
import csv
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic
from evaluation.alfworld_dev50 import load_frozen, read, sha, require, completed, spent, active_workers, PROFILE, FREEZE
from experience.alfworld_memory import AlfExperienceMemory
from scripts.analyze_alfworld_pilot import audit_episode, rows


def wilson(successes, n):
    z = 1.959963984540054
    phat = successes/n
    denominator = 1+z*z/n
    center = (phat+z*z/(2*n))/denominator
    half = z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/denominator
    return [max(0.,center-half),min(1.,center+half)]


def finalize(run):
    p,split,freeze = load_frozen()
    manifest = read(run/'manifest.json')
    require(sha(ROOT/FREEZE)==manifest['freeze_sha256'],'Method freeze changed')
    require((run/'complete.json').exists(),'No final means from incomplete cohort')
    expected={(r['task_id'],policy,seed) for r in split['evaluation_tasks'] for policy in p['policies'] for seed in p['rollout_seeds']}
    done=completed(run)
    require(set(done)==expected and len(done)==250,'Wrong task/policy/seed set')
    require(not active_workers(run),'Model workers still active')
    natives=[line for line in subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines()
             if str(ROOT/'scripts/alfworld_worker.py') in line and str(run) in line]
    require(not natives,'Owned native workers remain; inspect before publication')
    memory=AlfExperienceMemory(read(ROOT/p['memory_snapshot']),split)
    memory.model_fingerprint=freeze['actor_config_sha256']
    gate=read(ROOT/p['gate_config'])
    preflight=Path(manifest['preflight'])
    goals={r['task_id']:r for r in read(preflight/'goal_identity.json')}
    episodes,examples,changes=[],[],[]
    for key in sorted(done):
        value=done[key]
        entry=dict(task_id=key[0],policy=key[1],seed=key[2])
        result, ex, diff=audit_episode(Path(value['path']),entry,p,gate,memory,goals)
        result['raw_episode_path']=str(Path(value['path']).relative_to(ROOT))
        episodes.append(result)
        changes.extend(diff)
        # Compact representative evidence, not thousands of raw decision traces.
        if len([e for e in examples if e['policy']==key[1]])<8:
            examples.extend(ex[:2])
    audit_spent=spent(run)
    llm=Counter()
    for ledger in run.glob('attempt*/worker*/ledger.json'):
        calls=rows(ledger.parent/'generation_calls.jsonl')
        state=read(ledger)
        require(len(calls)==state['model_calls'],'Model receipt count mismatch')
        require(sum(c['metadata']['usage']['total_tokens'] for c in calls)==state['tokens'],'Token ledger mismatch')
        for c in calls:
            llm['calls']+=1
            for k in ['prompt_tokens','completion_tokens','total_tokens']:
                llm[k]+=c['metadata']['usage'][k]
    require(llm['calls']==audit_spent['model_calls'] and llm['total_tokens']==audit_spent['tokens'],'All-attempt accounting mismatch')
    proof=read(preflight/'complete.json')
    expected_spent=dict(audit_spent)
    expected_spent['transition_queries']+=proof['transition_queries']
    expected_spent['audit_capture_bundles']+=proof['audit_capture_bundles']
    require(expected_spent==read(run/'complete.json')['total_resource_spent'],'Global cost/preflight double count')
    policies=[]
    for policy in p['policies']:
        subset=[r for r in episodes if r['policy']==policy]
        require(len(subset)==50,'Unequal policy denominator')
        counts=sum((Counter(r['audit_counts']) for r in subset),Counter())
        costs=sum((Counter({k:v for k,v in r['costs'].items() if k!='peak_allocated_bytes'}) for r in subset),Counter())
        policies.append(dict(policy=policy,n=50,successes=sum(r['success'] for r in subset),success_rate=statistics.mean(r['success'] for r in subset),
            descriptive_wilson95=wilson(sum(r['success'] for r in subset),50),tokens=sum(r['tokens'] for r in subset),wm_calls=sum(r['wm_calls'] for r in subset),
            formal_steps=sum(r['formal_steps'] for r in subset),seconds=sum(r['total_seconds'] for r in subset),costs=dict(costs),audit_counts=dict(counts),
            family_results={family:dict(n=len(rs),successes=sum(r['success'] for r in rs)) for family in sorted({r['family'] for r in subset}) for rs in [[r for r in subset if r['family']==family]]}))
    index={(r['task_id'],r['policy']):r for r in episodes}
    pairs=[]
    comparisons=[]
    for reference in ['never','always','internal','experience']:
        differences=[int(index[(r['task_id'],'full')]['success'])-int(index[(r['task_id'],reference)]['success']) for r in split['evaluation_tasks']]
        pairs.append(dict(policy='full',reference=reference,n=50,mean_success_delta=statistics.mean(differences),positive=differences.count(1),zero=differences.count(0),negative=differences.count(-1),
            scope='Task-paired policy comparison, not same-state per-call causal utility'))
    for left,right in [('full','internal'),('experience','never')]:
        for row in split['evaluation_tasks']:
            aa=rows(ROOT/index[(row['task_id'],left)]['raw_episode_path']/'steps.jsonl')
            bb=rows(ROOT/index[(row['task_id'],right)]['raw_episode_path']/'steps.jsonl')
            equal=len(aa)==len(bb)
            comparisons.append(dict(task_id=row['task_id'],policy=left,reference=right,same_length=equal,
                same_actions=equal and all(a['final_action']==b['final_action'] for a,b in zip(aa,bb)),
                same_transitions=equal and all(a['actual']==b['actual'] for a,b in zip(aa,bb)),
                same_call_sequence=equal and all(a['gate']['call_wm']==b['gate']['call_wm'] for a,b in zip(aa,bb))))
    summary=dict(status='complete',episodes=episodes,policies=policies,paired_differences=pairs,
        external_added_decisions_on_same_full_states=changes,trajectory_comparisons=comparisons,all_attempt_llm_cost=dict(llm),
        preflight_cost=proof,whole_run=read(run/'complete.json'),source_commit=manifest['source_commit'],freeze_sha256=manifest['freeze_sha256'],
        limits=['One seed only; Wilson intervals are descriptive for this finite dev cohort, task templates/layouts may be dependent',
                'Not final140 valid_seen evaluation or8-seed paper-score reproduction','Always50 vs selective5 budgets; not matched-budget',
                'Ground-truth simulator has oracle-like information access; not learned WM prediction',
                'Extra Reviewer inference and feedback utility not separated; no long-horizon per-call causal test',
                'Known shared ScienceWorld word in Action placeholder retained; no silent prompt repair'])
    atomic(run/'summary.json',summary)
    atomic(run/'examples.json',examples)
    buffer=io.StringIO()
    columns=['task_id','policy','rollout_seed','family','success','formal_steps','wm_calls','verified_lookaheads','reviewer_calls','tokens','input_tokens','output_tokens','total_seconds','steps_sha256','raw_episode_path']
    writer=csv.DictWriter(buffer,fieldnames=columns,lineterminator='\n')
    writer.writeheader()
    writer.writerows({k:r[k] for k in columns} for r in episodes)
    (run/'episodes.csv').write_text(buffer.getvalue())
    atomic(run/'audit_complete.json',dict(status='pass',audited_episodes=250,summary_sha256=sha(run/'summary.json'),native_workers_remaining=[],model_workers_remaining=[],at_unix=time.time()))
    return summary


def publish(run):
    """Only publish own new report/results; never stage another session's changes."""
    try:
        require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Worktree changed; results saved, publication needs review')
        s=read(run/'summary.json')
        require(s['status']=='complete' and read(run/'audit_complete.json')['status']=='pass','Incomplete audit')
        dest=ROOT/'docs/alfworld_dev50_v1/results'
        dest.mkdir(parents=True,exist_ok=False)
        for name in ['summary.json','examples.json','episodes.csv','audit_complete.json','complete.json']:
            shutil.copy2(run/name,dest/name)
        lines=['# ALFWorld independent50-task development evaluation','',
            'Frozen Qwen3.5-9B Actor/Reviewer;50 new goal-disjoint TRAIN development tasks×seed0×five policies=250 Episodes. No final Test rollout or training. Methods, thresholds, prompts and Memory match the3-task pilot; no outcome-based tuning.','',
            '| Policy | Success | SR | Tokens | Lookaheads | Native API queries |','|---|---:|---:|---:|---:|---:|']
        for r in s['policies']:
            lines.append(f"| {r['policy']} | {r['successes']}/50 | {100*r['success_rate']:.1f}% | {r['tokens']:,} | {r['wm_calls']} | {r['costs']['api_queries']} |")
        lines+=['','Full-vs-Internal external-added current-state decisions: '+str(len(s['external_added_decisions_on_same_full_states'])),
            '', 'All per-task outcomes, paired policy differences, Reviewer/invalid-action/coverage/Unknown counts, replay checks, timings and costs are in [summary](alfworld_dev50_v1/results/summary.json) and [CSV](alfworld_dev50_v1/results/episodes.csv).',
            '', 'All recorded new-run LLM tokens (including interrupted attempts)='+str(s['all_attempt_llm_cost']['total_tokens'])+'. Preflight native costs are separate in summary; prior48-step calibration and frozen31-case Memory are reused, not recomputed or relabelled free historical construction. Model load/partial ledgers remain in server outputs.',
            '', 'Intervals are descriptive with one seed; no across-seed sample SD. Equal instance weighting, not category or batch weighting. Different policies have different trajectories. A revision or paired outcome does not establish per-call causal utility. This is oracle-like environment access, not a learned WM. No budget-matched or no-feedback Reviewer controls were run.',
            '', 'Known legacy Action placeholder wording and original512-token three-line Reviewer remain frozen; format/action failures are retained. No Phase6/final140×8 evaluation was started. See [protocol](ALFWORLD_DEV50.md).',
            '', 'Runtime source commit: '+s['source_commit']+'. Full audit passed; owned model/native workers released.']
        (ROOT/'docs/ALFWORLD_DEV50_REPORT.md').write_text('\n'.join(lines)+'\n')
        paths=['docs/ALFWORLD_DEV50_REPORT.md','docs/alfworld_dev50_v1/results']
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
        require(all(any(v==a or v.startswith(a+'/') for a in paths) for v in staged),'Unrelated staged changes; do not commit')
        subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Report frozen ALFWorld50-task five-policy independent development evaluation'],cwd=ROOT,check=True)
        env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=20')
        subprocess.run(['git','push','origin','HEAD:main'],cwd=ROOT,env=env,check=True)
        atomic(run/'publication.json',dict(status='complete_and_pushed',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    except Exception as e:
        atomic(run/'publication.json',dict(status='needs_review',error=str(e),scientific_results_preserved=True))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--publish',action='store_true')
    a=ap.parse_args()
    finalize(Path(a.run).absolute())
    if a.publish:
        publish(Path(a.run).absolute())
