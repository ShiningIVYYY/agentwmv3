#!/usr/bin/env python3
"""Bounded dev-only9B/calibration/four-policy diagnostic, not a benchmark sweep."""
import argparse
from dataclasses import asdict
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experience.collection import fields
from experience.memory import ExperienceMemory
from experience.schema import ExternalQuery
from experience.splits import validate_manifest, goal_key, file_hash
from gates.core import RuleGate, GateInput, ExternalInput, fingerprint
from gates.evidence import resolve_external
from internal_signals.decision import messages
from internal_signals.qwen35_actor import Qwen35Actor
from internal_signals.token_nll import selected_token_nll, teacher_forced_nll
from reflection.replay_review import PROTOCOL, review_messages, parse_review
from worldmodel.ground_truth_environment import ReplayScienceEnvironment, GroundTruthEnvironmentWM


def atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def request_seed(task, step, role):
    return int(hashlib.sha256(('phase57-rollout0|'+task+'|'+str(step)+'|'+role).encode()).hexdigest()[:8], 16) % (2**31)


def ranks(values):
    order = np.argsort(values, kind='stable')
    result = np.zeros(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index+1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        result[order[index:end]] = (index+end-1)/2
        index = end
    return result


def calibrate(records, actor):
    valid = [r['metadata']['internal_signals'] for r in records
             if r['metadata']['internal_signals']['status'] == 'ok']
    if len(valid) < 12:
        raise ValueError('Too few valid dev signals even for a pilot calibration')
    local = np.array([s['local_uncertainty'] for s in valid])
    glob = np.array([s['global_uncertainty'] for s in valid])
    quantiles = []
    for q in (.5, .75, .9):
        tl, tg = float(np.quantile(local, q)), float(np.quantile(glob, q))
        quantiles.append(dict(quantile=q, tau_local=tl, tau_global=tg,
                              expected_uncapped_call_rate=float(np.mean((local > tl) | (glob > tg)))))
    selected = quantiles[1]  # Prespecified q=.75, not selected using task rewards.
    config = dict(schema='training-free-gate-v1', policy='full', tau_local=selected['tau_local'],
                  tau_global=selected['tau_global'], allow_confirmed_action_skip=False,
                  memory_correction=False, missing_policy='global_then_call', max_forecasts_per_invocation=1,
                  review_protocol=PROTOCOL, reflection_max_rounds=1,
                  calibration_status='pilot_distribution_calibration_limited_goal_groups',
                  internal_signature=dict(backend=actor.config['backend'], dtype='bfloat16',
                    model_revision=actor.config['model_revision'], method='fixed_field_mean_token_nll_v1',
                    scoring_path='raw_generation_logits_before_temperature', score_unit='nats_per_token',
                    config_sha256=actor.fingerprint))
    report = dict(valid_steps=len(valid), total_steps=len(records), missing_steps=len(records)-len(valid),
                  independent_tasks=len({r['task_id'] for r in records}),
                  pearson=float(np.corrcoef(local, glob)[0, 1]),
                  spearman=float(np.corrcoef(ranks(local), ranks(glob))[0, 1]), quantiles=quantiles,
                  local=dict(min=float(local.min()), max=float(local.max()), mean=float(local.mean())),
                  global_nll=dict(min=float(glob.min()), max=float(glob.max()), mean=float(glob.mean())),
                  selected_quantile=.75, selected_without_outcomes=True, thresholds_fully_validated=False,
                  budgets_selected_independently=dict(never=0, always=20, internal=5, full=5))
    return config, report


def load_calibration_source(source, actor_config, calibration_ids):
    """Resume only identical model/scoring/split data, never select by outcomes."""
    source = Path(source).resolve()
    old = json.loads((source/'manifest.json').read_text())
    if old['model_config'] != actor_config or [r['task_id'] for r in old['calibration_tasks']] != calibration_ids:
        raise ValueError('Calibration source model/cohort mismatch')
    if old['split_sha256'] != file_hash(ROOT/'docs/phase3/split_manifest.json'):
        raise ValueError('Calibration source split mismatch')
    for name in ('internal_signals/qwen35_actor.py','internal_signals/hf_actor.py',
                 'internal_signals/provider.py','internal_signals/token_nll.py','worldmodel/ground_truth_environment.py'):
        if old['source_hashes'][name] != file_hash(ROOT/name):
            raise ValueError('Calibration computation changed: '+name)
    rows = [json.loads(s) for s in (source/'calibration_steps.jsonl').read_text().splitlines()]
    if {r['task_id'] for r in rows} != set(calibration_ids):
        raise ValueError('Incomplete calibration source')
    for row in rows:
        meta = row['metadata']
        if meta['actor_metadata']['config_sha256'] != fingerprint(actor_config):
            raise ValueError('Calibration source scoring fingerprint mismatch')
    return rows, dict(source_run=str(source), records_sha256=file_hash(source/'calibration_steps.jsonl'),
                      frozen_config=json.loads((source/'gate_config.json').read_text()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', default=str(ROOT/'configs/internal_signals/qwen35_9b_replay_v1.json'))
    parser.add_argument('--profile', default=str(ROOT/'configs/evaluation/phase57_replay_dev_v1.json'))
    parser.add_argument('--calibration-source', help='Reuse only a verified identical frozen development collection')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError('Use a new Run ID; no hidden retries')
    out.mkdir(parents=True)
    manifest = json.loads((ROOT/'docs/phase3/split_manifest.json').read_text())
    validate_manifest(manifest, require_native=True)
    dev = {r['task_id']: r for r in manifest['partitions']['development']}
    plan = json.loads((ROOT/'docs/phase5/development_plan.json').read_text())
    profile = json.loads(Path(args.profile).read_text())
    calibration_ids, evaluation_ids = profile['calibration_tasks'], profile['evaluation_tasks']
    if len(calibration_ids)>6 or len(evaluation_ids)>3 or profile['rollout_seeds']!=[0] or profile['calibration_quantile']!=.75:
        raise ValueError('Bounded prespecified pilot profile required')
    fixed = dict(maximum_interaction_turns=20,calibration_max_actions=8,environment_step_limit=21,
                 task_seed=0,simplifications='',simulation_depth=1,review_rounds=1,
                 memory_snapshot='docs/phase5/memory_snapshot_v2.json')
    if any(profile.get(k)!=v for k,v in fixed.items()):
        raise ValueError('Profile differs from implemented bounded protocol')
    actor_config=json.loads(Path(args.config).read_text())
    reused_calibration = None
    if args.calibration_source:
        reused_calibration = load_calibration_source(args.calibration_source, actor_config, calibration_ids)
    for public, actual in [('temperature','temperature'),('top_p','top_p'),('top_k','top_k'),
                           ('prompt_limit','max_prompt_tokens'),('response_limit','max_new_tokens'),('history_length','history_length')]:
        if profile[public]!=actor_config[actual]:
            raise ValueError('Actor/pilot profile mismatch: '+public)
    allowed_cal = {r['task_id'] for r in plan['calibration_tasks']}
    allowed_eval = {r['task_id'] for r in plan['validation_tasks']}
    if not set(calibration_ids) <= allowed_cal or not set(evaluation_ids) <= allowed_eval:
        raise ValueError('Dev plan identity mismatch')
    if {dev[k]['goal_group'] for k in calibration_ids} & {dev[k]['goal_group'] for k in evaluation_ids}:
        raise ValueError('Calibration/evaluation goal leakage')
    sources = [ROOT/f for f in ('scripts/run_phase57_replay.py', 'worldmodel/ground_truth_environment.py',
               'reflection/replay_review.py', 'internal_signals/qwen35_actor.py', 'internal_signals/hf_actor.py',
               'internal_signals/provider.py', 'internal_signals/token_nll.py', 'gates/core.py')]
    atomic(out/'manifest.json', dict(purpose='phase57_development_engineering_diagnostic_not_final_benchmark',
           git_base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip(),
           calibration_tasks=[dev[k] for k in calibration_ids], evaluation_tasks=[dev[k] for k in evaluation_ids],
           model_config=json.loads(Path(args.config).read_text()), pilot_profile=profile, calibration_max_actions=8,
           episode_max_actions=20, model_rollout_seed=0, simulator_seed=0, simplifications='',
           environment_step_limit=21, simulation_depth=1, review_rounds=1,
           feedback_permissions=['trial_observation', 'trial_reward', 'trial_done', 'trial_score'],
           query_policy='native_step_without_rng_mutating_valid_action_enumeration/post_load_seed',
           bounded_state_verification=True, complete_hidden_state_equivalence=False,
           no_training=True, no_learned_wm_calls=True,
           source_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
           memory_sha256=file_hash(ROOT/'docs/phase5/memory_snapshot_v2.json'),
           split_sha256=file_hash(ROOT/'docs/phase3/split_manifest.json'),
           paper_protocol_sha256=file_hash(ROOT/'docs/PAPER_ALIGNED_EVAL.md')))
    atomic(out/'progress.json', dict(stage='loading9b'))
    begin = time.perf_counter()
    actor = Qwen35Actor(args.config, args.model_dir, args.device)
    original_generate = actor.generate_role
    def audited_generate(*positional, **keywords):
        output, metadata = original_generate(*positional, **keywords)
        append(out/'generation_calls.jsonl', dict(available_at_ns=time.time_ns(), metadata=metadata))
        return output, metadata
    actor.generate_role = audited_generate
    load_seconds = time.perf_counter()-begin
    wm = GroundTruthEnvironmentWM()
    validation_usage = []
    smoke = dict(loading_info=actor.loading_info, load_seconds=load_seconds,
                 backend=actor.config['backend'], config_sha256=actor.fingerprint,
                 model_revision=actor.config['model_revision'], non_thinking_template=actor.render([dict(role='user', content='hello')]))
    atomic(out/'model_smoke.json', smoke)
    # Raw-logit/next-token check is separate from the formal generation-only path.
    captured = []
    def hook(module, arguments, output):
        if len(captured) < 8:
            captured.append(output.logits[0, -1].detach().float().clone())
    handle = actor.model.register_forward_hook(hook)
    try:
        text, meta = actor.generate_role([dict(role='user', content='Return exactly one word: hello')], 'validation', 901, True)
    finally:
        handle.remove()
    validation_usage.append(meta['usage'])
    atomic(out/'ordinary_validation_generation.json', meta)
    ids = meta['generation_record']['response_ids'][:len(captured)]
    raw = selected_token_nll(torch.stack(captured[:len(ids)]), ids).tolist()
    original = meta['generation_record']['token_nll'][:len(ids)]
    delta = max(abs(a-b) for a,b in zip(raw, original))
    cpu_values = selected_token_nll(torch.stack(captured[:len(ids)]).cpu(), ids).tolist()
    smoke.update(raw_gpu_nll_max_delta=delta, raw_gpu_nll_tolerance=1e-5,
                 cpu_fp32_diagnostic_max_delta=max(abs(a-b) for a,b in zip(cpu_values, original)),
                 cpu_path_used_for_thresholds=False)
    atomic(out/'model_smoke.json', smoke)
    if delta > 1e-5:
        raise ValueError('Captured raw logits and selected-token NLL disagree')
    forward = teacher_forced_nll(actor.model, meta['generation_record']['prompt_ids'], ids).tolist()
    smoke.update(ordinary_output=text, raw_nll_exact=delta==0, raw_nll_within_tolerance=delta<=1e-5, checked_tokens=len(ids),
                 extra_validation_forward_passes=1,
                 generation_vs_teacher_forced_max_delta=max(abs(a-b) for a,b in zip(raw, forward)),
                 actual_sampling_config=actor.model.generation_config.to_dict())
    atomic(out/'model_smoke.json', smoke)

    if reused_calibration is not None:
        calibration, reuse = reused_calibration
        shutil.copyfile(Path(args.calibration_source)/'calibration_steps.jsonl', out/'calibration_steps.jsonl')
        atomic(out/'calibration_provenance.json', reuse)
        source_smoke=json.loads((Path(args.calibration_source)/'model_smoke.json').read_text())
        smoke['scienceworld_validation_source'] = dict(run=args.calibration_source,
            source_smoke_sha256=file_hash(Path(args.calibration_source)/'model_smoke.json'),
            scored_unscored_response_ids_identical=source_smoke['scored_unscored_response_ids_identical'],
            repeated_generation_and_nll_exact=source_smoke['repeated_generation_and_nll_exact'])
        atomic(out/'model_smoke.json',smoke)
    else:
        calibration = []
        first = True
        for task_id in calibration_ids:
            row = dev[task_id]
            env = ReplayScienceEnvironment(dict(taskName=row['taskName'], variationIdx=row['variationIdx'],
                                         simplifications='', seed=0, environment_step_limit=21))
            past = []
            try:
                if goal_key(env.goal) != row['goal_group']:
                    raise ValueError('Native development goal differs')
                for step in range(8):
                    observation = env.current['observation']
                    prompt = messages(env.goal, observation, past, keep=2)
                    seed = request_seed(task_id, step, 'draft')
                    text, meta = actor.generate_role(prompt, 'calibration_draft', seed, True)
                    if first:
                        other, othermeta = actor.generate_role(prompt, 'parity', seed, False)
                        validation_usage.append(othermeta['usage'])
                        same = meta['generation_record']['response_ids'] == othermeta['generation_record']['response_ids']
                        repeated, repeated_meta = actor.generate_role(prompt, 'repeat_validation', seed, True)
                        validation_usage.append(repeated_meta['usage'])
                        if repeated!=text or repeated_meta['generation_record']['token_nll']!=meta['generation_record']['token_nll']:
                            raise ValueError('Repeated generation/scoring differs')
                        smoke.update(scored_unscored_response_ids_identical=same,
                                     repeated_generation_and_nll_exact=True,
                                     actual_scienceworld_draft=text,
                                     scienceworld_internal_signals=meta['internal_signals'])
                        atomic(out/'model_smoke.json', smoke)
                        if not same or '<think>' in text or '</think>' in text:
                            raise ValueError('Scored/unscored or nonthinking validation failed')
                        first = False
                    _, _, action = fields(text)
                    before_ns = time.time_ns()
                    actual = env.step(action)
                    record = dict(task_id=task_id, step=step, observation=observation, output=text,
                                  action=action, metadata=meta, actual=actual, executed_at_ns=before_ns)
                    calibration.append(record)
                    append(out/'calibration_steps.jsonl', record)
                    past.append(dict(observation=observation, action=action))
                    atomic(out/'progress.json', dict(stage='calibration', task_id=task_id, step=step+1, total=len(calibration)))
                    if actual['done']:
                        break
            finally:
                env.close()
    gate_config, report = calibrate(calibration, actor)
    if reused_calibration is not None and gate_config != reuse['frozen_config']:
        raise ValueError('Recomputed frozen calibration differs; no retuning permitted')
    atomic(out/'gate_config.json', gate_config)
    atomic(out/'calibration.json', report)  # Frozen BEFORE four-policy evaluation.
    memory = ExperienceMemory(json.loads((ROOT/'docs/phase5/memory_snapshot_v2.json').read_text()), manifest)
    episodes = []
    for policy in ('never', 'always', 'internal', 'full'):
        config = dict(gate_config, policy=policy)
        gate = RuleGate(config)
        for task_id in evaluation_ids:
            row = dev[task_id]
            env = ReplayScienceEnvironment(dict(taskName=row['taskName'], variationIdx=row['variationIdx'],
                                         simplifications='', seed=0, environment_step_limit=21))
            started = time.perf_counter()
            budget = profile['wm_invocation_budgets'][policy]
            past, traces = [], []
            try:
                if goal_key(env.goal) != row['goal_group']:
                    raise ValueError('Native development goal differs')
                for step in range(20):
                    observation = env.current['observation']
                    prompt = messages(env.goal, observation, past, keep=2)
                    text, meta = actor.generate_role(prompt, 'draft', request_seed(task_id, step, 'draft'), gate.needs_internal)
                    premise, consequence, draft = fields(text)
                    query = ExternalQuery(benchmark='scienceworld', task_id=task_id,
                              episode_id=out.name+'/'+policy+'/'+task_id, step_index=step+1,
                              available_at_ns=time.time_ns(), goal=env.goal, observation=observation,
                              draft_action=draft, premise=premise, expected_consequence=consequence)
                    external = memory.evaluate(query) if gate.needs_external else None
                    resolved = resolve_external(external, query, memory) if gate.needs_external else ExternalInput()
                    context = GateInput(decision_id=query.episode_id+'/'+str(step), remaining_wm_budget=budget,
                              remaining_forecast_budget=budget, internal=meta.get('internal_signals'),
                              external=resolved, draft_action_valid=meta['decision_format_valid'])
                    decision = gate.decide(context)
                    decided_ns = time.time_ns()
                    final = draft
                    simulation = review = review_meta = None
                    if decision['call_wm']:
                        budget -= 1  # Attempt counts even when reconstruction fails.
                        simulation = wm.simulate_one_step(env, draft)
                        if simulation['prediction_available']:
                            review_prompt = review_messages(env.goal, observation, past, draft, simulation)
                            if actor.count(review_prompt) <= actor.config['max_prompt_tokens']:
                                raw, review_meta = actor.generate_role(review_prompt, 'review', request_seed(task_id, step, 'review'), False)
                                review = dict(raw=raw, parsed=parse_review(raw, draft))
                                final = review['parsed']['final_action']
                            else:
                                review = dict(parsed=dict(decision='NOT_CALLED', final_action=draft, error='review_context_overflow'))
                        else:
                            review = dict(parsed=dict(decision='NOT_CALLED', final_action=draft, error='reconstruction_failed_keep_draft'))
                    execution_ns = time.time_ns()
                    if execution_ns <= decided_ns:
                        raise ValueError('Invalid pre-action decision ordering')
                    actual = env.step(final)
                    match = None
                    if simulation and simulation['prediction_available'] and final == draft:
                        match = all(simulation['candidate_transition'][k] == actual[k] for k in ('observation','reward','done','score','moves'))
                    trace = dict(policy=policy, task_id=task_id, step=step, goal=env.goal, observation=observation,
                                 draft_output=text, draft_action=draft, metadata=meta, external=external,
                                 resolved_external=asdict(resolved), gate=decision, decided_at_ns=decided_ns,
                                 simulation=simulation, review=review, review_metadata=review_meta,
                                 final_action=final, executed_at_ns=execution_ns, actual=actual,
                                 kept_draft_transition_match=match, budget_after=budget)
                    if simulation and simulation['prediction_available']:
                        trace['one_step_score_delta_actual_minus_trial_draft'] = actual['score']-simulation['predicted_score']
                    append(out/'evaluation_steps.jsonl', trace)
                    traces.append(trace)
                    atomic(out/'progress.json', dict(stage='evaluation', policy=policy, task_id=task_id, step=step+1, episodes=len(episodes)))
                    if match is False:
                        raise RuntimeError('Verified replay predicts different committed KEEP transition; stop')
                    past.append(dict(observation=observation, action=final))
                    if actual['done']:
                        break
                usage = [t['metadata']['usage'] for t in traces]+[t['review_metadata']['usage'] for t in traces if t['review_metadata']]
                episode = dict(policy=policy, task_id=task_id, task_score=env.current['score'],
                          success=env.current['done'] and env.current['score']==100,
                          formal_environment_actions=len(traces), simulator_moves=env.current['moves'],
                          done=env.current['done'], action_turn_cap_reached=len(traces)==20,
                          lookahead_attempts=sum(t['simulation'] is not None for t in traces),
                          verified_lookaheads=sum(bool(t['simulation'] and t['simulation']['prediction_available']) for t in traces),
                          reconstruction_failures=sum(bool(t['simulation'] and not t['simulation']['prediction_available']) for t in traces),
                          reviewer_calls=sum(t['review_metadata'] is not None for t in traces),
                          action_corrections=sum(t['draft_action']!=t['final_action'] for t in traces),
                          input_tokens=sum(u['prompt_tokens'] for u in usage),
                          output_tokens=sum(u['completion_tokens'] for u in usage), total_tokens=sum(u['total_tokens'] for u in usage),
                          seconds=time.perf_counter()-started)
                episodes.append(episode)
                append(out/'episodes.jsonl', episode)
            finally:
                env.close()
    summaries = []
    for policy in ('never', 'always', 'internal', 'full'):
        rows = [e for e in episodes if e['policy']==policy]
        totals = {k:sum(e[k] for e in rows) for k in ('formal_environment_actions','lookahead_attempts','verified_lookaheads',
                  'reconstruction_failures','reviewer_calls','action_corrections','input_tokens','output_tokens','total_tokens','seconds')}
        summaries.append(dict(policy=policy, instances=len(rows), mean_task_score=sum(e['task_score'] for e in rows)/len(rows),
                              success_rate=sum(e['success'] for e in rows)/len(rows), **totals))
    atomic(out/'summary.json', dict(purpose='small_development_functional_diagnostic', policies=summaries,
           rollout_seed_count=1, across_seed_sample_sd=None, positive_scientific_effectiveness_claimed=False,
           offline_new_experience_cost=0, calibration_source=args.calibration_source,
           calibration_reference_steps=len(calibration), calibration_model_calls=0 if args.calibration_source else len(calibration),
           calibration_tokens=0 if args.calibration_source else sum(r['metadata']['usage']['total_tokens'] for r in calibration),
           validation_model_calls=len(validation_usage), validation_tokens=sum(u['total_tokens'] for u in validation_usage),
           learned_wm_calls=0, new_model_training=False))
    atomic(out/'progress.json', dict(stage='complete', episodes=len(episodes)))


if __name__ == '__main__':
    main()
