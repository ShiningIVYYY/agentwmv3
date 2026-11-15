"""Read-only ALFWorld pilot audit; incomplete attempts never become zero scores."""
import argparse
import csv
import io
import json
import math
import statistics
import hashlib
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.independent_dev import atomic, request_seed
from experience.collection import fields
from experience.splits import file_hash
from experience.alfworld_memory import AlfExperienceMemory
from gates.core import RuleGate, GateInput, ExternalInput
from internal_signals.alfworld_prompts import draft_messages, review_messages, parse_review
from scripts.run_alfworld_parallel import verify
from worldmodel.alfworld_environment import fingerprint


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def require(test, message):
    if not test:
        raise ValueError(message)


def canonical(value):
    return json.loads(json.dumps(value))


def without(value, keys):
    return {k: v for k, v in value.items() if k not in keys}


def audit_generation(meta, prompt, role, seed, profile, scored):
    record = meta['generation_record']
    require(record['messages'] == prompt, 'Prompt/history/public-input mismatch')
    actual_seed = meta['actor_metadata']['generation_seed'] if scored else record['request_seed']
    require(actual_seed == seed, 'Role sampling seed mismatch')
    require(record['left_truncated_prompt_tokens'] == 0, 'Silent prompt truncation')
    require(meta['prompt_tokens'] <= profile['prompt_limit'], 'Prompt limit exceeded')
    require(meta['usage']['completion_tokens'] <= profile['response_limit'], 'Response limit exceeded')
    require(meta['shared_model_id'] == profile['model_id'], 'Actor/reviewer model identity differs')
    require(meta['shared_model_revision'] == profile['model_revision'], 'Model revision differs')
    require(record['dtype'] == 'bfloat16', 'Generation dtype differs')
    require(meta['role'] == role, 'Generation role mismatch')
    require(meta['usage']['total_tokens'] == meta['usage']['prompt_tokens'] + meta['usage']['completion_tokens'], 'Token sum mismatch')
    require(('internal_signals' in meta) == scored, 'Unused/missing internal computation')
    require(meta['internal_cost']['additional_backbone_forward_passes'] == 0, 'Unexpected extra backbone forward')
    if scored:
        signal = meta['internal_signals']
        require(signal['backend'] == 'hf_qwen35_generation_raw_bf16_v1', 'NLL path mismatch')
        require(signal['dtype'] == 'bfloat16', 'NLL dtype mismatch')
        require(signal['model_revision'] == profile['model_revision'], 'NLL model revision mismatch')
        nll = record['token_nll']
        details = signal['details']
        if signal['status'] == 'ok':
            require(nll and all(math.isfinite(v) and v >= 0 for v in nll), 'Invalid raw token NLL')
            require(math.isclose(signal['global_uncertainty'], statistics.mean(nll), abs_tol=1e-8), 'Global token range/reduction mismatch')
            values = {}
            for span in details['field_spans']:
                values[span['field']] = statistics.mean(nll[i] for i in span['token_indices'])
                require(math.isclose(values[span['field']], span['uncertainty'], abs_tol=1e-8), 'Field NLL mismatch')
            require(math.isclose(max(values.values()), signal['local_uncertainty'], abs_tol=1e-8), 'Local NLL mismatch')
    return record


def audit_episode(ep, entry, profile, config, memory, goals):
    result = read(ep / 'complete.json')
    traces = rows(ep / 'steps.jsonl')
    initial = read(ep / 'initial.json')
    require(result['steps_sha256'] == file_hash(ep / 'steps.jsonl'), 'Episode steps hash changed')
    task, policy, seed = entry['task_id'], entry['policy'], entry['seed']
    require((result['task_id'], result['policy'], result['rollout_seed']) == (task, policy, seed), 'Wrong task/policy/seed')
    require(fingerprint(initial['initial']) == goals[task]['initial_sha256'], 'Initial task state differs across policies')
    require(initial['goal'] == goals[task]['goal'], 'Native goal changed')
    require(0 < len(traces) <= 50, 'Invalid formal horizon')
    require(result['formal_steps'] == len(traces), 'Formal step accounting mismatch')
    gate = RuleGate(config, policy=policy)
    budget = profile['wm_invocation_budgets'][policy]
    previous = initial['initial']
    past = []
    cost = Counter(initializations=1, formal_steps=len(traces))
    counts = Counter()
    seconds = Counter()
    examples = []
    example_tags_seen = Counter()
    complement = []
    patterns = Counter()
    for step, t in enumerate(traces):
        require(t['step'] == step and t['task_id'] == task and t['policy'] == policy, 'Episode step identity mismatch')
        require(not previous['done'], 'Step after termination')
        require(t['observation'] == previous['observation'] and t['admissible_commands'] == previous['admissible_commands'], 'Current public state differs from previous actual transition')
        require(t['goal'] == initial['goal'], 'Goal changed')
        prompt = draft_messages(t['goal'], t['observation'], past, t['admissible_commands'])
        record = audit_generation(t['metadata'], prompt, 'draft', request_seed(task, step, 'draft', seed), profile, gate.needs_internal)
        require(record['response'] == t['draft_output'], 'Draft text differs from generation')
        require(fields(t['draft_output'])[2] == t['draft_action'], 'Draft action parsing mismatch')
        require(t['metadata']['config_sha256'] == memory.model_fingerprint, 'Model config signature changed')
        ext_values = dict(t['resolved_external'])
        for key in ['evidence_ids', 'matched_experience_ids', 'diagnostics']:
            ext_values[key] = tuple(ext_values[key])
        resolved = ExternalInput(**ext_values)
        if gate.needs_external:
            query = dict(task_id=task, episode_id=str(ep), goal=t['goal'], observation=t['observation'], draft_action=t['draft_action'],
                         commands=t['admissible_commands'], available_at_ns=t['external']['query_at_ns'])
            external, rederived = memory.evaluate(query)
            require(canonical(asdict(rederived)) == canonical(asdict(resolved)), 'External evidence/source/tri-state mismatch')
            # Legacy BM25 sums a Python set: cross-process term order can change
            # the final float by a few ulps. IDs/order/applicability remain exact.
            old_ranked = t['external']['retrieved_cards']
            require(len(external['retrieved_cards']) == len(old_ranked), 'Retrieved card count changed')
            for aa, bb in zip(external['retrieved_cards'], old_ranked):
                require(without(aa, {'score'}) == without(bb, {'score'}), 'Retrieved identity/order/applicability changed')
                require(math.isclose(aa['score'], bb['score'], rel_tol=1e-12, abs_tol=1e-12), 'BM25 numeric difference exceeds float-summation tolerance')
            omitted = {'retrieval_latency', 'judge_latency', 'retrieved_cards'}
            require(without(external, omitted) == without(t['external'], omitted), 'External applicability/evidence changed')
            require(query['available_at_ns'] < t['decided_at_ns'], 'External signal from after decision')
            seconds['retrieval'] += t['external']['retrieval_latency']
            seconds['external_rule_judge'] += t['external']['judge_latency']
            counts['coverage'] += external['coverage'] == 'true'
            counts['conflict'] += external['conflict'] == 'true'
            counts['sufficiency'] += external['sufficiency'] == 'true'
        else:
            require(t['external'] is None and resolved == ExternalInput(), 'Unused external computation')
        context = GateInput(t['gate']['decision_id'], budget, t['metadata'].get('internal_signals'), resolved, budget, t['metadata']['decision_format_valid'])
        recomputed = gate.decide(context)
        require(without(recomputed, {'gate_latency'}) == without(t['gate'], {'gate_latency'}), 'Gate replay differs from frozen rules')
        require(t['decided_at_ns'] < t['executed_at_ns'], 'Formal future feedback reached current Gate')
        require(t['draft_admissible'] == (t['draft_action'] in t['admissible_commands']), 'Draft admissibility mismatch')
        require(t['final_admissible'] == (t['final_action'] in t['admissible_commands']), 'Final admissibility mismatch')
        simulation = t['simulation']
        require(bool(simulation is not None) == t['gate']['call_wm'], 'Call/skip does not match actual simulation branch')
        counts['budget_exhausted'] += t['gate']['verification_unavailable']
        counts['draft_format_invalid'] += not t['metadata']['decision_format_valid']
        counts['draft_nonadmissible'] += not t['draft_admissible']
        counts['final_nonadmissible'] += not t['final_admissible']
        counts['nothing_happens'] += t['actual']['observation'].strip() == 'Nothing happens.'
        counts['internal_missing'] += gate.needs_internal and t['metadata']['internal_signals']['status'] != 'ok'
        patterns[t['final_action']] += 1
        if step:
            prior = traces[step - 1]
            counts['consecutive_same_action_and_feedback'] += (t['final_action'] == prior['final_action']
                and t['actual']['observation'] == prior['actual']['observation'])
        seconds['gate'] += t['gate']['gate_latency']
        if simulation:
            budget -= 1
            counts['lookahead_calls'] += 1
            require(simulation['source'] == 'ground_truth_environment' and simulation['benchmark'] == 'alfworld', 'Learned/other-domain simulator')
            require(simulation['formal_unchanged'] is True, 'Formal state pollution')
            require(t['decided_at_ns'] < simulation['available_at_ns'] < t['executed_at_ns'], 'Simulation chronology')
            sc = simulation['simulation_cost']
            for k in ['initializations', 'replay_steps', 'candidate_steps', 'formal_inspections']:
                cost[k] += sc[k]
            seconds['lookahead'] += simulation['simulation_latency']
            for k in ['initialization_seconds', 'replay_seconds', 'candidate_seconds', 'cleanup_seconds']:
                seconds[k] += sc.get(k, 0)
            if simulation['prediction_available']:
                counts['verified_lookaheads'] += 1
                require(simulation['worker_pid'] != simulation['formal_pid'] == initial['formal_pid'], 'Lookahead shares formal process')
                require(simulation['verified_prefix_steps'] == step, 'Replay omitted formal actions')
                require(sc['initializations'] == 1 and sc['replay_steps'] == step and sc['candidate_steps'] == 1, 'Simulation depth/replay accounting mismatch')
                require(simulation['candidate_state_sha256'] == fingerprint(simulation['candidate_transition']), 'Candidate state receipt mismatch')
                rp = review_messages(t['goal'], t['observation'], past, t['admissible_commands'], t['draft_action'], simulation)
                audit_generation(t['review_metadata'], rp, 'review', request_seed(task, step, 'review', seed), profile, False)
                parsed = parse_review(t['review']['raw'], t['draft_action'])
                require(parsed == t['review']['parsed'], 'Reviewer parser differs')
                require(parsed['final_action'] == t['final_action'], 'Final action differs from parsed review')
                counts['reviewer_calls'] += 1
                counts['review_decision_' + parsed['decision'].lower()] += 1
                counts['review_format_invalid'] += parsed['decision'] == 'INVALID'
                counts['review_length_limit'] += t['review_metadata']['finish_reason'] == 'length'
                counts['revisions'] += t['final_action'] != t['draft_action']
                counts['revision_admissible'] += t['final_action'] != t['draft_action'] and t['final_admissible']
                counts['revision_nonadmissible'] += t['final_action'] != t['draft_action'] and not t['final_admissible']
                counts['revision_nothing_happens'] += t['final_action'] != t['draft_action'] and t['actual']['observation'].strip() == 'Nothing happens.'
                counts['replace_label_unchanged_action'] += parsed['decision'] == 'REPLACE' and t['final_action'] == t['draft_action']
                if t['final_action'] == t['draft_action']:
                    require(t['kept_draft_transition_match'] is True and t['actual'] == simulation['candidate_transition'], 'KEEP/normal transition mismatch')
                    counts['keep_transition_matches'] += 1
                else:
                    # An immediate simulator-vs-formal difference is not a long-horizon causal utility estimate.
                    counts['immediate_success_gain'] += int(t['actual']['success']) > int(simulation['predicted_success'])
                    counts['immediate_success_loss'] += int(t['actual']['success']) < int(simulation['predicted_success'])
            else:
                counts['rejected_lookaheads'] += 1
                require(t['review_metadata'] is None and t['final_action'] == t['draft_action'], 'Unverified simulator feedback changed action')
        else:
            require(t['review'] is None and t['review_metadata'] is None and t['final_action'] == t['draft_action'], 'Skip incurred reviewer/action modification')
        require(t['budget_after'] == budget and budget >= 0, 'Budget accounting mismatch')
        require(t['actual']['formal_steps'] == step + 1, 'Trial consumed formal step counter')
        require(t['actual']['action'] == t['final_action'], 'Environment executed a different action')
        if policy == 'full':
            internal = RuleGate(config, policy='internal').decide(context)
            if t['gate']['call_wm'] != internal['call_wm']:
                complement.append(dict(task_id=task, step=step, rollout_seed=seed, action=t['draft_action'], final_action=t['final_action'],
                    full_call=t['gate']['call_wm'], internal_call_on_same_full_state=internal['call_wm'], local=t['gate']['local_uncertainty'],
                    global_nll=t['gate']['global_uncertainty'], evidence_ids=list(resolved.evidence_ids), final_admissible=t['final_admissible'],
                    immediate_success=t['actual']['success'], causal_long_horizon_utility='unknown'))
        for role, meta in [('actor', t['metadata']), ('reviewer', t['review_metadata'])]:
            if meta:
                cost[role + '_calls'] += 1
                for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                    cost[key] += meta['usage'][key]
                    cost[role + '_' + key] += meta['usage'][key]
                ic = meta['internal_cost']
                seconds[role + '_generation'] += ic['generation_seconds']
                seconds['internal_extraction'] += ic['extraction_seconds']
                cost['peak_allocated_bytes'] = max(cost['peak_allocated_bytes'], ic['peak_allocated_bytes'])
        tags = []
        if step < 2:
            tags.append('initial_step')
        if t['final_action'] != t['draft_action']:
            tags.append('reviewer_replacement')
        if t['review'] and t['review']['parsed']['decision'] == 'INVALID':
            tags.append('review_format_failure')
        if not t['final_admissible']:
            tags.append('nonadmissible_final_action')
        if t['external'] and t['external']['coverage'] == 'true':
            tags.append('external_evidence')
        selected_tags = [tag for tag in tags if example_tags_seen[tag] < 2]
        if selected_tags:
            example_tags_seen.update(selected_tags)
            examples.append(dict(task_id=task, policy=policy, step=step, categories=selected_tags, observation=t['observation'], draft=t['draft_action'],
                local=t['gate']['local_uncertainty'], global_nll=t['gate']['global_uncertainty'], gate=t['gate']['reason'],
                external=t['resolved_external'], trial_observation=simulation.get('predicted_observation') if simulation else None,
                reviewer=t['review'], final=t['final_action'], feedback=t['actual']['observation'], success=t['actual']['success']))
        past.append(dict(observation=t['observation'], action=t['final_action']))
        previous = t['actual']
    require(previous['done'] is True and result['done'] is True, 'Incomplete episode scored')
    require(result['success'] == previous['success'] and result['score_100_won'] == 100 * previous['success'], 'Wrong ALFWorld success/score extraction')
    require(result['horizon_truncated'] == previous['truncated'], 'Truncation differs')
    require(result['tokens'] == cost['total_tokens'], 'Episode tokens mismatch')
    for key, ck in [('wm_calls', 'lookahead_calls'), ('verified_lookaheads', 'verified_lookaheads'), ('reviewer_calls', 'reviewer_calls')]:
        require(result[key] == counts[ck], 'Episode calls mismatch: ' + key)
    require(result['retrieval_calls'] == len(traces) * gate.needs_external and result['internal_extractions'] == len(traces) * gate.needs_internal, 'Signal compute charged to unused baseline')
    cost['transition_queries'] = cost['initializations'] + cost['formal_steps'] + cost['replay_steps'] + cost['candidate_steps']
    cost['api_queries'] = cost['transition_queries'] + cost['formal_inspections']
    result.update(audit_counts=dict(counts), costs=dict(cost), timings=dict(seconds), top_actions=patterns.most_common(5))
    return result, examples, complement


def analyze(run, output, partial=False):
    manifest = read(run / 'manifest.json')
    source = Path(manifest['source_run'])
    original = verify(source)
    transitive = {}
    for name in ['evaluation/independent_dev.py', 'experience/collection.py', 'experience/splits.py', 'experience/memory.py',
                 'gates/generation.py', 'reflection/replay_review.py', 'worldmodel/resources.py',
                 'internal_signals/decision.py', 'internal_signals/token_nll.py', 'internal_signals/spanuq_adapter.py']:
        old_bytes = subprocess.check_output(['git', 'show', original['source_commit'] + ':' + name], cwd=ROOT)
        digest = hashlib.sha256(old_bytes).hexdigest()
        require(file_hash(ROOT / name) == digest, 'Shared transitive method source changed: ' + name)
        transitive[name] = digest
    for name, digest in manifest['new_source_hashes'].items():
        require(file_hash(ROOT / name) == digest, 'Continuation scheduling source changed: ' + name)
    p = manifest['profile']
    require(p == original['profile'], 'Parallel profile differs')
    split = original['split']
    groups = [{r['goal_group'] for r in split[k]} for k in ['memory_tasks', 'calibration_tasks', 'evaluation_tasks', 'final_valid_seen']]
    require(all(not (groups[i] & groups[j]) for i in range(4) for j in range(i)), 'Goal split leakage')
    require(len(manifest['schedule']) == 15 and len({(e['task_id'], e['policy'], e['seed']) for e in manifest['schedule']}) == 15, 'Schedule duplicates')
    config = read(source / 'gate_config.json')
    calibration_records = rows(source / 'calibration_steps.jsonl')
    for r in calibration_records:
        meta = r['metadata']
        audit_generation(meta, meta['generation_record']['messages'], 'draft', request_seed(r['task_id'], r['step'], 'draft', 0), p, True)
    require(len(calibration_records) == 48, 'Calibration cohort changed')
    memory = AlfExperienceMemory(read(ROOT / 'outputs/alfworld_memory_v2/snapshot.json'), split)
    memory.model_fingerprint = read(source / 'model.json')['config_sha256']
    goals = {r['task_id']: r for r in read(ROOT / 'docs/alfworld_pilot/native_goal_identity.json')}
    episodes, examples, complement, pending = [], [], [], []
    for entry in manifest['schedule']:
        ep = Path(entry['reused_source']) if 'reused_source' in entry else run / ('worker' + str(entry['worker'])) / 'episodes' / entry['task_id'] / entry['policy']
        if not (ep / 'complete.json').exists():
            pending.append(entry)
            continue
        result, ex, changes = audit_episode(ep, entry, p, config, memory, goals)
        result['raw_episode_path'] = str(ep.relative_to(ROOT))
        episodes.append(result)
        examples.extend(ex)
        complement.extend(changes)
    require(partial or not pending, 'Only completed runs may receive final results')
    by_policy = defaultdict(list)
    for result in episodes:
        by_policy[result['policy']].append(result)
    table = []
    for policy in p['policies']:
        values = by_policy[policy]
        if not values:
            continue
        summary = dict(policy=policy, episodes=len(values), successes=sum(v['success'] for v in values), success_rate=statistics.mean(v['success'] for v in values),
            formal_steps=sum(v['formal_steps'] for v in values), total_tokens=sum(v['tokens'] for v in values),
            wm_calls=sum(v['wm_calls'] for v in values), seconds=sum(v['total_seconds'] for v in values))
        summary['audit_counts'] = dict(sum((Counter(v['audit_counts']) for v in values), Counter()))
        summary['costs'] = dict(sum((Counter(without(v['costs'], {'peak_allocated_bytes'})) for v in values), Counter()))
        summary['peak_allocated_bytes'] = max(v['costs']['peak_allocated_bytes'] for v in values)
        summary['timings'] = dict(sum((Counter(v['timings']) for v in values), Counter()))
        table.append(summary)
    paired = []
    indexed = {(v['task_id'], v['rollout_seed'], v['policy']): v for v in episodes}
    for v in episodes:
        reference = indexed.get((v['task_id'], v['rollout_seed'], 'never'))
        if reference and v['policy'] != 'never':
            paired.append(dict(task_id=v['task_id'], seed=v['rollout_seed'], policy=v['policy'], reference='never',
                delta_success=int(v['success']) - int(reference['success']),
                delta_tokens=v['tokens'] - reference['tokens'], delta_formal_steps=v['formal_steps'] - reference['formal_steps'],
                note='Whole-policy paired task comparison; not same-state causal per-call utility'))
    trajectory_comparisons = []
    for policy, reference_policy in [('full', 'internal'), ('experience', 'never')]:
        for v in by_policy[policy]:
            ref = indexed.get((v['task_id'], v['rollout_seed'], reference_policy))
            if not ref:
                continue
            aa = rows(ROOT / v['raw_episode_path'] / 'steps.jsonl')
            bb = rows(ROOT / ref['raw_episode_path'] / 'steps.jsonl')
            comparable = len(aa) == len(bb)
            trajectory_comparisons.append(dict(task_id=v['task_id'], policy=policy, reference=reference_policy,
                equal_length=comparable,
                same_formal_actions=comparable and all(a['final_action'] == b['final_action'] for a, b in zip(aa, bb)),
                same_public_and_fact_transitions=comparable and all(a['actual'] == b['actual'] for a, b in zip(aa, bb)),
                same_draft_token_ids=comparable and all(a['metadata']['generation_record']['response_ids'] == b['metadata']['generation_record']['response_ids'] for a, b in zip(aa, bb)),
                same_call_sequence=comparable and all(a['gate']['call_wm'] == b['gate']['call_wm'] for a, b in zip(aa, bb))))
    for base in [source] + sorted(run.glob('worker*')):
        if not base.is_dir() or not (base / 'ledger.json').exists():
            continue
        ledger = read(base / 'ledger.json')
        generation = rows(base / 'generation_calls.jsonl') if (base / 'generation_calls.jsonl').exists() else []
        require(len(generation) == ledger['model_calls'] or partial, 'Generation request count differs from ledger')
        require(sum(g['metadata']['usage']['total_tokens'] for g in generation) == ledger['tokens'] or partial, 'Generation tokens differ from ledger')
        if base != source and not pending:
            selected = [v for v in episodes if str(base.relative_to(ROOT)) + '/' in v['raw_episode_path']]
            require(sum(v['costs']['transition_queries'] for v in selected) == ledger['transition_queries'], 'Native transition query ledger mismatch')
            require(sum(v['costs']['api_queries'] for v in selected) == ledger['audit_capture_bundles'], 'Native reply/inspection ledger mismatch')
    generation_costs = defaultdict(Counter)
    model_loads = []
    bases = [('original', source)] + [('continuation', b) for b in sorted(run.glob('worker*')) if b.is_dir()]
    for failure in manifest.get('failed_prior_runs', []):
        bases.extend(('failed_scheduler', b) for b in sorted(Path(failure['run']).glob('worker*')) if b.is_dir())
    for attempt, base in bases:
        if (base / 'model.json').exists():
            receipt = read(base / 'model.json')
            require(receipt['loading_info'] == dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[]), 'Strict model loading had discrepancies')
            require(receipt['config_sha256'] == memory.model_fingerprint, 'Worker model config differs')
            model_loads.append(dict(attempt=attempt, path=str((base / 'model.json').relative_to(ROOT)), seconds=receipt['load_seconds']))
        journal = rows(base / 'generation_calls.jsonl') if (base / 'generation_calls.jsonl').exists() else []
        for call in journal:
            where = call['where']
            if attempt == 'failed_scheduler':
                category = 'failed_scheduler_overhead'
            elif where.startswith('calibration/'):
                category = 'calibration'
            elif where in ['scored_unscored_parity', 'repeat_nll_parity']:
                category = 'numerical_validation'
            elif attempt == 'original' and not (source / 'episodes' / where.split('/')[1] / where.split('/')[0] / 'complete.json').exists():
                category = 'interrupted_partial_episode'
            else:
                category = 'completed_evaluation'
            group = generation_costs[category]
            group['model_calls'] += 1
            group[call['role'] + '_calls'] += 1
            group['generation_seconds'] += call['metadata']['internal_cost']['generation_seconds']
            group['internal_extraction_seconds'] += call['metadata']['internal_cost']['extraction_seconds']
            for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                group[key] += call['metadata']['usage'][key]
    if not pending:
        require(generation_costs['completed_evaluation']['total_tokens'] == sum(v['tokens'] for v in episodes), 'Completed evaluation cost scope mismatch')
        require(generation_costs['completed_evaluation']['model_calls'] == sum(v['formal_steps'] + v['reviewer_calls'] for v in episodes), 'Completed evaluation call scope mismatch')
        complete = read(run / 'complete.json')
        for final_key, call_key in [('tokens', 'total_tokens'), ('model_calls', 'model_calls')]:
            require(complete['total_resource_spent_including_initial_partial'][final_key] == sum(c[call_key] for c in generation_costs.values()), 'All-attempt model accounting mismatch')
    artifact = dict(status='partial' if pending else 'complete', episodes=episodes, policies=table, pending=pending,
        external_changes_on_same_full_states=complement, paired_against_never=paired, calibration=read(source / 'calibration.json'),
        matched_task_trajectory_comparisons=trajectory_comparisons,
        numerical_validation=without(read(source / 'numerical_validation.json'), {'internal_signals'}),
        freeze=read(source / 'threshold_freeze.json'), config=p, source_commit=original['source_commit'],
        continuation_commit=manifest['source_git_commit'], whole_run_resource_receipt=read(run / 'complete.json') if (run / 'complete.json').exists() else None,
        all_attempt_generation_costs={k: dict(v) for k, v in generation_costs.items()}, model_setup_attempts=model_loads,
        failed_attempt_receipt_limitations=read(ROOT / 'docs/alfworld_pilot/parallel_engineering_repair.json'),
        real_calibration_numeric_checks=len(calibration_records),
        unchanged_shared_transitive_source_hashes=transitive,
        limits=['Three development games, one rollout seed, no significance/generalization claim',
                'Public admissible commands provided equally to all five policies; not exact paper actor prompt',
                'Shared legacy STRUCTURE includes ScienceWorld word in Action placeholder despite ALF-native guidance; retained/frozen across all policies, a prompt limitation',
                'Native inspect returns cached last state receipt; independent processes/fact replay tested, not full process/RNG serialization',
                'Immediate action/feedback changes are not long-horizon causal utility; no controlled continuation experiment',
                'Always50 vs selective5 maximum calls, not matched-budget comparison',
                'Extra Reviewer computation and feedback information are not separated by a no-feedback Reviewer control',
                'Failed scheduler has bounded native receipt uncertainty; raw failed attempt is not assigned task score'])
    output.mkdir(parents=True, exist_ok=True)
    atomic(output / 'summary.json', artifact)
    atomic(output / 'examples.json', examples)
    flat = [without(v, {'audit_counts', 'costs', 'timings', 'top_actions'}) for v in episodes]
    if flat:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(flat[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(flat)
        (output / 'episodes.csv').write_text(buffer.getvalue())
    print(json.dumps(dict(status=artifact['status'], completed=len(episodes), policies=[without(r, {'audit_counts', 'costs', 'timings'}) for r in table]), indent=2))
    return artifact


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    analyze(Path(args.run).absolute(), Path(args.output).absolute(), args.partial)
