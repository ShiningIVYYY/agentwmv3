import json
from pathlib import Path
import requests

from reflection.controller import ReflectionController
from .backends import OracleWorldModel, Word2WorldModel, load_config, PredictionError
from .contexts import initial_context, alf_action, checkpoint_observation


def review_protocol(args):
    selected = getattr(args, 'reflection_review_protocol', 'auto')
    if selected != 'auto':
        return selected
    return 'ee-conservative-v2' if getattr(args, 'model', '').startswith('ee-') else 'json'


def make_controller(args, env, row, task, initial, factory, agent, root, oracle=None):
    backend = getattr(args, 'wm_backend', 'none')
    if backend == 'none':
        return None
    if backend == 'oracle':
        wm = oracle if oracle is not None else OracleWorldModel(factory, initial)
    else:
        path = args.wm_config or str(root / 'configs/worldmodel' / (env + '.json'))
        config = load_config(path, root, env)
        response = requests.get(config['url'].rstrip('/') + '/v1/models', timeout=15)
        response.raise_for_status()
        models = response.json()['data']
        matches = [m for m in models if m['id'] == config['served_model_name'] and
                   Path(m.get('root', '')).resolve() == Path(config['checkpoint']).resolve()]
        if not matches:
            raise PredictionError('World model service name/weight path mismatch')
        if matches[0].get('max_model_len') != config['max_model_len']:
            raise PredictionError('World model service context window differs from the fixed config')
        def count(messages):
            response = requests.post(args.tokenizer_url.rstrip('/') + '/count', json=dict(
                path=config['checkpoint'], messages=messages, disable_thinking=False), timeout=60)
            response.raise_for_status()
            result = response.json()
            if 'error' in result:
                raise PredictionError(str(result['error']))
            return result['count']
        wm = Word2WorldModel(config, initial_context(config, row, task, initial), count,
            to_action=alf_action if env == 'alfworld' else lambda x: x,
            from_observation=lambda text: checkpoint_observation(env, text, task))
    def record(stage, step, data):
        if agent is not None and hasattr(agent, 'out'):
            import time
            if stage == 'predict_start':
                progress = agent.out / 'progress.json'
                temporary = progress.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(dict(task_id=row['task_id'], step=step,
                    state='waiting_for_world_model', round=data['round'],
                    candidate=data['candidate'], updated=time.time())))
                temporary.replace(progress)
            with open(agent.out / 'reflection_events.jsonl', 'a') as f:
                f.write(json.dumps(dict(task_id=row['task_id'], step=step, stage=stage,
                    updated=time.time(), **data), ensure_ascii=False) + '\n')
    protocol = review_protocol(args)
    return ReflectionController(wm, agent, env, task, row['task_id'], args.reflection_max_rounds, record, protocol)


def select(controller, candidate, output, observation, steps, actions, step_number):
    if controller is None:
        return candidate, output, {}
    chosen, log = controller.select(candidate, observation, steps, actions, step_number)
    effective = output
    if chosen != candidate:
        chosen_reviews = [r for r in log['reviews'] if r['decision']['decision'] == 'replace' and
                          r['decision']['action'] == chosen]
        reason = chosen_reviews[-1]['decision']['reason']
        if getattr(controller, 'review_protocol', 'json') == 'agentwm-copied-v1' and controller.env == 'scienceworld':
            effective = chosen_reviews[-1]['execution_output']
        elif getattr(controller, 'review_protocol', 'json') in ('ee-conservative-v2', 'agentwm-copied-v1') and controller.env != 'alfworld':
            effective = chosen_reviews[-1]['output']
        elif getattr(controller, 'review_protocol', 'json') == 'ee-conservative-v2':
            effective = chosen_reviews[-1]['output']
        elif controller.env == 'scienceworld':
            effective = 'Thought: ' + reason + '\nAction: ' + chosen
        else:
            effective = '<think>' + reason + '</think>\n<action>' + chosen + '</action>'
    return chosen, effective, dict(initial_output=output, reflection=log)


def commit(controller, action, actual, metadata):
    if controller is not None:
        controller.commit(action, actual, metadata['reflection'])


def metrics(steps):
    records = [s['reflection'] for s in steps if 'reflection' in s]
    if not records:
        return {}
    rounds = sum(r['rounds'] for r in records)
    preds = [p for r in records for p in r['predictions'] if not p.get('cached', False)]
    usage = [v['generation'].get('usage') or {} for r in records for v in r['reviews']]
    replans = [v['replan'] for r in records for v in r['reviews'] if 'replan' in v]
    replan_usage = [v['generation'].get('usage') or {} for v in replans]
    wm_usage = [p.get('usage') or {} for p in preds]
    return dict(reflection_metrics=dict(steps=len(records), reflection_calls=rounds,
        wm_calls=len(preds), mean_rounds=rounds / len(records),
        wm_truncated_predictions=sum(p.get('observation_complete', True) is False and not p.get('empty_output', False) for p in preds),
        wm_empty_predictions=sum(p.get('empty_output', False) for p in preds),
        replacement_count=sum(r['replacements'] for r in records),
        invalid_reviews=sum(r['invalid_reviews'] for r in records),
        valid_review_calls=sum(v['decision']['decision']!='invalid' for r in records for v in r['reviews']),
        review_format_errors=sum(v.get('error_kind')=='invalid_format' for r in records for v in r['reviews']),
        invalid_replacement_reviews=sum(v.get('error_kind')=='invalid_action' for r in records for v in r['reviews']),
        guard_blocked_replacements=sum(r.get('guard_blocked_replacements', 0) for r in records),
        original_defects_detected=sum(bool(r.get('original_defect')) for r in records),
        review_budget_skips=sum(r.get('budget_skip', False) for r in records),
        changed_step_count=sum(r['initial_action'] != r['final_action'] for r in records),
        round_limit_count=sum(r['hit_round_limit'] for r in records),
        prediction_exact_matches=sum(r['prediction_exact_match'] for r in records),
        oracle_commit_verified=sum(r['backend'] == 'oracle' for r in records),
        wm_seconds=sum(p['seconds'] for p in preds),
        reflection_tokens=sum(u.get('total_tokens', 0) for u in usage),
        replan_calls=len(replans), replan_tokens=sum(u.get('total_tokens', 0) for u in replan_usage),
        review_replan_tokens=sum(u.get('total_tokens', 0) for u in usage + replan_usage),
        wm_tokens=sum(u.get('total_tokens', 0) for u in wm_usage)))


def audit_settings(args, root):
    """Return immutable feature settings only when explicitly enabled."""
    if args.wm_backend == 'none':
        if args.wm_config:
            raise ValueError('--wm-config requires --wm-backend word2world')
        return {}
    import hashlib
    files = sorted((root / 'worldmodel').glob('*.py')) + sorted((root / 'reflection').glob('*'))
    files = [p for p in files if p.is_file()]
    feature = dict(protocol='wm-reflection-v1', backend=args.wm_backend,
        max_rounds=args.reflection_max_rounds, final_round_policy='predict-final-replacement-and-execute',
        feedback_fields=['observation'], oracle_policy='isolated-reset-verified-prefix-replay',
        history_policy='commit-real-observations-only', sources={str(p.relative_to(root)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    feature['science_ambiguity_policy'] = 'stable-label-order-with-native-number-translation'
    feature['reflection_review_protocol'] = review_protocol(args)
    feature['selection_policy'] = 'evidence-gated-repair-v1' if getattr(args,'model','').startswith('ee-') else 'legacy-json'
    if getattr(args,'model','').startswith('ee-'):
        feature['repair_evidence'] = 'explicit-failure/current-action-unavailable/three-identical-feedback-repeats'
        feature['repair_prediction'] = 'same-real-prefix/complete-available/nonfailure/different-feedback'
        feature['science_review_context'] = 'complete-baseline-training-dialogue/trim-real-history-only'
    if review_protocol(args) == 'agentwm-copied-v1':
        from reflection.agentwm import POLICY
        for key in ('repair_evidence', 'repair_prediction', 'science_review_context'):
            feature.pop(key, None)
        feature.update(selection_policy=POLICY, prompt_source='local-copied-AgentWM-templates',
            prompt_provenance=json.loads((root / 'reflection/agentwm_prompt_provenance.json').read_text()),
            feedback_fields=['observation', 'reward', 'done', 'success', 'score'] if args.wm_backend == 'oracle' else ['observation', 'predicted_success'],
            unknown_numeric_feedback='None/no-real-environment-label-lookup',
            science_review_context='copied-critique-template/recent-five-real-steps',
            science_replan='copied-guarded-replan-template/separate-agent-call/current-real-state/WM-guard',
            review_output_format='JSON-schema-constrained-ALFWorld-and-ScienceWorld-review-only',
            original_actor_context='unchanged/full-science-training-dialogue',
            final_round_policy='direct-final-prediction/science-verified-replan-or-cached-original')
    if args.env == 'scienceworld' and args.wm_backend == 'oracle':
        feature['science_exit_comparison'] = 'unordered-full-door-descriptions/native-agent-text-preserved'
        feature['oracle_policy'] = 'live-jvm-heap-checkpoint/rollback-rejected/adopt-final-state-once'
        feature['snapshot_helper'] = json.loads((root / 'outputs/wm_reflection_setup/science_snapshot/provenance.json').read_text())
        feature['sources']['worldmodel/HeapSnapshot.java'] = hashlib.sha256((root / 'worldmodel/HeapSnapshot.java').read_bytes()).hexdigest()
    feature['temporary_directory_policy'] = 'owned-per-worker/on-experiment-volume'
    if args.env == 'alfworld':
        feature['pddl_resource_policy'] = 'two-exclusively-owned-native-engines/cleanup-and-reuse/no-early-dlclose'
    if args.wm_backend == 'word2world':
        feature['empty_prediction_policy'] = 'explicit-missing-evidence/reflect-with-actual-current-state'
        path = args.wm_config or str(root / 'configs/worldmodel' / (args.env + '.json'))
        feature['world_model'] = load_config(path, root, args.env)
        feature['config_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {'wm_reflection': feature}
