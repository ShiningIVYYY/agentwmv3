#!/usr/bin/env python3
"""Auditable, resumable environment evaluation. Run ONLY in the lab repository."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
import time
import traceback

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldmodel import runtime as wm_runtime
from worldmodel.contexts import alf_wm_observation, web_wm_observation, ScienceActionView
VERL = ROOT / 'external/verl-agent/agent_system'
PROTOCOL = 'audited-v1'
WEB_PROTOCOL = 'webshop-index-v2'
SCIENCE_PROTOCOL = 'scienceworld-context-v2'


def protocol_for_env(env):
    return {'webshop': WEB_PROTOCOL, 'scienceworld': SCIENCE_PROTOCOL}.get(env, PROTOCOL)


def validate_science_tasks(rows, games, frozen=None):
    """Validate the official 200 ordered IDs against the native flattened map."""
    if len(rows) != 200 or len({r['item_id'] for r in rows}) != 200:
        raise RuntimeError('ScienceWorld requires exactly 200 unique official test IDs')
    tasks = []
    for r in rows:
        if not re.fullmatch(r'sciworld_\d+', r['item_id']):
            raise RuntimeError('Invalid ScienceWorld task ID')
        index = int(r['item_id'].rsplit('_', 1)[1])
        if not 0 <= index < len(games):
            raise RuntimeError('ScienceWorld task index out of range')
        tasks.append(dict(task_id=r['item_id'], index=index, **games[index]))
    if frozen is not None:
        fields = ('task_id', 'index', 'taskName', 'variationIdx')
        if [{k:r[k] for k in fields} for r in tasks] != [{k:r[k] for k in fields} for r in frozen]:
            raise RuntimeError('ScienceWorld task mapping changed')
    return tasks


def science_messages(prelude, task, observation, steps, keep):
    """Native alternating dialogue; put the goal only in its first user turn."""
    recent = steps[-keep:] if keep else []
    messages = list(prelude)
    for i, s in enumerate(recent):
        messages += [dict(role='user', content=(task + '\n' if i == 0 else '') + s['observation']),
                     dict(role='assistant', content=s['output'])]
    messages.append(dict(role='user', content=('' if recent else task + '\n') + observation))
    return messages


def anchored_science_messages(prelude, task, observation, steps, keep):
    """Separate the training example and keep the real goal next to this state."""
    recent = steps[-keep:] if keep else []
    messages = list(prelude)
    boundary = ('The example episode above is complete. Start a separate task. '
                'Use the observations from this task to choose actions.\n\n')
    for i, record in enumerate(recent):
        content = record['observation']
        if i == 0:
            content = boundary + task + '\n' + content
        messages += [dict(role='user', content=content),
                     dict(role='assistant', content=record['output'])]
    content = 'Current task:\n' + task + '\n\nCurrent observation:\n' + observation
    if not recent:
        content = boundary + content
    messages.append(dict(role='user', content=content))
    return messages


def bounded_science(agent, prelude, task, observation, steps):
    if getattr(agent, "internal_actor", None) is not None:
        from internal_signals.decision import bounded
        return bounded(agent, task, observation, steps)
    # Full history first. Drop whole oldest observation/assistant pairs only
    # when the exact model tokenizer says the prompt exceeds the paper budget.
    for keep in range(len(steps), -1, -1):
        builder = anchored_science_messages if getattr(getattr(agent, 'args', None), 'science_context', None) == 'task-anchored-v1' else science_messages
        messages = builder(prelude, task, observation, steps, keep)
        if agent.count(messages) <= 4096:
            return messages, keep
    raise RuntimeError('ScienceWorld one-shot/current observation exceed 4096 tokens')


def validate_web_index(searcher, catalog_ids):
    """Fail closed when the Lucene index and loaded catalog do not match."""
    catalog_ids = set(catalog_ids)
    if searcher.num_docs != len(catalog_ids):
        raise RuntimeError('WebShop index/catalog size mismatch: %d vs %d' %
                           (searcher.num_docs, len(catalog_ids)))
    indexed_ids = []
    for i in range(searcher.num_docs):
        indexed_ids.append(json.loads(searcher.doc(i).raw())['id'])
    if len(set(indexed_ids)) != len(indexed_ids) or set(indexed_ids) != catalog_ids:
        raise RuntimeError('WebShop index/catalog ASIN mismatch')
    return hashlib.sha256('\n'.join(sorted(indexed_ids)).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def http(method, url, **kwargs):
    # Retrying a non-idempotent environment step could execute an action twice.
    response = requests.request(method, url, timeout=kwargs.pop('timeout', 180), **kwargs)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get('error'):
        raise RuntimeError(str(data['error']))
    return data


def history_text(steps, count=2):
    return '\n'.join(
        "[Observation %d: '%s', Action %d: '%s']" % (i + 1, s['observation'], i + 1, s['action'])
        for i, s in list(enumerate(steps))[-count:]
    ) if count else ''


def native_prompt(env, task, observation, actions, steps, history=2):
    prefix = 'ALFWORLD' if env == 'alfworld' else 'WEBSHOP'
    templates = runpy.run_path(str(VERL / 'environments/prompts' / (env + '.py')))
    fields = dict(task_description=task, current_observation=observation,
                  step_count=len(steps), current_step=len(steps) + 1,
                  history_length=min(history, len(steps)), action_history=history_text(steps, history))
    if env == 'alfworld':
        fields['admissible_actions'] = '\n '.join("'%s'" % a for a in actions if a != 'help')
    else:
        fields['available_actions'] = '\n'.join("'%s'," % a for a in actions)
    key = prefix + ('_TEMPLATE' if steps and history else '_TEMPLATE_NO_HIS')
    prompt = templates[key].format(**fields)
    if env == 'webshop' and len(prompt) > 13000:
        prompt = templates[prefix + '_TEMPLATE_NO_HIS'].format(**fields)
    return [{'role': 'user', 'content': prompt}]


def project(env, output, actions):
    namespace = runpy.run_path(str(VERL / 'environments/env_package' / env / 'projection.py'))
    if env == 'alfworld':
        projected, valid = namespace['alfworld_projection']([output], [actions])
    else:
        projected, valid = namespace['webshop_projection']([output])
    return projected[0], bool(valid[0])


def web_observation(obs, task):
    parts = obs.split(' [SEP] ')
    try:
        return ' [SEP] '.join("'%s'" % p for p in parts[parts.index(task) + 1:])
    except ValueError:
        return obs


class Agent:
    def __init__(self, args, out):
        self.args, self.out = args, out
        self.total_calls = 0
        self.generation_config = None
        self.internal_actor = None
        if getattr(args, "internal_signals_config", None):
            from internal_signals.hf_actor import HFInternalActor
            self.internal_actor = HFInternalActor(args.internal_signals_config, args.tokenizer_path, args.internal_device)
            return
        if args.disable_thinking:
            self.generation_config = json.loads((Path(args.tokenizer_path) / 'generation_config.json').read_text())
            self.generation_config.update(top_k=0, repetition_penalty=1.0, do_sample=True)

    def count(self, messages):
        if getattr(self, "internal_actor", None) is not None:
            return self.internal_actor.count(messages)
        return http('POST', self.args.tokenizer_url + '/count', json={
            'path': self.args.tokenizer_path, 'messages': messages,
            'disable_thinking': self.args.disable_thinking})['count']

    def reflection_context(self, env, task, observation, steps, actions, history):
        if env != 'scienceworld':
            return native_prompt(env,task,observation,actions,steps,history)
        if not hasattr(self, 'science_reflection_prelude'):
            source = Path(getattr(self.args, 'science_demo', None) or ROOT / 'data/paper_eval/sources/science_expert_sample.jsonl')
            with source.open() as stream:
                self.science_reflection_prelude = json.loads(stream.read() if getattr(self.args, 'science_demo', None) else stream.readline())['messages']
        builder = anchored_science_messages if getattr(self.args, 'science_context', None) == 'task-anchored-v1' else science_messages
        return builder(self.science_reflection_prelude, task, observation, steps, history)

    def generate(self, messages, task_id, step, response_schema=None, response_regex=None):
        if getattr(self, "internal_actor", None) is not None:
            atomic(self.out / "progress.json", dict(task_id=task_id, step=step, state="generating_internal_signals"))
            result = self.internal_actor.generate(messages, task_id, step, response_schema, response_regex)
            self.total_calls += 1
            return result
        if response_schema is not None and response_regex is not None:
            raise ValueError('Only one response format constraint may be supplied')
        count = self.count(messages)
        if count > 4096:
            raise RuntimeError('Prompt exceeds published 4096-token limit: %d' % count)
        payload = dict(model=self.args.model, messages=messages, temperature=.4,
                       max_tokens=self.args.max_tokens, top_p=1.0, seed=0)
        if response_schema is not None:
            payload['response_format'] = dict(type='json_schema', json_schema=dict(
                name='agentwm_review', strict=True, schema=response_schema))
        if response_regex is not None:
            payload['structured_outputs'] = dict(regex=response_regex)
        if self.args.disable_thinking:
            payload['chat_template_kwargs'] = {'enable_thinking': False}
            payload['generation_config'] = json.dumps(self.generation_config)
        else:
            # Transformers accepts these settings inside a JSON-string
            # generation_config, whereas vLLM accepts top-level extensions.
            payload.update(top_k=-1, repetition_penalty=1.0)
        atomic(self.out / 'progress.json', dict(task_id=task_id, step=step,
               state='waiting_for_model', prompt_tokens=count, updated=time.time()))
        # Transformers serve does not batch multi-GPU generation reliably. A common
        # lock serializes its three benchmark clients, without changing any prompt.
        lock = open(self.args.generation_lock, 'a') if self.args.generation_lock else None
        try:
            if lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
            for retry in range(3):
                try:
                    result = http('POST', self.args.url + '/v1/chat/completions',
                                  json=payload, timeout=900)
                    break
                except (requests.RequestException, RuntimeError):
                    if retry == 2:
                        raise
                    time.sleep(2 ** retry)
        finally:
            if lock:
                lock.close()
        choice = result['choices'][0]
        output = choice['message'].get('content') or ''
        if not output:
            raise RuntimeError('Empty assistant content: hidden reasoning cannot be parsed as an action')
        self.total_calls += 1
        metadata = dict(prompt_tokens=count, usage=result.get('usage'), finish_reason=choice.get('finish_reason'))
        if response_schema is not None:
            metadata['response_format'] = payload['response_format']
        if response_regex is not None:
            metadata['structured_outputs'] = payload['structured_outputs']
        return output, metadata

    def bounded_native(self, env, task, obs, acts, steps):
        for h in (2, 1, 0):
            messages = native_prompt(env, task, obs, acts, steps, h)
            if self.count(messages) <= 4096:
                return messages, h
        # Never silently delete admissible actions or truncate away the goal.
        raise RuntimeError('Current observation/actions alone exceed 4096 tokens')


class Alf:
    def __init__(self, args):
        if getattr(args, 'wm_backend', 'none') != 'none':
            from worldmodel.resources import install_pddl_cleanup
            install_pddl_cleanup()
        import yaml
        sys.path.insert(0, str(VERL / 'environments/env_package/alfworld'))
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        os.environ['ALFWORLD_DATA'] = '/home/shn/.cache/alfworld'
        config_path = VERL / 'environments/env_package/alfworld/configs/config_tw.yaml'
        self.cfg = yaml.safe_load(config_path.read_text())
        self.base = AlfredTWEnv(self.cfg, train_eval='eval_in_distribution')
        self.files = sorted(self.base.game_files)
        self.env = None
        self.args = args

    def tasks(self):
        assert len(self.files) == 140, len(self.files)
        return [dict(task_id='alfworld_%03d' % i, game_file=f, sha256=digest(f))
                for i, f in enumerate(self.files)]

    @staticmethod
    def transition(obs, info, done=False):
        won = bool(info.get('won', [False])[0])
        return dict(observation=obs[0], raw_observation=obs[0], done=bool(done),
                    success=won, score=100.0 * won,
                    wm_observation=alf_wm_observation(obs[0], info['admissible_commands'][0]))

    def branch(self, row):
        other = Alf(self.args)
        other.base.game_files = [row['game_file']]
        other.base.num_games = 1
        env = other.base.init_env(1)
        try:
            env.seed(1000)
            obs, info = env.reset()
            initial = self.transition(obs, info)
        except Exception:
            env.close()
            raise
        def step(action):
            obs, _, done, info = env.step([action])
            return self.transition(obs, info, done[0])
        return step, env.close, initial

    def run(self, row, agent):
        assert digest(row['game_file']) == row['sha256'], 'ALFWorld data changed'
        self.base.game_files = [row['game_file']]
        self.base.num_games = 1
        self.env = self.base.init_env(1)
        self.env.seed(1000)
        obs, info = self.env.reset()
        assert info['extra.gamefile'][0] == row['game_file']
        task = obs[0].split('Your task is to: ', 1)[1].strip()
        steps, won, done = [], False, [False]
        try:
            controller = wm_runtime.make_controller(self.args, 'alfworld', row, task,
                self.transition(obs, info), lambda: self.branch(row), agent, ROOT) if getattr(self.args, 'wm_backend', 'none') != 'none' else None
            for t in range(self.args.max_steps):
                acts = info['admissible_commands'][0]
                messages, h = agent.bounded_native('alfworld', task, obs[0], acts, steps)
                text, metadata = agent.generate(messages, row['task_id'], t + 1)
                act, valid = project('alfworld', text, acts)
                initial_act = act
                act, text, reflection = wm_runtime.select(controller, act, text, obs[0], steps, acts, t + 1)
                valid = True if act != initial_act else valid
                prior = obs[0]
                obs, _, done, info = self.env.step([act])
                if controller:
                    wm_runtime.commit(controller, act, self.transition(obs, info, done[0]), reflection)
                won = bool(info['won'][0])
                steps.append(dict(messages=messages, output=text, action=act,
                             valid_format=valid, admissible=act in acts, observation=prior,
                             next_observation=obs[0], history_length=h, **metadata, **reflection))
                agent.trace(row['task_id'], steps[-1])
                if done[0]:
                    break
            return dict(success=won, score=100.0 * won, done=bool(done[0]),
                        task_description=task, steps=steps, **wm_runtime.metrics(steps))
        finally:
            self.env.close()


class Science:
    def __init__(self, args):
        self.args = args
        self.url = args.science_url
        if getattr(args, "internal_signals_config", None):
            self.demo_path = None
            self.prelude = []
            return
        source = Path(getattr(args, 'science_demo', None) or ROOT / 'data/paper_eval/sources/science_expert_sample.jsonl')
        self.demo_path = source
        # Preserve the legacy full dialogue by default. The optional training-
        # only demo is pinned and replay-verified before a new cohort begins;
        # neither option claims to be the unpublished author inference one-shot.
        with source.open() as f:
            self.prelude = json.loads(f.read() if getattr(args, 'science_demo', None) else f.readline())['messages']
        assert len(self.prelude) % 2 == 0 and len(self.prelude) > 4
        assert [m['role'] for m in self.prelude] == ['user', 'assistant'] * (len(self.prelude) // 2)
        assert 'Action:' in self.prelude[-1]['content']

    def tasks(self):
        rows = json.loads((ROOT / 'data/paper_eval/sciworld_test.json').read_text())
        from scienceworld import ScienceWorldEnv
        env = ScienceWorldEnv()
        try:
            exceptions = {'5-1', '5-2', '9-1', '9-2', '9-3', '10-1', '10-2'}
            games = [dict(taskName=value, variationIdx=i)
                     for key, value in env.tasks.items() if key not in exceptions
                     for i in range(env.getMaxVariations(value))]
            old = json.loads((ROOT / 'data/paper_eval/audited/scienceworld.json').read_text())['tasks']
            tasks = validate_science_tasks(rows, games, old)
            for i, row in enumerate(tasks):
                env.load(row['taskName'], row['variationIdx'], simplificationStr='')
                row['task_description_sha256'] = hashlib.sha256(env.getTaskDescription().encode()).hexdigest()
                _, _, initial_done, initial_info = env.step('look around')
                assert not initial_done, 'Task already terminal at reset'
                row['initial_score'] = initial_info['score']
                if (i + 1) % 20 == 0:
                    print('Verified ScienceWorld goals:', i + 1, '/ 200', flush=True)
        finally:
            env.close()
        return tasks

    @staticmethod
    def transition(rst):
        score = float(rst['score'])
        return dict(observation=rst['observation'], raw_observation=rst['observation'],
            done=bool(rst['done']), success=bool(rst['done'] and score == 100), score=score,
            reward=rst['reward'], wm_observation=rst['observation'],
            native_observation=rst.get('native_observation', rst['observation']))

    def branch(self, row):
        # Candidate branches do not create sessions in the long-lived HTTP
        # server: AgentGym retains closed SDK objects, leaking gateway sockets.
        from scienceworld import ScienceWorldEnv
        env = ScienceWorldEnv()
        view = ScienceActionView()
        def close():
            env.close()
        try:
            env.load(row['taskName'], row['variationIdx'])
            observation, reward, done, info = env.step('look around')
            rst = view.observe(dict(observation=observation, reward=reward, done=done, score=info['score']))
        except Exception:
            close()
            raise
        def step(action):
            observation, reward, done, info = env.step(view.action(action))
            return self.transition(view.observe(dict(observation=observation, reward=reward, done=done, score=info['score'])))
        return step, close, self.transition(rst)

    def run(self, row, agent):
        snapshot_mode = getattr(self.args, 'wm_backend', 'none') == 'oracle'
        eid = None
        sdk = oracle = None
        steps = []
        try:
            if snapshot_mode:
                from scienceworld import ScienceWorldEnv
                from worldmodel.science_snapshot import ScienceSnapshotOracle
                sdk = ScienceWorldEnv()
                sdk.load(row['taskName'], row['variationIdx'])
                observation, reward, done, info = sdk.step('look around')
                rst = dict(observation=observation, reward=reward, done=done, score=info['score'],
                    task_name=row['taskName'], var_num=row['variationIdx'], task_description=sdk.getTaskDescription())
            else:
                eid = http('POST', self.url + '/create')['id']
                rst = http('POST', self.url + '/reset', json=dict(id=eid, data_idx=row['index']))
            assert rst['task_name'] == row['taskName'] and rst['var_num'] == row['variationIdx'], rst
            task = rst['task_description']
            view = ScienceActionView() if getattr(self.args, 'wm_backend', 'none') != 'none' else None
            if view:
                rst = view.observe(rst)
            if snapshot_mode:
                oracle = ScienceSnapshotOracle(sdk, view, self.transition, ROOT)
            if 'task_description_sha256' in row:
                assert hashlib.sha256(task.encode()).hexdigest() == row['task_description_sha256'], 'ScienceWorld goal changed'
                assert rst['score'] == row['initial_score'], 'ScienceWorld initial score changed'
            controller = wm_runtime.make_controller(self.args, 'scienceworld', row, task,
                self.transition(rst), lambda: self.branch(row), agent, ROOT, oracle=oracle) if getattr(self.args, 'wm_backend', 'none') != 'none' else None
            for t in range(self.args.max_steps):
                if rst['done']:
                    break
                acts = list(sdk.get_valid_action_object_combinations()) if snapshot_mode else http(
                    'GET', self.url + '/admissible_actions', params=dict(id=eid))['admissible_actions']
                if view:
                    acts = list(acts) + list(view.mapping)
                if controller is not None and (controller.review_protocol == 'agentwm-copied-v1' or controller.review_protocol.startswith('science-evidence-')):
                    from reflection.agentwm import science_action_information
                    hint = dict(possible_actions=list(sdk.get_possible_actions()),
                                possible_objects=list(sdk.get_possible_objects())) if snapshot_mode else http(
                        'GET', self.url + '/action_hint', params=dict(id=eid))
                    controller.action_information = science_action_information(
                        hint['possible_actions'], hint['possible_objects'], list(view.mapping) if view else [])
                # Use the released expert-data interface: action templates (including
                # connect/disconnect) are in the permanent introduction. No action
                # category is forbidden or removed, unlike the superseded evaluator.
                messages, h = bounded_science(agent, self.prelude, task, rst['observation'], steps)
                text, metadata = agent.generate(messages, row['task_id'], t + 1)
                # Match the AgentGym ScienceWorld response grammar, not ALFWorld tags.
                # BaseAdapter.parse_react uses the entire tail of the last
                # Action: separator, without first-line cleanup or fuzzy repair.
                parts = text.rsplit('Action:', 1)
                act = parts[-1].strip() if len(parts) == 2 else ''
                initial_act = act
                act, text, reflection = wm_runtime.select(controller, act, text, rst['observation'], steps, acts, t + 1)
                prior = rst['observation']
                native_action = view.action(act) if view else act
                rst = oracle.execute(act) if snapshot_mode else http(
                    'POST', self.url + '/step', json=dict(id=eid, action=native_action))
                if view and not snapshot_mode:
                    rst = view.observe(rst)
                if view:
                    reflection['native_action'] = native_action
                if controller:
                    wm_runtime.commit(controller, act, self.transition(rst), reflection)
                steps.append(dict(messages=messages, output=text, action=act,
                             valid_format=(metadata["decision_format_valid"] if getattr(self.args, "internal_signals_config", None) else bool(act != initial_act or (len(parts) == 2 and "Thought:" in parts[0]))), admissible=act in acts,
                             admissible_actions=acts, observation=prior,
                             next_observation=rst['observation'], score=rst['score'],
                             history_length=h, history_dropped=len(steps)-h, **metadata, **reflection))
                agent.trace(row['task_id'], steps[-1])
            score = float(rst['score'])
            return dict(success=bool(rst['done'] and score == 100.0), score=score,
                        done=rst['done'], task_description=task, steps=steps, **wm_runtime.metrics(steps))
        finally:
            if oracle is not None:
                oracle.close()
            if sdk is not None:
                sdk.close()
            if eid is not None:
                http('POST', self.url + '/close', json=dict(id=eid))


class Web:
    def __init__(self, args):
        self.args = args
        prefix = str(Path(sys.executable).resolve().parents[1])
        os.environ.setdefault('JAVA_HOME', prefix)
        os.environ['PATH'] = prefix + '/bin:' + os.environ.get('PATH', '')
        os.environ.setdefault('JVM_PATH', prefix + '/lib/jvm/lib/server/libjvm.so')
        sys.path.insert(0, str(VERL / 'environments/env_package/webshop/webshop'))
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        from web_agent_site.utils import DEFAULT_FILE_PATH, DEFAULT_ATTR_PATH
        self.env = WebAgentTextEnv(observation_mode='text', file_path=DEFAULT_FILE_PATH,
                                  attr_path=DEFAULT_ATTR_PATH, num_products=1000,
                                  human_goals=False, seed=1000)
        index = Path(DEFAULT_FILE_PATH).resolve().parents[1] / 'search_engine/indexes_1k'
        index_ids = validate_web_index(self.env.server.search_engine,
                                      self.env.server.product_item_dict)
        # Resolve the dependency root as well as the index: staged/worktree
        # evaluators can share this immutable catalog through an external symlink.
        self.data = dict(items=digest(DEFAULT_FILE_PATH), attributes=digest(DEFAULT_ATTR_PATH),
                         num_products=1000, index_path=str(index.relative_to(VERL.resolve().parents[2])),
                         index_asins_sha256=index_ids,
                         index_files={p.name: digest(p) for p in sorted(index.iterdir())
                                      if p.is_file() and p.name != 'write.lock'})

    def tasks(self):
        return [dict(task_id='webshop_%03d' % i, index=i,
                     goal_sha256=hashlib.sha256(json.dumps(self.env.server.goals[i],
                        sort_keys=True).encode()).hexdigest(), data_sha256=self.data)
                for i in range(500)]

    @staticmethod
    def actions(env):
        avail = env.get_available_actions()
        return (['search[<your query>]'] if avail['has_search_bar'] else []) + [
            'click[' + c + ']' for c in avail['clickables']]

    def transition(self, env, raw, task, reward=0.0, done=False):
        # step() automatically resets a terminal WebShop session. Never query
        # that NEW session's actions as if they belonged to the terminal page.
        actions = [] if done else self.actions(env)
        return dict(observation=web_observation(raw, task), raw_observation=raw,
            done=bool(done), success=bool(done and reward == 1.0), score=float(reward) * 100,
            reward=float(reward), wm_observation=web_wm_observation(raw, actions))

    def branch(self, row, task):
        import copy
        # Share immutable catalog/index only. All mutable session/browser state
        # lives in a separate server instance, including terminal auto-resets.
        server = copy.copy(self.env.server)
        server.user_sessions = {}
        env = type(self.env)(observation_mode='text', server=server, num_products=1000,
                             human_goals=False, seed=1000)
        raw, _ = env.reset(session=row['index'])
        assert env.instruction_text == task, 'Oracle WebShop goal changed'
        initial = self.transition(env, raw, task)
        def step(action):
            raw, reward, done, _ = env.step(action)
            return self.transition(env, raw, task, reward, done)
        return step, env.close, initial

    def run(self, row, agent):
        assert row['data_sha256'] == self.data, 'WebShop catalog changed'
        goal_hash = hashlib.sha256(json.dumps(self.env.server.goals[row['index']], sort_keys=True).encode()).hexdigest()
        assert row['goal_sha256'] == goal_hash, 'WebShop goal order changed'
        raw, _ = self.env.reset(session=row['index'])
        task = self.env.instruction_text
        obs = web_observation(raw, task)
        steps, reward, done = [], 0.0, False
        controller = wm_runtime.make_controller(self.args, 'webshop', row, task,
            self.transition(self.env, raw, task), lambda: self.branch(row, task), agent, ROOT) if getattr(self.args, 'wm_backend', 'none') != 'none' else None
        for t in range(self.args.max_steps):
            avail = self.env.get_available_actions()
            acts = (['search[<your query>]'] if avail['has_search_bar'] else []) + [
                'click[' + c + ']' for c in avail['clickables']]
            messages, h = agent.bounded_native('webshop', task, obs, acts, steps)
            text, metadata = agent.generate(messages, row['task_id'], t + 1)
            act, valid = project('webshop', text, acts)
            initial_act = act
            act, text, reflection = wm_runtime.select(controller, act, text, obs, steps, acts, t + 1)
            valid = True if act != initial_act else valid
            admissible = act in [a.lower() for a in acts] or bool(
                avail['has_search_bar'] and re.fullmatch(r'search\[.+\]', act))
            prior = obs
            state, reward, done, _ = self.env.step(act)
            if controller:
                wm_runtime.commit(controller, act, self.transition(self.env, state, task, reward, done), reflection)
            obs = web_observation(state, task)
            steps.append(dict(messages=messages, output=text, action=act,
                         valid_format=valid, admissible=admissible, observation=prior,
                         next_observation=obs, reward=reward, history_length=h, **metadata, **reflection))
            agent.trace(row['task_id'], steps[-1])
            if done:
                break
        return dict(success=bool(done and reward == 1.0), score=float(reward) * 100,
                    reward=float(reward), done=bool(done), task_description=task, steps=steps, **wm_runtime.metrics(steps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=['alfworld', 'scienceworld', 'webshop'], required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--manifest', required=True)
    p.add_argument('--url')
    p.add_argument('--model')
    p.add_argument('--tokenizer-path')
    p.add_argument('--tokenizer-url', default='http://127.0.0.1:8105')
    p.add_argument('--output')
    p.add_argument('--science-url', default='http://127.0.0.1:36001')
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--max-tokens', type=int, required=True)
    p.add_argument('--disable-thinking', action='store_true')
    p.add_argument('--generation-lock')
    p.add_argument('--limit', type=int)
    p.add_argument('--wm-backend', choices=['none', 'oracle', 'word2world'], default='none')
    p.add_argument('--wm-config', help='Word2World JSON config; defaults to configs/worldmodel/<env>.json')
    p.add_argument('--reflection-max-rounds', type=int, choices=[1, 2, 3], default=3)
    from reflection.science_evidence import PROTOCOLS as SCIENCE_EVIDENCE_PROTOCOLS
    p.add_argument('--reflection-review-protocol', choices=['auto', 'json', 'ee-native-action-v1', 'ee-conservative-v2', 'agentwm-copied-v1'] + list(SCIENCE_EVIDENCE_PROTOCOLS), default='auto')
    p.add_argument('--workers', type=int, choices=range(1, 33), default=1)
    p.add_argument('--task-queue', help=argparse.SUPPRESS)
    p.add_argument('--queue-run', help=argparse.SUPPRESS)
    p.add_argument('--queue-workers', type=int, help=argparse.SUPPRESS)
    p.add_argument('--science-context', choices=['task-anchored-v1'])
    p.add_argument('--science-demo', help='Pinned training-only one-shot JSON/JSONL file')
    args = p.parse_args()
    if bool(args.science_context) != bool(args.science_demo):
        p.error('--science-context and --science-demo must be specified together')
    if args.science_context and args.env != 'scienceworld':
        p.error('--science-context requires ScienceWorld')
    if args.workers > 1:
        if args.prepare:
            p.error('--workers requires evaluation mode')
        from experiments.parallel import run_parallel
        return run_parallel(args, ROOT)
    if bool(args.task_queue) != bool(args.queue_run) or bool(args.task_queue) != bool(args.queue_workers):
        p.error('Incomplete task queue worker configuration')
    temporary = None
    if not args.prepare and args.wm_backend != 'none':
        from worldmodel.resources import configure_temporary_directory
        temporary = configure_temporary_directory(args.output)
    backend = {'alfworld': Alf, 'scienceworld': Science, 'webshop': Web}[args.env](args)
    if args.prepare:
        atomic(args.manifest, dict(protocol=protocol_for_env(args.env), env=args.env, tasks=backend.tasks()))
        print('Prepared', args.manifest, flush=True)
        return
    manifest = json.loads(Path(args.manifest).read_text())
    assert manifest['protocol'] == protocol_for_env(args.env) and manifest['env'] == args.env, 'Manifest protocol mismatch; do not reuse pre-fix results'
    if args.env == 'scienceworld':
        import importlib.metadata
        from scienceworld.constants import JAR_PATH
        assert manifest['test_sha256'] == digest(ROOT / 'data/paper_eval/sciworld_test.json'), 'Official task file changed'
        assert manifest['demo_sha256'] == digest(ROOT / 'data/paper_eval/sources/science_expert_sample.jsonl'), 'ScienceWorld demo changed'
        assert manifest['scienceworld_version'] == importlib.metadata.version('scienceworld'), 'ScienceWorld version changed'
        assert manifest['simulator_jar_sha256'] == digest(JAR_PATH), 'ScienceWorld simulator changed'
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / '.runner.lock', 'a') as runner_lock:
        fcntl.flock(runner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved_args = dict(vars(args))
        for key in ('science_context', 'science_demo'):
            if saved_args[key] is None:
                saved_args.pop(key)
        for key in ('workers', 'task_queue', 'queue_run', 'queue_workers'):
            saved_args.pop(key)
        if args.reflection_review_protocol == 'auto':
            saved_args.pop('reflection_review_protocol')
        if args.wm_backend == 'none':
            # Preserve exact pre-feature config files for baseline resume.
            for key in ('wm_backend', 'wm_config', 'reflection_max_rounds'):
                saved_args.pop(key)
        config = dict(saved_args, protocol=protocol_for_env(args.env), manifest_sha256=digest(args.manifest),
                      **wm_runtime.audit_settings(args, ROOT))
        if args.env == 'scienceworld':
            config.update(science_prompt='full-train-dialogue-one-shot/native-goal-once/token-bounded-full-history',
                          science_demo_sha256=digest(ROOT / 'data/paper_eval/sources/science_expert_sample.jsonl'),
                          science_demo_messages=len(backend.prelude))
            if args.science_context:
                config.update(protocol='scienceworld-task-anchored-v1',
                    science_prompt='verified-short-train-one-shot/episode-boundary/current-goal/token-bounded-history',
                    science_demo_sha256=digest(backend.demo_path))
        config_file = out / 'config.json'
        if config_file.exists():
            assert json.loads(config_file.read_text()) == config, 'Cannot mix changed experiment configurations'
        else:
            atomic(config_file, config)
        tasks = manifest['tasks'][:args.limit] if args.limit else manifest['tasks']
        queue = None
        if args.task_queue:
            from experiments.task_queue import TaskQueue
            queue = TaskQueue(args.task_queue, args.manifest, tasks, args.queue_run, args.queue_workers)
            queue.publish_config(config)
            queue.ready(out.name)
        agent = Agent(args, out)
        def trace(task_id, step):
            with open(out / 'steps.jsonl', 'a') as f:
                f.write(json.dumps(dict(task_id=task_id, **step), ensure_ascii=False) + '\n')
                f.flush()
        agent.trace = trace
        for row in queue.claims() if queue else tasks:
            result_file = queue.result_file(row['task_id']) if queue else out / 'tasks' / (row['task_id'] + '.json')
            if result_file.exists() and json.loads(result_file.read_text()).get('status') == 'completed':
                continue
            for attempt in range(3):
                started = time.time()
                try:
                    result = backend.run(row, agent)
                    result.update(task_id=row['task_id'], status='completed', wall_seconds=time.time()-started)
                    atomic(result_file, result)
                    print(row['task_id'], 'score=', result['score'], 'success=', result['success'], flush=True)
                    break
                except Exception as e:
                    traceback.print_exc()
                    atomic(result_file, dict(task_id=row['task_id'], status='error',
                           error=repr(e), attempt=attempt+1, updated=time.time()))
                    if attempt < 2:
                        time.sleep(3)
        atomic(out / 'progress.json', dict(state='finished', updated=time.time()))


if __name__ == '__main__':
    sys.exit(main())
