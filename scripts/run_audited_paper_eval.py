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


def bounded_science(agent, prelude, task, observation, steps):
    # Full history first. Drop whole oldest observation/assistant pairs only
    # when the exact model tokenizer says the prompt exceeds the paper budget.
    for keep in range(len(steps), -1, -1):
        messages = science_messages(prelude, task, observation, steps, keep)
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
        if args.disable_thinking:
            self.generation_config = json.loads((Path(args.tokenizer_path) / 'generation_config.json').read_text())
            self.generation_config.update(top_k=0, repetition_penalty=1.0, do_sample=True)

    def count(self, messages):
        return http('POST', self.args.tokenizer_url + '/count', json={
            'path': self.args.tokenizer_path, 'messages': messages,
            'disable_thinking': self.args.disable_thinking})['count']

    def generate(self, messages, task_id, step):
        count = self.count(messages)
        if count > 4096:
            raise RuntimeError('Prompt exceeds published 4096-token limit: %d' % count)
        payload = dict(model=self.args.model, messages=messages, temperature=.4,
                       max_tokens=self.args.max_tokens, top_p=1.0, seed=0)
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
        return output, dict(prompt_tokens=count, usage=result.get('usage'),
                            finish_reason=choice.get('finish_reason'))

    def bounded_native(self, env, task, obs, acts, steps):
        for h in (2, 1, 0):
            messages = native_prompt(env, task, obs, acts, steps, h)
            if self.count(messages) <= 4096:
                return messages, h
        # Never silently delete admissible actions or truncate away the goal.
        raise RuntimeError('Current observation/actions alone exceed 4096 tokens')


class Alf:
    def __init__(self, args):
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
            for t in range(self.args.max_steps):
                acts = info['admissible_commands'][0]
                messages, h = agent.bounded_native('alfworld', task, obs[0], acts, steps)
                text, metadata = agent.generate(messages, row['task_id'], t + 1)
                act, valid = project('alfworld', text, acts)
                prior = obs[0]
                obs, _, done, info = self.env.step([act])
                won = bool(info['won'][0])
                steps.append(dict(messages=messages, output=text, action=act,
                             valid_format=valid, admissible=act in acts, observation=prior,
                             next_observation=obs[0], history_length=h, **metadata))
                agent.trace(row['task_id'], steps[-1])
                if done[0]:
                    break
            return dict(success=won, score=100.0 * won, done=bool(done[0]),
                        task_description=task, steps=steps)
        finally:
            self.env.close()


class Science:
    def __init__(self, args):
        self.args = args
        self.url = args.science_url
        source = ROOT / 'data/paper_eval/sources/science_expert_sample.jsonl'
        # Complete first published TRAIN dialogue, fixed before testing.
        # This is not the unpublished author inference one-shot, nor a
        # claim that this released dialogue reaches a terminal winning state.
        with source.open() as f:
            self.prelude = json.loads(f.readline())['messages']
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

    def run(self, row, agent):
        eid = http('POST', self.url + '/create')['id']
        steps = []
        try:
            rst = http('POST', self.url + '/reset', json=dict(id=eid, data_idx=row['index']))
            assert rst['task_name'] == row['taskName'] and rst['var_num'] == row['variationIdx'], rst
            task = rst['task_description']
            if 'task_description_sha256' in row:
                assert hashlib.sha256(task.encode()).hexdigest() == row['task_description_sha256'], 'ScienceWorld goal changed'
                assert rst['score'] == row['initial_score'], 'ScienceWorld initial score changed'
            for t in range(self.args.max_steps):
                if rst['done']:
                    break
                acts = http('GET', self.url + '/admissible_actions', params=dict(id=eid))['admissible_actions']
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
                prior = rst['observation']
                rst = http('POST', self.url + '/step', json=dict(id=eid, action=act))
                steps.append(dict(messages=messages, output=text, action=act,
                             valid_format=bool(len(parts) == 2 and 'Thought:' in parts[0]), admissible=act in acts,
                             admissible_actions=acts, observation=prior,
                             next_observation=rst['observation'], score=rst['score'],
                             history_length=h, history_dropped=len(steps)-h, **metadata))
                agent.trace(row['task_id'], steps[-1])
            score = float(rst['score'])
            return dict(success=bool(rst['done'] and score == 100.0), score=score,
                        done=rst['done'], task_description=task, steps=steps)
        finally:
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
        self.data = dict(items=digest(DEFAULT_FILE_PATH), attributes=digest(DEFAULT_ATTR_PATH),
                         num_products=1000, index_path=str(index.relative_to(ROOT)),
                         index_asins_sha256=index_ids,
                         index_files={p.name: digest(p) for p in sorted(index.iterdir())
                                      if p.is_file() and p.name != 'write.lock'})

    def tasks(self):
        return [dict(task_id='webshop_%03d' % i, index=i,
                     goal_sha256=hashlib.sha256(json.dumps(self.env.server.goals[i],
                        sort_keys=True).encode()).hexdigest(), data_sha256=self.data)
                for i in range(500)]

    def run(self, row, agent):
        assert row['data_sha256'] == self.data, 'WebShop catalog changed'
        goal_hash = hashlib.sha256(json.dumps(self.env.server.goals[row['index']], sort_keys=True).encode()).hexdigest()
        assert row['goal_sha256'] == goal_hash, 'WebShop goal order changed'
        raw, _ = self.env.reset(session=row['index'])
        task = self.env.instruction_text
        obs = web_observation(raw, task)
        steps, reward, done = [], 0.0, False
        for t in range(self.args.max_steps):
            avail = self.env.get_available_actions()
            acts = (['search[<your query>]'] if avail['has_search_bar'] else []) + [
                'click[' + c + ']' for c in avail['clickables']]
            messages, h = agent.bounded_native('webshop', task, obs, acts, steps)
            text, metadata = agent.generate(messages, row['task_id'], t + 1)
            act, valid = project('webshop', text, acts)
            admissible = act in [a.lower() for a in acts] or bool(
                avail['has_search_bar'] and re.fullmatch(r'search\[.+\]', act))
            prior = obs
            state, reward, done, _ = self.env.step(act)
            obs = web_observation(state, task)
            steps.append(dict(messages=messages, output=text, action=act,
                         valid_format=valid, admissible=admissible, observation=prior,
                         next_observation=obs, reward=reward, history_length=h, **metadata))
            agent.trace(row['task_id'], steps[-1])
            if done:
                break
        return dict(success=bool(done and reward == 1.0), score=float(reward) * 100,
                    reward=float(reward), done=bool(done), task_description=task, steps=steps)


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
    args = p.parse_args()
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
        config = dict(vars(args), protocol=protocol_for_env(args.env), manifest_sha256=digest(args.manifest))
        if args.env == 'scienceworld':
            config.update(science_prompt='full-train-dialogue-one-shot/native-goal-once/token-bounded-full-history',
                          science_demo_sha256=digest(ROOT / 'data/paper_eval/sources/science_expert_sample.jsonl'),
                          science_demo_messages=len(backend.prelude))
        config_file = out / 'config.json'
        if config_file.exists():
            assert json.loads(config_file.read_text()) == config, 'Cannot mix changed experiment configurations'
        else:
            atomic(config_file, config)
        agent = Agent(args, out)
        def trace(task_id, step):
            with open(out / 'steps.jsonl', 'a') as f:
                f.write(json.dumps(dict(task_id=task_id, **step), ensure_ascii=False) + '\n')
                f.flush()
        agent.trace = trace
        tasks = manifest['tasks'][:args.limit] if args.limit else manifest['tasks']
        for row in tasks:
            result_file = out / 'tasks' / (row['task_id'] + '.json')
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
    main()
