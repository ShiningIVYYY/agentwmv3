import copy
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
import requests


class PredictionError(RuntimeError):
    pass


class RandomIsolation:
    """Environment construction/reset must not change the live runner's RNGs."""
    def __enter__(self):
        self.python = random.getstate()
        self.numpy = sys.modules.get('numpy')
        self.torch = sys.modules.get('torch')
        self.nstate = self.numpy.random.get_state() if self.numpy else None
        self.tstate = self.torch.get_rng_state() if self.torch else None
        return self

    def __exit__(self, *args):
        random.setstate(self.python)
        if self.numpy:
            self.numpy.random.set_state(self.nstate)
        if self.torch:
            self.torch.set_rng_state(self.tstate)


def science_observation_key(text):
    """Ignore only the order of ScienceWorld's explicit room exit list.

    Separate JVMs enumerate these exits in different orders. Keep every full
    door description, including duplicates and its open/closed state, and keep
    the rest of the room response byte-for-byte. Unsupported layouts stay strict.
    This key is for Oracle validation only; Agent and checkpoint text stays native.
    """
    if not isinstance(text, str) or not text.startswith('Inside the '):
        return text
    head, separator, exits = text.rpartition('\n\n You also see: ')
    if not separator or not exits.endswith('.'):
        return text
    doors = exits[:-1].split(', ')
    if not all(re.fullmatch(r'A door to [^,\n]+ \(that is (?:open|closed)\)', door)
               for door in doors):
        return text
    return head + separator + ', '.join(sorted(doors)) + '.'


def transition_differences(expected, actual):
    expected = expected or {}
    science = 'native_observation' in expected and 'native_observation' in actual
    fields = ('observation', 'raw_observation', 'done', 'success', 'score', 'reward')
    differences = {}
    for key in fields:
        predicted, real = expected.get(key), actual.get(key)
        if science and key in ('observation', 'raw_observation'):
            equal = science_observation_key(predicted) == science_observation_key(real)
        else:
            equal = predicted == real
        if not equal:
            differences[key] = dict(predicted=predicted, actual=real)
    return differences


def same_transition(expected, actual):
    # Room exit enumeration is unordered; all contents and numeric feedback
    # remain strict. Stable ambiguity menus retain native-number translation.
    return not transition_differences(expected, actual)


class OracleWorldModel:
    """A fresh real simulator branch for each candidate; verified prefix replay."""
    backend = 'oracle'

    def __init__(self, factory, initial):
        self.factory = factory
        self.initial = copy.deepcopy(initial)
        self.history = []
        self.predictions = {}

    def predict(self, action):
        started = time.monotonic()
        with RandomIsolation():
            step, close, current = self.factory()
            try:
                if not same_transition(self.initial, current):
                    raise PredictionError('Oracle initial state differs from the live environment')
                for prior_action, expected in self.history:
                    current = step(prior_action)
                    if not same_transition(expected, current):
                        raise PredictionError('Oracle replay differs from the executed trajectory')
                result = step(action)
            finally:
                close()
        prediction = dict(result, backend=self.backend, seconds=time.monotonic() - started,
                          verified_prefix_steps=len(self.history))
        self.predictions[action] = copy.deepcopy(prediction)
        return prediction

    def commit(self, action, actual):
        expected = self.predictions.get(action)
        if expected is None or not same_transition(expected, actual):
            diff = transition_differences(expected, actual)
            raise PredictionError('Oracle candidate prediction differs from its actual committed transition: ' +
                                  json.dumps(diff, ensure_ascii=False))
        self.history.append((action, copy.deepcopy(actual)))
        self.predictions.clear()

    def close(self):
        pass


class Word2WorldModel:
    backend = 'word2world'

    def __init__(self, config, initial_context, count, to_action=lambda x: x,
                 from_observation=lambda x: x):
        self.config = config
        self.history = [dict(role='system', content=initial_context)]
        self.count = count
        self.to_action = to_action
        self.from_observation = from_observation
        self.provenance = config['provenance']

    def predict(self, action):
        started = time.monotonic()
        messages = self.history + [dict(role='user', content=self.to_action(action))]
        count = self.count(messages)
        if count + self.config['max_tokens'] > self.config['max_model_len']:
            raise PredictionError('World model full history exceeds its configured context window')
        payload = dict(model=self.config['served_model_name'], messages=messages,
                       temperature=0, top_p=1, max_tokens=self.config['max_tokens'])
        response = requests.post(self.config['url'].rstrip('/') + '/v1/chat/completions',
                                 json=payload, timeout=self.config.get('timeout', 180))
        response.raise_for_status()
        result = response.json()
        choice = result['choices'][0]
        raw = choice['message'].get('content')
        if not isinstance(raw, str) or not raw.strip():
            raise PredictionError('Empty world model prediction')
        success = ' [SUCCESS]' in raw
        observation = raw.split(' [SUCCESS]', 1)[0]
        return dict(backend=self.backend, raw_output=raw,
                    observation=self.from_observation(observation), predicted_success=success,
                    observation_complete=choice.get('finish_reason') != 'length',
                    finish_reason=choice.get('finish_reason'),
                    prompt_tokens=count, usage=result.get('usage'),
                    seconds=time.monotonic() - started)

    def commit(self, action, actual):
        # Teacher force the committed REAL observation. Rejected hypothetical
        # predictions never become assistant turns of the world model.
        self.history += [dict(role='user', content=self.to_action(action)),
                         dict(role='assistant', content=actual['wm_observation'])]

    def close(self):
        pass


def load_config(path, root, env):
    path = Path(path)
    config = json.loads(path.read_text())
    if config['env'] != env:
        raise PredictionError('World model config/environment mismatch')
    expected_repo = 'X1AOX1A/WorldModel-%s-Qwen2.5-7B' % {
        'alfworld': 'Alfworld', 'scienceworld': 'Sciworld', 'webshop': 'Webshop'}[env]
    if config['repo_id'] != expected_repo:
        raise PredictionError('Expected the official environment-specific Word2World checkpoint')
    for key in ('checkpoint', 'source_dir'):
        p = Path(config[key])
        config[key] = str(p if p.is_absolute() else root / p)
    provenance = json.loads((Path(config['checkpoint']) / 'provenance.json').read_text())
    if provenance['repo_id'] != config['repo_id']:
        raise PredictionError('Checkpoint provenance mismatch')
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        p = Path(config['checkpoint']) / name
        if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() != provenance['sha256'][name]:
            raise PredictionError('Checkpoint configuration/tokenizer changed: ' + name)
    config['provenance'] = provenance
    if config['temperature'] != 0 or config['top_p'] != 1 or config['max_tokens'] != 512:
        raise PredictionError('Online Word2World prediction must use the published 0/1/512 settings')
    return config
