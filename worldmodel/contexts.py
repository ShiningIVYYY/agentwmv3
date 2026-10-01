"""Author-provided initialization collectors, with explicit protocol adapters."""
import hashlib
import json
from pathlib import Path
import re
import sys
import types

from .backends import PredictionError, RandomIsolation


def collector(config, name):
    root = Path(config['source_dir'])
    if (root / 'REVISION').read_text().strip() != config['source_revision']:
        raise PredictionError('Word2World source revision mismatch')
    relative = 'scripts/collect_init_context/collect_wm_instruct_%s.py' % name
    source = (root / relative).read_text()
    hashes = json.loads((root / 'SHA256.json').read_text())
    if hashlib.sha256(source.encode()).hexdigest() != hashes[relative]:
        raise PredictionError('Word2World initialization collector changed')
    key = '_word2world_' + name
    module = types.ModuleType(key)
    module.__file__ = str(root / relative)
    sys.modules[key] = module
    # The released ALF collector uses builtin generic annotations. Deferred
    # annotations make the unchanged author source usable in Python 3.8.
    exec(compile('from __future__ import annotations\n' + source,
                 module.__file__, 'exec'), module.__dict__)
    return module


def alf_action(action):
    # AgentGym's released grammar renamed the leading placement verb only.
    return re.sub(r'^put (?=.+ in/on )', 'move ', action)


def alf_wm_observation(observation, actions):
    return observation + '\nAVAILABLE ACTIONS: ' + ','.join(
        alf_action(a) for a in actions)


def checkpoint_observation(env, text, task):
    if env == 'alfworld':
        return text.split('\nAVAILABLE ACTIONS', 1)[0].strip()
    if env == 'webshop':
        # The author's model predicts page text AND available-action suffix.
        text = text.split('\n\nYour admissible actions', 1)[0]
        parts = text.split(' [SEP] ')
        if task in parts:
            return ' [SEP] '.join("'%s'" % p for p in parts[parts.index(task) + 1:])
    if env == 'scienceworld':
        try:
            return science_menu(text)[0].strip()
        except PredictionError:
            # A learned or truncated prediction may have malformed menu text.
            return text.strip()
    return text.strip()


def science_menu(text):
    """Lossless stable choice labels; translate their numbers on execution."""
    header = 'Ambiguous request: Please enter the number for the action you intended (or blank to cancel):\n'
    if not text.startswith(header):
        return text, {}
    choices = []
    for line in text[len(header):].splitlines():
        match = re.fullmatch(r'(\d+):\t(.+)', line)
        if not match:
            raise PredictionError('Unsupported ScienceWorld ambiguity menu layout')
        choices.append((match[2], match[1]))
    if not choices or len({label for label, _ in choices}) != len(choices):
        raise PredictionError('ScienceWorld ambiguity choices lack unique labels')
    choices.sort()
    mapping = {str(i): native for i, (_, native) in enumerate(choices)}
    return header + ''.join('%d:\t%s\n' % (i, label) for i, (label, _) in enumerate(choices)), mapping


class ScienceActionView:
    def __init__(self):
        self.mapping = {}

    def observe(self, payload):
        native = payload['observation']
        observation, self.mapping = science_menu(native)
        return dict(payload, observation=observation, native_observation=native)

    def action(self, action):
        return self.mapping.get(action, action)


def web_wm_observation(raw, actions):
    return raw + '\n\nYour admissible actions of the current situation are: \n[\n' + \
        '\n'.join("'%s'," % a for a in actions) + '\n].'


def initial_context(config, row, task, initial):
    env = config['env']
    if env == 'alfworld':
        source = collector(config, 'alfworld')
        traj = Path(row['game_file']).with_name('traj_data.json')
        description = source.get_alfworld_description(traj)
    elif env == 'scienceworld':
        source = collector(config, 'sciworld')
        with RandomIsolation():
            description = source.get_sciworld_description(row['index'])
    else:
        return initial['wm_observation']
    instruction = initial['wm_observation']
    if env == 'scienceworld':
        instruction = task + '\n' + instruction
    return '# Environment Information (Only visible to Assistant)\n\n' + description + \
        '\n\n# User Environment Information (Displayed to User)\n\n' + instruction + '\n'
