"""Accurate ScienceWorld previews using reversible checkpoints of one live JVM."""
import copy
import hashlib
import json
from pathlib import Path
import time

from .backends import PredictionError, transition_differences
from .contexts import ScienceActionView


def load_bridge(env, root):
    out = root / 'outputs/wm_reflection_setup/science_snapshot'
    provenance = json.loads((out / 'provenance.json').read_text())
    if hashlib.sha256((root / 'worldmodel/HeapSnapshot.java').read_bytes()).hexdigest() != provenance['source_sha256']:
        raise PredictionError('ScienceWorld snapshot source changed; rebuild the helper')
    for relative, digest in provenance['classes'].items():
        if hashlib.sha256((out / relative).read_bytes()).hexdigest() != digest:
            raise PredictionError('ScienceWorld snapshot bytecode changed')
    jvm = env._gateway.jvm
    urls = env._gateway.new_array(jvm.java.net.URL, 1)
    urls[0] = jvm.java.net.URL((out / 'classes').as_uri() + '/')
    loader = jvm.java.net.URLClassLoader(urls, env.server.getClass().getClassLoader())
    cls = jvm.java.lang.Class.forName('agentwm.HeapSnapshot', True, loader)
    return cls.newInstance(), loader


class ScienceSnapshotOracle:
    backend = 'oracle'

    def __init__(self, env, view, transition, root):
        self.env, self.view, self.transition = env, view, transition
        self.bridge, self.loader = load_bridge(env, root)
        self.predictions = {}
        self.states = {}
        self.history = []
        self.adopted = None

    def python_state(self):
        # Python step() uses these fields for reward, the horizon and run logs.
        return {k:copy.deepcopy(v) for k,v in self.env.__dict__.items()
                if k not in ('_gateway', 'server', '_obj_tree_tempdir')}

    def restore_python(self, state):
        for key in list(self.env.__dict__):
            if key not in ('_gateway', 'server', '_obj_tree_tempdir') and key not in state:
                del self.env.__dict__[key]
        self.env.__dict__.update(copy.deepcopy(state))

    def predict(self, action):
        started = time.monotonic()
        before = self.bridge.take(self.env.server)
        python_before = self.python_state()
        mapping_before = dict(self.view.mapping)
        try:
            obs, reward, done, info = self.env.step(self.view.action(action))
            result = self.transition(self.view.observe(dict(observation=obs, reward=reward,
                done=done, score=info['score'])))
            after = self.bridge.take(self.env.server)
            python_after = self.python_state()
            mapping_after = dict(self.view.mapping)
            moves_after = self.env.get_num_moves()
        finally:
            before.restore()
            self.restore_python(python_before)
            self.view.mapping = mapping_before
        prediction = dict(result, backend='oracle', seconds=time.monotonic()-started,
            verified_prefix_steps=len(self.history), oracle_method='live-jvm-heap-checkpoint',
            snapshot_fields=before.objectFields(), snapshot_arrays=before.arrayCount(),
            snapshot_simulator_moves=moves_after)
        self.predictions[action] = copy.deepcopy(prediction)
        self.states[action] = (after, python_after, mapping_after)
        return prediction

    def execute(self, action):
        if action not in self.states:
            raise PredictionError('Final ScienceWorld action has no verified checkpoint prediction')
        after, python_after, mapping_after = self.states[action]
        after.restore()
        self.restore_python(python_after)
        self.view.mapping = dict(mapping_after)
        if self.env.get_num_moves() != self.predictions[action]['snapshot_simulator_moves']:
            raise PredictionError('ScienceWorld checkpoint adoption changed the simulator clock')
        self.adopted = action
        # This is the actual simulator transition saved after executing the
        # candidate. Adopt it once; rejected branches remain fully rolled back.
        return copy.deepcopy(self.predictions[action])

    def commit(self, action, actual):
        expected = self.predictions.get(action)
        if self.adopted != action or expected is None or transition_differences(expected, actual):
            raise PredictionError('ScienceWorld adopted checkpoint differs from its simulator transition')
        if not self.states[action][0].matches():
            raise PredictionError('ScienceWorld live heap differs from the adopted checkpoint')
        self.history.append((action, copy.deepcopy(actual)))
        self.states.clear()
        self.predictions.clear()
        self.adopted = None

    def close(self):
        self.states.clear()
        self.predictions.clear()
        self.loader.close()
