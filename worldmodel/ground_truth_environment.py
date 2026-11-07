"""Independent ScienceWorld reset/replay lookahead; no formal preview/rollback.

The checked projection includes public transitions and selected private audit
fingerprints. It is NOT a claim of complete hidden-state equivalence.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

from evaluation.profile import canonicalize_observation
from worldmodel.backends import RandomIsolation
from worldmodel.contexts import ScienceActionView

JAR_SHA256 = 'f4e1fdce0cdabe1bab7128a7e78ea820210d9db3c28d7272bcbbc3f3a1c5f0d7'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rng_state(env):
    cls = env._gateway.jvm.java.lang.Class.forName('scala.util.Random$', True, env.server.getClass().getClassLoader())
    rng = cls.getField('MODULE$').get(None).self()
    result = {}
    current = rng.getClass()
    while current is not None:
        for field in current.getDeclaredFields():
            if field.getName() in ('seed', 'nextNextGaussian', 'haveNextNextGaussian'):
                field.setAccessible(True)
                value = field.get(rng)
                if not isinstance(value, (int, float, bool)):
                    value = value.get()
                result[field.getName()] = value
        current = current.getSuperclass()
    if 'seed' not in result:
        raise RuntimeError('Cannot verify selected Scala/Java RNG state')
    return result


class ReplayScienceEnvironment:
    """One owned SDK/JVM and an immutable record of every committed API action."""
    def __init__(self, spec):
        from scienceworld import ScienceWorldEnv
        from scienceworld.constants import JAR_PATH
        if hashlib.sha256(Path(JAR_PATH).read_bytes()).hexdigest() != JAR_SHA256:
            raise ValueError('Unvalidated ScienceWorld JAR')
        self.spec = copy.deepcopy(spec)
        self.env = ScienceWorldEnv()
        self.view = ScienceActionView()
        self.history = []
        self.capture_count = 0
        self.closed = False
        try:
            cls = self.env._gateway.jvm.java.lang.Class.forName('scala.util.Random$', True, self.env.server.getClass().getClassLoader())
            cls.getField('MODULE$').get(None).setSeed(spec['seed'])
            self.env.envStepLimit = spec['environment_step_limit']
            self.env.load(spec['taskName'], spec['variationIdx'], spec['simplifications'], False)
            # SDK load reseeds Scala RNG from variation, overriding the caller.
            # Set the declared transition seed after loading in both A and B.
            cls.getField('MODULE$').get(None).setSeed(spec['seed'])
            # SDK.taskdescription adds a UI label; AgentGym identity uses native text.
            self.goal = str(self.env.server.getTaskDescription())
            self._step('look around')  # Same explicit initialization query in A/B.
            self.initial = self.capture()
        except BaseException:
            self.close()
            raise

    @property
    def pid(self):
        return self.env._gateway.java_process.pid

    def _step(self, action):
        native_action = self.view.action(action)
        started = time.perf_counter()
        # SDK.step additionally calls get_valid_action_object_combinations().
        # That supposedly informational query consumes Scala RNG differently
        # across JVMs. Use the same public native transition/scoring semantics
        # without that unnecessary, state-perturbing enumeration. No SDK/JAR edit.
        obs = self.env.server.step(native_action)
        score = int(round(100 * self.env.server.getScore()))
        moves = self.env.get_num_moves()
        reward = score - self.env.lastStepScore
        self.env.lastStepScore = score
        done = bool(self.env.server.getCompleted()) or moves > self.env.envStepLimit or score < 0
        shown = self.view.observe(dict(observation=obs))['observation']
        self.current = dict(observation=canonicalize_observation(shown), raw_observation=obs,
                            reward=reward, done=bool(done), score=score, moves=moves)
        return dict(self.current, native_action=native_action, action=action,
                    step_seconds=time.perf_counter() - started)

    def capture(self):
        self.capture_count += 1
        return dict(transition=copy.deepcopy(self.current),
                    room=canonicalize_observation(self.env.look()),
                    inventory=canonicalize_observation(self.env.inventory()),
                    simulator_moves=self.env.get_num_moves(), sdk_last_score=self.env.lastStepScore,
                    native_completed=bool(self.env.server.getCompleted()),
                    executed_action_history=str(self.env.server.getActionHistory()),
                    object_tree_sha256=digest(self.env.getObjectTree()), selected_rng=rng_state(self.env),
                    formal_action_count=len(self.history), ambiguity_mapping=dict(self.view.mapping))

    def step(self, action):
        if self.current['done']:
            raise RuntimeError('Cannot execute after episode termination')
        before = self.capture()
        result = self._step(action)
        self.history.append(dict(action=action, transition=result, before=before))
        self.history[-1]['after'] = self.capture()
        return result

    def close(self):
        if self.closed:
            return
        process = self.env._gateway.java_process
        try:
            self.env.close()
        except BrokenPipeError:
            # SDK shutdown may win the race with its own final stdin newline.
            if process.poll() is None:
                process.wait(timeout=10)
        try:
            process.wait(timeout=10)
        except __import__('subprocess').TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
        self.closed = True


def differences(expected, actual):
    # Raw row enumeration is retained in logs; only the existing official
    # presentation canonicalization is used for feedback comparison.
    e, a = copy.deepcopy(expected), copy.deepcopy(actual)
    e['transition'].pop('raw_observation', None)
    a['transition'].pop('raw_observation', None)
    return {k: dict(expected=e.get(k), actual=a.get(k)) for k in set(e) | set(a) if e.get(k) != a.get(k)}


class GroundTruthEnvironmentWM:
    backend = 'ground_truth_environment_replay'

    def simulate_one_step(self, formal, draft_action):
        started = time.perf_counter()
        before = formal.capture()
        worker = None
        cost = dict(jvm_initializations=0, initialization_step_calls=0, replay_step_calls=0,
                    candidate_step_calls=0, initialization_seconds=0.0, replay_seconds=0.0,
                    candidate_seconds=0.0, cleanup_seconds=0.0, audit_capture_bundles=0)
        result = dict(source='ground_truth_environment', backend=self.backend,
                      status='reconstruction_failed', prediction_available=False,
                      verified_prefix_steps=0, full_hidden_state_equivalence_claimed=False)
        try:
            if formal.current['done']:
                raise RuntimeError('Formal episode already done')
            with RandomIsolation():
                begin = time.perf_counter()
                cost['jvm_initializations'] = 1
                worker = ReplayScienceEnvironment(formal.spec)
                cost['initialization_seconds'] = time.perf_counter() - begin
                cost['initialization_step_calls'] = 1
                result['worker_pid'] = worker.pid
                result['formal_pid'] = formal.pid
                if worker.pid == formal.pid:
                    raise RuntimeError('Worker shares the formal JVM')
                diff = differences(formal.initial, worker.initial)
                if diff:
                    result['mismatch'] = dict(stage='initial', fields=diff)
                    raise RuntimeError('Initial reconstruction mismatch')
                for index, prior in enumerate(formal.history):
                    begin = time.perf_counter()
                    worker.step(prior['action'])  # Includes invalid/numbered actions.
                    cost['replay_step_calls'] += 1
                    cost['replay_seconds'] += time.perf_counter() - begin
                    diff = differences(prior['after'], worker.capture())
                    if diff:
                        result['mismatch'] = dict(stage='prefix', step_index=index, fields=diff)
                        raise RuntimeError('Committed prefix reconstruction mismatch')
                    result['verified_prefix_steps'] += 1
                diff = differences(before, worker.capture())
                if diff:
                    result['mismatch'] = dict(stage='current', fields=diff)
                    raise RuntimeError('Current reconstruction mismatch')
                begin = time.perf_counter()
                transition = worker.step(draft_action)
                cost['candidate_step_calls'] = 1
                cost['candidate_seconds'] = time.perf_counter() - begin
                result.update(status='ok', prediction_available=True,
                              predicted_observation=transition['observation'], predicted_reward=transition['reward'],
                              predicted_done=transition['done'], predicted_score=transition['score'],
                              predicted_moves=transition['moves'], action_valid=None,
                              environment_feedback=transition['observation'], candidate_transition=transition)
        except Exception as error:
            result.update(status='reconstruction_failed', prediction_available=False,
                          error_type=type(error).__name__, error=str(error))
            # No unverified feedback may reach a Reviewer, even on partial failure.
            for key in list(result):
                if key.startswith('predicted_') or key in ('environment_feedback', 'candidate_transition'):
                    del result[key]
        finally:
            if worker is not None:
                cost['audit_capture_bundles'] = worker.capture_count
                begin = time.perf_counter()
                worker.close()
                cost['cleanup_seconds'] = time.perf_counter() - begin
        after = formal.capture()
        if before != after:
            raise RuntimeError('FATAL: independent worker altered the formal environment; stop episode')
        result.update(formal_unchanged=True, checked_state_sha256=digest(before),
                      simulation_cost=cost, simulation_latency=time.perf_counter() - started,
                      available_at_ns=time.time_ns())
        return result

    predict = simulate_one_step

    def close(self):
        pass  # Every candidate owns and closes its fresh worker.
