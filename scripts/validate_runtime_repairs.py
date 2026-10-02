#!/usr/bin/env python3
"""Live simulator fixtures for snapshot isolation and native library ownership."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def science():
    from scienceworld import ScienceWorldEnv
    from worldmodel.contexts import ScienceActionView
    from worldmodel.science_snapshot import ScienceSnapshotOracle
    from worldmodel.backends import same_transition
    spec = importlib.util.spec_from_file_location('ev', ROOT / 'scripts/run_audited_paper_eval.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    rows = json.loads((ROOT / 'data/paper_eval/audited/scienceworld_context_v2.json').read_text())['tasks']
    failed = {'sciworld_2932','sciworld_1093','sciworld_1131','sciworld_1822','sciworld_3523',
        'sciworld_1934','sciworld_4335','sciworld_4312','sciworld_565','sciworld_3515','sciworld_2511'}
    report = []
    for row in rows:
        if row['task_id'] not in failed:
            continue
        env = ScienceWorldEnv()
        wm = None
        try:
            env.load(row['taskName'], row['variationIdx'])
            obs, reward, done, info = env.step('look around')
            view = ScienceActionView()
            view.observe(dict(observation=obs))
            wm = ScienceSnapshotOracle(env, view, ev.Science.transition, ROOT)
            actions = ['look around','wait','inventory','look around']
            if row['task_id'] in ('sciworld_4335','sciworld_4312'):
                actions += ['open door to hallway','go hallway','open door to greenhouse',
                    'go greenhouse','open bee hive'] + ['look in bee hive','wait'] * 30
            exact = 0
            for action in actions:
                before = wm.bridge.take(env.server)
                moves = env.get_num_moves()
                tree = env.getObjectTree()
                python_before = wm.python_state()
                a = wm.predict(action)
                b = wm.predict(action)
                assert same_transition(a,b), 'Repeated snapshot prediction differed'
                assert before.matches(), 'Preview changed hidden JVM fields or RNG'
                assert env.get_num_moves() == moves and env.getObjectTree() == tree, 'Preview changed world or time'
                assert wm.python_state() == python_before, 'Preview changed Python SDK state'
                # Preview a rejected alternative too, then adopt the earlier candidate.
                wm.predict('inventory')
                assert before.matches(), 'Rejected alternative changed the live state'
                actual = wm.execute(action)
                assert wm.states[action][0].matches(), 'Accepted heap differs from saved transition'
                wm.commit(action, actual)
                assert same_transition(a,actual) and a['observation'] == actual['observation']
                assert env.get_num_moves() == a['snapshot_simulator_moves']
                exact += 1
            report.append(dict(task_id=row['task_id'], steps=exact, repeat_equal=True,
                rejected_preview_isolated=True, full_heap_and_rng_verified=True, committed_exact=True))
            print(json.dumps(report[-1]), flush=True)
        finally:
            if wm is not None:
                wm.close()
            env.close()
    assert len(report) == len(failed)
    return dict(env='scienceworld', passed=True, fixture_only=True, checks=report)


def alf():
    import gc
    from worldmodel.resources import configure_temporary_directory
    from worldmodel.backends import OracleWorldModel
    output = ROOT / 'outputs/wm_reflection_setup/runtime_repairs/alf_temp'
    owned = configure_temporary_directory(output)
    spec = importlib.util.spec_from_file_location('ev', ROOT / 'scripts/run_audited_paper_eval.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    rows = json.loads((ROOT / 'data/paper_eval/audited/alfworld.json').read_text())['tasks']
    backend = ev.Alf(SimpleNamespace(wm_backend='oracle'))
    factory = lambda: backend.branch(rows[0])
    step, close, initial = factory()
    def mappings():
        return sum('libdownward.so' in line for line in Path('/proc/self/maps').read_text().splitlines())
    try:
        wm = OracleWorldModel(factory,initial)
        baseline = mappings()
        sizes = []
        for _ in range(80):
            prediction = wm.predict('look')
            assert prediction['observation']
            sizes.append(mappings())
            # Pool lazily allocates one live engine and one preview engine.
            assert sizes[-1] <= 10, 'PDDL libraries grew beyond the two-engine pool'
            gc.collect()
        wm.commit('look',step('look'))
    finally:
        close()
        after_close = mappings()
    replay_checks = []
    trace = Path('/data1/shn/agentwmv2/outputs/wm_reflection/full_7b_v3/oracle/alfworld_qwen25/steps.jsonl')
    recorded = []
    if trace.exists():
        with trace.open() as handle:
            for line in handle:
                value = json.loads(line)
                if value['task_id'] == 'alfworld_115':
                    recorded.append(value['action'])
                    if len(recorded) == 15:
                        break
    for index in (112,115,0):
        branch = lambda row=rows[index]: backend.branch(row)
        live_step, live_close, initial = branch()
        oracle = OracleWorldModel(branch, initial)
        actions = recorded if index == 115 and recorded else ['look','inventory'] * 8
        try:
            for action in actions:
                prediction = oracle.predict(action)
                oracle.commit(action,live_step(action))
                gc.collect()
                assert mappings() <= 10, 'Native engine pool grew during prefix replay'
            replay_checks.append(dict(task_id=rows[index]['task_id'], verified_steps=len(actions)))
        finally:
            live_close()
    owned.cleanup()
    from textworld.envs.pddl.pddl import PddlEnv
    pool = PddlEnv.close._agentwm_pool
    assert len(pool.loaded) == 2 and not pool.active and len(pool.idle) == 2
    assert after_close <= 10, 'Pool exceeded two retained native engines'
    assert len(set(sizes)) == 1, 'Native library mappings grew across repeated previews'
    return dict(env='alfworld', passed=True, fixture_only=True, branches=80,
        pooled_libraries=len(pool.loaded), idle_libraries=len(pool.idle),
        prefix_replay_checks=replay_checks,
        stable_native_library_mappings=sizes, mappings_after_close=after_close,
        temporary_directory=str(output), private_temporary_directory_cleaned=not Path(owned.name).exists())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=['scienceworld','alfworld'], required=True)
    args = p.parse_args()
    result = science() if args.env == 'scienceworld' else alf()
    path = ROOT / 'outputs/wm_reflection_setup/runtime_repairs' / (args.env + '_validation.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
