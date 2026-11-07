#!/usr/bin/env python3
"""Minimal real SDK reproduction of RNG-consuming valid-action queries and heat drift."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scienceworld import ScienceWorldEnv
from worldmodel.ground_truth_environment import ReplayScienceEnvironment, rng_state


def tree_differences(left, right, path='', limit=20):
    result = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            result.extend(tree_differences(left.get(key), right.get(key), path+'/'+key, limit))
            if len(result) >= limit:
                break
    elif left != right:
        result.append(dict(path=path, left=left, right=right))
    return result[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    queries = []
    for _ in range(2):
        env = ScienceWorldEnv()
        try:
            env.load('use-thermometer', 8, '', False)
            before = rng_state(env)
            env.get_valid_action_object_combinations()
            queries.append(dict(before=before, after=rng_state(env)))
        finally:
            env.close()
    spec = dict(taskName='use-thermometer', variationIdx=8, simplifications='', seed=0, environment_step_limit=21)
    envs = [ReplayScienceEnvironment(spec), ReplayScienceEnvironment(spec)]
    try:
        for action in ['look around', 'inventory', 'this is an invalid action', 'wait']:
            for env in envs:
                env.step(action)
        trees = [env.env.getObjectTree() for env in envs]
        result = dict(task=spec, valid_action_query_rng=queries,
                      prefix=['look around', 'inventory', 'this is an invalid action', 'wait'],
                      selected_rng=[rng_state(e.env) for e in envs],
                      tree_differences=tree_differences(*trees), full_hidden_state_equivalence_claimed=False)
    finally:
        for env in envs:
            env.close()
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
