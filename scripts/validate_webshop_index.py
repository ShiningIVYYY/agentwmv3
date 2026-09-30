#!/usr/bin/env python3
"""Real JVM/search/goal/purchase checks, without model calls or benchmark scores."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).resolve().with_name('run_audited_paper_eval.py')
spec = importlib.util.spec_from_file_location('evaluator', path)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def main():
    backend = ev.Web(SimpleNamespace(max_steps=15))
    previous = json.loads((ev.ROOT / 'data/paper_eval/audited/webshop.json').read_text())['tasks']
    current = backend.tasks()
    assert len(current) == len(previous) == 500
    assert [(r['index'], r['goal_sha256']) for r in current] == [
        (r['index'], r['goal_sha256']) for r in previous], 'Heldout goals changed'
    from web_agent_site.engine.engine import get_top_n_product_from_keywords
    searches = []
    old = ev.ROOT / 'outputs/paper_eval/audited/webshop_ee_il/tasks'
    for file in sorted(old.glob('*.json')):
        row = json.loads(file.read_text())
        if row['success']:
            continue
        step = next((s for s in row['steps'] if s['action'].startswith('search[')), None)
        if step and 'Total results: 0' in step['next_observation']:
            query = step['action'][7:-1]
            products = get_top_n_product_from_keywords(query.split(), backend.env.server.search_engine,
                backend.env.server.all_products, backend.env.server.product_item_dict)
            assert products, 'Previously empty query is still empty: ' + query
            searches.append(dict(task_id=row['task_id'], products=len(products)))
        if len(searches) == 20:
            break
    assert len(searches) == 20
    # Test fixture: target-aware replay ONLY checks environment semantics; it is
    # not a model evaluation and never enters benchmark task output directories.
    purchases = []
    for index in (0, 2, 3):
        backend.env.reset(session=index)
        goal = backend.env.server.goals[index]
        target = goal['asin']
        obs, reward, done, _ = backend.env.step('search[<q> ' + goal['query'] + ']')
        for page in range(100):
            available = [a.lower() for a in backend.env.get_available_actions()['clickables']]
            if target.lower() in available:
                break
            assert 'next >' in available, 'Fixture target not reachable'
            backend.env.step('click[next >]')
        assert target.lower() in available
        backend.env.step('click[' + target.lower() + ']')
        for option in goal['goal_options'].values():
            backend.env.step('click[' + option.lower() + ']')
        obs, reward, done, _ = backend.env.step('click[buy now]')
        assert done and reward == 1.0, (index, reward, obs)
        assert 'Purchased' in obs
        purchases.append(dict(index=index, done=done, reward=reward))
    report = dict(status='passed', catalog_size=len(backend.env.server.product_item_dict),
        index_documents=backend.env.server.search_engine.num_docs,
        fixed_goals_unchanged=500, recovered_searches=searches, fixture_purchases=purchases,
        data_sha256=backend.data)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
