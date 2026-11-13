"""Preregister ONLY the missing Experience-Only arm on the frozen Phase5.9 cohort."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic
from experience.splits import file_hash


def main():
    baseline=ROOT/'outputs/phase59_independent_v2/manifest.json';m=json.loads(baseline.read_text());base=m['profile']
    current=json.loads((ROOT/'configs/evaluation/phase59_independent_dev_v1.json').read_text())
    assert current==base,'Phase5.9 profile differs from its actual run'
    out=ROOT/'docs/phase59_external';out.mkdir(exist_ok=True)
    source=subprocess.check_output(['git','show',m['source_git_commit']+':evaluation/independent_dev.py'],cwd=ROOT)
    expected=m['code_hashes']['evaluation/independent_dev.py'];assert hashlib.sha256(source).hexdigest()==expected
    (out/'frozen_runtime.py').write_bytes(source)
    for path,sha in base['method_hashes'].items():assert file_hash(ROOT/path)==sha,path
    p=dict(base);p['wm_invocation_budgets']=dict(base['wm_invocation_budgets'],experience=5)
    p.update(purpose='phase59_missing_experience_only_arm_same_frozen_method',added_policy='experience',
        baseline_profile='configs/evaluation/phase59_independent_dev_v1.json',baseline_profile_sha256=file_hash(ROOT/'configs/evaluation/phase59_independent_dev_v1.json'),
        baseline_actual_manifest_sha256=file_hash(baseline),baseline_runtime_commit=m['source_git_commit'],
        frozen_runtime='docs/phase59_external/frozen_runtime.py',frozen_runtime_sha256=expected,
        no_internal_extraction=True,reviewer_version='original',reviewer_response_cap=512,
        limits=dict(max_workers=2,tokens=650000,model_calls=650,transition_queries=3500,audit_capture_bundles=18000,max_wall_seconds=3600),
        worker_limits=dict(tokens=325000,model_calls=325,transition_queries=1750,audit_capture_bundles=9000))
    atomic(ROOT/'configs/evaluation/phase59_external_only_v1.json',p)
    print(json.dumps(dict(tasks=len(p['tasks']),seeds=p['rollout_seeds'],new_episodes=24,policy='experience',budget=5,reviewer='original',runtime_sha256=expected)))


if __name__=='__main__':main()
