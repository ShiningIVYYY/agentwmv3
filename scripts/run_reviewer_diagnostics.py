#!/usr/bin/env python3
"""Reuse the unmodified Phase5 runner/Science.run for a single named diagnostic."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_scienceworld_validation as base
from diagnostics.reviewer import PredictionOnly,ValidityChecked
from experience.splits import file_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True,choices=['never','original','prediction_only','validity_checked','fixed_initializer_original'])
    p.add_argument('--output',required=True);p.add_argument('--science-url',default='http://127.0.0.1:36055')
    p.add_argument('--wm-url',default='http://127.0.0.1:18212');p.add_argument('--model-dir',required=True)
    p.add_argument('--wm-checkpoint',required=True);p.add_argument('--science-python',required=True)
    a=p.parse_args();out=Path(a.output).resolve()
    if out.exists():raise ValueError('Use a new diagnostic run ID')
    setup=ROOT/'outputs/phase55_setup';setup.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/'docs/phase5/development_plan.json').read_text())
    plan['policies']=['never' if a.variant=='never' else 'always']
    plan['diagnostic_variant']=a.variant
    plan_path=setup/(a.variant+'_plan.json')
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Frozen plan changed')
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    sources=[Path(__file__),*sorted((ROOT/'diagnostics').glob('*.py'))]
    provenance={'variant':a.variant,'algorithm_change':a.variant in ('prediction_only','validity_checked'),
        'original_algorithm_unchanged':True,'source_hashes':{str(f.relative_to(ROOT)):file_hash(f) for f in sources},
        'frozen_inputs':{f:file_hash(ROOT/f) for f in ('configs/gates/dev_distribution_phase5_v1.json','docs/phase5/memory_snapshot_v2.json','configs/internal_signals/scienceworld_token_nll_v1.json','configs/evaluation/paper_aligned_v1/scienceworld.json')},
        'plan_sha256':file_hash(plan_path),'rollout_seed':0,'tasks':plan['validation_tasks']}
    provenance['initializer_policy']='validated-room-initializer-v1' if a.variant=='fixed_initializer_original' else 'released-collector-unchanged'
    if a.variant=='fixed_initializer_original':
        provenance['initializer_files']={str(f.relative_to(ROOT)):file_hash(f) for f in sorted((setup/'validated_initializers').glob('sciworld_*.json')) if '.attempt' not in f.name}
    (setup/(a.variant+'_diagnostic_manifest.json')).write_text(json.dumps(provenance,indent=2)+'\n')
    archived=setup/(a.variant+'_source');archived.mkdir()
    for source in sources:
        dest=archived/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    old=base.runner.wm_runtime.make_controller
    old_setup=base.ExistingWorldModelSetup
    if a.variant=='fixed_initializer_original':
        from diagnostics.initialization import ValidatedWorldModelSetup
        base.ExistingWorldModelSetup=ValidatedWorldModelSetup
    def make(*args,**kwargs):
        ctrl=old(*args,**kwargs)
        if a.variant=='prediction_only':return PredictionOnly(ctrl)
        if a.variant=='validity_checked':return ValidityChecked(ctrl)
        return ctrl
    base.runner.wm_runtime.make_controller=make
    sys.argv=['run_scienceworld_validation.py','--mode','episodes','--plan',str(plan_path),
        '--model-dir',a.model_dir,'--wm-checkpoint',a.wm_checkpoint,'--science-python',a.science_python,
        '--science-url',a.science_url,'--wm-url',a.wm_url,'--threshold-config','configs/gates/dev_distribution_phase5_v1.json',
        '--snapshot','docs/phase5/memory_snapshot_v2.json','--output',str(out)]
    try:base.main()
    finally:
        base.runner.wm_runtime.make_controller=old
        base.ExistingWorldModelSetup=old_setup
        if out.exists():
            (out/'diagnostic_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
            shutil.copytree(archived,out/'executed_diagnostic_source')


if __name__=='__main__':main()
