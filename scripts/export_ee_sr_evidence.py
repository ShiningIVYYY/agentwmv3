#!/usr/bin/env python3
"""Compact genuine SR compatibility/prompt-control evidence; no inference."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluation.independent_dev import atomic,read_rows
from experience.splits import file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',default='outputs/ee_sr_analysis_v1');ap.add_argument('--output',default='docs/ee_sr_trial/results');a=ap.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('New evidence output required')
    shutil.copytree(a.analysis,out);runs={'sr':Path('outputs/ee_sr_actor_trial_v1'),'qwen35_native_control':Path('outputs/ee_qwen35_native_control_v1')}
    for label,run in runs.items():
        for n in ['manifest.json','model.json','numerical_validation.json','ledger.json','complete.json']:
            if (run/n).exists():shutil.copyfile(run/n,out/(label+'_'+n))
        tracefile=out/(label+'_decisions.jsonl');count=0
        with tracefile.open('w') as f:
            for path in sorted((run/'episodes').rglob('steps.jsonl')):
                for t in read_rows(path):
                    m=t['metadata'];t={k:v for k,v in t.items() if k!='metadata'}
                    t['model_metadata']=dict(model_revision=m['shared_model_revision'],config_sha256=m['config_sha256'],usage=m['usage'],finish_reason=m['finish_reason'],
                        generation_seconds=m['internal_cost']['generation_seconds'],peak_allocated_bytes=m['internal_cost']['peak_allocated_bytes'],internal_signals=m.get('internal_signals'))
                    f.write(json.dumps(t,ensure_ascii=False)+'\n');count+=1
        failures=[]
        for path in sorted((run/'episodes').rglob('parse_failures.jsonl')):
            for t in read_rows(path):
                m=t['metadata'];failures.append(dict(path=str(path.relative_to(run)),raw_unexecuted_output=t['raw'],parsed=t['parsed'],usage=m['usage'],finish_reason=m['finish_reason'],
                    internal_signals=m.get('internal_signals'),response_ids=m['generation_record']['response_ids'],prompt_messages=m['generation_record']['messages']))
        atomic(out/(label+'_format_failures.json'),failures)
        initial=[dict(path=str(p.relative_to(run)),**json.loads(p.read_text())) for p in sorted((run/'episodes').rglob('initial.json'))]
        atomic(out/(label+'_initial_states.json'),initial)
        atomic(out/(label+'_raw_index.json'),[dict(path=str(p.relative_to(run)),sha256=file_hash(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.suffix in ('.json','.jsonl')])
    atomic(out/'export_provenance.json',dict(script_sha256=file_hash(Path(__file__)),analysis_script_sha256=file_hash(ROOT/'scripts/analyze_ee_sr_trial.py'),
        structured_unexecuted_outputs_are_not_environment_feedback=True,no_clean_independent_generalization_claim=True))
    files=[p for p in out.rglob('*') if p.is_file()];assert not [p for p in files if p.stat().st_size>1048576]
    print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files)),indent=2))


if __name__=='__main__':main()
