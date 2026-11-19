"""CPU audit of original real generations, offset compatibility and failed tokens."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from internal_signals.actual_token_nll import ActualTokenNLLProvider
from internal_signals.actual_token_alignment import response_alignment,field_masks,VERSION
from evaluation.independent_dev import atomic
from scripts.run_utility_sampling import rows,read


def main():
    start=time.perf_counter();dest=ROOT/'docs/recovery_v4';dest.mkdir(exist_ok=True)
    t=AutoTokenizer.from_pretrained(ROOT/'checkpoints/qwen35_9b',local_files_only=True)
    cfg=read(ROOT/'configs/internal_signals/qwen35_9b_activation_v1.json');provider=ActualTokenNLLProvider(t,cfg)
    exact=cal=0
    for run in ['activation_experiment_v1','utility_sampling_v2','utility_expansion_v3']:
        for p in (ROOT/'outputs'/run).glob('*/worker*/cases.jsonl'):
            for c in rows(p):
                record=c['metadata']['generation_record'];old=c['metadata']['internal_signals'];new=provider.extract(record).to_dict()
                if old['status']=='ok':
                    if new['status']!='ok':raise ValueError('Formerly valid signal rejected')
                    for k in ['local_uncertainty','global_uncertainty','premise_uncertainty','consequence_uncertainty']:
                        if old[k]!=new[k]:raise ValueError('Canonical NLL changed: '+c['case_id'])
                    exact+=1
                    if run=='activation_experiment_v1' and c['role']=='gate_calibration':cal+=1
    last=rows(ROOT/'outputs/utility_expansion_v3/train_pool/worker0/generation_calls.jsonl')[-1]
    record=last['metadata']['generation_record'];result=provider.extract(record).to_dict()
    if result['status']!='ok':raise ValueError('Actual failed input not repaired')
    text,offsets=response_alignment(t,record['prompt_text'],record['prompt_ids'],record['response_ids'])
    if text!=record['response'] or len(offsets)!=len(record['response_ids']):raise ValueError('Response/offset coverage mismatch')
    if any(a<0 or a>=b or b>len(text) for a,b in offsets):raise ValueError('Invalid repaired offsets')
    if result['global_uncertainty']!=__import__('math').fsum(record['token_nll'])/len(record['token_nll']):raise ValueError('Raw NLL sequence changed')
    if cal!=144:raise ValueError('Original144 calibration scores not checked')
    atomic(dest/'failing_generation_record.json',record)
    proof=dict(status='pass',alignment_version=VERSION,canonical_signals_numerically_identical=exact,original_calibration_scores_exact=cal,
        failed_actual_response_tokens=len(record['response_ids']),failed_reencoded_response_tokens=len(t(record['prompt_text']+text,add_special_tokens=False)['input_ids'])-len(record['prompt_ids']),
        raw_ids_and_nll_not_modified=True,repaired_signal=result,no_model_calls=True,seconds=time.perf_counter()-start)
    atomic(dest/'alignment_cpu_audit.json',proof);print(json.dumps({k:proof[k] for k in ['status','canonical_signals_numerically_identical','original_calibration_scores_exact','failed_actual_response_tokens','failed_reencoded_response_tokens','seconds']},indent=2))


if __name__=='__main__':main()
