"""Offsets for actual constrained output IDs, including noncanonical segmentation.

Forced syntax can split a BPE token (e.g. '.\\n'). Retokenizing the resulting
text would change the model's generated IDs and cannot be used to score them.
We verify the original prompt suffix and exact joint/per-token decoding instead.
Byte fragments, empty pieces, specials and decode-boundary drift fail closed.
"""
import math,time
from internal_signals.base import InternalSignals
from internal_signals.token_nll import summarize_fields


def generated_alignment(tokenizer,record):
    prompt=record['prompt_text'];pids=record['prompt_ids'];ids=record['response_ids']
    all_prompt=tokenizer(prompt,add_special_tokens=False)['input_ids']
    removed=record.get('left_truncated_prompt_tokens',0)
    if isinstance(removed,bool) or not isinstance(removed,int) or not 0<=removed<len(all_prompt):raise ValueError('Invalid prompt truncation')
    if all_prompt[removed:]!=pids:raise ValueError('Original prompt token suffix mismatch')
    if not ids or any(i in tokenizer.all_special_ids for i in ids):raise ValueError('Empty/special response tokens')
    pieces=tokenizer.batch_decode([[i] for i in ids],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    if any(not p or '\ufffd' in p for p in pieces):raise ValueError('Empty or incomplete byte-decoding piece')
    response=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    if ''.join(pieces)!=response or response!=record['response']:raise ValueError('Per-token/joint/record decoding mismatch')
    if tokenizer.decode(all_prompt+ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=prompt+response:
        raise ValueError('Actual token stream crosses prompt boundary')
    offsets=[];at=0
    for p in pieces:offsets.append((at,at+len(p)));at+=len(p)
    canonical=tokenizer(prompt+response,add_special_tokens=False)['input_ids']==all_prompt+ids
    return response,offsets,canonical


def extract_signal(provider,record):
    start=time.perf_counter();c=provider.config
    for k,v in [('backend',provider.backend),('dtype',c['dtype']),('model_revision',c['model_revision'])]:
        if record[k]!=v:raise ValueError('Signal identity mismatch: '+k)
    values=record['token_nll'];errors=[];details={};local=premise=consequence=glob=None
    try:
        if not values or len(values)!=len(record['response_ids']) or any(not math.isfinite(v) or v<0 for v in values):
            raise ValueError('Invalid actual-token NLL')
        text,offsets,canonical=generated_alignment(provider.tokenizer,record)
        details=summarize_fields(text,offsets,values)
        details.update(offsets=offsets,alignment_method='actual_generated_token_stream_offsets_v1',canonical_retokenization_equal=canonical)
        glob=details['global_uncertainty'];local=details['local_uncertainty'];premise=details['premise_uncertainty'];consequence=details['consequence_uncertainty']
        if details['missing_span']:errors.append(details['missing_reason'])
    except ValueError as e:errors.append('alignment_or_numeric_error: '+str(e))
    if record.get('truncated'):errors.append('generation_truncated')
    fatal=any(e.startswith(('alignment_or_numeric_error','generation_truncated')) for e in errors)
    if fatal:
        local=premise=consequence=glob=None
        details.update(missing_span=True,format_valid=False,missing_reason='; '.join(errors))
    missing=details.get('missing_span',True)
    fallback=glob if missing and not fatal and c['missing_policy']=='global_only' else None
    details.update(fallback_policy=c['missing_policy'],fallback_uncertainty=fallback,fallback_applied=fallback is not None,
        effective_uncertainty=local if not missing else fallback,method='fixed_field_mean_token_nll_v1',score_unit='nats_per_token',
        calibrated_error_probability=False,prompt_tokens=len(record['prompt_ids']),generation_tokens_including_eos=record['generation_tokens'],
        scoring_path='raw_generation_logits_before_temperature',additional_backbone_forward_passes=0,raw_logit_reduction_seconds=record['reduction_seconds'])
    return InternalSignals(premise,consequence,local,glob,bool(missing),len(values),record['reduction_seconds']+time.perf_counter()-start,
        provider.backend,c['dtype'],c['model_revision'],'partial' if missing else 'ok',errors,details)
