"""Offsets for the actual generated IDs; never replace them by canonical IDs."""
from internal_signals.hidden_states import response_alignment as canonical_alignment

VERSION='actual-generated-id-alignment-v1'


def response_alignment(tokenizer,prompt_text,prompt_ids,response_ids,*,left_truncated_tokens=0):
    if not response_ids:raise ValueError('No response tokens')
    original_prompt=tokenizer(prompt_text,add_special_tokens=False)['input_ids']
    if isinstance(left_truncated_tokens,bool) or not isinstance(left_truncated_tokens,int) or not 0<=left_truncated_tokens<len(original_prompt):
        raise ValueError('Invalid left-truncation boundary')
    if original_prompt[left_truncated_tokens:]!=prompt_ids:raise ValueError('Actual model prompt is not the declared exact suffix')
    if any(i in tokenizer.all_special_ids for i in response_ids):raise ValueError('Special token in response scoring range')
    text=tokenizer.decode(response_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=tokenizer(prompt_text+text,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']==original_prompt+response_ids:
        # Preserve the old official offsets exactly for all previously valid inputs.
        return canonical_alignment(tokenizer,prompt_text,prompt_ids,response_ids,left_truncated_tokens=left_truncated_tokens)
    if not text.isascii():raise ValueError('Noncanonical non-ASCII response requires a separately verified byte-offset path')
    if tokenizer.decode(original_prompt+response_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=prompt_text+text:
        raise ValueError('Actual full-token decode does not preserve declared prompt/response boundary')
    offsets=[];previous=''
    for end in range(1,len(response_ids)+1):
        prefix=tokenizer.decode(response_ids[:end],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if not prefix.startswith(previous) or not text.startswith(prefix) or len(prefix)<=len(previous):
            raise ValueError('Actual-ID prefix decoding is not a strict monotone character partition')
        offsets.append((len(previous),len(prefix)));previous=prefix
    if previous!=text or len(offsets)!=len(response_ids):raise ValueError('Incomplete actual-ID offset coverage')
    return text,offsets


def field_masks(actor,meta):
    from internal_signals.token_nll import FORMAT
    r=meta['generation_record'];text,offsets=response_alignment(actor.tokenizer,r['prompt_text'],r['prompt_ids'],r['response_ids'],
        left_truncated_tokens=r.get('left_truncated_prompt_tokens',0))
    if text!=r['response']:raise ValueError('Response text changed')
    m=FORMAT.fullmatch(text)
    if m is None:raise ValueError('Cannot mask an invalid decision')
    return {name:[i for i,(a,b) in enumerate(offsets) if a<m.end(name) and b>m.start(name)] for name in ['premise','consequence','action']}
