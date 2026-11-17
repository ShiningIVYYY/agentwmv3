"""Post-token response-only hooks on actual discovered Qwen3.5 text blocks."""
import inspect,time,torch


class ActivationCapture:
    def __init__(self,actor,layers=None):
        self.actor=actor
        cfg=actor.model.config.text_config
        matches=[(name,module) for name,module in actor.model.named_modules()
                 if isinstance(module,torch.nn.ModuleList) and len(module)==cfg.num_hidden_layers
                 and 'DecoderLayer' in type(module[0]).__name__]
        if len(matches)!=1:raise ValueError('Ambiguous decoder discovery: '+str([(n,len(m)) for n,m in matches]))
        self.path,self.blocks=matches[0];self.dim=cfg.hidden_size
        self.layers=layers or sorted(set([cfg.num_hidden_layers//3,cfg.num_hidden_layers//2-1,2*cfg.num_hidden_layers//3]))
        if any(not 0<=i<len(self.blocks) for i in self.layers):raise ValueError('Invalid layer')
        if 'logits_to_keep' not in inspect.signature(actor.model.forward).parameters:raise ValueError('Unverified low-memory forward API')
        self.description=dict(schema='post-token-full-forward-v1',hidden_dim=self.dim,num_layers=len(self.blocks),layers=self.layers,
            capture_point='decoder_block_output_before_final_norm',module_path=self.path,
            layer_types={str(i):cfg.layer_types[i] for i in self.layers},dtype='bfloat16',response_only=True)

    @torch.inference_mode()
    def extract(self,record):
        allowed={'prompt_ids','response_ids','prompt_text','response'}
        if set(record)!=allowed:raise ValueError('Feature whitelist rejects outcomes or future metadata')
        p=len(record['prompt_ids']);t=len(record['response_ids'])
        if not p or not t:raise ValueError('Empty aligned response')
        actor=self.actor
        if actor.tokenizer.decode(record['response_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False)!=record['response']:
            raise ValueError('Response token alignment changed')
        data=torch.tensor([record['prompt_ids']+record['response_ids']],device=actor.device,dtype=torch.long)
        values={};handles=[]
        def hook(index):
            def save(module,args,output):
                h=output[0] if isinstance(output,tuple) else output
                if h.shape!=(1,p+t,self.dim):raise ValueError('Unexpected hidden tensor shape: '+str(h.shape))
                values[str(index)]=h[0,p:p+t].detach().to(device='cpu',dtype=torch.bfloat16).clone()
            return save
        actor.sync()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats(actor.device)
        start=time.perf_counter()
        try:
            for i in self.layers:handles.append(self.blocks[i].register_forward_hook(hook(i)))
            actor.model(input_ids=data,attention_mask=torch.ones_like(data),use_cache=False,logits_to_keep=1,return_dict=True)
            actor.sync()
        finally:
            for h in handles:h.remove()
        if set(values)!={str(i) for i in self.layers}:raise ValueError('Missing hook receipt')
        if any(not torch.isfinite(h).all() for h in values.values()):raise ValueError('Nonfinite hidden states')
        return values,dict(seconds=time.perf_counter()-start,additional_backbone_forward_passes=1,prompt_tokens=p,response_tokens=t,
            bytes=sum(h.numel()*h.element_size() for h in values.values()),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(actor.device) if torch.cuda.is_available() else None,
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(actor.device) if torch.cuda.is_available() else None,**self.description)


def feature_record(meta):
    r=meta['generation_record']
    return {k:r[k] for k in ['prompt_ids','response_ids','prompt_text','response']}
