"""Single-generation format automata. They constrain syntax, never environment truth.

Raw model logits are collected BEFORE this processor. Fixed label/delimiter tokens
are explicit decoding constraints, not sampled evidence or fabricated field values.
"""
import json
import torch
from transformers import LogitsProcessor


class FormatMachine(LogitsProcessor):
    def __init__(self, tokenizer, prompt_length, token_limit, kind='draft', draft=None):
        if kind not in ('draft','review'):raise ValueError('Unknown format')
        self.tok=tokenizer;self.prompt_length=prompt_length;self.limit=token_limit
        self.kind=kind;self.draft=draft;self.position=0;self.pending=[];self.after=None
        self.state='literal';self.content='';self.field_index=0;self.choice='';self.decision=None
        self.forced_budget_transitions=0;self.mask_cache={}
        self.eos=tokenizer.eos_token_id
        self.nl=tokenizer.encode('\n',add_special_tokens=False)
        self.quote=tokenizer.encode('"',add_special_tokens=False)
        if len(self.nl)!=1 or len(self.quote)!=1:raise ValueError('Unverified delimiter tokenization')
        tables=getattr(tokenizer,'_agentwm_format_tables_v1',None)
        if tables is None:
            special=set(tokenizer.all_special_ids)
            pieces=tokenizer.batch_decode([[i] for i in range(len(tokenizer))],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            line=[i for i,s in enumerate(pieces) if i not in special and s and not s[-1].isspace() and '\ufffd' not in s and not any(ord(c)<32 for c in s) and '<|' not in s]
            js=[i for i in line if '"' not in pieces[i] and '\\' not in pieces[i]]
            tables=(pieces,line,js,[i for i in line if pieces[i].startswith(' ') and pieces[i].strip()],[i for i in js if pieces[i].strip()])
            tokenizer._agentwm_format_tables_v1=tables
            tokenizer._agentwm_format_masks_v1={}
        pieces,line,js,nonempty_line,nonempty_json=tables
        self.pieces=pieces
        self.line_ids=line;self.json_ids=js
        # A leading space belongs to the MODEL'S first value token, avoiding a
        # forced standalone-space token that would violate exact retokenization.
        self.nonempty_line=nonempty_line;self.nonempty_json=nonempty_json
        self.pool_cache=getattr(tokenizer,'_agentwm_format_pools_v1',{})
        tokenizer._agentwm_format_pools_v1=self.pool_cache
        self.literal('Premise:' if kind=='draft' else '{"decision":"','field' if kind=='draft' else 'decision')

    def literal(self, text, after):
        self.pending=self.tok.encode(text,add_special_tokens=False);self.state='literal';self.after=after
        if not self.pending:self.state=after

    def enum_start(self, values, after):
        self.options={value:self.tok.encode(value,add_special_tokens=False) for value in values}
        self.enum_prefix=[];self.state='enum';self.enum_after=after

    def consume(self, token):
        if self.state=='literal':
            if not self.pending or token!=self.pending[0]:raise ValueError('Literal constraint violation')
            self.pending.pop(0)
            if not self.pending:
                self.state=self.after
                if self.state=='decision':self.enum_start(['KEEP','REPLACE'],'after_decision')
                elif self.state=='reason':
                    self.enum_start(['trial_supported','no_grounded_alternative'] if self.decision=='KEEP' else ['trial_failed','trial_harmful'],'after_reason')
            return
        if self.state=='enum':
            self.enum_prefix.append(token)
            matches={k:v for k,v in self.options.items() if v[:len(self.enum_prefix)]==self.enum_prefix}
            if not matches:raise ValueError('Enum constraint violation')
            full=[k for k,v in matches.items() if len(v)==len(self.enum_prefix)]
            if full:
                if any(len(v)>len(self.enum_prefix) for v in matches.values()):raise ValueError('Ambiguous enum prefix')
                value=full[0]
                if self.enum_after=='after_decision':
                    self.decision=value
                    if value=='KEEP':
                        self.literal('","final_action":'+json.dumps(self.draft,ensure_ascii=False)+',"reason_code":"','reason')
                    else:self.literal('","final_action":"','field')
                else:self.literal('"}','done')
            return
        if self.state=='field':
            if self.kind=='draft' and token==self.nl[0]:
                if not self.content.strip() or self.field_index>=2:raise ValueError('Empty/unexpected line')
                self.field_index+=1;self.content='';self.literal(['','Expected Consequence:','Action:'][self.field_index],'field')
            elif self.kind=='review' and token==self.quote[0]:
                if not self.content.strip():raise ValueError('Empty revised action')
                self.literal(',"reason_code":"','reason')
            elif token==self.eos:
                if self.kind!='draft' or self.field_index!=2 or not self.content.strip():raise ValueError('Premature EOS')
                self.state='finished'
            else:self.content+=self.pieces[token]
            return
        if self.state=='done' and token==self.eos:self.state='finished';return
        raise ValueError('Generation outside grammar')

    def allowed(self, used):
        left=self.limit-used
        if self.state=='literal':return [self.pending[0]]
        if self.state=='enum':return sorted({v[len(self.enum_prefix)] for v in self.options.values() if v[:len(self.enum_prefix)]==self.enum_prefix})
        if self.state in ('done','finished'):return [self.eos]
        if self.state!='field':raise ValueError('Invalid grammar state')
        has=bool(self.content.strip())
        if self.kind=='draft':
            if self.field_index<2:
                suffix=('\nExpected Consequence: x\nAction: x' if self.field_index==0 else '\nAction: x')
                reserve=len(self.tok.encode(suffix,add_special_tokens=False))+2
                if has and left<=reserve:
                    self.forced_budget_transitions+=1;return self.nl
                if not has:return self.nonempty_line
                return self.pool('line_newline') if not self.content[-1].isspace() else self.line_ids
            if has and left<=1:self.forced_budget_transitions+=1;return [self.eos]
            return self.pool('line_eos') if has else self.nonempty_line
        # JSON string limits come from the existing Reviewer schema, not task-specific rules.
        suffix='",'+'"reason_code":"trial_harmful"}'
        reserve=len(self.tok.encode(suffix,add_special_tokens=False))+2
        if has and (left<=reserve or len(self.content)>=256):
            self.forced_budget_transitions+=1;return self.quote
        return self.pool(('json',has,256-len(self.content)))

    def pool(self,key):
        if key not in self.pool_cache:
            if key=='line_newline':value=self.line_ids+self.nl
            elif key=='line_eos':value=self.line_ids+[self.eos]
            else:
                _,has,remaining=key
                value=[i for i in (self.json_ids if has else self.nonempty_json) if len(self.pieces[i])<=remaining]
                if has:value+=self.quote
            self.pool_cache[key]=value
        return self.pool_cache[key]

    def __call__(self, input_ids, scores):
        if input_ids.shape[0]!=1:raise ValueError('Only single unpadded generation validated')
        emitted=input_ids[0,self.prompt_length:].tolist()
        while self.position<len(emitted):self.consume(emitted[self.position]);self.position+=1
        allowed=self.allowed(len(emitted))
        # Fresh tensor: do not mutate HF's retained pre-processor logits.
        key=(str(scores.device),scores.shape[-1],id(allowed))
        masks=self.tok._agentwm_format_masks_v1
        if len(allowed)<30:
            result=torch.full_like(scores,-torch.inf);result[:,allowed]=scores[:,allowed];return result
        if key not in masks:
            mask=torch.ones(scores.shape[-1],device=scores.device,dtype=torch.bool);mask[allowed]=False;masks[key]=mask
        return scores.masked_fill(masks[key],-torch.inf)

    def verify_finished(self, emitted):
        while self.position<len(emitted):self.consume(emitted[self.position]);self.position+=1
        if self.state!='finished':raise ValueError('Format generation did not finish within unchanged token cap')
        return dict(version='three-line-and-compact-json-fsm-v1',kind=self.kind,
                    budget_forced_transitions=self.forced_budget_transitions,
                    content_source='frozen_model_generated_tokens',labels='forced_syntax',
                    nll_source='raw_model_logits_before_format_mask_and_sampling')
