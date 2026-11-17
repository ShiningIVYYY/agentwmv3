"""Project-specific supervised mean/MLP/EMA-QV probes; foundation model not trained."""
import math,torch
from torch import nn


class ActivationProbe(nn.Module):
    def __init__(self,dim,kind='linear',outputs=1,width=128,heads=4,rho=.99):
        super().__init__();self.kind=kind;self.rho=rho
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('std',torch.ones(dim))
        if kind=='linear':self.head=nn.Linear(dim,outputs)
        else:
            self.mlp=nn.Sequential(nn.Linear(dim,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
            if kind=='ema':
                self.query=nn.Parameter(torch.randn(heads,width)/math.sqrt(width));self.value=nn.Linear(width,heads)
                self.head=nn.Linear(heads,outputs)
            elif kind=='mlp':self.head=nn.Linear(width,outputs)
            else:raise ValueError('Unknown probe architecture')

    def forward(self,h,mask):
        if not mask.any(dim=1).all():raise ValueError('Empty sequence mask')
        x=(h.float()-self.mean)/self.std.clamp_min(1e-3)
        if self.kind!='linear':x=self.mlp(x)
        if self.kind=='ema':
            scores=torch.einsum('btd,hd->bth',x,self.query)/math.sqrt(x.shape[-1])
            age=mask.sum(1)[:,None]-1-torch.arange(h.shape[1],device=h.device)[None,:]
            scores=scores+age[:,:,None]*math.log(self.rho)
            weights=scores.masked_fill(~mask[:,:,None],float('-inf')).softmax(1)
            pooled=(weights*self.value(x)).sum(1)
        else:pooled=(x*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        return self.head(pooled)

    def stream_step(self,h,state=None):
        if self.training:raise ValueError('Streaming step is inference-only; train the differentiable masked batch path')
        if self.kind!='ema':raise ValueError('Streaming pooling requires EMA architecture')
        x=self.mlp((h.float()-self.mean)/self.std.clamp_min(1e-3))
        score=(self.query*x).sum(-1)/math.sqrt(x.shape[-1]);value=self.value(x)
        if state is None:
            maximum=score;numerator=value;denominator=torch.ones_like(score)
        else:
            oldmax,n,d=state;decayed=oldmax+math.log(self.rho);maximum=torch.maximum(decayed,score)
            a=(decayed-maximum).exp();b=(score-maximum).exp();numerator=a*n+b*value;denominator=a*d+b
        return self.head(numerator/denominator),tuple(x.detach() for x in (maximum,numerator,denominator))


def pad_batch(values):
    if not values:raise ValueError('Empty batch')
    length=max(v.shape[0] for v in values);dim=values[0].shape[1]
    out=torch.zeros(len(values),length,dim,dtype=torch.float32);mask=torch.zeros(len(values),length,dtype=torch.bool)
    for i,v in enumerate(values):out[i,:len(v)]=v.float();mask[i,:len(v)]=True
    return out,mask


def binary_metrics(labels,scores):
    import numpy as np
    y=np.asarray(labels,dtype=int);s=np.asarray(scores,dtype=float)
    if len(y)==0:return dict(n=0,positive=0,auroc=None,average_precision=None)
    pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:return dict(n=len(y),positive=pos,auroc=None,average_precision=None)
    order=np.argsort(-s,kind='stable');y=y[order];s=s[order]
    tp=fp=0;ap=area=0.;last_recall=last_fpr=0.;i=0
    while i<len(y):
        j=i+1
        while j<len(y) and s[j]==s[i]:j+=1
        tp+=int(y[i:j].sum());fp+=(j-i)-int(y[i:j].sum())
        recall=tp/pos;fpr=fp/neg
        ap+=(recall-last_recall)*tp/(tp+fp)
        area+=(fpr-last_fpr)*(recall+last_recall)/2
        last_recall,last_fpr=recall,fpr;i=j
    return dict(n=len(y),positive=pos,auroc=float(area),average_precision=float(ap),calibrated_probability=False)
