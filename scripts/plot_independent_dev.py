#!/usr/bin/env python3
"""Standalone descriptive research figure from completed genuine results."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    d=Path(a.analysis);run=Path(a.run);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    pol=json.loads((d/'policy_summary.json').read_text());paired=json.loads((d/'paired_summary.json').read_text())
    extra=json.loads((d/'external_added_calls.json').read_text());gate=json.loads((run/'manifest.json').read_text())['profile']
    config=json.loads((Path(__file__).resolve().parents[1]/gate['gate_config']).read_text())
    colors=dict(never='#777777',always='#D98B26',internal='#397BA8',full='#35946B')
    fig,axes=plt.subplots(2,2,figsize=(15,11),layout='constrained')
    ax=axes[0,0]
    for r in pol:
        x=r['total_tokens']/r['instances']/1000;y=r['mean_task_score'];ci=r['score_resampling_interval'];c=colors[r['policy']]
        ax.vlines(x,ci['low'],ci['high'],color=c,alpha=.65,linewidth=2)
        ax.scatter([x],[y],color=c,s=65,zorder=3)
        ax.annotate(r['policy'],(x,y),xytext={'never':(6,8),'always':(6,8),'internal':(6,-15),'full':(-30,12)}[r['policy']],textcoords='offset points',fontsize=10,color=c)
    ax.axhline(0,color='black',alpha=.2);ax.set(xlabel='Mean LLM tokens / episode (thousands)',ylabel='Mean raw Task Score',
        title='A  Score–tokens points\nDescriptive goal-cluster intervals')
    effects=next(r for r in paired if r['comparison']=='full-never')['per_task_mean_score_delta']
    items=sorted(effects.items(),key=lambda p:p[1]);ax=axes[0,1]
    ax.barh([k.replace('sciworld_','') for k,v in items],[v for k,v in items],color=['#35946B' if v>=0 else '#BA605F' for k,v in items])
    ax.axvline(0,color='black',linewidth=.8);ax.set(xlabel='Full − Never Task Score, mean over two seeds',ylabel='Task ID',title='B  Task contributions (no task removed)')
    ax=axes[1,0];locals_=[];globals_=[]
    for path in (d/'decisions').glob('*-full.csv'):
        with path.open() as f:
            for r in csv.DictReader(f):
                if r['local_nll']!='' and r['global_nll']!='':locals_.append(float(r['local_nll']));globals_.append(float(r['global_nll']))
    if locals_:ax.scatter(locals_,globals_,s=18,color='#397BA8',alpha=.35,label='Full realized states')
    ex=[c for c in extra['calls'] if c['internal_signals'] and c['internal_signals']['local_uncertainty'] is not None and c['internal_signals']['global_uncertainty'] is not None]
    if ex:ax.scatter([c['internal_signals']['local_uncertainty'] for c in ex],[c['internal_signals']['global_uncertainty'] for c in ex],marker='*',s=105,color='#D98B26',label='External-added Call')
    ax.axvline(config['tau_local'],color='#777777',linestyle='--',linewidth=1)
    ax.axhline(config['tau_global'],color='#777777',linestyle='--',linewidth=1)
    ax.set(xlabel='Local token NLL (nats/token)',ylabel='Global token NLL (nats/token)',title='C  Full states; fixed thresholds and actual added calls')
    ax.legend(fontsize=9)
    ax=axes[1,1];x=np.arange(len(pol));bottom=np.zeros(len(pol))
    parts=[('Formal + formal init',[r['formal_steps']+r['instances'] for r in pol],'#8CA1B3'),
        ('Worker init',[r['simulation_cost'].get('initialization_step_calls',0) for r in pol],'#77B6A0'),
        ('History replay',[r['simulation_cost'].get('replay_step_calls',0) for r in pol],'#E4AC61'),
        ('Candidate trial',[r['simulation_cost'].get('candidate_step_calls',0) for r in pol],'#5279AC')]
    for label,values,c in parts:
        v=np.asarray(values);ax.bar(x,v,bottom=bottom,label=label,color=c);bottom+=v
    ax.set_xticks(x,[r['policy'] for r in pol]);ax.set(ylabel='Native transition queries',title='D  Environment query cost, separate from LLM tokens')
    ax.legend(fontsize=9)
    fig.suptitle('Independent development: 12 tasks × 2 rollout seeds, 8 goal clusters\nFrozen Qwen3.5-9B + Oracle-Like ScienceWorld 1.3.0 lookahead',fontsize=14)
    fig.savefig(out.with_suffix('.png'),dpi=160);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)
    print(out.with_suffix('.png'))


if __name__=='__main__':main()
