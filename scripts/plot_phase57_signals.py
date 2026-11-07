#!/usr/bin/env python3
"""Plot actual frozen dev NLL records; no policy/outcome optimization."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=Path(a.run)
    rows=[json.loads(s) for s in (run/'calibration_steps.jsonl').read_text().splitlines()]
    valid=[r['metadata']['internal_signals'] for r in rows if r['metadata']['internal_signals']['status']=='ok']
    x=np.array([s['global_uncertainty'] for s in valid]);y=np.array([s['local_uncertainty'] for s in valid])
    c=json.loads((run/'gate_config.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),layout='constrained')
    axes[0].scatter(x,y,s=24,alpha=.8,color='#245a91')
    axes[0].axhline(c['tau_local'],linestyle='--',color='#a94b32',label='q=.75 thresholds')
    axes[0].axvline(c['tau_global'],linestyle='--',color='#a94b32')
    axes[0].set(xlabel='Global NLL (nats/token)',ylabel='Local NLL (nats/token)',title=f'Pearson r={np.corrcoef(x,y)[0,1]:.3f}')
    axes[0].legend(fontsize=8)
    axes[1].hist([x,y],bins=8,label=['Global','Local'],color=['#8daac6','#c57b60'],alpha=.8)
    axes[1].set(xlabel='NLL (nats/token)',ylabel='Decision steps',title='Observed distributions')
    axes[1].legend(fontsize=8)
    fig.suptitle(f'Development pilot: {len(valid)} steps / {len(set(r["task_id"] for r in rows))} tasks; not threshold stability evidence',fontsize=10)
    fig.savefig(a.output,dpi=160);plt.close(fig)


if __name__=='__main__':main()
