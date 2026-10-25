"""Standalone plots of retained experiments; no smoothing or success selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
cases=[('Original','original-empty-s0'),('C','c-empty-s0'),('D','d-empty-s0'),
       ('D+C','dc-empty-s0'),('E','e-empty-s0')]
stages=['proximal_out','proximal_return','distal_out','distal_return']
labels=['Proximal out','Proximal return','Wrist out','Wrist return']
data={name:{s['name']:s for s in json.loads((a.root/run/'metrics.json').read_text())['segments']}
      for name,run in cases}
fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
metrics=[('ramp_command_rmse_rad','Moving error vs upper command','rad'),
         ('static_abs_bias_mean_rad','Mean absolute holding bias','rad'),
         ('settle_within_1deg_s','Settling within 1 degree','wall-clock s')]
x=np.arange(4);width=.15
for ax,(key,title,unit) in zip(axes,metrics):
    for i,(name,_) in enumerate(cases):
        values=[data[name][stage][key] for stage in stages]
        positions=x+(i-2)*width
        ax.bar(positions,[0 if value is None else value for value in values],width,label=name)
        for position,value in zip(positions,values):
            if value is None:
                ax.text(position,.03,'N/R',rotation=90,ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation=20,ha='right')
    ax.set_title(title);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.25)
axes[0].legend(ncol=3,fontsize=9)
fig.suptitle('Fixed empty-arm probe: every controller candidate retained',fontsize=14)
fig.text(.5,-.025,'One independent seed-0 reset per candidate. N/R: did not settle in the observation window. Research evidence; no official task-success claim.',ha='center',fontsize=9)
fig.savefig(a.output/'empty-ablation.png',dpi=180,bbox_inches='tight')
fig.savefig(a.output/'empty-ablation.pdf',bbox_inches='tight')
plt.close(fig)

comparison=a.root/'comparison.json'
if comparison.exists():
    trials=[r for r in json.loads(comparison.read_text())['tasks'] if r.get('requested_parcels')==4]
    if trials:
        fig,ax=plt.subplots(figsize=(10,max(3,len(trials)*.6)),constrained_layout=True)
        for i,r in enumerate(trials):
            count=r.get('visual_confirmed_parcels') or 0
            ax.barh(i,count,color='#32874e' if r.get('four_parcel_visual_complete') else '#b5683d')
            duration=r.get('policy_duration_wall_s')
            ax.text(count+.05,i,f"{count}/4; {duration:.1f} s" if duration is not None else f'{count}/4',va='center',fontsize=9)
        ax.set_yticks(np.arange(len(trials)))
        ax.set_yticklabels([r['label'] for r in trials]);ax.invert_yaxis()
        ax.set_xlim(0,5.1);ax.set_xticks(range(5));ax.set_xlabel('Parcels passing the retained visual completion checks')
        ax.set_title('All four-parcel attempts, including failures')
        ax.grid(axis='x',alpha=.25)
        fig.text(.5,-.04,'Failed-run duration is not a speed benefit. Official task_success remains separately reported, including null.',ha='center',fontsize=9)
        fig.savefig(a.output/'full-attempts.png',dpi=180,bbox_inches='tight')
        fig.savefig(a.output/'full-attempts.pdf',bbox_inches='tight')
        plt.close(fig)
