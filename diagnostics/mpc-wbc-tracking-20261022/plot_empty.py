"""Standalone scientific plot of raw sensed wrist angles in matched probes."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--include-pd',action='store_true');a=ap.parse_args()
runs=[('baseline-empty-v2','Baseline','#555555'),('candidate-empty','Reference index fix','#2c987c'),('gains-empty','Wrist gains 120 / 12.4924','#2475c7')]
if a.include_pd:runs.append(('pd-empty-v2','Post-WBC bounded PD','#d97706'))
fig,axes=plt.subplots(3,2,figsize=(11,8),sharex=True)
for run,label,color in runs:
    seg=[json.loads(l) for l in open('/tmp/'+run+'-probe.jsonl') if json.loads(l).get('kind')=='segment']
    start=next(s for s in seg if s['name']=='distal_out')['start_monotonic_ns']/1e9
    t=[];q=[];cmdt=[];cmd=[]
    for l in open('/tmp/'+run+'-chain.jsonl'):
        r=json.loads(l)
        if 'topic' not in r:continue
        dt=r['callback_monotonic_ns']/1e9-start
        if not -.2<dt<14.1:continue
        if r['topic']=='/sensors_data_raw':t.append(dt);q.append(r['message']['joint_data']['joint_q'][13:27])
    for l in open('/tmp/'+run+'-probe.jsonl'):
        r=json.loads(l)
        if r.get('kind')=='command':
            dt=r['monotonic_ns']/1e9-start
            if -.2<dt<14.1:cmdt.append(dt);cmd.append(r['q_rad'])
    q=np.rad2deg(q);cmd=np.rad2deg(cmd)
    for row,j in enumerate([4,5,6]):
        for side in [0,1]:
            ax=axes[row,side];ix=j+7*side
            ax.plot(t,q[:,ix],label=label,color=color,linewidth=1.3,alpha=.85)
            if run==runs[0][0]:
                ax.plot(cmdt,cmd[:,ix],label='Published target',color='black',linestyle='--',linewidth=1.2)
                ax.axvspan(0,2,color='#eeeeee');ax.axvspan(7,9,color='#eeeeee')
                ax.set_title(('Left' if side==0 else 'Right')+' wrist joint '+str(j+1))
                ax.grid(alpha=.2);ax.set_ylabel('Angle (degree)')
for ax in axes[-1]:ax.set_xlabel('Wall seconds from outward ramp start')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.99))
fig.suptitle('Official scene1, seed 0; empty-hand probe; unchanged physics and limits',y=1.025,fontsize=13)
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(a.output,dpi=170,bbox_inches='tight');plt.close(fig)
