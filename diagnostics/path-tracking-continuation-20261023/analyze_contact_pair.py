"""Matched semantic stages and bounded-PD events; no object truth or SDK torque.

Run inside the simulation container after run_analysis.py; all inputs are files.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('/tmp/scene1-contact-analysis')
sys.path.insert(0,str(BASE))
from scene1.kinematics import UrdfArmKinematics
from analyze_task import nearest

ap=argparse.ArgumentParser()
ap.add_argument('--urdf',required=True)
ap.add_argument('--output',required=True)
ap.add_argument('--figure',required=True)
ap.add_argument('--baseline-label',default='baseline-contact-seed0')
ap.add_argument('--candidate-label',default='pd-contact-seed0')
a=ap.parse_args()
fk=UrdfArmKinematics(description=Path(a.urdf).read_text())
WRIST=[4,5,6,11,12,13]
report={}
fig,axes=plt.subplots(2,2,figsize=(12,8))
for label,title,color in [(a.baseline_label,'Baseline','#4b5563'),
                          (a.candidate_label,'Bounded wrist PD','#d97706')]:
    rows=[json.loads(l) for l in (BASE/label/'steps.jsonl').open()]
    st=np.array([r['send_monotonic_s'] for r in rows])
    states=[r['cartesian_alignment']['state'] for r in rows]
    first_grasp=states.index('grasp_contact_confirmed')
    weigh=states.index('small_weighing')
    retreat=states.index('small_short_retreat')
    stage_indices={'approach_and_grasp':(0,first_grasp+1),
                   'loaded_transport':(first_grasp+1,weigh),
                   'weigh_and_release':(weigh,retreat),
                   'completed_prefix':(0,retreat)}
    sensor=[];wbc=[];pd=[]
    for line in (BASE/label/'chain.jsonl').open():
        r=json.loads(line)
        if 'topic' not in r:continue
        t=r['callback_monotonic_ns']/1e9;m=r['message']
        if r['topic']=='/sensors_data_raw':
            stamp=m['sensor_time'];jd=m['joint_data']
            sensor.append((t,stamp['secs']+stamp['nsecs']/1e9,jd['joint_q'][13:27],jd['joint_v'][13:27]))
        elif r['topic']=='/humanoid_controller/optimizedState_wbc_mrt_origin':
            wbc.append((t,m['data'][-14:]))
        elif r['topic']=='/scene1_tracking/wrist_pd':
            pd.append((t,m['data']))
    t=np.array([r[0] for r in sensor]);sim=np.array([r[1] for r in sensor])
    q=np.array([r[2] for r in sensor]);v=np.array([r[3] for r in sensor])
    wt=np.array([r[0] for r in wbc]);wq=np.array([r[1] for r in wbc])
    ref,age=nearest(t,wt,wq)
    error=ref-q
    out=dict(stages={},pd=None)
    for name,(lo,hi) in stage_indices.items():
        start,end=st[lo],st[hi]
        selected=(t>=start)&(t<end)&(age<.025)
        joints=error[selected];times=t[selected]
        idx=np.flatnonzero(selected)[::5]
        tcp=np.array([[np.linalg.norm(fk.matrix(ref[i,7*j:7*j+7],j)[:3,3]-
                                     fk.matrix(q[i,7*j:7*j+7],j)[:3,3]) for j in [0,1]] for i in idx])
        endpoint_idx=[int(np.argmin(abs(t-x))) for x in [start,end]]
        sr=rows[lo:hi]
        offset_total=0.;offset_pairs=0
        for old,new in zip(sr,sr[1:]):
            oi=old['cartesian_alignment'];ni=new['cartesian_alignment']
            if oi['state']==ni['state'] and 'offset_rad' in oi and 'offset_rad' in ni:
                offset_total+=float(np.linalg.norm(np.array(ni['offset_rad'])-np.array(oi['offset_rad'])))
                offset_pairs+=1
        out['stages'][name]=dict(steps=hi-lo,first_step=sr[0]['step'],last_step=sr[-1]['step'],
                wall_s=float(end-start),sensor_clock_s=float(sim[endpoint_idx[1]]-sim[endpoint_idx[0]]),
                endpoint_pair_error_ms=[float(abs(t[i]-x)*1000) for i,x in zip(endpoint_idx,[start,end])],
                samples=int(selected.sum()),wrist_wbc_rmse_rad=float(np.sqrt(np.mean(joints[:,WRIST]**2))),
                arm_wbc_rmse_rad=float(np.sqrt(np.mean(joints**2))),
                joint_mean_error_rad=np.mean(joints,axis=0).tolist(),
                joint_rmse_rad=np.sqrt(np.mean(joints**2,axis=0)).tolist(),
                tcp_fk_mean_error_mm=(np.mean(tcp,axis=0)*1000).tolist(),
                tcp_fk_p95_error_mm=(np.quantile(tcp,.95,axis=0)*1000).tolist(),
                frozen_act_steps=sum(bool(r['policy_paused_for_alignment']) for r in sr),
                hold_or_wait_steps=sum(bool(r['cartesian_alignment'].get('pose_hold')) or 'waiting' in r['cartesian_alignment']['state'] or 'sensor_hold' in r['cartesian_alignment']['state'] for r in sr),
                logged_visual_offset_variation_rad=offset_total if offset_pairs else None,
                logged_visual_offset_pairs=offset_pairs,
                bounds_clipped_steps=sum(bool(r['motion']['bounds_clipped']) for r in sr))
        if name=='weigh_and_release':
            axes[0,0].plot(times-start,np.sqrt(np.mean(joints[:,WRIST]**2,axis=1)),color=color,label=title,lw=1.2)
            axes[0,1].plot(t[idx]-start,tcp[:,1]*1000,color=color,label=title,lw=1.2)
    if pd:
        pt=np.array([r[0] for r in pd]);px=np.array([r[1] for r in pd])
        raw=np.max(abs(px[:,14:28][:,WRIST]),axis=1)
        enabled=px[:,56]>.5
        clips=enabled&(raw>.5+1e-9)
        events=[];start_i=None
        for i in range(len(clips)+1):
            is_clip=bool(clips[i]) if i<len(clips) else False
            if is_clip and start_i is None:start_i=i
            if not is_clip and start_i is not None:
                end_i=i-1
                ix=int(np.searchsorted(st,pt[start_i],side='right')-1)
                endix=int(np.searchsorted(st,pt[end_i],side='right')-1)
                events.append(dict(first_wall_s=float(pt[start_i]-st[0]),last_wall_s=float(pt[end_i]-st[0]),
                    observed_duration_wall_s=float(pt[end_i]-pt[start_i]),samples=end_i-start_i+1,
                    raw_pd_peak_nm=float(raw[start_i:end_i+1].max()),
                    first_step=rows[ix]['step'] if ix>=0 else None,
                    last_step=rows[endix]['step'] if endix>=0 else None,
                    first_phase=states[ix] if ix>=0 else 'initialization',
                    clipped_joint_indices=sorted({j for x in px[start_i:end_i+1] for j in WRIST if abs(x[14+j])>.5+1e-9})))
                start_i=None
        out['pd']=dict(clip_events=events,
                       semantics='raw PD is exact controller diagnostic; 50Hz recorded duration under-resolves event boundaries')
        subset=enabled&(pt>=st[0]-5)&(pt<st[-1])
        axes[1,0].plot(pt[subset]-st[0],raw[subset],color=color,lw=1.1,label='Max raw PD over 6 wrists')
        axes[1,0].axhline(.5,color='#991b1b',ls='--',label='0.5 Nm added-torque limit')
        axes[1,0].axvspan(-5,0,color='#eeeeee',label='End of go_run initialization')
        for x,txt in [(st[first_grasp],'Grasp'),(st[weigh],'Weigh')]:
            axes[1,0].axvline(x-st[0],color='#9ca3af',ls=':',lw=.8)
            axes[1,0].text(x-st[0],.02,txt,rotation=90,va='bottom',fontsize=8)
    report[label]=out

for i,(label,title,color) in enumerate([(a.baseline_label,'Baseline','#4b5563'),(a.candidate_label,'Bounded wrist PD','#d97706')]):
    vals=[report[label]['stages'][k]['wall_s'] for k in ['approach_and_grasp','loaded_transport','weigh_and_release']]
    axes[1,1].bar(np.arange(3)+(i-.5)*.32,vals,width=.32,color=color,label=title)
axes[1,1].set_xticks(np.arange(3));axes[1,1].set_xticklabels(['Approach + grasp','Loaded transport','Weigh + release'])
axes[0,0].set(title='Wrist tracking during weighing',xlabel='Wall seconds from weighing entry',ylabel='6-joint instantaneous RMS error (rad)')
axes[0,1].set(title='Right tool FK reference-to-feedback difference',xlabel='Wall seconds from weighing entry',ylabel='Derived FK difference (mm)')
axes[1,0].set(title='Candidate PD: initialization tail and active policy',xlabel='Wall seconds from first policy action',ylabel='Raw requested added torque (Nm)')
axes[1,1].set(title='Complete stages before guarded stop',ylabel='Wall seconds')
for ax in axes.flat:
    ax.grid(alpha=.18);ax.legend(fontsize=8)
fig.suptitle('Official scene1 seed 0 | same v69 policy | one reset per controller\nClosed-loop references differ; neither run completes a parcel',fontsize=12)
fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(a.figure,dpi=170);plt.close(fig)
report['limitations']=['One development seed, one run each; no success-rate inference',
    'No matched static plateau in candidate; stage mean error is not static bias',
    'Complete weighing stage excludes proof of stable post-release support; settling/regrasp/handover/bin untested',
    'FK is robot-model reference-to-feedback difference, not independent tool or object truth',
    'Visual offset coverage varies; absent fields do not mean zero correction or zero vision compute']
Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v['stages'] for k,v in report.items() if isinstance(v,dict)},indent=2))
