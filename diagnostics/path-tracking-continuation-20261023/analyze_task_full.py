"""Observed chain and phase metrics. No scene/object truth; no SDK torque aliases."""
import argparse
import collections
import json
from pathlib import Path
import sys
import numpy as np


def stamp_ns(s):
    return s['secs']*1000000000+s['nsecs']


def stats(a):
    a=np.asarray(a)
    return dict(n=int(a.size),mean=float(np.mean(a)),p95=float(np.quantile(a,.95)),max=float(np.max(a))) if a.size else None


def nearest(target, times, values):
    ix=np.searchsorted(times,target).clip(0,len(times)-1)
    prev=np.maximum(ix-1,0)
    ix=np.where(abs(times[prev]-target)<abs(times[ix]-target),prev,ix)
    return values[ix],abs(times[ix]-target)


def main():
    ap=argparse.ArgumentParser()
    for x in ['trace','steps','output','project','urdf']:
        ap.add_argument('--'+x,required=True)
    args=ap.parse_args()
    sys.path.insert(0,args.project)
    from scene1.kinematics import UrdfArmKinematics
    k=UrdfArmKinematics(description=Path(args.urdf).read_text())
    steps=[json.loads(l) for l in open(args.steps)]
    st=np.array([r['send_monotonic_s'] for r in steps])
    phases=np.array([r.get('cartesian_alignment',{}).get('state','unknown') for r in steps])
    joints=[*range(7),*range(8,15)]
    st_dt=np.r_[np.diff(st),steps[-1]['step_s']]
    rtime=collections.defaultdict(list);data=collections.defaultdict(list)
    counts=meta=None
    for line in open(args.trace):
        r=json.loads(line)
        if r.get('kind')=='counts':counts=r;continue
        if r.get('kind')=='metadata':meta=r;continue
        if 'topic' not in r:continue
        t=r['callback_monotonic_ns']/1e9
        if t<st[0]-1 or t>st[-1]+steps[-1]['step_s']+1:continue
        topic=r['topic'];m=r['message']
        if topic=='/sensors_data_raw':
            jd=m['joint_data'];value=[stamp_ns(m['header']['stamp'])/1e9,stamp_ns(m['sensor_time'])/1e9,jd['joint_q'][13:27],jd['joint_v'][13:27],jd['joint_torque'][13:27]]
        elif topic=='/kuavo_arm_traj':value=[stamp_ns(m['header']['stamp'])/1e9,np.deg2rad(m['position']).tolist()]
        elif topic=='/joint_cmd':value=m['tau'][13:27]
        elif topic in ['/humanoid_controller/optimizedState_wbc_mrt_origin','/humanoid_controller/optimizedInput_wbc_mrt_origin','/humanoid_controller/wbc_planned_joint_acc']:
            value=m['data'][-14:]
        elif topic in ['/monitor/time_cost/controller_loop_time','/monitor/time_cost/wbc','/monitor/time_cost/mpc']:
            value=m['data']
        else:continue
        rtime[topic].append(t);data[topic].append(value)
    sensor=data['/sensors_data_raw'];t=np.array(rtime['/sensors_data_raw'])
    source_t=np.array([v[0] for v in sensor]);sim_t=np.array([v[1] for v in sensor])
    q=np.array([v[2] for v in sensor]);qd=np.array([v[3] for v in sensor]);tau=np.array([v[4] for v in sensor])
    cp=data['/kuavo_arm_traj'];ct=np.array([v[0] for v in cp]);cq=np.array([v[1] for v in cp])
    ci=np.searchsorted(ct,source_t,side='right')-1
    command=cq[ci.clip(0,len(cq)-1)]
    wtopic='/humanoid_controller/optimizedState_wbc_mrt_origin'
    wq,age=nearest(t,np.array(rtime[wtopic]),np.array(data[wtopic]))
    tvtopic='/humanoid_controller/optimizedInput_wbc_mrt_origin'
    wv,vage=nearest(t,np.array(rtime[tvtopic]),np.array(data[tvtopic]))
    torquecmd,tage=nearest(t,np.array(rtime['/joint_cmd']),np.array(data['/joint_cmd']))
    step_sim_time=np.interp(np.r_[st,st[-1]+steps[-1]['step_s']],t,sim_t)
    step_sim_duration=np.diff(step_sim_time)
    phase_index=np.searchsorted(st,t,side='right')-1
    valid=(phase_index>=0)&(t<=st[-1]+steps[-1]['step_s'])&(ci>=0)&(age<.025)&(vage<.025)
    sample_phase=phases[phase_index.clip(0,len(phases)-1)]
    limit=np.array([66,75,57,75,14.1,14.1,14.1]*2)
    command_difference=q-command;wbc_difference=q-wq
    # Fixed plateau criterion: previous 1s command/reference ranges <=3mrad,
    # present measured max speed <=0.03rad/s. Report if no eligible samples.
    plateau=np.zeros(len(t),dtype=bool)
    for i in range(len(t)):
        start=np.searchsorted(t,t[i]-1.)
        if i-start>=30 and (np.ptp(command[start:i+1],axis=0).max()<.003
                            and np.ptp(wq[start:i+1],axis=0).max()<.003
                            and abs(qd[i]).max()<.03):plateau[i]=True
    # FK difference of robot joint reference and measured joint state in the
    # same base frame. Not an independent optical TCP measurement/calibration.
    fk_index=np.flatnonzero(valid)[::5]
    fk_cmd=[];fk_wbc=[]
    for i in fk_index:
        a=[];b=[]
        for side in [0,1]:
            sl=slice(7*side,7*side+7)
            observed=k.matrix(q[i,sl],side)
            a.append(np.linalg.norm(k.matrix(command[i,sl],side)[:3,3]-observed[:3,3]))
            b.append(np.linalg.norm(k.matrix(wq[i,sl],side)[:3,3]-observed[:3,3]))
        fk_cmd.append(a);fk_wbc.append(b)
    fk_cmd=np.array(fk_cmd);fk_wbc=np.array(fk_wbc)
    report=dict(run_id=meta['run_id'],task_success=None,independent_full_success=None,
        interpretation={'tracking':'time-aligned error includes dynamic lag, not necessarily static error',
                        'wbc_alignment':'unstamped vectors matched by callback arrival <=25ms',
                        'force':'raw joint_torque is simulation qfrc_actuator per publisher source; tau is requested torque; tau_max is not treated as Nm',
                        'fk':'URDF FK difference, same base frame; not an independent end-effector measurement',
                        'visual':'outer residual variation includes intended visual trajectory changes; not pure corrective motion or vision compute',
                        'plateau':'1s command and WBC reference ranges <=0.003rad and measured speed <=0.03rad/s'},
        duration_s=float(st[-1]-st[0]+steps[-1]['step_s']),steps=len(steps),
        policy_sim_elapsed_s=float(step_sim_time[-1]-step_sim_time[0]),
        sampled_sim_time_per_wall_time=float((sim_t[-1]-sim_t[0])/(t[-1]-t[0])),
        wbc_pair_error_ms=stats(age[valid]*1000),counts=counts,
        missing=['QP active constraint flags','precise WBC sample timestamp','actuator apply timestamp','independent TCP measurement','vision-only compute time'],phases={})
    raw=np.array([r['act_action'] for r in steps])[:,joints]
    proposed=np.array([r['proposed_action'] for r in steps])[:,joints]
    sent=np.array([r['sent_action'] for r in steps])[:,joints]
    joint_limits=np.vstack([k.limits[0],k.limits[1]])
    command_at_bound=np.minimum(sent-joint_limits[:,0],joint_limits[:,1]-sent)<=1e-6
    feedback_near_bound=np.minimum(q-joint_limits[:,0],joint_limits[:,1]-q)<=.001
    feedback_outside_bound=np.minimum(q-joint_limits[:,0],joint_limits[:,1]-q)<-.0005
    report['joint_bound_definition']=dict(source='runtime URDF limits, not QP active flags',joint_order='left7,right7',command_distance_rad=1e-6,feedback_distance_rad=.001,feedback_outside_tolerance_rad=.0005,limits_rad=joint_limits.tolist())
    residual=proposed-raw
    delta_sent=np.r_[0,np.linalg.norm(np.diff(sent,axis=0),axis=1)]
    delta_residual=np.r_[0,np.linalg.norm(np.diff(residual,axis=0),axis=1)]
    same_phase=np.r_[False,phases[1:]==phases[:-1]]
    for phase in dict.fromkeys(phases):
        ps=phases==phase;vs=valid&(sample_phase==phase);ss=vs&plateau
        fs=sample_phase[fk_index]==phase
        record=dict(steps=int(ps.sum()),elapsed_s=float(st_dt[ps].sum()),
          elapsed_sim_s=float(step_sim_duration[ps].sum()),tracking_samples=int(vs.sum()),
          proximal_wbc_rmse_rad=float(np.sqrt(np.mean(wbc_difference[vs][:,[0,1,2,3,7,8,9,10]]**2))) if vs.any() else None,
          per_joint_wbc_rmse_rad=np.sqrt(np.mean(wbc_difference[vs]**2,axis=0)).tolist() if vs.any() else None,
          command_tracking_rmse_rad=float(np.sqrt(np.mean(command_difference[vs]**2))) if vs.any() else None,
          wbc_tracking_rmse_rad=float(np.sqrt(np.mean(wbc_difference[vs]**2))) if vs.any() else None,
          wrist_wbc_rmse_rad=float(np.sqrt(np.mean(wbc_difference[vs][:,[4,5,6,11,12,13]]**2))) if vs.any() else None,
          plateau_samples=int(ss.sum()),plateau_bias_rad=np.mean(wbc_difference[ss],axis=0).tolist() if ss.any() else None,
          command_fk_error_m=stats(fk_cmd[fs]),wbc_fk_error_m=stats(fk_wbc[fs]),
          actuator_limit_near_samples=int(np.any(abs(tau[vs])>=.99*limit,axis=1).sum()),
          requested_torque_above_actuator_range_samples=int(np.any(abs(torquecmd[vs])>limit,axis=1).sum()),
          command_at_joint_bound_steps_per_joint=np.sum(command_at_bound[ps],axis=0).tolist(),
          feedback_near_joint_bound_samples_per_joint=np.sum(feedback_near_bound[vs],axis=0).tolist(),
          feedback_outside_joint_bound_samples_per_joint=np.sum(feedback_outside_bound[vs],axis=0).tolist(),
          feedback_max_joint_bound_excess_rad_per_joint=np.maximum(0,-np.minimum(q[vs]-joint_limits[:,0],joint_limits[:,1]-q[vs])).max(axis=0).tolist() if vs.any() else None,
          bounds_clipped_steps=sum(bool(r['motion']['bounds_clipped']) for r,p in zip(steps,ps) if p),
          velocity_limited_steps=sum(bool(r['motion']['velocity_limited']) for r,p in zip(steps,ps) if p),
          frozen_policy_steps=sum(bool(r['policy_paused_for_alignment']) for r,p in zip(steps,ps) if p),
          explicit_hold_steps=sum(bool(r.get('cartesian_alignment',{}).get('pose_hold')) or 'waiting' in phase or 'sensor_hold' in phase for r,p in zip(steps,ps) if p),
          within_phase_command_variation_rad=float(delta_sent[ps&same_phase].sum()),
          within_phase_outer_residual_variation_rad=float(delta_residual[ps&same_phase].sum()),
          visual_pose_errors=[r['cartesian_alignment']['pose_errors'] for r,p in zip(steps,ps) if p and 'pose_errors' in r['cartesian_alignment']])
        # Summarize pose errors separately without treating planned parcel pose as truth.
        pe=record.pop('visual_pose_errors')
        record['visual_pose_error_records']=len(pe)
        report['phases'][phase]=record
    parcel=1;parcel_steps=[];completed=set()
    for r in steps:
        parcel_steps.append(parcel)
        if r['cartesian_alignment'].get('state')=='visual_cycle_complete':
            completed.add(parcel);parcel+=1
    parcel_steps=np.asarray(parcel_steps)
    sample_parcel=parcel_steps[phase_index.clip(0,len(steps)-1)]
    report['per_parcel']={}
    for number in np.unique(parcel_steps):
        ps=parcel_steps==number;vs=valid&(sample_parcel==number)
        fs=sample_parcel[fk_index]==number
        report['per_parcel'][str(number)]=dict(
            visually_complete=bool(number in completed),steps=int(ps.sum()),
            command_at_joint_bound_steps_per_joint=np.sum(command_at_bound[ps],axis=0).tolist(),
            feedback_near_joint_bound_samples_per_joint=np.sum(feedback_near_bound[vs],axis=0).tolist(),
            feedback_outside_joint_bound_samples_per_joint=np.sum(feedback_outside_bound[vs],axis=0).tolist(),
            feedback_max_joint_bound_excess_rad_per_joint=np.maximum(0,-np.minimum(q[vs]-joint_limits[:,0],joint_limits[:,1]-q[vs])).max(axis=0).tolist() if vs.any() else None,
            elapsed_s=float(st_dt[ps].sum()),elapsed_sim_s=float(step_sim_duration[ps].sum()),
            tracking_samples=int(vs.sum()),
            wrist_wbc_rmse_rad=float(np.sqrt(np.mean(wbc_difference[vs][:,[4,5,6,11,12,13]]**2))) if vs.any() else None,
            proximal_wbc_rmse_rad=float(np.sqrt(np.mean(wbc_difference[vs][:,[0,1,2,3,7,8,9,10]]**2))) if vs.any() else None,
            wbc_fk_error_m=stats(fk_wbc[fs]))
        detail={}
        for phase in dict.fromkeys(phases[ps]):
            pps=ps&(phases==phase);pvs=vs&(sample_phase==phase);pfs=fs&(sample_phase[fk_index]==phase)
            pss=pvs&plateau
            item=dict(steps=int(pps.sum()),elapsed_s=float(st_dt[pps].sum()),
                      elapsed_sim_s=float(step_sim_duration[pps].sum()),tracking_samples=int(pvs.sum()),
                      plateau_samples=int(pss.sum()),
                      per_joint_wbc_rmse_rad=np.sqrt(np.mean(wbc_difference[pvs]**2,axis=0)).tolist() if pvs.any() else None,
                      per_joint_command_rmse_rad=np.sqrt(np.mean(command_difference[pvs]**2,axis=0)).tolist() if pvs.any() else None,
                      plateau_bias_rad=np.mean(wbc_difference[pss],axis=0).tolist() if pss.any() else None,
                      wbc_fk_error_m=stats(fk_wbc[pfs]),
                      frozen_policy_steps=sum(bool(r['policy_paused_for_alignment']) for r,include in zip(steps,pps) if include),
                      explicit_hold_steps=sum(bool(r.get('cartesian_alignment',{}).get('pose_hold')) or 'waiting' in phase or 'sensor_hold' in phase for r,include in zip(steps,pps) if include))
            for group,indices in [('proximal',[0,1,2,3,7,8,9,10]),('wrist',[4,5,6,11,12,13])]:
                item[group+'_wbc_rmse_rad']=float(np.sqrt(np.mean(wbc_difference[pvs][:,indices]**2))) if pvs.any() else None
                item[group+'_command_rmse_rad']=float(np.sqrt(np.mean(command_difference[pvs][:,indices]**2))) if pvs.any() else None
                item[group+'_upper_to_wbc_rmse_rad']=float(np.sqrt(np.mean((wq[pvs]-command[pvs])[:,indices]**2))) if pvs.any() else None
                item[group+'_measured_velocity_rms_rad_s']=float(np.sqrt(np.mean(qd[pvs][:,indices]**2))) if pvs.any() else None
            detail[str(phase)]=item
        report['per_parcel'][str(number)]['phases']=detail
    for topic in ['/monitor/time_cost/controller_loop_time','/monitor/time_cost/wbc','/monitor/time_cost/mpc']:
        report[topic]=stats(data[topic])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['run_id','duration_s','steps','sampled_sim_time_per_wall_time']},indent=2))


if __name__=='__main__':
    main()
