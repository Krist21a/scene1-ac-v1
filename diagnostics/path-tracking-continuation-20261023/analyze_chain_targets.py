"""Separate advertised arm interpolation, MPC state and WBC filtering errors."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

PROX = [0,1,2,3,7,8,9,10]
WRIST = [4,5,6,11,12,13]


def evaluate_trajectory(message, t):
    times=np.asarray(message['timeTrajectory'])
    states=np.asarray([s['value'] for s in message['stateTrajectory']])
    if states.ndim != 2 or states.shape[1] != 14 or len(times) != len(states):
        raise ValueError('Expected explicit 14-joint arm trajectory')
    if len(times)==1:
        return states[0]
    index=min(len(times)-2,max(0,int(np.searchsorted(times,t,side='right'))-1))
    dt=times[index+1]-times[index]
    alpha=np.clip((t-times[index])/dt,0.,1.) if dt>0 else float(t>=times[index+1])
    return states[index]*(1-alpha)+states[index+1]*alpha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trace',required=True);ap.add_argument('--steps',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();s=defaultdict(list)
    names={'/humanoid_controller/optimizedState_mrt_origin':'mpc',
           '/humanoid_controller/optimizedState_wbc_mrt_origin':'wbc'}
    for line in Path(a.trace).open():
        r=json.loads(line);topic=r.get('topic')
        if topic is None:continue
        t=r['callback_monotonic_ns']/1e9;m=r['message']
        if topic=='/sensors_data_raw':
            stamp=m['sensor_time'];s['sensor'].append((t,stamp['secs']+stamp['nsecs']/1e9,m['joint_data']['joint_q'][13:27]))
        elif topic=='/humanoid_mpc_observation':s['observation'].append((t,m['time']))
        elif topic=='/kuavo_arm_traj':s['upper'].append((t,np.deg2rad(m['position'])))
        elif topic=='/humanoid_mpc_arm_commanded':s['arm'].append((t,m))
        elif topic in names:s[names[topic]].append((t,m['data'][-8:] if names[topic]=='mpc' else m['data'][-14:]))
    steps=[json.loads(l) for l in Path(a.steps).open()]
    st=np.array([r['send_monotonic_s'] for r in steps]);sp=np.array([r['cartesian_alignment']['state'] for r in steps])
    parcel=[];n=1
    for r in steps:
        parcel.append(n)
        if r['cartesian_alignment']['state']=='visual_cycle_complete':n+=1
    parcel=np.array(parcel)
    t=np.array([x[0] for x in s['sensor']]);sensor_epoch=np.array([x[1] for x in s['sensor']]);q=np.array([x[2] for x in s['sensor']])
    obs=np.array(s['observation']);controller_time=np.interp(t,obs[:,0],obs[:,1])
    valid=(t>=max(st[0],*(s[k][0][0] for k in ['upper','arm','mpc','wbc'])))&(t<=st[-1]+steps[-1]['step_s'])
    data={'measured':q}
    ages={}
    for k in ['upper','arm','mpc','wbc']:
        times=np.array([x[0] for x in s[k]])
        if k in ['upper','arm']:
            index=(np.searchsorted(times,t,side='right')-1).clip(0,len(times)-1)
        else:
            index=np.searchsorted(times,t).clip(0,len(times)-1);previous=np.maximum(0,index-1)
            index=np.where(abs(t-times[previous])<abs(t-times[index]),previous,index)
            valid &= abs(t-times[index])<.025
        ages[k]=abs(t-times[index])
        if k=='arm':data[k]=np.array([evaluate_trajectory(s[k][i][1],when) for i,when in zip(index,controller_time)])
        else:data[k]=np.array([s[k][i][1] for i in index])
    def summarize(mask):
        out={'samples':int(mask.sum())}
        if not mask.any():return out
        for name,left,right,indices in [
            ('upper_to_timed_arm_proximal','upper','arm',PROX),
            ('upper_to_timed_arm_wrist','upper','arm',WRIST),
            ('timed_arm_to_mpc_proximal','arm','mpc',PROX),
            ('mpc_to_wbc_proximal','mpc','wbc',PROX),
            ('wbc_to_measured_proximal','wbc','measured',PROX),
            ('wbc_to_measured_wrist','wbc','measured',WRIST)]:
            l=data[left][mask] if left=='mpc' else data[left][mask][:,indices]
            r=data[right][mask] if right=='mpc' else data[right][mask][:,indices]
            out[name+'_rmse_rad']=float(np.sqrt(np.mean((l-r)**2)))
        return out
    phase_index=(np.searchsorted(st,t,side='right')-1).clip(0,len(st)-1)
    result=dict(all=summarize(valid),per_parcel={},
                clock_note='Absolute sensor epoch is mapped through observed MPC relative time before evaluating trajectory knot times',
                sensor_epoch_minus_mpc_time_spread_ms=float(np.std((sensor_epoch-controller_time)[valid])*1000),
                alignment_age_p95_ms={k:float(np.quantile(v[valid],.95)*1000) for k,v in ages.items()},
                limitations=['Reference vectors lack controller-cycle timestamps; arrival alignment is approximate',
                             'Advertised arm trajectory is evaluated at its knot times, not mistaken for its future endpoint',
                             'This observes solver output, not internal active constraints or cost matrices'])
    for number in np.unique(parcel):
        mask=valid&(parcel[phase_index]==number)
        result['per_parcel'][str(number)]=dict(all=summarize(mask),phases={
            str(phase):summarize(mask&(sp[phase_index]==phase)) for phase in dict.fromkeys(sp[parcel==number])})
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['all'],indent=2))


if __name__=='__main__':main()
