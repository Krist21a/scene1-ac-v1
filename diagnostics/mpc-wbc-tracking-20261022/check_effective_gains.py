"""Plateau acceleration/error slope, corroborates config use without torque alias."""
import argparse
import json
from pathlib import Path
import collections
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--trace',required=True);ap.add_argument('--probe',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
segments=[json.loads(l) for l in open(a.probe) if json.loads(l).get('kind')=='segment']
windows=[(s['start_monotonic_ns']/1e9+5,s['start_monotonic_ns']/1e9+7) for s in segments if s['name'].startswith('distal')]
r=collections.defaultdict(list)
for line in open(a.trace):
    x=json.loads(line)
    if 'topic' not in x:continue
    t=x['callback_monotonic_ns']/1e9
    if not any(lo<t<hi for lo,hi in windows):continue
    topic=x['topic'];m=x['message']
    if topic=='/sensors_data_raw':v=m['joint_data']['joint_q'][13:27]
    elif topic in ['/humanoid_controller/optimizedState_wbc_mrt_origin','/humanoid_controller/wbc_planned_joint_acc']:v=m['data'][-14:]
    else:continue
    r[topic].append((t,v))
sensor=r['/sensors_data_raw'];t=np.array([x[0] for x in sensor]);q=np.array([x[1] for x in sensor]);out=[]
for topic in ['/humanoid_controller/optimizedState_wbc_mrt_origin','/humanoid_controller/wbc_planned_joint_acc']:
    tt=np.array([x[0] for x in r[topic]]);v=np.array([x[1] for x in r[topic]])
    ix=np.searchsorted(tt,t).clip(1,len(tt)-1);ix=np.where(abs(tt[ix-1]-t)<abs(tt[ix]-t),ix-1,ix);out.append(v[ix])
err=out[0]-q;acc=out[1]
result={}
for j in [4,5,6,11,12,13]:
    ok=abs(err[:,j])>.01
    result[str(j)]={'samples':int(ok.sum()),'median_acceleration_per_error_s2':float(np.median(acc[ok,j]/err[ok,j]))}
Path(a.output).write_text(json.dumps(dict(joints=result,limitation='Approximate static slope using arrival-aligned raw sensor q; WBC acceleration output is planned, not measured. Damping and QP residual remain; not full gain identification.'),indent=2)+'\n')
print(json.dumps(result))
