"""Check first reference knot against named v52 arm and shifted joint order."""
import argparse
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--trace',required=True)
ap.add_argument('--output',required=True)
args=ap.parse_args()
st=[];sq=[];targets=[]
for l in open(args.trace):
    r=json.loads(l)
    if r.get('topic')=='/sensors_data_raw':
        st.append(r['callback_monotonic_ns']/1e9)
        sq.append(r['message']['joint_data']['joint_q'])
    elif r.get('topic')=='/humanoid_mpc_target_arm':
        m=r['message']
        targets.append((r['callback_monotonic_ns']/1e9,m['stateTrajectory'][0]['value'],m['timeTrajectory']))
st=np.array(st);sq=np.array(sq)
good=[];shift=[];age=[];horizon=[]
for t,q,tt in targets:
    i=np.searchsorted(st,t).clip(1,len(st)-1)
    i=i-1 if abs(st[i-1]-t)<abs(st[i]-t) else i
    if abs(st[i]-t)>.025:continue
    q=np.array(q)
    if len(q)!=14:continue
    good.append(q-sq[i,13:27]);shift.append(q-sq[i,12:26]);age.append(abs(st[i]-t))
    horizon.append(tt[-1]-tt[0])
report=dict(samples=len(good),correct_arm_order_rmse_rad=float(np.sqrt(np.mean(np.array(good)**2))),
            waist_shifted_order_rmse_rad=float(np.sqrt(np.mean(np.array(shift)**2))),
            callback_pair_error_ms_p95=float(np.quantile(age,.95)*1000),
            reference_horizon_s_median=float(np.median(horizon)),
            limitation='Reference generated from estimator WBC observation; comparison uses raw robot sensors and nearest callback arrival, not exact source matching.')
Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
