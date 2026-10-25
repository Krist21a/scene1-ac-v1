"""Offline probe analysis. Arrival aligned vectors, stable plateaus separated."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True)
    ap.add_argument('--probe', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    rows = collections.defaultdict(list)
    metadata = counts = None
    for line in open(args.trace):
        r = json.loads(line)
        if r.get('kind') == 'metadata':
            metadata = r
        elif r.get('kind') == 'counts':
            counts = r
        elif 'topic' in r:
            rows[r['topic']].append(r)
    segments = [json.loads(l) for l in open(args.probe) if json.loads(l).get('kind') == 'segment']
    sensor = rows['/sensors_data_raw']
    times = np.array([r['callback_monotonic_ns']/1e9 for r in sensor])
    q = np.array([r['message']['joint_data']['joint_q'][13:27] for r in sensor])
    v = np.array([r['message']['joint_data']['joint_v'][13:27] for r in sensor])
    tau = np.array([r['message']['joint_data']['joint_torque'][13:27] for r in sensor])

    def align(topic, field, part):
        rr = rows[topic]
        tt = np.array([r['callback_monotonic_ns']/1e9 for r in rr])
        vv = np.array([r['message'][field][part] for r in rr])
        ix = np.searchsorted(tt, times).clip(0, len(tt)-1)
        prev = np.maximum(ix-1, 0)
        ix = np.where(abs(tt[prev]-times) < abs(tt[ix]-times), prev, ix)
        return vv[ix], abs(tt[ix]-times)

    wq, ageq = align('/humanoid_controller/optimizedState_wbc_mrt_origin', 'data', slice(-14, None))
    wv, agev = align('/humanoid_controller/optimizedInput_wbc_mrt_origin', 'data', slice(-14, None))
    result = dict(metadata=metadata, counts=counts,
                  alignment='nearest callback arrival; unstamped WBC vectors; <=25ms included; not exact controller-cycle alignment',
                  timestamp_pair_error_ms_p95=float(np.quantile(ageq,.95)*1000),
                  topics={k:len(x) for k,x in rows.items()}, segments=[],
                  task_success=None, independent_full_success=False,
                  missing=['QP active constraint flags', 'actuator application timestamp'])
    for seg in segments:
        dt = times-seg['start_monotonic_ns']/1e9
        target = np.array(seg['target'])
        source = np.array(seg['source'])
        ix = np.where(abs(target-source) > .05)[0]
        if not len(ix):
            continue
        command = source + np.minimum(np.maximum(dt,0)/2,1)[:,None]*(target-source)
        valid = (ageq < .025) & (agev < .025)
        moving = (dt>.2) & (dt<2.) & valid
        late = (dt>5) & (dt<7) & valid
        entire = (dt>0) & (dt<7) & valid
        bias = np.mean(q[late]-target,axis=0)
        err = (q-command)[:,ix]
        werr = (q-wq)[:,ix]
        static_speed = abs(v[late][:,ix])
        settle = None
        # Time from end of command ramp, 0.5s continuously within 1 degree.
        good = np.max(abs(q[:,ix]-target[ix]),axis=1) < np.deg2rad(1.)
        for k in np.flatnonzero((dt>=2)&(dt<6.5)&valid):
            window = (times>=times[k]) & (times<times[k]+.5)
            if window.sum()>10 and np.all(good[window]&valid[window]):
                settle=float(dt[k]-2);break
        signed = (q[:,ix]-target[ix])*np.sign(target[ix]-source[ix])
        rec=dict(name=seg['name'], joints=ix.tolist(), samples=int(entire.sum()),
                 ramp_command_rmse_rad=float(np.sqrt(np.mean(err[moving]**2))),
                 ramp_command_rmse_bias_removed_rad=float(np.sqrt(np.mean((err[moving]-bias[ix])**2))),
                 ramp_wbc_rmse_rad=float(np.sqrt(np.mean(werr[moving]**2))),
                 static_bias_rad=bias[ix].tolist(),
                 static_abs_bias_mean_rad=float(np.mean(abs(bias[ix]))),
                 static_speed_p95_rad_s=float(np.quantile(static_speed,.95)),
                 static_position_std_rad=float(np.mean(np.std(q[late][:,ix],axis=0))),
                 settle_within_1deg_s=settle,
                 overshoot_rad=float(max(0,np.max(signed[entire]))),
                 wbc_reference_speed_abs_mean_rad_s=np.mean(abs(wv[moving][:,ix]),axis=0).tolist(),
                 observed_speed_abs_mean_rad_s=np.mean(abs(v[moving][:,ix]),axis=0).tolist(),
                 actuator_force_abs_peak_nm=np.max(abs(tau[entire][:,ix]),axis=0).tolist())
        result['segments'].append(rec)
    for topic in ['/monitor/time_cost/controller_loop_time','/monitor/time_cost/wbc','/monitor/time_cost/mpc']:
        vals=np.array([r['message']['data'] for r in rows[topic]])
        result[topic]={'p50':float(np.quantile(vals,.5)),'p95':float(np.quantile(vals,.95)), 'max':float(vals.max())} if len(vals) else None
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['timestamp_pair_error_ms_p95','segments']},indent=2))


if __name__ == '__main__':
    main()
