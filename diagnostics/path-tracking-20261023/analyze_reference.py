"""Arrival-aligned reference consistency, delay and static statistics.

Sensor simulation time is interpolated onto unstamped ROS vectors. All delay
estimates are approximate and retain individual joint results. No object truth.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

GROUPS = dict(proximal=[0,1,2,3,7,8,9,10], wrist=[4,5,6,11,12,13])
TOPICS = dict(wbc_q='/humanoid_controller/optimizedState_wbc_mrt_origin',
              wbc_v='/humanoid_controller/optimizedInput_wbc_mrt_origin',
              arm_q='/humanoid_mpc_arm_commanded',
              mpc_q='/humanoid_controller/optimizedState_mrt_origin',
              mpc_v='/humanoid_controller/optimizedInput_mrt_origin')


def collect(path):
    streams = defaultdict(list)
    for line in Path(path).open():
        r = json.loads(line)
        if 'topic' not in r:
            continue
        topic, m = r['topic'], r['message']; t = r['callback_monotonic_ns']/1e9
        if topic == '/sensors_data_raw':
            stamp = m['sensor_time']; sim = stamp['secs'] + stamp['nsecs']/1e9
            jd = m['joint_data']
            streams['sensor'].append((t, sim, jd['joint_q'][13:27], jd['joint_v'][13:27]))
        elif topic == '/kuavo_arm_traj':
            streams['published_q'].append((t, np.deg2rad(m['position'])))
        elif topic in TOPICS.values():
            key = next(k for k,v in TOPICS.items() if v == topic)
            value = (m['stateTrajectory'][-1]['value'] if key == 'arm_q'
                     else m['data'][-8:] if key.startswith('mpc_') else m['data'][-14:])
            streams[key].append((t, value))
    return streams


def stats(ref, actual, velocity, reference_velocity, dt):
    result = {}
    # Strict settled segments: one simulated second with stable reference and
    # low measured speed. An absent plateau is reported, never filled with 0.
    hold = np.zeros(len(ref), dtype=bool); window = int(np.ceil(1/dt))
    for i in range(window, len(ref)):
        hold[i] = np.max(np.ptp(ref[i-window:i+1], axis=0)) <= .003 and np.max(abs(velocity[i])) <= .03
    for name, indices in GROUPS.items():
        error = actual[:,indices]-ref[:,indices]
        r = dict(rmse_rad=float(np.sqrt(np.mean(error**2))),
                 per_joint_rmse_rad=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                 static_samples=int(hold.sum()), static_bias_rad=None, static_position_std_rad=None)
        if np.any(hold):
            r['static_bias_rad'] = np.mean(error[hold], axis=0).tolist()
            # Error variation, so different static target levels are not mixed.
            r['static_position_std_rad'] = np.std(error[hold], axis=0).tolist()
        lag = []
        for j in indices:
            x = reference_velocity[:,j]
            if np.std(x) < .01:
                lag.append(None); continue
            options = []
            for shift in range(0, int(.4/dt)+1):
                a = x[:len(x)-shift or None]; b = velocity[shift:,j]
                if len(a) < 50 or np.std(b) < 1e-6:
                    continue
                score = float(np.corrcoef(a,b)[0,1]); options.append((score,shift))
            best = max(options) if options else None
            lag.append(None if best is None else dict(sim_lag_s=best[1]*dt,
                velocity_correlation=best[0], at_search_boundary=best[1] in (0,int(.4/dt))))
        r['approximate_dynamic_lag_per_joint'] = lag
        result[name] = r
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--trace', required=True)
    ap.add_argument('--output', required=True); ap.add_argument('--steps')
    a = ap.parse_args(); streams = collect(a.trace)
    sensor = streams['sensor']; wall = np.array([s[0] for s in sensor]); sim = np.array([s[1] for s in sensor])
    # Remove duplicate/non-progressing simulation stamps before interpolation.
    valid = np.r_[True,np.diff(sim)>0]; wall,sim = wall[valid],sim[valid]
    q = np.array([s[2] for s in sensor])[valid]; v = np.array([s[3] for s in sensor])[valid]
    dt = .01
    start,end = sim[0]+1, sim[-1]-.5
    if a.steps:
        steps = [json.loads(l) for l in Path(a.steps).open()]
        start = max(start, float(np.interp(steps[0]['send_monotonic_s'], wall, sim)))
        end = min(end, float(np.interp(steps[-1]['send_monotonic_s'], wall, sim)))
    for name in ['wbc_q','wbc_v','published_q']:
        start = max(start,float(np.interp(streams[name][0][0],wall,sim)))
        end = min(end,float(np.interp(streams[name][-1][0],wall,sim)))
    grid = np.arange(start,end,dt)
    def interpolate(t, values):
        values = np.asarray(values)
        return np.stack([np.interp(grid,t,values[:,j]) for j in range(values.shape[1])],axis=1)
    data = dict(measured_q=interpolate(sim,q), measured_v=interpolate(sim,v))
    for key, rows in streams.items():
        if key == 'sensor':continue
        t = np.interp([r[0] for r in rows],wall,sim)
        data[key] = interpolate(t,[r[1] for r in rows])
    derivative = np.gradient(data['wbc_q'],dt,axis=0)
    out = dict(sample_dt_sim_s=dt, samples=len(grid), groups=GROUPS,
               tracking=stats(data['wbc_q'],data['measured_q'],data['measured_v'],derivative,dt),
               reference_consistency={})
    out['upper_to_wbc'] = stats(data['published_q'],data['wbc_q'],derivative,
                              np.gradient(data['published_q'],dt,axis=0),dt)
    if 'mpc_q' in data:
        ix = GROUPS['proximal']
        out['mpc_to_wbc_proximal_rmse_rad'] = float(np.sqrt(np.mean(
            (data['mpc_q']-data['wbc_q'][:,ix])**2)))
    for name, ix in GROUPS.items():
        moving = abs(derivative[:,ix]) > .02
        error = data['wbc_v'][:,ix]-derivative[:,ix]
        out['reference_consistency'][name] = dict(
            position_derivative_rms_rad_s=float(np.sqrt(np.mean(derivative[:,ix]**2))),
            published_velocity_rms_rad_s=float(np.sqrt(np.mean(data['wbc_v'][:,ix]**2))),
            position_velocity_disagreement_rms_rad_s=float(np.sqrt(np.mean(error**2))),
            moving_joint_samples=int(moving.sum()), zero_velocity_while_moving_fraction=(
                float(np.mean(abs(data['wbc_v'][:,ix][moving]) < 1e-5)) if moving.any() else None))
    out['limitations'] = [
        'Unstamped references aligned by callback arrival and interpolated sensor simulation clock',
        '10ms grid; not exact application delay, QP constraint audit, or whole robot stability proof',
        'Dynamic lag uses velocity correlation over 0..400ms; boundary/low-correlation fits are inconclusive',
        'Static variation aggregates eligible plateaus; no plateau yields null',
        'Position derivative inferred from sampled filtered positions; high-frequency content can alias']
    Path(a.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))


if __name__ == '__main__':main()
