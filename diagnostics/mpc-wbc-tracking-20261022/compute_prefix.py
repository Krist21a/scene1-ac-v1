"""Recompute common completed prefix and sensor-clock elapsed time from raw logs."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = {}
for name in ['baseline-first-parcel', 'gains-first-parcel']:
    rows = [json.loads(line) for line in (HERE / name / 'output/steps.jsonl').open()]
    first_lift = next(i for i, row in enumerate(rows) if row['cartesian_alignment']['state'] == 'visual_cycle_lift')
    endpoints = [rows[0]['send_monotonic_s'], rows[first_lift]['send_monotonic_s']]
    best = [None, None]
    for line in (HERE / name / 'chain.jsonl').open():
        r = json.loads(line)
        if r.get('topic') != '/sensors_data_raw':
            continue
        t = r['callback_monotonic_ns'] / 1e9
        stamp = r['message']['sensor_time']
        sim = stamp['secs'] + stamp['nsecs'] / 1e9
        for i, end in enumerate(endpoints):
            error = abs(t - end)
            if best[i] is None or error < best[i][0]:
                best[i] = (error, sim)
    out[name] = dict(through_regrasp_steps=first_lift, wall_s=endpoints[1] - endpoints[0],
                     sensor_sim_elapsed_s=best[1][1] - best[0][1],
                     pair_error_ms=[x[0] * 1000 for x in best])
(HERE / 'prefix-recomputed.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
