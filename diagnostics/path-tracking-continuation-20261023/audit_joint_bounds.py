"""Quantify observed physical joint-bound excursions; does not control anything."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('label')
a = ap.parse_args()
d = Path(__file__).resolve().parent
run = d / a.label
assert json.loads((run / 'experiment.json').read_text()).get('ended_at')
out = d / 'analysis-full' / a.label
metrics = json.loads((out / 'metrics-full.json').read_text())
bounds = metrics['joint_bound_definition']['limits_rad']
rows = [json.loads(s) for s in (run / 'output/steps.jsonl').open()]
lo = rows[0]['send_monotonic_s']
hi = rows[-1]['send_monotonic_s'] + rows[-1]['step_s']
peak, counts, longest = [0.] * 14, [0] * 14, [0.] * 14
starts, lasts = [None] * 14, [None] * 14
samples = 0
for line in (run / 'chain.jsonl').open():
    if '/sensors_data_raw' not in line:
        continue
    r = json.loads(line)
    if r.get('topic') != '/sensors_data_raw':
        continue
    t = r['callback_monotonic_ns'] / 1e9
    if not lo <= t <= hi:
        continue
    samples += 1
    for j, (q, (low, high)) in enumerate(zip(r['message']['joint_data']['joint_q'][13:27], bounds)):
        excess = max(low - q, q - high, 0.)
        peak[j] = max(peak[j], excess)
        if excess > .0005:
            counts[j] += 1
            if starts[j] is None or t - lasts[j] > .06:
                starts[j] = t
            lasts[j] = t
            longest[j] = max(longest[j], t - starts[j])
        else:
            starts[j] = None
assert samples > 0, 'Missing sensor samples; no zero-count success claim'
state = json.loads((run / 'experiment.json').read_text())
short_failed_run = samples <= 1000 and state.get('policy_summary', {}).get('reason') != 'visual_cycle_target_reached'
assert samples > 1000 or short_failed_run, 'Successful task requires the original minimum sample coverage'
report = dict(samples=samples, short_failed_run=short_failed_run,
    coverage_status='short_failed_run_partial_only' if short_failed_run else 'original_minimum_sample_count_met',
    full_task_acceptance_claimed=False, units='rad', raw_sensor_indices='13:27, left7/right7',
    threshold_rad=.0005, max_excess_rad=peak, counts_over_threshold=counts,
    longest_observed_over_threshold_s=longest,
    interpretation='Measured simulation joint excursions relative to runtime URDF.50Hz sampled intervals, not exact durations or QP active flags. No command-bound relaxation.')
(out / 'joint-bound-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
