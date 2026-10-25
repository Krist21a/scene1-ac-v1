"""Audit same-controller-cycle PD diagnostics, not estimated actuator commands."""
import argparse
import json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--trace', required=True)
ap.add_argument('--output', required=True)
a = ap.parse_args()
rows = []
for line in open(a.trace):
    r = json.loads(line)
    if r.get('topic') == '/scene1_tracking/wrist_pd':
        rows.append((r['callback_monotonic_ns'] / 1e9, r['message']['data']))
assert rows, 'PD diagnostic topic missing'
t = np.array([r[0] for r in rows])
x = np.array([r[1] for r in rows])
assert x.shape[1] == 57 and np.isfinite(x).all()
base, raw, added, command = [x[:, start:start + 14] for start in [0, 14, 28, 42]]
enabled = x[:, 56] > .5
wrist = [4, 5, 6, 11, 12, 13]
proximal = [i for i in range(14) if i not in wrist]
limited = np.clip(raw, -.5, .5)
expected = base.copy()
for j in wrist:
    expected[enabled, j] = np.clip(base[enabled, j] + limited[enabled, j], -14.1, 14.1)
clip = enabled & np.any(abs(raw[:, wrist]) > .5 + 1e-9, axis=1)
total_clip = enabled & np.any(abs(base[:, wrist] + limited[:, wrist]) > 14.1, axis=1)
run_start = None
longest = 0.
for i, is_clip in enumerate(clip | total_clip):
    if is_clip:
        if run_start is None:
            run_start = t[i]
        longest = max(longest, t[i] - run_start)
    else:
        run_start = None
report = dict(samples=len(t), enabled_samples=int(enabled.sum()),
              raw_added_torque_clip_samples=int(clip.sum()), total_torque_clip_samples=int(total_clip.sum()),
              longest_observed_continuous_clip_s=float(longest),
              wrist_added_peak_nm=np.max(abs(added[:, wrist]), axis=0).tolist(),
              wrist_command_peak_nm=np.max(abs(command[:, wrist]), axis=0).tolist(),
              proximal_added_peak_nm=float(np.max(abs(added[:, proximal]))),
              exact_cycle_addition_identity_max_error_nm=float(np.max(abs(command - base - added))),
              exact_cycle_pd_clip_formula_max_error_nm=float(np.max(abs(command - expected))),
              limitation='50Hz recorded output; transient events between samples not excluded. Raw and total command are controller diagnostics, actuator force is in sensors_data_raw.')
Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
