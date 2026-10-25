"""Audit emitted normal-route wrist references; sampled evidence, not QP state."""
import argparse
import json
import math
from pathlib import Path

WRIST = [4, 5, 6, 11, 12, 13]


def audit(trace, start=-math.inf, end=math.inf):
    n = active = clipped = updated = invalid = 0
    peak_raw = peak_sent = error = longest = 0.
    clip_start = None
    limit_values = set()
    for line in Path(trace).open():
        r = json.loads(line)
        if r.get('topic') != '/scene1_tracking/wrist_velocity_reference':
            continue
        t = r['callback_monotonic_ns'] / 1e9
        if not start <= t <= end:
            continue
        n += 1
        x = r['message']['data']
        if len(x) != 47 or not all(map(math.isfinite, x)):
            invalid += 1
            continue
        if x[43] <= .5:
            clip_start = None
            continue
        active += 1
        updated += x[46] > .5
        limit_values.add(x[45])
        peak_raw = max(peak_raw, *(abs(x[14+j]) for j in WRIST))
        peak_sent = max(peak_sent, *(abs(x[28+j]) for j in WRIST))
        error = max(error, *(abs(x[28+j]-max(-x[45], min(x[45], x[14+j]))) for j in WRIST))
        if x[44] > .5:
            clipped += 1
            if clip_start is None:
                clip_start = t
            longest = max(longest, t-clip_start)
        else:
            clip_start = None
    return dict(samples=n, active_samples=active, updated_samples=updated,
                invalid_samples=invalid, clipped_samples=clipped,
                longest_sampled_clip_s=longest, peak_raw_rad_s=peak_raw,
                peak_sent_rad_s=peak_sent, velocity_limits_rad_s=sorted(limit_values),
                sent_vs_bounded_raw_max_error_rad_s=error,
                evidence_valid=bool(active and not invalid and error < 1e-9),
                limitations=['50 Hz downsampled callback-arrival evidence; high-rate guard is separate',
                             'No exact cycle reconstruction or QP active-constraint visibility'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--steps')
    a = ap.parse_args()
    result = dict(all=audit(a.trace))
    if a.steps:
        first = last = None
        for line in Path(a.steps).open():
            r = json.loads(line)
            if first is None:
                first = r['send_monotonic_s']
            last = r['send_monotonic_s']
        result['formal_policy'] = audit(a.trace, first, last)
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
