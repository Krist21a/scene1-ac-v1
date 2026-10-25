"""Summarize complete loop spans, from a file or a live stdin stream."""
import argparse
from collections import defaultdict
import json
import sys


def summarize(rows):
    spans, phases = defaultdict(float), defaultdict(lambda: [0, 0.])
    total = 0.
    n = 0
    first = last = None
    for row in rows:
        n += 1
        total += row['cycle_s']
        start = row['started_monotonic_s']
        first = start if first is None else min(first, start)
        end = start+row['cycle_s']
        last = end if last is None else max(last, end)
        phases[row.get('phase', 'unknown')][0] += 1
        phases[row.get('phase', 'unknown')][1] += row['cycle_s']
        for k, v in row['spans_s'].items():
            spans[k] += v
    return dict(steps=n, total_s=total, formal_wall_s=last-first if n else None,
                mean_cycle_s=total/max(n, 1),
                spans_s=dict(sorted(spans.items(), key=lambda kv: -kv[1])),
                phases=dict(sorted(phases.items(), key=lambda kv: -kv[1][1])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default='-')
    a = ap.parse_args()
    stream = sys.stdin if a.path == '-' else open(a.path)
    rows = []
    for line in stream:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # A live writer may not have completed its final line.
    print(json.dumps(summarize(rows), indent=2))
