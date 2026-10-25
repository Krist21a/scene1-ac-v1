"""Observe completed upper-policy log rows. Never publish robot commands."""
import argparse
import json
from pathlib import Path
import time
from clip_guard import ClipGuard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', required=True)
    ap.add_argument('--before', required=True)
    ap.add_argument('--threshold', type=int, default=5)
    a = ap.parse_args()
    root = Path(a.outputs)
    before = set(json.loads(a.before))
    guard = ClipGuard(a.threshold)
    offset = 0
    run = None
    heartbeat = 0
    print(json.dumps(dict(kind='ready', monotonic_s=time.monotonic())), flush=True)
    while True:
        now = time.monotonic()
        new = {p.name for p in root.iterdir() if p.is_dir()} - before
        if len(new) > 1:
            raise RuntimeError(('More than one new policy output', sorted(new)))
        if new:
            run = new.pop()
            p = root / run / 'steps.jsonl'
            if p.exists():
                with p.open('rb') as f:
                    f.seek(offset)
                    data = f.read(1024 * 1024)
                end = data.rfind(b'\n')
                if end >= 0:
                    rows = [json.loads(line) for line in data[:end+1].splitlines()]
                    offset += end + 1
                    trigger = guard.consume([dict(step=r['step'], clipped=bool(r['motion']['bounds_clipped'])) for r in rows])
                    if trigger:
                        print(json.dumps(dict(kind='trigger', reason='consecutive_upper_target_clipping',
                                              monotonic_s=now, run=run, **trigger)), flush=True)
                        return
        if now - heartbeat >= .5:
            print(json.dumps(dict(kind='heartbeat', monotonic_s=now, run=run,
                                  last_step=guard.last_step, consecutive=guard.consecutive)), flush=True)
            heartbeat = now
        time.sleep(.02)


if __name__ == '__main__':
    main()
