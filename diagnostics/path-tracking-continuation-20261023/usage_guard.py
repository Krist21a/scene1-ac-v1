"""Current-session quota stop requested by user: stop with 10% remaining."""
import json
from pathlib import Path
SESSION = Path('/home/krist/.codex/sessions/2026/10/25/rollout-2026-10-25T18-09-11-01a17289-1e0c-7e12-b67a-3b92a975bbe3.jsonl')
def latest():
    with SESSION.open('rb') as f:
        f.seek(max(0, SESSION.stat().st_size - 4*1024*1024))
        lines = f.read().splitlines()
    for line in reversed(lines):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        p = r.get('payload', {})
        if r.get('type') != 'event_msg' or p.get('type') != 'token_count':
            continue
        limits = p.get('rate_limits') or {}
        windows = {k: limits[k] for k in ('primary', 'secondary') if limits.get(k)}
        if not windows:
            continue
        used = max(float(w['used_percent']) for w in windows.values())
        return dict(timestamp=r['timestamp'], windows=windows, used_percent=used,
                    remaining_percent=100-used, threshold_percent=90, enforced=True,
                    reached=used >= 90, source=str(SESSION))
    raise RuntimeError('Current-session quota unavailable; stop before starting further work')
