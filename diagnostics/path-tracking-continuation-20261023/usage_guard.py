"""Read only quota counters from the explicitly selected current session log."""
import datetime
import json
from pathlib import Path

SESSION = Path('/home/krist/.codex/sessions/2026/10/23/rollout-2026-10-23T15-41-48-01a167b5-783f-7a10-a1d8-2501750159d1.jsonl')
LIMIT = 80.
# User resumed after this stop was fulfilled: continue to multi-seed acceptance.
ENFORCE_REMAINING20_STOP = False

def latest():
    # Tail bounds avoid repeatedly parsing historical tool outputs or credentials.
    with SESSION.open('rb') as stream:
        stream.seek(max(0, SESSION.stat().st_size-4*1024*1024))
        lines=stream.read().splitlines()
    for line in reversed(lines):
        try:r=json.loads(line)
        except ValueError:continue
        p=r.get('payload',{})
        if r.get('type')!='event_msg' or p.get('type')!='token_count':continue
        limits=p.get('rate_limits') or {}
        windows={k:v for k in ('primary','secondary') if (v:=limits.get(k)) is not None}
        if not windows:continue
        used=max(float(v['used_percent']) for v in windows.values())
        return dict(timestamp=r['timestamp'],used_percent=used,windows=windows,
                    remaining_percent=100.-used,threshold_percent=LIMIT,
                    enforced=ENFORCE_REMAINING20_STOP, historical_threshold_reached=used>=LIMIT,
                    reached=ENFORCE_REMAINING20_STOP and used>=LIMIT,source=str(SESSION))
    raise RuntimeError('No quota counter in current session tail; cannot enforce usage stop')

if __name__=='__main__': print(json.dumps(latest()))
