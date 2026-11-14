"""Live quota guard with an opt-in, once-only reset at 1% for this task."""
import json
import os
import datetime
import queue
import subprocess
import threading
import time
import fcntl
import uuid
from pathlib import Path

_cached = None
_cached_at = 0.

def account_rpc(method, params=None):
    # A transient account endpoint failure must not tear down a live physics
    # run on its first occurrence. Retry only reads; never return stale quota
    # or automatically repeat a credit mutation here.
    attempts = 3 if method == 'account/rateLimits/read' else 1
    for attempt in range(attempts):
        try:
            return _account_rpc_once(method, params)
        except (RuntimeError, queue.Empty, OSError):
            if attempt + 1 == attempts:
                raise
            time.sleep(.5 * (attempt + 1))


def _account_rpc_once(method, params=None):
    """Use authenticated account RPC without opening another model turn."""
    command = ['/home/krist/.local/bin/codex', 'app-server', '--stdio']
    if os.geteuid() == 0:
        command = ['sudo', '-n', '-u', 'krist', '-H', *command]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True, bufsize=1)
    messages = queue.Queue()
    def receive():
        for line in process.stdout:
            try: messages.put(json.loads(line))
            except ValueError: pass
    threading.Thread(target=receive, daemon=True).start()
    def send(value):
        process.stdin.write(json.dumps(value)+'\n'); process.stdin.flush()
    def response(identity):
        deadline = time.monotonic()+20.
        while True:
            value = messages.get(timeout=max(.001, deadline-time.monotonic()))
            if value.get('id') == identity:
                if 'error' in value: raise RuntimeError('Live quota RPC failed')
                return value['result']
            if time.monotonic() >= deadline: raise RuntimeError('Live quota RPC timed out')
    try:
        send(dict(id=1, method='initialize', params=dict(clientInfo=dict(name='scene1_usage_guard',version='1.0'),
                                                      capabilities=dict(experimentalApi=True))))
        response(1); send(dict(method='initialized'))
        request = dict(id=2, method=method)
        if params is not None:
            request['params'] = params
        send(request)
        return response(2)
    finally:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)


def live_latest():
    """Read live usage, optionally consuming the user's single authorized card."""
    raw = account_rpc('account/rateLimits/read')
    limits = raw['rateLimits']
    windows = {k: dict(used_percent=v['usedPercent'], window_minutes=v['windowDurationMins'],
                      resets_at=v['resetsAt']) for k in ('primary','secondary')
               if (v := limits.get(k)) is not None}
    if not windows or any(not 0 <= float(v['used_percent']) <= 100 for v in windows.values()):
        raise RuntimeError('Live quota unavailable or invalid; stop before further work')
    used = max(float(v['used_percent']) for v in windows.values())
    receipt_name = os.environ.get('SCENE1_RESET_CREDIT_RECEIPT')
    threshold = 99 if receipt_name else 90
    if used >= 99 and receipt_name:
        attempted = False
        receipt = Path(receipt_name)
        with receipt.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            attempt = json.loads(receipt.read_text()) if receipt.exists() else None
            if attempt is None:
                credits = (raw.get('rateLimitResetCredits') or {}).get('credits') or []
                available = [c for c in credits if c['status'] == 'available' and c['resetType'] == 'codexRateLimits']
                if available:
                    attempt = dict(account_id=raw.get('accountId'), before_used_percent=used,
                                   params=dict(creditId=available[0]['id'], idempotencyKey=str(uuid.uuid4())))
                    with receipt.open('x') as f:
                        json.dump(attempt, f, indent=2); f.flush(); os.fsync(f.fileno())
            if attempt and 'response' not in attempt:
                if attempt['account_id'] != raw.get('accountId'):
                    raise RuntimeError('Account changed during authorized reset')
                # Reuse the exact credit and idempotency key after an uncertain response.
                attempt['response'] = account_rpc('account/rateLimitResetCredit/consume', attempt['params'])
                temporary = receipt.with_suffix('.tmp')
                with temporary.open('w') as f:
                    json.dump(attempt, f, indent=2); f.flush(); os.fsync(f.fileno())
                temporary.replace(receipt)
                attempted = True
        # Release the inter-process lock before re-reading or checking the receipt.
        if attempted:
            return live_latest()
    return dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), windows=windows,
                used_percent=used, remaining_percent=100-used, threshold_percent=threshold, enforced=True,
                reached=used>=threshold or raw.get('ordinaryUsageAllowed') is False,
                reset_credits_available=(raw.get('rateLimitResetCredits') or {}).get('availableCount'),
                source='authenticated Codex account/rateLimits/read',
                thread_id=os.environ.get('CODEX_THREAD_ID') or os.environ.get('CODEX_SESSION_ID'))

def current_session():
    identity = os.environ.get('CODEX_THREAD_ID') or os.environ.get('CODEX_SESSION_ID')
    if not identity or any(c not in '0123456789abcdef-' for c in identity):
        raise RuntimeError('Current Codex session identity unavailable')
    matches = list(Path('/home/krist/.codex/sessions').rglob('*-'+identity+'.jsonl'))
    if len(matches) != 1:
        raise RuntimeError('Cannot uniquely resolve current Codex quota source')
    return matches[0]

def latest():
    global _cached, _cached_at
    if _cached is not None and time.monotonic()-_cached_at < 60.:
        return _cached
    _cached = live_latest()
    _cached_at = time.monotonic()
    return _cached

def historical_latest():
    """Historical receipt lookup only; never used to authorize a new trial."""
    SESSION = current_session()
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
