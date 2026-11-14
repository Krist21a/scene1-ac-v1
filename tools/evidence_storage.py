"""Explicit evidence destination shared by orchestration, audit and archival."""
import os
from pathlib import Path

EVIDENCE_ROOT = Path(os.environ.get('SCENE1_EVIDENCE_ROOT', '/media/krist/USB')).absolute()
STORE = EVIDENCE_ROOT/'scene1-speed-20261025'
ARCHIVE_ROOT = EVIDENCE_ROOT/'scene1-path-tracking-continuation-20261023'


def require_storage():
    mount = next(p for p in (EVIDENCE_ROOT, *EVIDENCE_ROOT.parents) if p.is_mount())
    if str(EVIDENCE_ROOT).startswith('/media/') and mount == Path('/'):
        raise RuntimeError('Evidence volume is not mounted: '+str(EVIDENCE_ROOT))
    if os.statvfs(mount).f_flag & os.ST_RDONLY:
        raise RuntimeError('Evidence volume is read-only: '+str(mount))
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    return EVIDENCE_ROOT
