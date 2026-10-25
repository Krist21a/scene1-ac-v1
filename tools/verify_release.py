"""Verify exported frozen policy and optionally extracted controller assets."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,expected in json.loads((ROOT/'evidence/policy-manifest.json').read_text()).items():
 assert sha(ROOT/name)==expected,name
m=json.loads((ROOT/'evidence/control-runtime-manifest.json').read_text())
present=[(ROOT/name).exists() for name in m['files']]
if any(present):
 assert all(present),'Incomplete control asset extraction'
 for name,expected in m['files'].items():assert sha(ROOT/name)==expected,name
print('Frozen policy verified; controller binaries '+('verified' if all(present) else 'not extracted (download Release asset)'))
