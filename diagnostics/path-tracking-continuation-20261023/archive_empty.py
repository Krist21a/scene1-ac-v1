"""Verify retained empty-probe evidence before removing duplicate /tmp traces."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import environment as e


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument('labels', nargs='+')
a = ap.parse_args()
e.wait_idle(e.ALGO)
e.wait_idle(e.SIM)
for label in a.labels:
    d = e.HERE / label
    state = json.loads((d/'experiment.json').read_text())
    assert state['kind'] == 'empty' and state.get('ended_at')
    assert d.is_symlink() and str(d.resolve()).startswith('/media/krist/KINGSTON/scene1-path-tracking-continuation-20261023/')
    files = {str(p.relative_to(d)): sha(p) for p in d.rglob('*') if p.is_file()}
    receipt = dict(label=label, retained_directory=str(d.resolve()), files=files, removed_duplicates=[])
    for filename, suffix in [('chain.jsonl', '-chain.jsonl'), ('probe.jsonl', '-probe.jsonl')]:
        if filename not in files:
            continue
        remote = '/tmp/' + label + suffix
        code = '''import pathlib,hashlib,sys
p=pathlib.Path(sys.argv[1]);h=hashlib.sha256()
assert p.parent==pathlib.Path('/tmp')
if not p.exists(): print('already absent');sys.exit()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
assert h.hexdigest()==sys.argv[2]
p.unlink();print('byte-verified duplicate removed')
'''
        result = e.command('docker','exec',e.SIM,'python3','-c',code,remote,files[filename])
        receipt['removed_duplicates'].append(dict(path=remote,result=result))
    (e.HERE/(label+'-archive-manifest.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(label + ': retained and verified')
