"""Atomic controller-library switch; preserve source and require fully idle containers."""
import argparse
import datetime
import hashlib
import json
import subprocess
import environment as e
from controller_variants import CHOICES, QR_RELATIVE, available, variant

ap = argparse.ArgumentParser()
ap.add_argument('version', choices=CHOICES)
args = ap.parse_args()
e.wait_idle(e.ALGO)
e.wait_idle(e.SIM)
target = '/root/kuavo_ws/devel/.private/humanoid_controllers/lib/libhumanoid_controllers.so'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
selected = variant(args.version)
source = selected['source']
variants=available()
allowed = [v['sha256'] for v in variants.values()]
qr_target='/root/kuavo_ws/'+QR_RELATIVE
qr_allowed=[v['qr_sha256'] for v in variants.values()]
subprocess.run(['docker', 'cp', str(source), e.SIM + ':/tmp/tracking-selected-wrist-pd'], check=True)
subprocess.run(['docker','cp',str(selected['qr_source']),e.SIM+':/tmp/tracking-selected-dynamic-qr'],check=True)
code = '''import pathlib,hashlib,os,shutil,json
t=pathlib.Path(%r);s=pathlib.Path('/tmp/tracking-selected-wrist-pd')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=sha(t);assert old in %r,old
assert sha(s)==%r
q=pathlib.Path(%r);qs=pathlib.Path('/tmp/tracking-selected-dynamic-qr')
qold=sha(q);assert qold in %r,qold
assert sha(qs)==%r
tmp=t.with_name(t.name+'.tracking-switch');shutil.copy2(s,tmp);tmp.chmod(t.stat().st_mode);os.replace(tmp,t)
qt=q.with_name(q.name+'.tracking-switch');shutil.copy2(qs,qt);qt.chmod(q.stat().st_mode);os.replace(qt,q)
print(json.dumps(dict(before=old,after=sha(t),dynamic_qr_before=qold,dynamic_qr_after=sha(q))))
''' % (target, allowed, sha(source),qr_target,qr_allowed,selected['qr_sha256'])
result = json.loads(e.command('docker', 'exec', e.SIM, 'python3', '-c', code))
receipt = dict(at=datetime.datetime.now().astimezone().isoformat(), factor=selected['factor'],
               version=args.version, file=target, **result,
               source_note='Controller source untouched; explicit dynamic QR selection, original matrix restored for other variants')
with (e.HERE / 'switches.jsonl').open('a') as f:
    f.write(json.dumps(receipt) + '\n')
print(json.dumps(receipt, indent=2))
