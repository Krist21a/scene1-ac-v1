"""Atomic controller-library switch; preserve source and require fully idle containers."""
import argparse
import datetime
import hashlib
import json
import subprocess
import environment as e
from controller_variants import CHOICES, QR_RELATIVE, TASK_RELATIVE, VR_RELATIVE, available, variant

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
task_target='/root/kuavo_ws/'+TASK_RELATIVE
task_allowed=[v['task_sha256'] for v in variants.values()]
vr_target='/root/kuavo_ws/'+VR_RELATIVE
vr_allowed=[v['vr_sha256'] for v in variants.values()]
subprocess.run(['docker', 'cp', str(source), e.SIM + ':/tmp/tracking-selected-wrist-pd'], check=True)
subprocess.run(['docker','cp',str(selected['qr_source']),e.SIM+':/tmp/tracking-selected-dynamic-qr'],check=True)
subprocess.run(['docker','cp',str(selected['task_source']),e.SIM+':/tmp/tracking-selected-task'],check=True)
subprocess.run(['docker','cp',str(selected['vr_source']),e.SIM+':/tmp/tracking-selected-vr'],check=True)
code = '''import pathlib,hashlib,os,shutil,json
t=pathlib.Path(%r);s=pathlib.Path('/tmp/tracking-selected-wrist-pd')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=sha(t);assert old in %r,old
assert sha(s)==%r
q=pathlib.Path(%r);qs=pathlib.Path('/tmp/tracking-selected-dynamic-qr')
qold=sha(q);assert qold in %r,qold
assert sha(qs)==%r
k=pathlib.Path(%r);ks=pathlib.Path('/tmp/tracking-selected-task')
kold=sha(k);assert kold in %r,kold
assert sha(ks)==%r
v=pathlib.Path(%r);vs=pathlib.Path('/tmp/tracking-selected-vr')
vold=sha(v);assert vold in %r,vold
assert sha(vs)==%r
tmp=t.with_name(t.name+'.tracking-switch');shutil.copy2(s,tmp);tmp.chmod(t.stat().st_mode);os.replace(tmp,t)
qt=q.with_name(q.name+'.tracking-switch');shutil.copy2(qs,qt);qt.chmod(q.stat().st_mode);os.replace(qt,q)
kt=k.with_name(k.name+'.tracking-switch');shutil.copy2(ks,kt);kt.chmod(k.stat().st_mode);os.replace(kt,k)
vt=v.with_name(v.name+'.tracking-switch');shutil.copy2(vs,vt);vt.chmod(v.stat().st_mode);os.replace(vt,v)
print(json.dumps(dict(before=old,after=sha(t),dynamic_qr_before=qold,dynamic_qr_after=sha(q),task_before=kold,task_after=sha(k),vr_before=vold,vr_after=sha(v))))
''' % (target, allowed, sha(source),qr_target,qr_allowed,selected['qr_sha256'],task_target,task_allowed,selected['task_sha256'],vr_target,vr_allowed,selected['vr_sha256'])
result = json.loads(e.command('docker', 'exec', e.SIM, 'python3', '-c', code))
receipt = dict(at=datetime.datetime.now().astimezone().isoformat(), factor=selected['factor'],
               version=args.version, file=target, **result,
               source_note='Controller source untouched; explicit registered task and dynamic QR selection, originals restored for other variants')
with (e.HERE / 'switches.jsonl').open('a') as f:
    f.write(json.dumps(receipt) + '\n')
print(json.dumps(receipt, indent=2))
