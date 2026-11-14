"""Preserve new isolated experiments on external storage after byte verification.

Never touches production/history. Host experiment path becomes a symlink.
Only verified redundant container trace and isolated output copies are removed.
"""
import argparse, datetime, hashlib, json, re, shutil, subprocess
from pathlib import Path
import environment as e
from trace_io import open_trace, digest

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def manifest(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}

ap=argparse.ArgumentParser();ap.add_argument('label');a=ap.parse_args()
assert re.fullmatch(r'[a-z0-9][a-z0-9_-]*',a.label)
e.wait_idle(e.ALGO);e.wait_idle(e.SIM)
source=e.HERE/a.label
assert source.is_dir()
state=json.loads((source/'experiment.json').read_text())
assert state.get('ended_at') and state['label']==a.label
store=e.ARCHIVE_ROOT
e.require_storage()
store.mkdir(exist_ok=True)
target=store/a.label;temporary=store/(a.label+'.copying')
expected=manifest(source)
already_external=source.is_symlink()
if already_external:
    assert source.resolve()==target.resolve() and target.is_dir()
else:
    assert not target.exists() and not temporary.exists()
    shutil.copytree(source,temporary)
    assert manifest(temporary)==expected, 'Copy mismatch; originals retained'
    temporary.rename(target)
# Keep authoritative byte manifest separate so the recorded artifact stays immutable.
receipt=dict(label=a.label,at=datetime.datetime.now().astimezone().isoformat(),host_path=str(source),archive_path=str(target),files=expected,removed_redundant_copies=[])
(source.parent/(a.label+'-archive-manifest.json')).write_text(json.dumps(receipt,indent=2)+'\n')
if not already_external:
    shutil.rmtree(source)
    source.symlink_to(target,target_is_directory=True)
assert manifest(source)==expected
trace_sha = expected.get('chain.jsonl')
if 'chain.jsonl.gz' in expected:
    transport = json.loads((source/'chain-receipt.json').read_text())
    assert transport.get('verified') is True
    assert transport['compressed_sha256'] == expected['chain.jsonl.gz']
    with open_trace(source, 'rb') as stream:
        assert digest(stream) == transport['original_sha256']
    trace_sha = transport['original_sha256']
if trace_sha is not None:
    code='''import pathlib,hashlib,sys
p=pathlib.Path(sys.argv[1]);expected=sys.argv[2]
assert p.parent==pathlib.Path('/tmp') and p.name.endswith('-chain.jsonl')
if p.exists():
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==expected
 p.unlink();print(str(p))
'''
    removed=e.command('docker','exec',e.SIM,'python3','-c',code,'/tmp/'+a.label+'-chain.jsonl',trace_sha)
    receipt['trace_original_sha256'] = trace_sha
    receipt['removed_redundant_copies'].append(dict(container=e.SIM,path=removed.strip()))
if len(state.get('output_runs',[]))==1:
    project=state['policy_root'];run=state['output_runs'][0]
    assert project.startswith('/root/scene1_speed/') and re.fullmatch(r'\d{8}-\d{6}',run)
    expected_output={k[len('output/'):]:v for k,v in expected.items() if k.startswith('output/')}
    # A guard may stop SDK initialization before the first policy record.
    # Empty output is valid failure evidence; verify remote emptiness below.
    receipt['empty_policy_output'] = not bool(expected_output)
    code='''import pathlib,hashlib,json,shutil,sys
p=pathlib.Path(sys.argv[1]);expected=json.load(sys.stdin)
assert str(p).startswith('/root/scene1_speed/') and p.parent.name=='outputs'
if p.exists():
 found={}
 for x in p.rglob('*'):
  if x.is_file():
   h=hashlib.sha256()
   with x.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
   found[str(x.relative_to(p))]=h.hexdigest()
 assert found==expected, 'Container output differs; retained'
 shutil.rmtree(p);print(str(p))
'''
    result=subprocess.run(['docker','exec','-i',e.ALGO,'python3','-c',code,project+'/outputs/'+run],input=json.dumps(expected_output),text=True,capture_output=True,check=True)
    receipt['removed_redundant_copies'].append(dict(container=e.ALGO,path=result.stdout.strip()))
receipt['verified']=True
(source.parent/(a.label+'-archive-manifest.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))
