"""Remove the explicitly retired SAM2 installation; keep historical evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import environment as e


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    e.wait_idle(e.ALGO); e.wait_idle(e.SIM)
    report = dict(at=datetime.datetime.now().astimezone().isoformat(), removed=[],
                  retained='Historical inputs, masks, results, benchmark scripts and reports; production and v63 unchanged')
    root = Path('/home/krist/r3d-research-20261021')
    targets = [root/'downloads/sam2.1_hiera_tiny.pt', root/'downloads/sam2-2b90b9f.tar.gz',
               root/'sam2-2b90b9f5ceec907a1c18123530e92e794ad901a4']
    weight_sha = '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
    assert sha(targets[0]) == weight_sha
    report['host_before'] = {str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for target in targets
                             for p in ([target] if target.is_file() else target.rglob('*')) if p.is_file()}
    code = '''import pathlib,json,hashlib
root=pathlib.Path('/root/r3d-research-20261021')
paths=[root/'sam2-code',root/'sam2.1_hiera_tiny.pt']
paths += list((root/'venv/lib/python3.12/site-packages').glob('*sam*'))
out={}
for target in paths:
 for p in ([target] if target.is_file() else target.rglob('*')):
  if p.is_file():out[str(p)]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(out))
'''
    report['container_before'] = json.loads(e.command('docker','exec',e.ALGO,'python3','-c',code))
    assert report['container_before']['/root/r3d-research-20261021/sam2.1_hiera_tiny.pt']['sha256'] == weight_sha
    path = e.HERE/'sam2-removal.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    subprocess.run(['docker','exec',e.ALGO,'/root/r3d-research-20261021/venv/bin/python',
                    '-m','pip','uninstall','-y','SAM-2'],check=True)
    cleanup = '''import pathlib,shutil,hashlib
root=pathlib.Path('/root/r3d-research-20261021')
p=root/'sam2.1_hiera_tiny.pt'
assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
p.unlink();shutil.rmtree(root/'sam2-code')
''' % weight_sha
    e.command('docker','exec',e.ALGO,'python3','-c',cleanup)
    for target in targets:
        if target.is_dir():shutil.rmtree(target)
        else:target.unlink()
        report['removed'].append(str(target))
    for python in ['/root/r3d-research-20261021/venv/bin/python','/root/miniforge3/envs/kls_dev/bin/python']:
        result=e.command('docker','exec','-w','/root',e.ALGO,python,'-c',
            'import importlib.util; assert importlib.util.find_spec("sam2") is None; print("sam2 absent")')
        report[python] = result
    report['removed_container']=['SAM-2 distribution','/root/r3d-research-20261021/sam2-code',
                                 '/root/r3d-research-20261021/sam2.1_hiera_tiny.pt']
    report['complete']=True
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_before')},indent=2))


if __name__ == '__main__':main()
