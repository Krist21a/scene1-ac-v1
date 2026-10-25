"""Archive completed analysis staging, verify bytes, then remove verified copies."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SIM = 'kuavo_challenge_container_GPU_046e4301'
SOURCE = '/tmp/scene1-integrated-analysis'
stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
destination = Path('/media/krist/KINGSTON/scene1-path-tracking-continuation-20261023') / ('analysis-staging-' + stamp)
assert not destination.exists()
subprocess.run(['docker', 'cp', SIM + ':' + SOURCE, str(destination)], check=True)
verified = {}
for directory in sorted(destination.iterdir()):
    if not directory.is_dir():
        continue
    state = HERE / directory.name / 'experiment.json'
    pipeline = HERE / (directory.name + '-pipeline.json')
    if not state.exists() or not json.loads(state.read_text()).get('ended_at'):
        continue
    if not pipeline.exists() or json.loads(pipeline.read_text())['current_stage'] != 'done':
        continue
    local = {str(f.relative_to(directory)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in directory.rglob('*') if f.is_file()}
    remote_path = SOURCE + '/' + directory.name
    code = "import pathlib,hashlib,json; p=pathlib.Path(" + repr(remote_path) + "); print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}))"
    remote = json.loads(subprocess.check_output(['docker', 'exec', SIM, 'python3', '-c', code]))
    assert local and local == remote, directory.name
    subprocess.run(['docker', 'exec', SIM, 'rm', '-rf', '--', remote_path], check=True)
    verified[directory.name] = local
receipt = dict(destination=str(destination), verified=verified)
(HERE / ('analysis-staging-archive-' + stamp + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(destination=str(destination), verified=list(verified))))
