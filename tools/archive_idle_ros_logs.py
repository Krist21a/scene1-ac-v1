"""Archive idle-container ROS logs before reclaiming verified source bytes."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
STORE = Path('/media/krist/KINGSTON/scene1-speed-20261025')
SCAN = r'''
import hashlib,json
from pathlib import Path
root=Path('/root/.ros/log')
print(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in root.rglob('*') if p.is_file() and not p.is_symlink()}))
'''
REMOVE = r'''
import hashlib,json,sys
from pathlib import Path
root=Path('/root/.ros/log');manifest=json.load(sys.stdin);removed=0;size=0
for name,digest in manifest.items():
 p=root/name
 if not p.is_file() or p.is_symlink():continue
 if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:continue
 size+=p.stat().st_size;p.unlink();removed+=1
print(json.dumps(dict(files=removed,bytes=size)))
'''


def archive_if_needed(label, environment):
    # Called only after controller shutdown, before any reset/formal timer.
    if shutil.disk_usage('/').free >= 5_000_000_000:
        return
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', label):
        raise ValueError('Invalid archive label')
    for container in (environment.ALGO, environment.SIM):
        environment.wait_idle(container)
        manifest = json.loads(subprocess.check_output(
            ['docker', 'exec', container, 'python3', '-c', SCAN], text=True))
        if not manifest:
            continue
        for name in manifest:
            p = Path(name)
            if p.is_absolute() or '..' in p.parts:
                raise ValueError('Unsafe log path')
        archive = STORE/(label+'-'+container+'-idle-ros-logs.tar')
        with archive.open('xb') as output:
            subprocess.run(['docker', 'exec', '-i', container, 'tar',
                '-C', '/root/.ros/log', '-cf', '-', '--null', '-T', '-'],
                input=b'\0'.join(name.encode() for name in manifest)+b'\0',
                stdout=output, check=True)
            output.flush()
        with tarfile.open(archive) as saved:
            for name, digest in manifest.items():
                if hashlib.sha256(saved.extractfile(name).read()).hexdigest() != digest:
                    raise RuntimeError('Log archive verification failed; originals retained')
        receipt = dict(container=container, archive=str(archive), sha256=manifest)
        receipt_path = archive.with_suffix('.json')
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
        # Recheck every original. Files changed since archiving are retained.
        receipt['removed'] = json.loads(subprocess.check_output(
            ['docker', 'exec', '-i', container, 'python3', '-c', REMOVE],
            input=json.dumps(manifest), text=True))
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(dict(log_archive=str(archive), **receipt['removed'])), flush=True)
