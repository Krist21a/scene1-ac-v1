"""Build the optional, ROS-independent ordered union-find kernel."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import uuid

root = Path(__file__).resolve().parents[1]
source = root/'scene1/components_native.c'
output = root/'scene1/_components.so'
flags = ['-std=c99', '-O3', '-shared', '-fPIC']
if shutil.which('cc'):
    command = ['cc', *flags, '-o', str(output), str(source)]
    subprocess.run(command, check=True)
else:
    container = 'kuavo-ros-noetic-s12-ready'
    temporary = '/tmp/scene1-components-'+uuid.uuid4().hex+'.so'
    command = ['docker', 'exec', '-i', container, 'cc', *flags, '-x', 'c', '-', '-o', temporary]
    subprocess.run(command, input=source.read_bytes(), check=True)
    try:
        subprocess.run(['docker', 'cp', container+':'+temporary, str(output)], check=True)
    finally:
        subprocess.run(['docker', 'exec', container, 'rm', '--', temporary], check=True)
print(json.dumps(dict(command=command, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     binary_sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
