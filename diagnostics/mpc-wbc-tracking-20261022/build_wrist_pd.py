"""Rebuild one isolated controller object and library with cached CMake recipes."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument('--directory', required=True, type=Path)
args = ap.parse_args()
d = args.directory.resolve()
source = d / 'humanoidController.cpp'
assert source.is_file()
build = Path('/root/kuavo_ws/build/humanoid_controllers')
recipes = json.loads((build / 'compile_commands.json').read_text())
item = next(r for r in recipes if r['file'].endswith('/humanoidController.cpp'))
argv = shlex.split(item['command'])
obj = d / 'humanoidController.cpp.o'
argv[argv.index('-o') + 1] = str(obj)
argv[argv.index('-c') + 1] = str(source)
link = shlex.split((build / 'CMakeFiles/humanoid_controllers.dir/link.txt').read_text())
link = [str(obj) if v.endswith('/humanoidController.cpp.o') else v for v in link]
lib = d / 'libhumanoid_controllers.so'
link[link.index('-o') + 1] = str(lib)
manifest = dict(compile=argv, link=link, cwd=str(build),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(d / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
subprocess.run(argv, cwd=build, check=True)
subprocess.run(link, cwd=build, check=True)
manifest['library_sha256'] = hashlib.sha256(lib.read_bytes()).hexdigest()
(d / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({k: v for k, v in manifest.items() if k not in ['compile', 'link']}))
