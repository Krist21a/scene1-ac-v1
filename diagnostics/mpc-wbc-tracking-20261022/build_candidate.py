"""Compile/link isolated candidate using the workspace's exact cached build recipe.

Run inside simulator container with --directory /tmp/reference-origin-build.
No shared workspace source/object/executable is overwritten by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',required=True,type=Path)
    args=ap.parse_args()
    d=args.directory.resolve()
    assert (d/'HumanoidVRHandCommandNode.cpp').is_file()
    ws=Path('/root/kuavo_ws')
    build=ws/'build/humanoid_interface_ros'
    recipes=json.loads((build/'compile_commands.json').read_text())
    item=next(r for r in recipes if r['file'].endswith('/HumanoidVRHandCommandNode.cpp'))
    argv=shlex.split(item['command'])
    obj=d/'HumanoidVRHandCommandNode.cpp.o'
    argv[argv.index('-o')+1]=str(obj)
    argv[argv.index('-c')+1]=str(d/'HumanoidVRHandCommandNode.cpp')
    link=shlex.split((build/'CMakeFiles/humanoid_VR_hand_control.dir/link.txt').read_text())
    link=[str(obj) if v.endswith('HumanoidVRHandCommandNode.cpp.o') else v for v in link]
    exe=d/'humanoid_VR_hand_control'
    link[link.index('-o')+1]=str(exe)
    manifest=dict(compile=argv,link=link,cwd=str(build),source_sha256=hashlib.sha256((d/'HumanoidVRHandCommandNode.cpp').read_bytes()).hexdigest())
    (d/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(argv,cwd=build,check=True)
    subprocess.run(link,cwd=build,check=True)
    manifest['executable_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest()
    (d/'build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['compile','link']}))


if __name__=='__main__':
    main()
