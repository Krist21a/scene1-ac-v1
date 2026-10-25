"""Read-only verification of the restored production environment and backups."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'diagnostics/mpc-wbc-tracking-20261022'
WS = ROOT.parent / 'workspace/leju-kuavo-challenge-cup-2026'
ALGO = 'kuavo-ros-noetic-s12-ready'
SIM = 'kuavo_challenge_container_GPU_046e4301'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(base, expected):
    missing, mismatch = [], []
    for name, digest in expected.items():
        p = base / name
        if not p.is_file():
            missing.append(name)
        elif sha(p) != digest:
            mismatch.append(name)
    return dict(files=len(expected), missing=missing, mismatch=mismatch)


def main():
    initial = json.loads((OLD / 'initial-manifest.json').read_text())
    production = initial['production']
    out = dict(at=datetime.datetime.now().astimezone().isoformat(),
               production_host=check(ROOT, production),
               production_snapshot=check(OLD / 'baseline/production', production),
               baseline_official=check(WS, initial['baseline_official']),
               root_free_bytes=shutil.disk_usage('/').free,
               errors=[])
    code = '''import hashlib,json,pathlib,sys
expected=json.load(sys.stdin);root=pathlib.Path('/root/scene1_act_optimization')
missing=[];mismatch=[]
for name,digest in expected.items():
 p=root/name
 if not p.is_file():missing.append(name)
 elif hashlib.sha256(p.read_bytes()).hexdigest()!=digest:mismatch.append(name)
print(json.dumps(dict(files=len(expected),missing=missing,mismatch=mismatch)))
'''
    proc = subprocess.run(['docker', 'exec', '-i', ALGO, 'python3', '-c', code],
                          input=json.dumps(production), text=True, capture_output=True, check=True)
    out['production_container'] = json.loads(proc.stdout)
    v63 = json.loads((ROOT / 'diagnostics/full-visual-v63-hashes-20261017.json').read_text())
    out['v63_backup'] = check(ROOT / 'backups/full_visual_v63_initial_close_height', v63)
    binary = 'devel/.private/humanoid_interface_ros/lib/humanoid_interface_ros/humanoid_VR_hand_control'
    source = 'src/humanoid-control/humanoid_interface_ros/src/HumanoidVRHandCommandNode.cpp'
    out['restored_reference'] = check(WS, {
        binary: sha(OLD / 'baseline/vr-reference/humanoid_VR_hand_control'),
        source: sha(OLD / 'baseline/vr-reference/HumanoidVRHandCommandNode.cpp')})
    out['restored_executable_sha256'] = sha(WS / binary)
    controller_backup = OLD / 'baseline/controller-library/libhumanoid_controllers.so'
    if controller_backup.exists():
        controller_relative = 'devel/.private/humanoid_controllers/lib/libhumanoid_controllers.so'
        out['restored_controller_library'] = check(WS, {controller_relative: sha(controller_backup)})
        out['restored_controller_library_sha256'] = sha(WS / controller_relative)
        if out['restored_controller_library']['missing'] or out['restored_controller_library']['mismatch']:
            out['errors'].append('controller library not restored')
    out['official_commit'] = subprocess.check_output(['git', '-C', str(WS), 'rev-parse', 'HEAD'], text=True).strip()
    qr_snapshot=HERE/'baseline-config/manifest.json'
    if qr_snapshot.exists():
        qr=json.loads(qr_snapshot.read_text())
        out['restored_dynamic_qr']=check(WS,{qr['path']:qr['sha256']})
        if out['restored_dynamic_qr']['missing'] or out['restored_dynamic_qr']['mismatch']:
            out['errors'].append('dynamic QR not restored')
    out['official_git_status'] = subprocess.check_output(['git', '-C', str(WS), 'status', '--porcelain'], text=True).strip()
    out['containers'] = {}
    for container in [ALGO, SIM]:
        ps = subprocess.check_output(['docker', 'exec', container, 'ps', '-eo', 'pid,args'], text=True).splitlines()[1:]
        active = [line for line in ps if not any(line.strip().endswith(x) for x in [' zsh', ' /bin/bash', ' ps -eo pid,args'])]
        inspect = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]
        out['containers'][container] = dict(status=inspect['State']['Status'], image_id=inspect['Image'],
                                           started_at=inspect['State']['StartedAt'], processes=ps,
                                           active_work_processes=active,
                                           mounts=[dict(source=x['Source'], destination=x['Destination'], type=x['Type']) for x in inspect['Mounts']])
        if active:
            out['errors'].append('active processes: ' + container)
    for key in ['production_host', 'production_snapshot', 'production_container', 'v63_backup', 'baseline_official', 'restored_reference']:
        if out[key]['missing'] or out[key]['mismatch']:
            out['errors'].append('hash mismatch: ' + key)
    if out['official_commit'] != initial['official_commit'] or out['official_git_status']:
        out['errors'].append('official workspace changed')
    (HERE / (__import__('sys').argv[1] if len(__import__('sys').argv)>1 else 'final-verification.json')).write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'containers'}, ensure_ascii=False, indent=2))
    if out['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
