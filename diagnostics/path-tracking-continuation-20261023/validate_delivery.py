"""Verify source patches, every retained trial, and final baseline restoration.

Run when all experiments have ended. This audits evidence; it does not turn a
failed trial or a null official verdict into a task success.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import controller_variants as variants
import environment as e

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'path-tracking-20261023'
SOURCE_RELATIVE = 'src/humanoid-control/humanoid_controllers/src/humanoidController.cpp'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_manifest(root, manifest):
    for relative, expected in manifest.items():
        assert sha(root / relative) == expected, str(root / relative)


def apply(root, path):
    result = subprocess.run(['patch', '--batch', '--fuzz=0', '--no-backup-if-mismatch',
                             '-p1', '-i', str(path)], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(path=str(path), sha256=sha(path))


def main():
    receipt = dict(at=datetime.datetime.now().astimezone().isoformat(),
                   path_patches=[], control_patches=[], experiments=[], errors=[])
    try:
        states = [(p.parent, read(p)) for p in sorted(HERE.glob('*/experiment.json'))]
        assert states and all(x.get('ended_at') for _, x in states), 'An experiment is still active'
        e.wait_idle(e.ALGO)
        e.wait_idle(e.SIM)
        subprocess.run([sys.executable, str(HERE/'verify_path_delivery.py')], check=True)
        receipt['continuation_path'] = read(HERE/'path-delivery-verification.json')
        parent = HERE.parent / 'mpc-wbc-integrated-20261022/candidate-policy-reobserve'
        initial = read(HERE/'parent-manifest.json')
        check_manifest(parent, initial)
        with tempfile.TemporaryDirectory(prefix='scene1-final-delivery-') as temporary:
            work = Path(temporary)
            path = work/'path'
            for relative in initial:
                target = path/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(parent/relative, target)
            for patch, candidate in [('axis.patch', 'axis'), ('axis-sweep.patch', 'axis-sweep'),
                                     ('preflight-feasibility.patch', 'axis-feasible')]:
                row = apply(path, PRIOR/patch)
                expected = read(HERE/(candidate+'-manifest.json'))
                check_manifest(path, expected)
                check_manifest(HERE/('candidate-policy-'+candidate), expected)
                row['manifest_sha256'] = sha(HERE/(candidate+'-manifest.json'))
                receipt['path_patches'].append(row)
            controller = work/'C'
            target = controller/SOURCE_RELATIVE
            target.parent.mkdir(parents=True)
            shutil.copyfile(e.OLD/'baseline'/SOURCE_RELATIVE, target)
            row = apply(controller, PRIOR/'control-c.patch')
            build = read(e.OLD/'candidate-wrist-pd/build-manifest.json')
            assert sha(target) == sha(e.OLD/'candidate-wrist-pd/humanoidController.cpp') == build['source_sha256']
            assert sha(e.OLD/'candidate-wrist-pd/libhumanoid_controllers.so') == build['library_sha256'] == e.PD_SHA
            row.update(label='C', source_sha256=sha(target), library_sha256=e.PD_SHA,
                       interpretation='Bounded compensation after WBC output; not internal QP optimization')
            receipt['control_patches'].append(row)
            controller = work/'A'
            vr_relative='src/humanoid-control/humanoid_interface_ros/src/HumanoidVRHandCommandNode.cpp'
            target=controller/vr_relative
            target.parent.mkdir(parents=True)
            shutil.copyfile(e.OLD/'baseline/vr-reference/HumanoidVRHandCommandNode.cpp',target)
            row=apply(controller,e.OLD/'candidate-reference-origin/change.patch')
            build=read(e.OLD/'candidate-reference-origin/build-manifest.json')
            assert sha(target)==sha(e.OLD/'candidate-reference-origin/HumanoidVRHandCommandNode.cpp')==build['source_sha256']
            assert sha(e.OLD/'candidate-reference-origin/humanoid_VR_hand_control')==build['executable_sha256']
            row.update(label='A',source_sha256=sha(target),executable_sha256=build['executable_sha256'],
                       interpretation='Correct arm first-knot indexing in ROS target generation before MPC')
            receipt['control_patches'].append(row)
            controller = work/'G'
            target = controller/SOURCE_RELATIVE
            target.parent.mkdir(parents=True)
            shutil.copyfile(e.OLD/'candidate-wrist-pd/humanoidController.cpp', target)
            row = apply(target.parent, HERE/'candidate-wrist-damping/wrist-damping.patch')
            build = read(HERE/'candidate-wrist-damping/build-manifest.json')
            assert sha(target) == sha(HERE/'candidate-wrist-damping/humanoidController.cpp') == build['source_sha256']
            assert sha(HERE/'candidate-wrist-damping/libhumanoid_controllers.so') == build['library_sha256']
            row.update(label='G', source_sha256=sha(target), library_sha256=build['library_sha256'],
                       selection='Rejected: increased post-WBC damping degraded empty tracking and failed loaded preparation')
            receipt['control_patches'].append(row)
            controller = work/'F'
            target = controller/variants.TASK_RELATIVE
            target.parent.mkdir(parents=True)
            shutil.copyfile(HERE/'baseline-task/task.info', target)
            row = apply(controller, HERE/'candidate-proximal-kp/proximal-kp.patch')
            assert sha(target) == sha(HERE/'candidate-proximal-kp/task.info')
            row.update(label='F', task_sha256=sha(target), selection='Rejected after loaded contact test')
            receipt['control_patches'].append(row)
            controller = work/'H'
            target = controller/variants.TASK_RELATIVE
            target.parent.mkdir(parents=True)
            shutil.copyfile(HERE/'baseline-task/task.info', target)
            row = apply(controller, HERE/'candidate-arm-weight/arm-weight.patch')
            registration = read(HERE/'candidate-arm-weight/registration.json')
            assert sha(target) == sha(HERE/'candidate-arm-weight/task.info') == registration['candidate_sha256']
            row.update(label='H', task_sha256=sha(target), selection='Empty-only research; not selected for loaded work')
            receipt['control_patches'].append(row)
            controller = work/'I'
            target = controller/SOURCE_RELATIVE
            target.parent.mkdir(parents=True)
            shutil.copyfile(e.OLD/'candidate-wrist-pd/humanoidController.cpp', target)
            row = apply(controller, HERE/'candidate-pd-scale/pd-scale.patch')
            build = read(HERE/'candidate-pd-scale/build-manifest.json')
            assert sha(target) == sha(HERE/'candidate-pd-scale/humanoidController.cpp') == build['source_sha256']
            assert sha(HERE/'candidate-pd-scale/libhumanoid_controllers.so') == build['library_sha256']
            row.update(label='I', source_sha256=sha(target), library_sha256=build['library_sha256'],
                       interpretation='C post-WBC PD amplitude x2 before original torque bounds; not QP source optimization')
            receipt['control_patches'].append(row)
            controller = work/'J'
            target = controller/SOURCE_RELATIVE
            target.parent.mkdir(parents=True)
            shutil.copyfile(e.OLD/'candidate-wrist-pd/humanoidController.cpp', target)
            row = apply(controller, HERE/'candidate-pd-scale15/pd-scale15.patch')
            build = read(HERE/'candidate-pd-scale15/build-manifest.json')
            assert sha(target) == sha(HERE/'candidate-pd-scale15/humanoidController.cpp') == build['source_sha256']
            assert sha(HERE/'candidate-pd-scale15/libhumanoid_controllers.so') == build['library_sha256']
            row.update(label='J', source_sha256=sha(target), library_sha256=build['library_sha256'],
                       interpretation='C post-WBC PD amplitude x1.5 before original torque bounds; not QP source optimization')
            receipt['control_patches'].append(row)

        identities = variants.available()
        run_ids = set()
        for directory, state in states:
            label = directory.name
            expected = identities[state['version']]
            archive_path = HERE/(label+'-archive-manifest.json')
            archive = read(archive_path)
            # Archive manifests include failures, instrumentation errors and all raw frames.
            check_manifest(directory, archive['files'])
            row = dict(label=label, status=state['status'], version=state['version'], seed=state['seed'],
                       archive_sha256=sha(archive_path), verified_files=len(archive['files']))
            runtime_path = directory/'runtime.json'
            if runtime_path.exists():
                runtime = read(runtime_path)
                mapped = [digest for process in runtime['processes']
                          for name, digest in process['libraries'].items()
                          if name.endswith('/libhumanoid_controllers.so')]
                assert mapped == [expected['sha256']], (label, 'mapped library')
                vr_mapped=[p['sha256'] for p in runtime['processes']
                           if Path(p['exe']).name=='humanoid_VR_hand_control']
                assert vr_mapped==[expected['vr_sha256']], (label,'arm-reference executable')
                for parameter, digest in [('/taskFile', expected['task_sha256']),
                                          ('/dynamicQrFile', expected['qr_sha256'])]:
                    assert runtime['files'][runtime['params'][parameter]] == digest, (label, parameter)
                run_id = runtime['params']['/run_id']
                assert run_id == state['ros_run_id'] and run_id not in run_ids, label
                run_ids.add(run_id)
                row.update(runtime_sha256=sha(runtime_path), ros_run_id=run_id,
                           library_sha256=expected['sha256'], task_sha256=expected['task_sha256'],
                           vr_sha256=expected['vr_sha256'])
            if state.get('policy_manifest_sha256'):
                name = state['policy_variant'].replace('_', '-')+'-manifest.json'
                assert state['policy_manifest_sha256'] == sha(HERE/name), label
                row['policy_sha256'] = state['policy_manifest_sha256']
            for name, digest in state.get('orchestration_files', {}).items():
                assert sha(directory/'orchestration'/name) == digest, (label, name)
            if (directory/'output/summary.json').exists():
                summary = read(directory/'output/summary.json')
                row.update(visual_confirmed_parcels=summary['visual_confirmed_parcels'],
                           official_task_success=summary['task_success'],
                           four_parcel_visual_complete=state.get('independent_four_parcel_visual_completion', False))
            receipt['experiments'].append(row)
            print(json.dumps(dict(label=label, verified_files=row['verified_files'])), flush=True)
        assert read(HERE/'final-verification.json')['errors'] == []
        assert read(PRIOR/'sam2-removal.json')['complete']
        receipt['unique_ros_runs'] = len(run_ids)
        receipt['heldout_seeds_used'] = sorted({s['seed'] for _, s in states if s['seed'] != 0})
        receipt['status'] = 'passed'
    except Exception as error:
        receipt['status'] = 'failed'
        receipt['errors'].append(repr(error))
        raise
    finally:
        (HERE/'delivery-verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({key: receipt.get(key) for key in ['status', 'errors']}))


if __name__ == '__main__':
    main()
