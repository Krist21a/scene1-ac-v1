"""Reconstruct delivered patches and audit retained experiment identities.

Uses disposable files only. No controller switch, simulator or policy launch.
"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import controller_variants as variants

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'mpc-wbc-tracking-20261022'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(directory, manifest):
    for name, expected in manifest.items():
        assert sha(directory / name) == expected, str(directory / name)


def apply(directory, patch):
    result = subprocess.run(['patch', '--batch', '--fuzz=0', '--no-backup-if-mismatch',
                             '-p1', '-i', str(patch)], cwd=directory, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return dict(patch=str(patch.relative_to(HERE)), sha256=sha(patch), output=result.stdout)


def main():
    receipt = dict(at=datetime.datetime.now().astimezone().isoformat(), path_patches=[],
                   control_patches=[], experiments=[], errors=[])
    try:
        parent = HERE.parent / 'mpc-wbc-integrated-20261022/candidate-policy-reobserve'
        manifest = read(HERE / 'parent-manifest.json')
        check_manifest(parent, manifest)
        with tempfile.TemporaryDirectory(prefix='scene1-delivery-') as temporary:
            root = Path(temporary)
            policy = root / 'policy'
            for name in manifest:
                target = policy / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(parent / name, target)
            for patch, directory, manifest_name in [
                    ('axis.patch', 'candidate-policy-axis', 'axis-manifest.json'),
                    ('axis-sweep.patch', 'candidate-policy-axis-sweep', 'axis-sweep-manifest.json'),
                    ('preflight-feasibility.patch', 'candidate-policy-axis-feasible', 'axis-feasible-manifest.json')]:
                row = apply(policy, HERE / patch)
                expected = read(HERE / manifest_name)
                check_manifest(policy, expected)
                check_manifest(HERE / directory, expected)
                row.update(files_verified=len(expected), candidate=directory,
                           manifest_sha256=sha(HERE / manifest_name))
                receipt['path_patches'].append(row)

            original = OLD / 'baseline/src/humanoid-control/humanoid_controllers/src/humanoidController.cpp'
            for label, source, patch, target in [
                    ('C', original, HERE / 'control-c.patch', OLD / 'candidate-wrist-pd'),
                    ('D', original, HERE / 'candidate-velocity/change.patch', HERE / 'candidate-velocity'),
                    ('D+C', OLD / 'candidate-wrist-pd/humanoidController.cpp',
                     HERE / 'candidate-velocity-pd/change.patch', HERE / 'candidate-velocity-pd')]:
                directory = root / label
                directory.mkdir()
                relative = ('src/humanoid-control/humanoid_controllers/src/humanoidController.cpp'
                            if label == 'C' else 'humanoidController.cpp')
                destination = directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                row = apply(directory, patch)
                for file in directory.rglob('*'):
                    if file.is_file():
                        assert sha(file) == sha(target / file.name), file.name
                row.update(label=label, source_sha256=sha(target / 'humanoidController.cpp'))
                receipt['control_patches'].append(row)
            qr = root / 'E' / variants.QR_RELATIVE
            qr.parent.mkdir(parents=True)
            shutil.copyfile(HERE / 'baseline-config/dynamic_qr.info', qr)
            row = apply(root / 'E', HERE / 'mpc-arm-q.patch')
            assert sha(qr) == sha(HERE / 'candidate-mpc-arm-q/dynamic_qr.info')
            row.update(label='E', qr_sha256=sha(qr))
            receipt['control_patches'].append(row)

        identity = variants.available()
        receipt['controller_artifacts'] = {name: dict(library_sha256=value['sha256'],
                                                       dynamic_qr_sha256=value['qr_sha256'])
                                           for name, value in identity.items()}
        empty_versions = {'original-empty-s0': 'baseline', 'c-empty-s0': 'candidate',
                          'd-empty-s0': 'velocity', 'dc-empty-s0': 'velocity_pd', 'e-empty-s0': 'mpc_q'}
        comparison = read(HERE / 'comparison.json')
        for row in comparison['empty'] + comparison['tasks']:
            label = row['label']
            state = read(HERE / label / 'experiment.json')
            version = empty_versions.get(label, state.get('version'))
            expected = identity[version]
            loaded = state['runtime_controller_sha256']
            if isinstance(loaded, str):
                loaded = [loaded]
            assert loaded == [expected['sha256']], label
            if state.get('dynamic_qr_sha256'):
                assert state['dynamic_qr_sha256'] == expected['qr_sha256'], label
            assert state['seed'] == 0 and state['status'] in ['completed', 'policy_finished', 'policy_error'], label
            archive_path = HERE / (label + '-archive-manifest.json')
            archive = read(archive_path)
            if label not in empty_versions:
                assert archive['verified'], label
                name = 'axis-feasible-manifest.json' if label.startswith('feasible-') else 'axis-sweep-manifest.json'
                assert state['policy_manifest_sha256'] == sha(HERE / name), label
                assert row['official_task_success'] is None, label
            for filename in ['experiment.json', 'runtime.json']:
                assert sha(HERE / label / filename) == archive['files'][filename], (label, filename)
            receipt['experiments'].append(dict(label=label, status=state['status'], version=version,
                                                archive_manifest_sha256=sha(archive_path),
                                                runtime_sha256=sha(HERE / label / 'runtime.json')))
        assert read(HERE / 'final-verification.json')['errors'] == []
        assert read(HERE / 'sam2-removal.json')['complete']
        assert len(comparison['empty']) == 5 and len(comparison['tasks']) == 6
        receipt['heldout_seeds_used'] = []
        receipt['status'] = 'passed'
    except Exception as error:
        receipt['status'] = 'failed'
        receipt['errors'].append(repr(error))
        raise
    finally:
        (HERE / 'delivery-verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({key: receipt.get(key) for key in ['status', 'errors']}))


if __name__ == '__main__':
    main()
