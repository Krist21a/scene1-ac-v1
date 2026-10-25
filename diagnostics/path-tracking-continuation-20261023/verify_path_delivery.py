"""Reconstruct every continuation path patch without touching active policy."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
CHAIN = [
    ('executor', 'executor-certificate.patch'),
    ('executor-geometry', 'release-geometry.patch'),
    ('executor-spatial', 'executor-spatial.patch'),
    ('executor-release', 'executor-release.patch'),
    ('executor-physical', 'executor-physical.patch'),
    ('executor-context', 'executor-context.patch'),
    ('executor-hold', 'executor-hold.patch'),
    ('executor-open', 'executor-open.patch'),
    ('executor-project', 'executor-project.patch'),
    ('executor-obb', 'executor-obb.patch'),
    ('executor-fresh', 'executor-fresh.patch'),
    ('executor-fast', 'executor-fast.patch'),
    ('executor-mpc-hold', 'executor-mpc-hold.patch'),
    ('executor-numerics', 'executor-numerics.patch'),
    ('executor-prepare', 'executor-prepare.patch'),
    ('executor-prepare-adapt', 'executor-prepare-adapt.patch'),
    ('executor-grasp-frame', 'executor-grasp-frame.patch'),
    ('executor-grasp-gain', 'executor-grasp-gain.patch'),
    ('executor-grasp-settle', 'executor-grasp-settle.patch'),
    ('executor-support-heading', 'executor-support-heading.patch'),
    ('executor-support-gain', 'executor-support-gain.patch'),
    ('executor-pending-hold', 'executor-pending-hold.patch'),
    ('executor-first-gain', 'executor-first-gain.patch'),
    ('executor-arm-prepare', 'executor-arm-prepare.patch'),
    ('executor-soft-open', 'executor-soft-open.patch'),
    ('executor-support-prepare', 'executor-support-prepare.patch'),
    ('executor-insert-pitch', 'executor-insert-pitch.patch'),
    ('executor-support-recover', 'executor-support-recover.patch'),
    ('executor-placement-physical', 'executor-placement-physical.patch'),
    ('executor-release-quiet', 'executor-release-quiet.patch'),
    ('executor-first-transport', 'executor-first-transport.patch'),
    ('executor-first-pitch', 'executor-first-pitch.patch'),
    ('executor-first-pickup', 'executor-first-pickup.patch'),
    ('executor-first-select', 'executor-first-select.patch'),
    ('executor-pickup-sweep', 'executor-pickup-sweep.patch'),
    ('executor-first-clear', 'executor-first-clear.patch'),
    ('executor-first-reobserve', 'executor-first-reobserve.patch'),
    ('executor-pickup-hold-sweep', 'executor-pickup-hold-sweep.patch'),
    ('executor-first-bin', 'executor-first-bin.patch'),
    ('executor-bin-components', 'executor-bin-components.patch'),
    ('executor-bin-dense', 'executor-bin-dense.patch'),
    ('executor-table-dense', 'executor-table-dense.patch'),
    ('executor-pad-dense', 'executor-pad-dense.patch'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = dict(at=datetime.datetime.now().astimezone().isoformat(),
                   candidates=[], errors=[], status='running')
    try:
        previous = json.loads((HERE/'axis-feasible-manifest.json').read_text())
        parent = HERE/'candidate-policy-axis-feasible'
        with tempfile.TemporaryDirectory(prefix='scene1-path-reconstruction-') as name:
            working = Path(name)
            for relative, digest in previous.items():
                assert sha(parent/relative) == digest, relative
                target = working/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(parent/relative, target)
            for candidate, patch in CHAIN:
                source = HERE/('candidate-policy-'+candidate)
                manifest_path = HERE/(candidate+'-manifest.json')
                expected = json.loads(manifest_path.read_text())
                # Raw sensor fixtures/public robot description are separate
                # immutable assets. Never copy implementation to repair patches.
                assets = []
                for relative in sorted(expected.keys()-previous.keys()):
                    if relative.startswith('tests/fixtures/'):
                        assert Path(relative).suffix in ('.npz', '.json', '.urdf')
                        target = working/relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source/relative, target)
                        assets.append(dict(path=relative, sha256=sha(target)))
                result = subprocess.run(['patch', '--batch', '--fuzz=0',
                    '--no-backup-if-mismatch', '-p1', '-i', str(HERE/patch)],
                    cwd=working, text=True, capture_output=True)
                assert result.returncode == 0, result.stdout+result.stderr
                for relative, digest in expected.items():
                    assert sha(working/relative) == digest, ('reconstruction', candidate, relative)
                    assert sha(source/relative) == digest, ('candidate', candidate, relative)
                receipt['candidates'].append(dict(candidate=candidate,
                    manifest_sha256=sha(manifest_path), patch=patch,
                    patch_sha256=sha(HERE/patch), files=len(expected), assets=assets))
                previous = expected
        receipt['status'] = 'passed'
    except Exception as error:
        receipt['status'] = 'failed'
        receipt['errors'].append(repr(error))
        raise
    finally:
        (HERE/'path-delivery-verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(dict(status=receipt['status'], errors=receipt['errors'],
                              candidates=len(receipt['candidates']))))


if __name__ == '__main__':
    main()
