"""Verify downloaded source/runtime bytes. This does not validate a robot task."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(name):
    manifest = json.loads((ROOT / 'recovery' / name).read_text())
    for relative, record in manifest.items():
        path = ROOT / record.get('stored_as', relative)
        if not path.is_file() or path.stat().st_size != record['size'] or digest(path) != record['sha256']:
            raise SystemExit('Missing or changed file: ' + str(path))
    print('Verified', name, len(manifest), 'files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true')
    args = parser.parse_args()
    verify('source-files.json')
    if args.runtime:
        verify('runtime-files.json')
