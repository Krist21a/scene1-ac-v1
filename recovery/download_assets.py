"""Restore pinned non-Docker assets from this repository's release."""
import argparse
import json
import subprocess
import tarfile
from pathlib import Path
from verify import digest

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'Krist21a/scene1-ac-v1'
TAG = 'scene1-latest-20261115'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true')
    parser.add_argument('--simulator', action='store_true')
    args = parser.parse_args()
    selected = {key for key in ('runtime', 'simulator') if getattr(args, key)} or {'runtime'}
    downloads = ROOT / 'downloads'
    downloads.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / 'recovery/release-assets.json').read_text())
    for asset in manifest['assets']:
        if asset['kind'] not in selected:
            continue
        path = downloads / asset['name']
        if not path.exists():
            subprocess.run(['gh', 'release', 'download', TAG, '--repo', REPOSITORY,
                            '--pattern', asset['name'], '--dir', str(downloads)], check=True)
        if path.stat().st_size != asset['size'] or digest(path) != asset['sha256']:
            raise SystemExit('Download hash mismatch; remove and retry: ' + str(path))
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            for member in members:
                destination = (ROOT / member.name).resolve()
                if member.name.startswith('/') or not destination.is_relative_to(ROOT):
                    raise ValueError('Unsafe archive path: ' + member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError('Unexpected archive member: ' + member.name)
                if destination.is_symlink():
                    raise ValueError('Refusing to replace symlink: ' + member.name)
            archive.extractall(ROOT, members=members)
        print('Restored', asset['name'])
    if 'runtime' in selected:
        subprocess.run(['python3', str(ROOT / 'recovery/verify.py'), '--runtime'], check=True)


if __name__ == '__main__':
    main()
