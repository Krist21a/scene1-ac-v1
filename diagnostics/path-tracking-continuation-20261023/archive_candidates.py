"""Byte-verify immutable research policy copies before freeing root duplicates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ARCHIVE = Path('/media/krist/KINGSTON/scene1-path-tracking-continuation-20261023/policies')


def files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()
            and '__pycache__' not in p.parts and p.suffix != '.pyc'}


ap = argparse.ArgumentParser()
ap.add_argument('names', nargs='+')
args = ap.parse_args()
assert ARCHIVE.parents[1].is_mount()
ARCHIVE.mkdir(exist_ok=True)
for name in args.names:
    assert name.startswith('executor') and all(c.isalnum() or c == '-' for c in name)
    source = HERE/('candidate-policy-'+name)
    assert source.is_dir() and not source.is_symlink(), source
    expected = json.loads((HERE/(name+'-manifest.json')).read_text())
    original = files(source)
    assert all(original.get(k) == v for k, v in expected.items()), source
    destination = ARCHIVE/source.name
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assert files(destination) == original
    temporary = source.with_name(source.name+'.verified-duplicate')
    source.rename(temporary)
    try:
        source.symlink_to(destination, target_is_directory=True)
    except Exception:
        temporary.rename(source)
        raise
    shutil.rmtree(temporary)
    (HERE/(name+'-policy-archive.json')).write_text(json.dumps(dict(
        destination=str(destination), files=original, verified=True), indent=2)+'\n')
    print(name+': verified external copy; original path retained as symlink')
