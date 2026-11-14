"""Lossless trace transport; audits always read the original JSONL byte stream."""
import gzip
import hashlib
import json
from pathlib import Path


def trace_path(run):
    raw = Path(run)/'chain.jsonl'
    return raw if raw.is_file() else Path(run)/'chain.jsonl.gz'


def open_trace(run, mode='rt'):
    path = trace_path(run)
    return gzip.open(path, mode) if path.suffix == '.gz' else path.open(mode)


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1048576), b''):
        result.update(block)
    return result.hexdigest()


def collect(container, remote, run, command):
    # Compress only after recording stops. Keep the raw container trace until
    # archive_completed independently verifies and removes its redundant copy.
    code = '''import gzip,hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);out=p.with_suffix(p.suffix+'.gz')
h=hashlib.sha256()
with p.open('rb') as src,gzip.open(out,'wb',compresslevel=6) as dst:
 for block in iter(lambda:src.read(1048576),b''):
  h.update(block);dst.write(block)
original=h.hexdigest();h=hashlib.sha256()
with out.open('rb') as src:
 for block in iter(lambda:src.read(1048576),b''):h.update(block)
print(json.dumps(dict(original_sha256=original,original_bytes=p.stat().st_size,
 compressed_sha256=h.hexdigest(),compressed_bytes=out.stat().st_size)))
'''
    receipt = json.loads(command('docker','exec',container,'python3','-c',code,remote))
    target = Path(run)/'chain.jsonl.gz'
    command('docker','cp',container+':'+remote+'.gz',str(target))
    with target.open('rb') as stream:
        if digest(stream) != receipt['compressed_sha256']:
            raise RuntimeError('Compressed trace transfer mismatch; container originals retained')
    with gzip.open(target, 'rb') as stream:
        if digest(stream) != receipt['original_sha256']:
            raise RuntimeError('Decompressed trace differs; container originals retained')
    receipt['compression_level'] = 6
    receipt['verified'] = True
    receipt['format'] = 'gzip; exact original JSONL bytes after decompression'
    (Path(run)/'chain-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    command('docker','exec',container,'python3','-c',
            'import pathlib,sys;pathlib.Path(sys.argv[1]).unlink()',remote+'.gz')
    return receipt
