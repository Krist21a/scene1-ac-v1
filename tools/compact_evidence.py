"""Losslessly compact an archived run's large control trace after audit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
from evidence_storage import STORE, require_storage


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''):
        result.update(block)
    return result.hexdigest()


def compact(label):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', label):
        raise ValueError('Invalid label')
    driver = ROOT/'diagnostics/path-tracking-continuation-20261023'
    pipeline = json.loads((STORE/(label+'-pipeline.json')).read_text())
    if any(pipeline['stages'].get(stage) != 0 for stage in ('analysis', 'audit', 'archive', 'restore')):
        raise RuntimeError('Require completed audit, archive and restoration before compaction')
    archive = json.loads((driver/(label+'-archive-manifest.json')).read_text())
    run = driver/label
    source = run/'chain.jsonl'
    output = run/'chain.jsonl.gz'
    receipt = STORE/(label+'-compaction.json')
    if not source.exists() and (run/'chain-receipt.json').is_file():
        transport = json.loads((run/'chain-receipt.json').read_text())
        if not transport.get('verified') or not output.is_file():
            raise RuntimeError('Incomplete native trace transport')
        with output.open('rb') as stream:
            if digest(stream) != archive['files']['chain.jsonl.gz']:
                raise RuntimeError('Native compressed archive changed')
        with gzip.open(output, 'rb') as stream:
            if digest(stream) != transport['original_sha256']:
                raise RuntimeError('Native trace decompression mismatch')
        return
    if receipt.exists():
        if not output.is_file():
            raise RuntimeError('Compacted trace missing')
        with gzip.open(output, 'rb') as stream:
            if digest(stream) != archive['files']['chain.jsonl']:
                raise RuntimeError('Compacted trace differs from historical archive')
        return
    if output.exists():
        raise RuntimeError('Unregistered compressed trace; retain originals for investigation')
    with source.open('rb') as stream:
        original = digest(stream)
    if original != archive['files']['chain.jsonl']:
        raise RuntimeError('Trace differs from historical archive')
    temporary = output.with_suffix('.gz.partial')
    with source.open('rb') as stream, gzip.open(temporary, 'wb', compresslevel=1) as compressed:
        shutil.copyfileobj(stream, compressed, 1024*1024)
    with gzip.open(temporary, 'rb') as stream:
        if digest(stream) != original:
            raise RuntimeError('Compression verification failed; original retained')
    temporary.rename(output)
    with output.open('rb') as stream:
        compressed_sha = digest(stream)
    receipt.write_text(json.dumps(dict(label=label, original_sha256=original,
        compressed_sha256=compressed_sha, original_bytes=source.stat().st_size,
        compressed_bytes=output.stat().st_size,
        note='Historical archive unchanged. Decompress chain.jsonl.gz to recover its exact original bytes.'),
        indent=2)+'\n')
    source.unlink()


def prune_periodic(label):
    """Retain keyframe evidence; explicitly retire redundant debug snapshots."""
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', label):
        raise ValueError('Invalid label')
    pipeline = json.loads((STORE/(label+'-pipeline.json')).read_text())
    if pipeline.get('current_stage') != 'done' or any(
            pipeline['stages'].get(stage) != 0 for stage in ('analysis', 'audit', 'archive', 'restore')):
        raise RuntimeError('Require completed audit, archive and restoration')
    driver = ROOT/'diagnostics/path-tracking-continuation-20261023'
    run = driver/label
    archive = json.loads((driver/(label+'-archive-manifest.json')).read_text())
    receipt = STORE/(label+'-snapshot-retention.json')
    if receipt.exists():
        if json.loads(receipt.read_text()).get('status') != 'complete':
            raise RuntimeError('Incomplete snapshot retirement; inspect receipt before resuming')
        return
    keep = {1}
    last = None
    for line in (run/'output/steps.jsonl').open():
        row = json.loads(line)
        last = row['step']
        info = row.get('cartesian_alignment') or {}
        if info.get('transition_to') or info.get('state') in (
                'grasp_contact_confirmed', 'weighing_release', 'short_retreat_complete',
                'placement_stable', 'pregrasp_aligned', 'regrasp_pose_verified',
                'visual_cycle_complete'):
            keep.add(last)
    candidates = []
    for path in (run/'output').iterdir():
        match = re.fullmatch(r'(?:vision|wrist-left|wrist-right|sensors)-(\d+)\.(?:jpg|npz)', path.name)
        if match:
            candidates.append((path, int(match[1])))
    if candidates:
        keep.add(max(step for _, step in candidates))
    removed = {}
    # Verify the whole removal set before deleting any file. The original
    # archive manifest remains unchanged; this separate receipt records loss.
    for path, step in candidates:
        if step in keep:
            continue
        relative = str(path.relative_to(run))
        with path.open('rb') as stream:
            actual = digest(stream)
        if actual != archive['files'].get(relative):
            raise RuntimeError(('Snapshot changed since archive', relative))
        removed[relative] = dict(sha256=actual, bytes=path.stat().st_size)
    receipt.write_text(json.dumps(dict(label=label, status='planned', retained_keyframe_steps=sorted(keep),
        removed=removed, removed_bytes=sum(r['bytes'] for r in removed.values()),
        note='Periodic debugging images retired after audit. All control/timing logs, keyframes and failure captures retained; historical manifest unchanged.'), indent=2)+'\n')
    for relative in removed:
        (run/relative).unlink()
    completed = json.loads(receipt.read_text())
    completed['status'] = 'complete'
    receipt.write_text(json.dumps(completed, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--prune-periodic', action='store_true')
    args = parser.parse_args()
    compact(args.label)
    if args.prune_periodic:
        prune_periodic(args.label)
