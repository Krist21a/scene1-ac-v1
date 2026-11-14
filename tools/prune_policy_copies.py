"""Remove only byte-verified container copies of archived, finished trial sources."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from evidence_storage import STORE, require_storage
ALGO = 'kuavo-ros-noetic-s12-ready'
SIM = 'kuavo_challenge_container_GPU_046e4301'
REMOTE = r'''
import hashlib,json,sys
from pathlib import Path
data=json.load(sys.stdin)
root=Path(data['root'])
checked=[]
for name,digest in data['manifest'].items():
    path=root/name
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(('Missing/non-regular frozen source',str(path)))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise RuntimeError(('Frozen source changed; nothing removed',str(path)))
    checked.append((path,path.stat().st_size))
for path,size in checked:
    path.unlink()
print(json.dumps(dict(files=len(checked),bytes=sum(size for _,size in checked),
                     remaining_files=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()])))
'''


def prune(label):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', label):
        raise ValueError('Invalid trial label')
    receipt_path = STORE/(label+'-policy-prune.json')
    if receipt_path.exists():
        return json.loads(receipt_path.read_text())
    pipeline = json.loads((STORE/(label+'-pipeline.json')).read_text())
    # A failed experiment's sources are also reproducible from the verified
    # external snapshot. Preserve all generated evidence and require completed
    # analysis/archive/restoration regardless of the policy's outcome.
    required = ('analysis', 'audit', 'archive', 'restore')
    if pipeline.get('current_stage') != 'done' or any(pipeline['stages'].get(s) != 0 for s in required):
        raise RuntimeError('Only fully analyzed, archived and restored trials can be pruned')
    manifest_path = STORE/'policies'/(label+'-manifest.json')
    manifest = json.loads(manifest_path.read_text())
    frozen = STORE/'policies'/label
    for name, digest in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path')
        path = frozen/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(('External copy failed verification; nothing removed', name))
    result = subprocess.run(['docker', 'exec', '-i', ALGO,
                             '/root/miniforge3/envs/kls_dev/bin/python', '-c', REMOTE],
                            input=json.dumps(dict(root='/root/scene1_speed/'+label, manifest=manifest)),
                            text=True, capture_output=True, check=True)
    receipt = dict(label=label, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   external_copy=str(frozen), removed=json.loads(result.stdout),
                   restoration='docker cp external_copy/. container:/root/scene1_speed/'+label)
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def prune_analysis(label):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', label):
        raise ValueError('Invalid trial label')
    receipt_path = STORE/(label+'-analysis-prune.json')
    if receipt_path.exists():
        return json.loads(receipt_path.read_text())
    pipeline = json.loads((STORE/(label+'-pipeline.json')).read_text())
    if pipeline.get('current_stage') != 'done' or any(pipeline['stages'].get(s) != 0
            for s in ('analysis', 'audit', 'archive', 'restore')):
        raise RuntimeError('Only fully archived/restored analysis copies can be pruned')
    driver = Path(__file__).resolve().parents[1]/'diagnostics/path-tracking-continuation-20261023'
    names = ('pd-audit.json', 'chain-targets.json', 'reference-analysis.json',
             'steps.jsonl', 'release-transients.json', 'metrics-full.json')
    manifest = {}
    for name in names:
        source = (driver/label/'output'/name if name == 'steps.jsonl'
                  else driver/'analysis-full'/label/name)
        if name == 'pd-audit.json' and not source.is_file():
            continue  # Older analysis drivers did not emit this optional secondary report.
        manifest[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    result = subprocess.run(['docker', 'exec', '-i', SIM, 'python3', '-c', REMOTE],
        input=json.dumps(dict(root='/tmp/scene1-integrated-analysis/'+label, manifest=manifest)),
        text=True, capture_output=True, check=True)
    receipt = dict(label=label, source_hashes=manifest, removed=json.loads(result.stdout))
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', nargs='+')
    parser.add_argument('--analysis', action='store_true')
    args = parser.parse_args()
    for label in args.labels:
        receipt = prune(label)
        print(json.dumps(dict(label=label, removed_bytes=receipt['removed']['bytes'])), flush=True)
        if args.analysis:
            receipt = prune_analysis(label)
            print(json.dumps(dict(label=label, removed_analysis_bytes=receipt['removed']['bytes'])), flush=True)
