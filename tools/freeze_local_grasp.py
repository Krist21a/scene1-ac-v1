"""Freeze the validation winner and both final-test profiles before testing."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

from evidence_storage import EVIDENCE_ROOT, STORE, require_storage
from speed_acceptance import inspect

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--ranking', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', args.label):
        raise ValueError('Invalid final policy label')
    require_storage()
    args.output = args.output.absolute()
    args.output.relative_to(EVIDENCE_ROOT)
    selection = json.loads(args.selection.read_text())
    ranking = json.loads(args.ranking.read_text())
    registry = json.loads(args.registry.read_text())
    split_hash = hashlib.sha256(json.dumps(registry['seed_splits'], sort_keys=True).encode()).hexdigest()
    if (selection['status'] != 'complete' or selection['final_test_read'] is not False
            or registry['status'] != 'complete' or selection['ranking_sha256'] != digest(args.ranking)
            or selection['collection_registry_sha256'] != digest(args.registry)
            or selection['seed_split_sha256'] != split_hash or registry['seed_split_sha256'] != split_hash
            or selection['selector_sha256'] != digest(ROOT/'tools/select_local_grasp_checkpoint.py')):
        raise ValueError('Require unchanged completed validation selection and original seed registry')
    winners = [row for row in selection['candidates'] if row['update'] == selection['selected_update']]
    nominated = [row for row in ranking['top_three'] if row['update'] == selection['selected_update']]
    if len(winners) != 1 or len(nominated) != 1:
        raise ValueError('Selected checkpoint was not nominated by validation loss')
    winner, nominated = winners[0], nominated[0]
    for candidate in selection['candidates']:
        for row in candidate['runs']:
            if inspect(row['label'], ROOT, STORE, 650.) != row['acceptance']:
                raise ValueError('Validation round evidence changed: '+row['label'])
    test_seeds = registry['seed_splits']['test']
    collected_labels = {row['label'] for row in registry['runs']}
    for path in STORE.glob('*-pipeline.json'):
        run = json.loads(path.read_text())
        if run.get('seed') in test_seeds and run['label'] not in collected_labels:
            raise ValueError('A final-test seed was used before this model/configuration freeze')
    model_source = EVIDENCE_ROOT/winner['model_relative_path']
    hashes = nominated['files_sha256']
    if {str(p.relative_to(model_source)) for p in model_source.rglob('*') if p.is_file()} != set(hashes):
        raise ValueError('Selected checkpoint file set changed')
    for name, expected in hashes.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or (model_source/relative).is_symlink()
                or digest(model_source/relative) != expected):
            raise ValueError('Selected checkpoint or preprocessing changed')
    source_run = winner['runs'][0]['acceptance']
    source = STORE/'policies'/source_run['label']
    source_manifest = source.parent/(source_run['label']+'-manifest.json')
    if digest(source_manifest) != selection['policy_manifest_sha256']:
        raise ValueError('Validated policy manifest changed')
    manifest = json.loads(source_manifest.read_text())
    for name, expected in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or digest(source/relative) != expected:
            raise ValueError('Validated policy source changed: '+name)
    destination = STORE/'policies'/args.label
    manifest_path = destination.parent/(args.label+'-manifest.json')
    model_destination = args.output.parent/(args.label+'-model')
    if any(path.exists() for path in (args.output, destination, manifest_path, model_destination)):
        raise ValueError('Final freeze destination already exists')
    mount = json.loads((EVIDENCE_ROOT/'traditional-local-act-1107/training-storage-mount.json').read_text())
    container_root = Path(mount['container_target'])
    container_model = str(container_root/model_destination.relative_to(EVIDENCE_ROOT))
    container_manifest = str(container_root/args.output.relative_to(EVIDENCE_ROOT))
    import yaml
    baseline = yaml.safe_load((source/source_run['profile']).read_text())
    for key in ('grasp_execution', 'local_act_model', 'local_act_manifest'):
        baseline['visual_cycle'].pop(key, None)
    baseline['visual_cycle']['grasp_execution'] = 'traditional'
    learned = yaml.safe_load(yaml.safe_dump(baseline))
    learned['visual_cycle'].update(grasp_execution='local_act', local_act_model=container_model,
                                   local_act_manifest=container_manifest)
    profiles = {f'configs/{args.label}-traditional.yaml':baseline,
                f'configs/{args.label}-local-act.yaml':learned}
    if set(profiles) & set(manifest):
        raise ValueError('Final profiles would overwrite validated source files')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_source, model_destination)
    for name in hashes:
        if digest(model_destination/name) != hashes[name]:
            raise RuntimeError('Final model copy verification failed')
    destination.mkdir(parents=True)
    for name in manifest:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
        if digest(target) != manifest[name]:
            raise RuntimeError('Final policy copy verification failed')
    for name, profile in profiles.items():
        (destination/name).write_text(yaml.safe_dump(profile, sort_keys=False))
        manifest[name] = digest(destination/name)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    freeze = dict(stage='frozen_final', frozen_at=datetime.now(timezone.utc).isoformat(),
        training_updates=20000, checkpoint_update=winner['update'],
        training_receipt_sha256=selection['training_receipt_sha256'], files_sha256=hashes,
        model_relative_path=str(model_destination.relative_to(EVIDENCE_ROOT)),
        seed_split_sha256=split_hash, test_seeds=test_seeds, policy_label=args.label,
        policy_manifest_sha256=digest(manifest_path),
        traditional_profile=f'configs/{args.label}-traditional.yaml',
        local_act_profile=f'configs/{args.label}-local-act.yaml',
        controller=source_run['controller'], cpu_layout=source_run['cpu_layout'], record_video=True,
        selection_sha256=digest(args.selection), ranking_sha256=digest(args.ranking),
        collection_registry_sha256=digest(args.registry), freezer_sha256=digest(Path(__file__)),
        selected_validation_successes=winner['successful_rounds'], final_test_read=False,
        replacement_approved=False)
    args.output.write_text(json.dumps(freeze, indent=2)+'\n')
    print(json.dumps(dict(frozen_policy=args.label, checkpoint_update=winner['update'],
        model_manifest=str(args.output), replacement_approved=False)))


if __name__ == '__main__':
    main()
