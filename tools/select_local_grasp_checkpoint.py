"""Select among the three loss-ranked ACT checkpoints using 30 full rounds.

Each candidate must retain one attempted round for each registered validation
seed. Task success takes precedence over mean time among successful rounds.
Final-test recordings and results are never read by this command.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from evidence_storage import EVIDENCE_ROOT, STORE, ARCHIVE_ROOT
from speed_acceptance import inspect

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranking', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--rounds', type=Path, required=True,
        help='Completed preregistered validate_local_grasp.py batch receipt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Checkpoint selection already exists')
    ranking = json.loads(args.ranking.read_text())
    registry = json.loads(args.registry.read_text())
    batch = json.loads(args.rounds.read_text())
    rounds = batch['candidate_rounds']
    seeds = registry['seed_splits']['validation']
    split_hash = hashlib.sha256(json.dumps(registry['seed_splits'], sort_keys=True).encode()).hexdigest()
    candidates = ranking['top_three']
    if (registry['status'] != 'complete' or registry['seed_split_sha256'] != split_hash
            or ranking['status'] != 'complete' or ranking['final_test_read'] is not False
            or ranking['requested_seeds'] != seeds or len(set(seeds)) != 10
            or len(candidates) != 3 or len({c['update'] for c in candidates}) != 3
            or set(rounds) != {str(c['update']) for c in candidates}
            or batch['status'] != 'complete' or batch['final_test_read'] is not False
            or batch['validation_seeds'] != seeds
            or batch['provenance']['ranking_sha256'] != digest(args.ranking)
            or batch['provenance']['collection_registry_sha256'] != digest(args.registry)):
        raise ValueError('Require all three nominated checkpoints and the original ten validation seeds')
    labels = [label for candidate in candidates for label in rounds[str(candidate['update'])]]
    if len(labels) != 30 or len(set(labels)) != 30:
        raise ValueError('Require thirty distinct attempted full rounds; failures cannot be replaced')
    if len(batch['runs']) != 30 or {r['label'] for r in batch['runs']} != set(labels):
        raise ValueError('Attempt outcomes differ from the preregistered validation batch')
    registered_at = datetime.fromisoformat(batch['registered_at'])
    if registered_at.tzinfo is None:
        raise ValueError('Validation registration timestamp must include a timezone')
    mount = json.loads((EVIDENCE_ROOT/'traditional-local-act-1107/training-storage-mount.json').read_text())
    container_root = Path(mount['container_target'])

    def host_path(value):
        relative = Path(value).relative_to(container_root)
        if '..' in relative.parts:
            raise ValueError('Unsafe checkpoint evidence path')
        return EVIDENCE_ROOT/relative

    import yaml
    common_profile = common_runtime = common_manifest = None
    rows = []
    for candidate in candidates:
        model = host_path(candidate['path'])
        manifest_path = host_path(candidate['deployment_manifest'])
        manifest = json.loads(manifest_path.read_text())
        if (manifest['stage'] != 'validation_candidate'
                or manifest['training_updates'] != 20000
                or manifest['checkpoint_update'] != candidate['update']
                or manifest['training_receipt_sha256'] != ranking['training_receipt_sha256']
                or manifest['files_sha256'] != candidate['files_sha256']):
            raise ValueError('Validation model provenance changed')
        for name, expected in candidate['files_sha256'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or digest(model/relative) != expected:
                raise ValueError('Validation checkpoint or preprocessing changed')
        result = dict(update=candidate['update'], normalized_l1=candidate['normalized_l1'],
            model_relative_path=str(model.relative_to(EVIDENCE_ROOT)),
            deployment_manifest_sha256=digest(manifest_path), runs=[])
        for seed, label in zip(seeds, rounds[str(candidate['update'])], strict=True):
            pipeline = json.loads((STORE/(label+'-pipeline.json')).read_text())
            if pipeline['seed'] != seed:
                raise ValueError('Validation rounds must follow the original seed order')
            experiment = json.loads((ARCHIVE_ROOT/label/'experiment.json').read_text())
            if datetime.fromisoformat(experiment['started_at']) <= registered_at:
                raise ValueError('Validation round preceded its batch registration')
            # Even unsuccessful policy attempts must have complete archival and
            # restoration before their outcome is eligible for model selection.
            if any(pipeline['stages'].get(key) != 0 for key in ('analysis', 'audit', 'archive', 'restore')):
                raise ValueError('Validation attempt has incomplete evidence or restoration: '+label)
            accepted = inspect(label, ROOT, STORE, 650.)
            recorded = next(r for r in batch['runs'] if r['label'] == label)
            if recorded.get('acceptance') != accepted or recorded['checkpoint_update'] != candidate['update']:
                raise ValueError('Validation evidence changed since the recorded attempt')
            profile = yaml.safe_load((STORE/'policies'/label/accepted['profile']).read_text())
            cycle = profile['visual_cycle']
            if (cycle.pop('grasp_execution') != 'local_act'
                    or cycle.pop('local_act_model') != candidate['path']
                    or cycle.pop('local_act_manifest') != candidate['deployment_manifest']):
                raise ValueError('Round did not use the declared validation checkpoint')
            runtime = {key:accepted[key] for key in ('runtime', 'controller', 'cpu_layout', 'record_video')}
            if common_profile is None:
                common_profile, common_runtime, common_manifest = profile, runtime, accepted['manifest']
            if profile != common_profile or runtime != common_runtime or accepted['manifest'] != common_manifest:
                raise ValueError('Validation candidates changed the frozen policy, environment or recovery rules')
            local = json.loads((ARCHIVE_ROOT/label/'output/local-act-grasps.json').read_text())
            if local['model_manifest_sha256'] != result['deployment_manifest_sha256']:
                raise ValueError('Executed model does not match its validation manifest')
            non_timing = {k:v for k,v in accepted['checks'].items()
                if k not in ('under_wall_limit', 'old_timing_under_wall_limit')}
            result['runs'].append(dict(label=label, seed=seed,
                successful=all(non_timing.values()), under_650=accepted['passed'],
                formal_wall_s=accepted['formal_wall_s'], acceptance=accepted))
        completed = [r for r in result['runs'] if r['successful']]
        result.update(successful_rounds=len(completed), success_rate=len(completed)/10,
            mean_successful_wall_s=(sum(r['formal_wall_s'] for r in completed)/len(completed)
                                    if completed else None))
        rows.append(result)
    ordered = sorted(rows, key=lambda r:(-r['successful_rounds'],
        r['mean_successful_wall_s'] if r['mean_successful_wall_s'] is not None else float('inf'),
        r['normalized_l1'], r['update']))
    winner = ordered[0]
    report = dict(status='complete', selected_update=winner['update'],
        selected_validation_successes=winner['successful_rounds'],
        selection_rule='guarded full-round success count descending, mean successful wall time ascending; loss/update break exact ties',
        candidates=ordered, seed_split_sha256=split_hash, validation_seeds=seeds,
        ranking_sha256=digest(args.ranking), collection_registry_sha256=digest(args.registry),
        rounds_manifest_sha256=digest(args.rounds), selector_sha256=digest(Path(__file__)),
        training_receipt_sha256=ranking['training_receipt_sha256'],
        policy_manifest_sha256=common_manifest, final_test_read=False,
        final_freeze_required=True, task_acceptance=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(selected_update=report['selected_update'], final_freeze_required=True)))


if __name__ == '__main__':
    main()
