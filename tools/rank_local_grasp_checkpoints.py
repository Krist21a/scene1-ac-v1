"""Rank ACT checkpoints on validation skills using frozen training normalization.

The three lowest deterministic, non-padding action-chunk L1 losses nominate
checkpoints for full validation rounds. This loss is not task acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-receipt', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--conversion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument('--toolchain', type=Path, default=Path('/root/LeTools-Learning'))
    args = parser.parse_args()
    training = json.loads(args.training_receipt.read_text())
    conversion = json.loads(args.conversion.read_text())
    if training.get('status') != 'complete' or training.get('updates') != 20000:
        raise ValueError('Only the completed fixed training run can be ranked')
    if (conversion.get('status') != 'complete' or conversion.get('split') != 'validation'
            or conversion.get('final_test_recordings_read') is not False):
        raise ValueError('Only the validation split can rank checkpoints')
    train_conversion_path = next((Path(name) for name, digest in training['files_sha256'].items()
        if digest == training['conversion_sha256']), None)
    if train_conversion_path is None:
        raise ValueError('Training conversion provenance is missing')
    if hashlib.sha256(train_conversion_path.read_bytes()).hexdigest() != training['conversion_sha256']:
        raise ValueError('Training conversion provenance changed')
    train_conversion = json.loads(train_conversion_path.read_text())
    if (train_conversion['seed_split_sha256'] != conversion['seed_split_sha256']
            or set(train_conversion['requested_seeds']) & set(conversion['requested_seeds'])):
        raise ValueError('Validation split must be disjoint within the same registered collection')
    if not conversion.get('metadata_sha256'):
        raise ValueError('Validation conversion metadata is not frozen')
    for name, expected in conversion['metadata_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe dataset metadata path')
        if hashlib.sha256((args.dataset/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Validation metadata changed: '+name)
    if args.output.exists():
        raise ValueError('Checkpoint ranking already exists')
    sys.path.insert(0, str(args.toolchain/'third_party/lerobot/src'))
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    report = dict(status='running', training_receipt_sha256=hashlib.sha256(args.training_receipt.read_bytes()).hexdigest(),
        conversion_sha256=hashlib.sha256(args.conversion.read_bytes()).hexdigest(),
        requested_seeds=conversion['requested_seeds'],
        metric='normalized absolute action error / non-padding scalar count; deterministic latent zero',
        normalization='saved training checkpoint processors; no validation statistics supplied',
        final_test_read=False, task_acceptance=False, checkpoints=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    save()
    try:
        for checkpoint in training['checkpoints']:
            directory = Path(checkpoint['path'])/'pretrained_model'
            policy = ACTPolicy.from_pretrained(directory).to('cuda').eval()
            policy.reset()
            if policy.config.chunk_size != 30 or policy.config.action_feature.shape != (16,):
                raise ValueError('Checkpoint no longer matches the 30 x 16 skill')
            preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=str(directory),
                preprocessor_overrides={'device_processor':{'device':'cuda'}})
            meta = LeRobotDatasetMetadata(conversion['repo_id'], root=args.dataset)
            dataset = LeRobotDataset(conversion['repo_id'], root=args.dataset,
                delta_timestamps=resolve_delta_timestamps(policy.config, meta), return_uint8=True)
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                shuffle=False, drop_last=False, num_workers=0)
            total, count, frames = 0., 0, 0
            with torch.inference_mode():
                for batch in loader:
                    for key in meta.camera_keys:
                        if batch[key].dtype == torch.uint8:
                            batch[key] = batch[key].float()/255.
                    normalized = preprocessor(batch)
                    predicted = policy.predict_action_chunk(normalized)
                    mask = (~normalized['action_is_pad']).unsqueeze(-1).expand_as(predicted)
                    error = (predicted-normalized['action']).abs()
                    if not torch.isfinite(error).all():
                        raise ValueError('Non-finite validation prediction')
                    total += error.masked_select(mask).sum().item()
                    count += mask.sum().item()
                    frames += predicted.shape[0]
            if count == 0 or frames != dataset.num_frames:
                raise ValueError('Incomplete validation coverage')
            hashes = {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(directory.rglob('*')) if p.is_file()}
            report['checkpoints'].append(dict(update=checkpoint['update'], path=str(directory),
                normalized_l1=total/count, valid_scalar_count=count, frames=frames, files_sha256=hashes))
            save()
            del policy, preprocessor, dataset, loader
            torch.cuda.empty_cache()
        report['top_three'] = sorted(report['checkpoints'], key=lambda row:(row['normalized_l1'], row['update']))[:3]
        if len(report['top_three']) != 3:
            raise ValueError('Need three distinct checkpoints for complete-round validation')
        for row in report['top_three']:
            manifest_path = Path(row['path']).parent/'local-grasp-validation-manifest.json'
            if manifest_path.exists():
                raise ValueError('Validation candidate manifest already exists')
            manifest_path.write_text(json.dumps(dict(stage='validation_candidate',
                training_updates=training['updates'], checkpoint_update=row['update'],
                training_receipt_sha256=report['training_receipt_sha256'],
                files_sha256=row['files_sha256']), indent=2)+'\n')
            row['deployment_manifest'] = str(manifest_path)
        report['status'] = 'complete'; save()
    except Exception as exc:
        report.update(status='failed', error=repr(exc)); save()
        raise


if __name__ == '__main__':
    main()
