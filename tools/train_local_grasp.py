"""Run the installed LeTools ACT trainer with the frozen short-skill recipe.

Invoke inside the existing LeTools environment. Microbatches split each full
eight-example batch and accumulate before one optimizer/scheduler update.
No validation or final-test data is loaded by this entry point.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolchain', type=Path, default=Path('/root/LeTools-Learning'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--conversion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microbatch', type=int, choices=(1, 2, 4, 8), default=8)
    args = parser.parse_args()
    conversion = json.loads(args.conversion.read_text())
    if (conversion.get('status') != 'complete' or conversion.get('split') != 'train'
            or not conversion.get('traditional_acceptance_sha256')
            or not 1 <= conversion.get('episode_count', 0) <= 160
            or conversion.get('frame_count', 0) < 8
            or conversion.get('final_test_recordings_read') is not False):
        raise ValueError('Require a completed training-only expert conversion')
    if args.output.exists():
        raise ValueError('Training output exists; never overwrite an earlier attempt')
    if not conversion.get('metadata_sha256'):
        raise ValueError('Dataset conversion has no frozen metadata hashes')
    for name, expected_digest in conversion['metadata_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe converted metadata path')
        if hashlib.sha256((args.dataset/relative).read_bytes()).hexdigest() != expected_digest:
            raise ValueError('Converted training dataset metadata changed: '+name)
    sys.path.insert(0, str(args.toolchain))
    sys.path.insert(0, str(args.toolchain/'third_party/lerobot/src'))
    from kuavo_model.train import _load_yaml, _deep_merge, _convert_to_lerobot_train_config
    defaults = args.toolchain/'configs/train/lerobot/total/act_total.yaml'
    merged = _deep_merge(_load_yaml(defaults), _load_yaml(args.config))
    merged['dataset']['root'] = str(args.dataset.resolve())
    merged['dataset']['repo_id'] = conversion['repo_id']
    merged['training']['output_dir'] = str(args.output.resolve())
    config = _convert_to_lerobot_train_config(merged)
    expected = dict(batch_size=8, steps=20000, save_freq=2000, resume=False)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('The initial training budget and effective batch are fixed')
    if (config['policy']['type'] != 'act' or config['policy']['vision_backbone'] != 'resnet18'
            or config['policy']['chunk_size'] != 30 or config['policy']['n_action_steps'] != 1
            or config['dataset']['use_imagenet_stats'] is not False
            or config.get('use_rabc', False)):
        raise ValueError('Unexpected ACT recipe or normalization')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output.with_name(args.output.name+'-training-receipt.json')
    resolved_path = args.output.with_name(args.output.name+'-resolved-config.json')
    metrics_path = args.output.with_name(args.output.name+'-updates.jsonl')
    if any(p.exists() for p in (receipt_path, resolved_path, metrics_path)):
        raise ValueError('Training attempt receipts already exist')
    resolved_path.write_text(json.dumps(config, indent=2)+'\n')
    files = [defaults, args.config, args.conversion, Path(__file__),
             args.toolchain/'kuavo_model/train.py',
             args.toolchain/'third_party/lerobot/src/lerobot/scripts/lerobot_train.py',
             args.toolchain/'third_party/lerobot/src/lerobot/policies/act/modeling_act.py',
             args.toolchain/'third_party/lerobot/src/lerobot/policies/act/configuration_act.py']
    files.extend(sorted((args.dataset/'meta').glob('*.json')))
    receipt = dict(status='prepared', files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        dataset=str(args.dataset.resolve()), conversion_sha256=hashlib.sha256(args.conversion.read_bytes()).hexdigest(),
        microbatch=args.microbatch, effective_batch=8, updates=0, checkpoints=[],
        incomplete_batches_skipped=0, normalization_source='train dataset only',
        validation_loaded=False, final_test_loaded=False,
        batch_policy='Skip incomplete epoch-tail batches; exactly eight examples per optimizer update')

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')

    save()
    import torch
    from lerobot.scripts import lerobot_train as native
    original_cycle, original_update = native.cycle, native.update_policy
    original_save = native.save_checkpoint

    def full_batches(loader):
        for batch in original_cycle(loader):
            size = batch['action'].shape[0]
            if size != 8:
                receipt['incomplete_batches_skipped'] += 1
                continue
            yield batch

    def update(metrics, policy, batch, optimizer, grad_clip_norm, accelerator,
               lr_scheduler=None, lock=None, rabc_weights_provider=None):
        if accelerator.num_processes != 1 or batch['action'].shape[0] != 8:
            raise ValueError('This recipe requires one process and effective batch eight')
        if rabc_weights_provider is not None:
            raise ValueError('Reward weighting is outside this expert ACT recipe')
        if args.microbatch == 8:
            metrics, outputs = original_update(metrics, policy, batch, optimizer, grad_clip_norm,
                accelerator, lr_scheduler, lock, rabc_weights_provider)
        else:
            started = time.perf_counter()
            policy.train()
            optimizer.zero_grad()
            loss_total, outputs = 0., {}
            for start in range(0, 8, args.microbatch):
                part = {key:(value[start:start+args.microbatch] if torch.is_tensor(value)
                             and value.ndim and value.shape[0] == 8 else value)
                        for key, value in batch.items()}
                weight = args.microbatch/8.
                with accelerator.autocast():
                    loss, details = policy.forward(part)
                accelerator.backward(loss*weight)
                loss_total += loss.detach().item()*weight
                for key, value in details.items():
                    if isinstance(value, (float, int)):
                        outputs[key] = outputs.get(key, 0.)+value*weight
                del loss, part, details
            if grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'), error_if_nonfinite=False)
            with lock if lock is not None else nullcontext():
                optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()
            unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
            if native.has_method(unwrapped, 'update'):
                unwrapped.update()
            metrics.loss, metrics.grad_norm = loss_total, grad_norm.item()
            metrics.lr = optimizer.param_groups[0]['lr']
            metrics.update_s = time.perf_counter()-started
        if (not math.isfinite(metrics.loss.val) or not math.isfinite(metrics.grad_norm.val)
                or accelerator.optimizer_step_was_skipped):
            receipt['failed_update_attempt'] = receipt['updates']+1
            raise RuntimeError('Non-finite or skipped optimizer update; fixed training run is invalid')
        receipt['updates'] += 1
        with metrics_path.open('a') as stream:
            stream.write(json.dumps(dict(update=receipt['updates'], effective_batch=8,
                microbatch=args.microbatch, loss=metrics.loss.val, lr=metrics.lr.val,
                update_s=metrics.update_s.val, wall_time_s=time.time()))+'\n')
        return metrics, outputs

    def checkpoint(*positional, **keywords):
        original_save(*positional, **keywords)
        receipt['resolved_runtime_config'] = keywords['cfg'].to_dict()
        receipt['checkpoints'].append(dict(update=keywords['step'], path=str(keywords['checkpoint_dir'])))
        save()

    native.cycle, native.update_policy, native.save_checkpoint = full_batches, update, checkpoint
    sys.argv = [str(Path(__file__)), '--config_path='+str(resolved_path)]
    try:
        receipt['status'] = 'running'; save()
        native.main()
        if receipt['updates'] != 20000 or [c['update'] for c in receipt['checkpoints']] != list(range(2000, 20001, 2000)):
            raise RuntimeError('Training did not complete the fixed update/checkpoint budget')
        receipt['status'] = 'complete'; save()
    except Exception as exc:
        receipt.update(status='failed', error=repr(exc)); save()
        raise
    finally:
        native.cycle, native.update_policy, native.save_checkpoint = original_cycle, original_update, original_save


if __name__ == '__main__':
    main()
