"""Convert preregistered expert short skills with the installed LeTools/LeRobot.

The collection registry must contain a completed 40/10/10 seed split, all 60
run outcomes, and the SHA256/path of a passing traditional acceptance report.
Final-test raw recordings are deliberately not convertible through this entry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def under(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Registry evidence paths must be relative and stay inside the evidence root')
    return root/relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    if registry.get('schema_version') != 1 or registry.get('status') != 'complete':
        raise ValueError('Require a complete version-1 collection registry, including failed rounds')
    splits = registry['seed_splits']
    all_seeds = []
    for split, expected in (('train', 40), ('validation', 10), ('test', 10)):
        values = splits[split]
        if len(values) != expected or any(type(seed) is not int for seed in values):
            raise ValueError('Expected preregistered 40/10/10 integer seed split')
        all_seeds.extend(values)
    if len(set(all_seeds)) != 60:
        raise ValueError('Seed splits overlap')
    held_out = set(splits['validation']+splits['test'])
    if held_out.intersection(registry['development_seeds']):
        raise ValueError('Development seeds leaked into validation/final test')
    split_digest = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    if split_digest != registry['seed_split_sha256']:
        raise ValueError('Preregistered split changed')
    receipt = registry['traditional_acceptance']
    acceptance_path = under(args.evidence_root, receipt['relative_path'])
    if digest(acceptance_path) != receipt['sha256']:
        raise ValueError('Traditional acceptance report changed')
    acceptance = json.loads(acceptance_path.read_text())
    if (acceptance.get('passed') is not True or len(acceptance.get('runs', [])) < 10
            or not all(acceptance['checks'].values())
            or not all(run['passed'] and run['formal_wall_s'] < 650. for run in acceptance['runs'])):
        raise ValueError('Expert collection requires full traditional-policy acceptance')
    runs = registry['runs']
    if len(runs) != 60 or {run['seed'] for run in runs} != set(all_seeds):
        raise ValueError('Every preregistered seed needs one outcome; failures cannot be omitted/replaced')
    if any(run.get('frozen_match') is not True for run in runs):
        raise ValueError('Every collected outcome must match the preregistered expert runtime')
    if args.output.exists():
        raise ValueError('Dataset destination exists; never overwrite a prior conversion')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from scene1.skill_alignment import align_recording

    by_seed = {run['seed']:run for run in runs}
    report = dict(status='running', split=args.split, registry_sha256=digest(args.registry),
        dataset_root=str(args.output.resolve()), repo_id=args.repo_id,
        seed_split_sha256=split_digest, traditional_acceptance_sha256=receipt['sha256'],
        requested_seeds=splits[args.split], rounds=[], episodes=[],
        alignment_clock='ROS source clock', fps=10,
        final_test_recordings_read=False, task='scene1')
    dataset = None
    expected_images = None
    names = [*[f'left_joint_{i}' for i in range(7)], 'left_gripper',
             *[f'right_joint_{i}' for i in range(7)], 'right_gripper']
    audit_path = args.output.parent/(args.output.name+'-conversion.json')
    if audit_path.exists():
        raise ValueError('Conversion receipt already exists')

    def save_report():
        audit_path.write_text(json.dumps(report, indent=2)+'\n')

    try:
        for seed in splits[args.split]:
            run = by_seed[seed]
            round_report = dict(seed=seed, label=run['label'], outcome=run['outcome'], segments=[])
            report['rounds'].append(round_report)
            if run.get('recording_relative') is None:
                round_report['rejection'] = 'no raw skill recording; failed round retained'
                save_report()
                continue
            recording = under(args.evidence_root, run['recording_relative'])
            try:
                metadata, decisions = align_recording(recording)
            except (OSError, ValueError, KeyError) as exc:
                round_report['rejection'] = str(exc)
                save_report()
                continue
            if len(decisions) > 4 or len({d['parcel_index'] for d in decisions}) != len(decisions):
                raise ValueError('Expert run contains duplicate parcel segments or more than four skills')
            if expected_images is not None and (metadata['image_features'] != expected_images
                    or list(metadata['image_features']) != list(expected_images)):
                raise ValueError('Camera order/shapes changed within the frozen expert collection')
            expected_images = metadata['image_features']
            if dataset is None:
                features = {'observation.state':dict(dtype='float32', shape=(16,), names=names),
                            'action':dict(dtype='float32', shape=(16,), names=names)}
                for key, (channels, height, width) in expected_images.items():
                    if channels != 3:
                        raise ValueError('Expected three-channel RGB')
                    features[key] = dict(dtype='video', shape=(height, width, 3),
                                         names=['height', 'width', 'channels'])
                dataset = LeRobotDataset.create(args.repo_id, fps=10, features=features,
                    root=args.output, robot_type='kuavo_scene1', image_writer_threads=2,
                    vcodec='h264', encoder_threads=2)
            with (recording/'images.bin').open('rb') as packed:
                packed_size = packed.seek(0, 2)
                for decision in decisions:
                    frames = decision.pop('frames', [])
                    round_report['segments'].append(decision)
                    if not decision['accepted']:
                        continue
                    audit = []
                    try:
                        for frame in frames:
                            sample = {'task':'scene1', 'observation.state':frame['observation_state'],
                                      'action':frame['action']}
                            for key, image in frame['images'].items():
                                offset, size = image['image_offset'], image['image_bytes']
                                if offset < 0 or size <= 0 or offset+size > packed_size:
                                    raise ValueError('compressed RGB payload outside packed image file')
                                packed.seek(offset)
                                rgb = cv2.imdecode(np.frombuffer(packed.read(size), np.uint8), cv2.IMREAD_COLOR)
                                if rgb is None:
                                    raise ValueError('invalid compressed RGB frame')
                                _, height, width = expected_images[key]
                                sample[key] = cv2.cvtColor(cv2.resize(rgb, (width, height)), cv2.COLOR_BGR2RGB)
                            dataset.add_frame(sample)
                            audit.append({key:value for key, value in frame.items()
                                          if key not in ('observation_state', 'action')})
                    except (ValueError, cv2.error) as exc:
                        dataset.clear_episode_buffer()
                        decision.update(accepted=False, rejection=str(exc))
                        continue
                    episode = dataset.num_episodes
                    dataset.save_episode(parallel_encoding=False)
                    provenance = args.output/'skill-provenance'
                    provenance.mkdir(exist_ok=True)
                    (provenance/f'episode-{episode:04d}.json').write_text(json.dumps(dict(
                        seed=seed, source_label=run['label'], parcel_index=decision['parcel_index'],
                        recording_relative=run['recording_relative'], frames=audit), indent=2)+'\n')
                    report['episodes'].append(dict(episode_index=episode, seed=seed,
                        parcel_index=decision['parcel_index'], frames=len(frames)))
            save_report()
        if dataset is None or not report['episodes']:
            raise ValueError('No successful, causally aligned expert skills')
        dataset.finalize()
        report.update(status='complete', episode_count=len(report['episodes']),
            frame_count=sum(e['frames'] for e in report['episodes']),
            metadata_sha256={str(p.relative_to(args.output)):digest(p)
                             for p in sorted((args.output/'meta').rglob('*')) if p.is_file()},
            preprocessing_rule='Training must use only the train dataset statistics; validation statistics are not normalization inputs')
        save_report()
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        save_report()
        raise
    finally:
        if dataset is not None:
            dataset.finalize()
    print(json.dumps({key:report[key] for key in ('status', 'split', 'episode_count', 'frame_count')}))


if __name__ == '__main__':
    main()
