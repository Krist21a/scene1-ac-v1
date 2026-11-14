"""Causal 10 Hz alignment of GraspSkillRecorder evidence, without future frames."""
from bisect import bisect_right
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np


def align_recording(directory):
    """Return per-segment decisions; any missing/ambiguous frame rejects a segment.

    ROS-grid observations use the latest sample at or before the grid time.
    ROS receipt times additionally prohibit using a message that arrived after
    the grid point. Monotonic, simulation and wall clocks are retained separately;
    neither replaces the ROS source clock. No nearest-neighbor matching is used.
    """
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if (metadata['schema_version'] != 1 or metadata['requested_alignment_hz'] != 10.
            or metadata['alignment_clock'] != 'ROS source clock'):
        raise ValueError('Unsupported grasp-recording schema or clock rate')
    groups = defaultdict(list)
    for line in (directory/'events.jsonl').open():
        event = json.loads(line)
        if 'segment_id' in event:
            groups[event['segment_id']].append(event)
    decisions = []
    for segment in summary['segments']:
        decision = dict(segment_id=segment['id'], parcel_index=segment['parcel_index'],
            success=segment['success'], accepted=False, recording=str(directory),
            full_round_confirmed_parcels=summary['independently_confirmed_parcels'])
        decisions.append(decision)
        if not summary['raw_recording_valid']:
            decision['rejection'] = 'raw recording contains drops or errors'
            continue
        if not segment['success']:
            decision['rejection'] = 'trial lift was not externally confirmed'
            continue
        try:
            frames = _align_segment(groups[segment['id']], segment, metadata)
        except (ValueError, KeyError) as exc:
            decision['rejection'] = str(exc)
            continue
        decision.update(accepted=True, frames=frames, frame_count=len(frames))
    return metadata, decisions


def _align_segment(events, segment, metadata):
    streams = defaultdict(list)
    for event in events:
        key = event.get('feature') if event['kind'] == 'image' else event['kind']
        streams[key].append(event)
    for key, stream in streams.items():
        if key == 'phase':
            stream.sort(key=lambda e: e['receipt']['ros_s'])
        else:
            stream.sort(key=lambda e: e['sequence'])
    arms = streams['arm_state']
    if len(arms) < 2:
        raise ValueError('missing measured arm clock')
    # Duplicated source updates are harmless; backwards time is not.
    if any(b['sensor_sim_s'] < a['sensor_sim_s'] or b['source_ros_s'] < a['source_ros_s']
           for a, b in zip(arms, arms[1:])):
        raise ValueError('arm source/simulation clock moved backwards')
    start, end = segment['start'], segment['end']
    eligible = [a for a in arms if start['ros_s'] <= a['source_ros_s'] <= end['ros_s']
                and start['monotonic_s'] <= a['receipt']['monotonic_s'] <= end['monotonic_s']]
    if not eligible:
        raise ValueError('no state sample inside successful skill boundary')
    arm_times = [a['source_ros_s'] for a in arms]
    first = math.ceil(start['ros_s']*10.)
    last = math.floor(end['ros_s']*10.)
    if last-first < 1:
        raise ValueError('skill has fewer than two 10 Hz samples')
    cameras = tuple(metadata['image_features'])
    required = (*cameras, 'gripper_state', 'arm_command', 'gripper_command', 'phase')
    times = {}
    for key in required:
        stream = streams[key]
        if not stream:
            raise ValueError('missing stream: '+key)
        # Source stamps select observations; successful publish-return time
        # selects commands. A command's header can precede the actual publish.
        for event in stream:
            event['_time'] = (event['source_ros_s'] if key in (*cameras, 'gripper_state')
                              else event['receipt']['ros_s'])
        stream.sort(key=lambda e: (e['_time'], e['receipt']['monotonic_s']))
        times[key] = [e['_time'] for e in stream]

    frames = []
    for tick in range(first, last+1):
        anchor = tick/10.
        index = bisect_right(arm_times, anchor)-1
        if index < 0:
            raise ValueError('no causal arm sample')
        arm = arms[index]
        # Do not equate simulated ROS seconds with wall/monotonic seconds.
        # Both source and receipt must precede the same ROS grid point.
        while index >= 0 and arms[index]['receipt']['ros_s'] > anchor:
            index -= 1
        if index < 0:
            raise ValueError('no received arm sample at ROS grid point')
        arm = arms[index]
        if not 0. <= anchor-arm['source_ros_s'] <= .03:
            raise ValueError('ROS-grid state age exceeds 30 ms')
        chosen = {}
        for key in required:
            stream = streams[key]
            index = bisect_right(times[key], anchor)-1
            while index >= 0 and stream[index]['receipt']['ros_s'] > anchor:
                index -= 1
            if index < 0:
                raise ValueError('no causal sample in '+key)
            event = stream[index]
            limit = .25 if key in cameras else .05 if key == 'gripper_state' else .3
            if key != 'phase' and not 0. <= anchor-event['_time'] <= limit:
                raise ValueError('stale '+key)
            if key in ('arm_command', 'gripper_command'):
                # Publication timing is an interval, not an invented instant.
                # Reject anchors overlapping the next completed publication.
                if any(next_event['publish_before']['ros_s'] <= anchor < next_event['receipt']['ros_s']
                       for next_event in stream[index+1:index+3]):
                    raise ValueError('sensor anchor overlaps command publication')
            chosen[key] = event
        q = np.asarray(arm['values'], float)
        grip = np.asarray(chosen['gripper_state']['values'], float)
        target = np.asarray(chosen['arm_command']['values'], float)
        closure = np.asarray(chosen['gripper_command']['values'], float)
        if q.shape != (14,) or target.shape != (14,) or grip.shape != (2,) or closure.shape != (2,):
            raise ValueError('joint/gripper dimensions differ from the canonical 16-vector')
        state = np.r_[q[:7], grip[0], q[7:], grip[1]]
        action = np.r_[target[:7], closure[0], target[7:], closure[1]]
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('non-finite state or published action')
        if np.any(closure < 0.) or np.any(closure > 1.):
            raise ValueError('published gripper target outside normalized bounds')
        frames.append(dict(timestamp=(tick-first)/10., ros_grid_s=anchor,
            sensor_sim_sample_s=arm['sensor_sim_s'], source_ros_s=arm['source_ros_s'],
            receipt=arm['receipt'], phase=chosen['phase']['stage'],
            observation_state=state.astype(np.float32), action=action.astype(np.float32),
            images={key: {name:chosen[key][name] for name in
                         ('image_offset', 'image_bytes', 'source_ros_s', 'receipt', 'sequence')}
                    for key in cameras},
            source_sequences=dict(arm_state=arm['sequence'], **{
                key:event.get('sequence') for key, event in chosen.items()}),
            command_publications={key: dict(before=chosen[key]['publish_before'],
                                            after=chosen[key]['receipt'])
                                  for key in ('arm_command', 'gripper_command')}))
    return frames
