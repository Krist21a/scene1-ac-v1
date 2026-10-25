"""Task checks, original fields and formal-policy versus initialization PD limits."""
import argparse,collections,json,math
from pathlib import Path
import environment as e
ap=argparse.ArgumentParser();ap.add_argument('label');a=ap.parse_args();d=e.HERE/a.label
derived=e.HERE/'analysis-full'/a.label;derived.mkdir(parents=True,exist_ok=True)
rows=[json.loads(l) for l in (d/'output/steps.jsonl').open()];summary=json.loads((d/'output/summary.json').read_text());state=json.loads((d/'experiment.json').read_text())
start=rows[0]['send_monotonic_s'];end=rows[-1]['send_monotonic_s']+rows[-1]['step_s']
grip=[rows[0]['measured_state'][j]>.6 for j in [7,15]];events=[];bins=[];completes=[];first_cycle_end=None;act_after_first=0
for r in rows:
    info=r['cartesian_alignment']
    for s,j in enumerate([7,15]):
        val=r['sent_action'][j];new=grip[s] if .25<val<.6 else val>=.6
        if new!=grip[s]:events.append(dict(step=r['step'],side=['left','right'][s],action='close' if new else 'open',phase=info['state']));grip[s]=new
    if info.get('bin_visual_confirmed'):
        bins.append(dict(step=r['step'],slot=info.get('bin_verification_slot'),new_points=info.get('new_bin_points'),stamp=r.get('rgbd_stamp')))
    if info['state']=='visual_cycle_complete':
        completes.append(dict(step=r['step'],completed_parcels=info['completed_parcels']))
        if first_cycle_end is None:first_cycle_end=r['step']
    if first_cycle_end and r['step']>first_cycle_end and not r['policy_paused_for_alignment']:act_after_first+=1
pd={k:dict(samples=0,raw_clipped_samples=0,total_clipped_samples=0,raw_peak_nm=0.,added_peak_nm=0.,wrist_command_peak_nm=0.,longest_observed_clip_s=0.,previous_clip_start=None) for k in ['initialization','formal_policy','after_policy']}
wrist=[4,5,6,11,12,13]
for line in (d/'chain.jsonl').open():
    r=json.loads(line)
    if r.get('topic')!='/scene1_tracking/wrist_pd':continue
    t=r['callback_monotonic_ns']/1e9;x=r['message']['data']
    key='initialization' if t<start else 'formal_policy' if t<=end else 'after_policy'
    p=pd[key];p['samples']+=1
    if x[56]<.5:continue
    raw=max(abs(x[14+j]) for j in wrist);added=max(abs(x[28+j]) for j in wrist);command=max(abs(x[42+j]) for j in wrist)
    clipped=raw>.5+1e-9;total=any(abs(x[j]+max(-.5,min(.5,x[14+j])))>14.1+1e-9 for j in wrist)
    p['raw_clipped_samples']+=clipped;p['total_clipped_samples']+=total
    p['raw_peak_nm']=max(p['raw_peak_nm'],raw);p['added_peak_nm']=max(p['added_peak_nm'],added);p['wrist_command_peak_nm']=max(p['wrist_command_peak_nm'],command)
    if clipped or total:
        if p['previous_clip_start'] is None:p['previous_clip_start']=t
        p['longest_observed_clip_s']=max(p['longest_observed_clip_s'],t-p['previous_clip_start'])
    else:p['previous_clip_start']=None
for p in pd.values():
    p.pop('previous_clip_start')
    p['available']=p['samples']>0
    if not p['available']:
        for k in list(p):
            if k not in ['samples','available']:p[k]=None
report=dict(label=a.label,steps=len(rows),step_sequence_contiguous=[r['step'] for r in rows]==list(range(1,len(rows)+1)),summary=summary,ros_run_id=state['ros_run_id'],controller_sha=state['runtime_controller_sha256'],policy_sha=state['policy_manifest_sha256'],requested_parcels=state['parcels'],bin_confirmations=bins,complete_events=completes,gripper_events=events,gripper_event_count=len(events),unpaused_ACT_steps_after_first_complete=act_after_first,target_clips=sum(bool(r['motion']['bounds_clipped']) for r in rows),velocity_limited_steps=sum(bool(r['motion']['velocity_limited']) for r in rows),regrasp_sweep_block_steps=sum(r['cartesian_alignment'].get('open_fingers_clear') is False for r in rows),regrasp_forward_gate_hold_steps=sum('regrasp_forward_gate' in r['cartesian_alignment'] and not r['cartesian_alignment']['regrasp_forward_gate']['centered'] for r in rows),pd_limits_50hz=pd,task_success=summary['task_success'],limitations='Gripper events are upper command threshold crossings (0.25/0.6 hysteresis), not proof of contact. Original visual checks and raw feedback evidence retained. 50Hz clipping can miss shorter events; online guard is separate. null is not an official verdict.')
transfers=[]
for r in rows:
    t=r['cartesian_alignment'].get('release_reference_transfer')
    if t is None:
        continue
    offset=[q-b for q,b in zip(t['bounded_target_rad'],t['original_reference_rad'])]
    assert len(offset)==7 and max(map(abs,offset))<=t['correction_cap_rad']+1e-9
    assert all(math.isfinite(q) for q in t['bounded_target_rad'])
    transfers.append(dict(step=r['step'],bound_active_count=sum(t['bound_active']),
        original_cap_rad=t['correction_cap_rad'],max_offset_rad=max(map(abs,offset)),
        remaining_reference_error_rad=t['remaining_target_error_rad'],
        max_requested_change_rad=max(map(abs,t['requested_change_rad'])),
        motion_filter_velocity_limited=r['motion']['velocity_limited'],
        motion_filter_bounds_clipped=r['motion']['bounds_clipped']))
report['release_reference_transfers']=transfers
report['release_reference_transfer_interpretation']='Single-event bounded target construction, separately reported from MotionFilter clipping. Does not establish measured tracking or reset correction budgets.'
refreshes=[]
for r in rows:
    info=r['cartesian_alignment']
    f=info.get('post_level_geometry_refresh')
    if f is None:
        continue
    support=info['post_level_settlement']
    assert f['support_reconfirmed'] and support['confirmed']
    refreshes.append(dict(step=r['step'],xy_shift_m=f['xy_shift_m'],
        total_phase_steps=f['total_phase_steps'],support=support,
        gripper_command=r['sent_action'][15]))
report['post_level_geometry_refreshes']=refreshes
goals=[];seen=set()
for row in rows:
    goal=row.get('cartesian_alignment',{}).get('retreat_goal_check')
    if not goal:continue
    key=(goal['observation_stamp'],tuple(goal['goal_base_m']))
    if key in seen:continue
    seen.add(key)
    assert goal['position_residual_m']<=.014
    assert goal['orientation_residual_degrees']<=8.
    assert all(math.isfinite(q) and lo<=q<=hi for q,lo,hi in zip(
        goal['feasibility_joint_rad'],goal['lower_joint_rad'],goal['upper_joint_rad']))
    goals.append(dict(step=row['step'],**goal))
report['retreat_goal_certificates']=goals
failed_path=d/'output/failed_observation.json'
report['rejected_retreat_certificate']=(json.loads(failed_path.read_text()).get('retreat_preflight_audit')
                                      if failed_path.exists() else None)
(derived/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['label','steps','gripper_event_count','target_clips','regrasp_sweep_block_steps','regrasp_forward_gate_hold_steps','pd_limits_50hz']},ensure_ascii=False))
