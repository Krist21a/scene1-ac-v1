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
    assert len(offset)==7
    if t.get('reference_mode') in ['measured_physical','mpc_continuous']:
        assert t['correction_cap_rad'] is None
        assert all(lo<=q<=hi for q,lo,hi in zip(t['bounded_target_rad'],
                   t['physical_lower_joint_rad'],t['physical_upper_joint_rad']))
        if t['reference_mode']=='mpc_continuous':
            prepared=t.get('prepared_before_open',False)
            assert t['preserved_action_indices']==(list(range(8,15)) if prepared else [8,9,10,11])
            assert t['bounded_target_rad'][:4]==t['previous_sent_command_rad'][:4]
            assert max(map(abs,t['requested_change_rad'][:4]))==0.
            if prepared:
                assert t['bounded_target_rad']==t['previous_sent_command_rad']
                assert max(map(abs,t['requested_change_rad']))==0.
                prior=rows[r['step']-2]
                prep=prior['cartesian_alignment']['release_preparation']
                assert prep['stable_steps']>=3 and prep.get('support_checks_passed',prep['original_support_checks_passed'])
                if not prep['original_support_checks_passed']:
                    assert prep['support_position_rule']=='prepared_footprint'
                    assert prior['cartesian_alignment']['support_footprint_fit']
                if prep.get('whole_arm'):
                    motion_limit=prep.get('release_motion_limit_rad',.003)
                    assert 0. < motion_limit <= .003
                    assert prep['tracking_error_rad']<=.015 and prep['settled_movement_rad']<=motion_limit
                assert prep['sent_error_rad']<=.005 and prep['wrist_error_rad']<=.015
                assert prep['wrist_movement_rad']<=.003 and prior['sent_action'][15]>=.9
            else:
                assert max(map(abs,t['remaining_target_error_rad'][4:]))<=5e-4
        else:
            assert max(map(abs,t['remaining_target_error_rad']))<=5e-4
    else:
        assert max(map(abs,offset))<=t['correction_cap_rad']+1e-9
    assert all(math.isfinite(q) for q in t['bounded_target_rad'])
    transfers.append(dict(step=r['step'],bound_active_count=sum(t['bound_active']),
        original_cap_rad=t['correction_cap_rad'],reference_mode=t.get('reference_mode','original_act_budget'),max_offset_rad=max(map(abs,offset)),
        remaining_reference_error_rad=t['remaining_target_error_rad'],
        max_requested_change_rad=max(map(abs,t['requested_change_rad'])),
        motion_filter_velocity_limited=r['motion']['velocity_limited'],
        motion_filter_bounds_clipped=r['motion']['bounds_clipped']))
report['release_reference_transfers']=transfers
report['release_preparation']=[dict(step=r['step'],**r['cartesian_alignment']['release_preparation'])
    for r in rows if 'release_preparation' in r['cartesian_alignment']]
report['release_reference_transfer_interpretation']='Physical bounds and MotionFilter checked separately. measured_physical intentionally replaces obsolete ACT anchoring with a new visual-phase reference; this is not measured tracking evidence.'
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
    if goal.get('solver_method') == 'nominal_incremental_executor_sweep':
        assert goal['certified'] and goal['nominal_steps']<=goal['remaining_steps']
        assert all(r['proposed_sweep']['clear'] and r['filtered_sweep']['clear'] and not r['motion']['bounds_clipped'] for r in goal['nominal_records'])
    else:
        assert goal['orientation_residual_degrees']<=8.
    assert all(math.isfinite(q) and lo<=q<=hi for q,lo,hi in zip(
        goal['feasibility_joint_rad'],goal['lower_joint_rad'],goal['upper_joint_rad']))
    goals.append(dict(step=row['step'],**goal))
report['retreat_goal_certificates']=goals
projections=[]
for row in rows:
    info=row.get('cartesian_alignment',{})
    projection=info.get('retreat_projection')
    if projection is None:continue
    assert projection['applied'] and projection['margin_m']==.003
    assert projection['max_step_rad']<=.006000001
    assert projection['final']['proposed']['clear'] and projection['final']['filtered']['clear']
    assert not projection['final']['bounds_clipped']
    assert info['command_sweep']['filtered']['clear']
    projections.append(dict(step=row['step'],**projection))
report['retreat_projections']=projections
fresh_feedback=[]
for row in rows:
    sweep=row.get('cartesian_alignment',{}).get('command_sweep')
    if sweep is None:continue
    feedback=sweep.get('feedback')
    if state['policy_variant'] in ['executor_fresh','executor_fast','executor_mpc_hold']:
        assert feedback is not None
    if feedback is None:continue
    assert feedback['source']=='fresh_sdk_get_obs_after_planning'
    assert len(feedback['state'])==16 and all(math.isfinite(q) for q in feedback['state'])
    assert feedback['capture_started_monotonic_s']<=feedback['captured_monotonic_s']<=feedback['checked_monotonic_s']<=row['send_monotonic_s']
    assert sweep['filtered']['clear'] and sweep['margin_m']==.003
    fresh_feedback.append(dict(step=row['step'],
        captured_to_send_s=row['send_monotonic_s']-feedback['captured_monotonic_s'],
        max_joint_change_from_planning_rad=feedback['max_joint_change_from_planning_rad']))
report['fresh_sweep_feedback']=dict(samples=len(fresh_feedback),
    max_captured_to_send_s=max((x['captured_to_send_s'] for x in fresh_feedback),default=None),
    max_joint_change_from_planning_rad=max((x['max_joint_change_from_planning_rad'] for x in fresh_feedback),default=None),
    records=fresh_feedback)
failed_path=d/'output/failed_observation.json'
report['rejected_retreat_certificate']=(json.loads(failed_path.read_text()).get('retreat_preflight_audit')
                                      if failed_path.exists() else None)
(derived/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['label','steps','gripper_event_count','target_clips','regrasp_sweep_block_steps','regrasp_forward_gate_hold_steps','pd_limits_50hz']},ensure_ascii=False))
