"""Independent release gate audit, including all seven joints when enabled."""
import json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
label=sys.argv[1];rows=[json.loads(s) for s in (root/label/'output/steps.jsonl').read_text().splitlines()]
blocks=[];active=[]
for row in rows:
    prep=row['cartesian_alignment'].get('release_preparation')
    if prep and prep.get('whole_arm'):
        assert row['sent_action'][15]>=.9
        assert not row['motion']['bounds_clipped']
        active.append(dict(step=row['step'],phase_step=row['cartesian_alignment'].get('phase_steps'),**prep))
    elif active:
        transfer=row['cartesian_alignment'].get('release_reference_transfer')
        if transfer:
            last=active[-1]
            assert last['stable_steps']>=3 and last['tracking_error_rad']<=.015
            motion_limit=last.get('release_motion_limit_rad',.003)
            assert 0. < motion_limit <= .003
            assert last['settled_movement_rad']<=motion_limit and last.get('support_checks_passed',last['original_support_checks_passed'])
            if not last['original_support_checks_passed']:
                assert last['support_position_rule']=='prepared_footprint'
                assert rows[last['step']-1]['cartesian_alignment']['support_footprint_fit']
            assert max(map(abs,transfer['requested_change_rad']))==0.
        blocks.append(dict(preparation_steps=len(active),first=active[0],last=active[-1],released=bool(transfer)))
        active=[]
if active:blocks.append(dict(preparation_steps=len(active),first=active[0],last=active[-1],released=False))
out=dict(label=label,blocks=blocks,all_checks_passed=True)
target=root/'analysis-full'/label/'arm-preparation-audit.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
