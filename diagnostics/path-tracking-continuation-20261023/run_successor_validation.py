"""Complete registered regressions before freezing fresh seeds 4 and 5."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
receipt_path = P / 'successor-validation-20261025.json'
assert not receipt_path.exists(), 'No implicit retry'
state = dict(status='running', current_trial='executor-pad-dense-ac-p2-s3-1025', stages=[])

def save():
    state['updated_at'] = datetime.datetime.now().astimezone().isoformat()
    receipt_path.write_text(json.dumps(state, indent=2) + '\n')

def run(script, args, log):
    with (P / log).open('x') as output:
        rc = subprocess.run([sys.executable, str(P/script), *args], stdout=output, stderr=subprocess.STDOUT).returncode
    state['stages'].append(dict(script=script, args=args, returncode=rc, log=log))
    save()
    assert rc == 0, (script, args, rc)

def audit_success(label, parcels):
    pipeline = json.loads((P / (label+'-pipeline.json')).read_text())
    experiment = json.loads((P/label/'experiment.json').read_text())
    assert pipeline['current_stage']=='done' and all(v['returncode']==0 for v in pipeline['stages'].values())
    assert experiment['policy_summary']['visual_confirmed_parcels']==parcels
    assert experiment['visual_completion_checks_passed']
    for script in ['audit_first_pickup.py','audit_arm_preparation.py','audit_soft_open.py','audit_support_recovery.py']:
        run(script,[label],label+'-'+script+'.log')
    run('archive_analysis_staging.py',[],label+'-staging-successor.log')

save()
try:
    label = state['current_trial']
    while True:
        pipeline = P/(label+'-pipeline.json')
        if pipeline.exists() and json.loads(pipeline.read_text())['current_stage']=='done':
            break
        time.sleep(10)
    audit_success(label,2)
    development = 'executor-pad-dense-ac-p4-s0-1025'
    regression = 'executor-pad-dense-ac-p4-s2-regression-1025'
    for label,seed in [(development,0),(regression,2)]:
        state['current_trial']=label
        save()
        run('run_one.py',['origin_pd',label,'--policy','executor_pad_dense','--seed',str(seed),'--parcels','4'],label+'-successor.log')
        audit_success(label,4)
    freeze='path-control-freeze-pad-dense-20261025.json'
    run('freeze_successor_candidate.py',['--policy','executor_pad_dense','--development',development,'--regression',regression,'--output',freeze],'freeze-pad-dense-20261025.log')
    state['phase']='frozen_acceptance'
    save()
    run('run_frozen_acceptance.py',[freeze,'--receipt','frozen-pad-acceptance-20261025.json'],'frozen-pad-acceptance-20261025.log')
    state['status']='completed'
except BaseException as error:
    state['status']='stopped'
    state['error']=repr(error)
    raise
finally:
    save()
