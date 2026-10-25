"""One registered extended empty probe, analysis and archival; no retries."""
import argparse
import datetime
import json
import subprocess
import sys
import environment as e
from controller_variants import CHOICES, variant

ap=argparse.ArgumentParser()
ap.add_argument('version',choices=CHOICES)
ap.add_argument('label')
a=ap.parse_args()
destination=e.HERE/(a.label+'-pipeline.json')
if destination.exists() or (e.HERE/a.label).exists():
    raise RuntimeError('Label already used; failed attempts are never retried implicitly')
selected=variant(a.version)
receipt=dict(label=a.label,version=a.version,kind='empty',stages={})
def run(stage,script,arguments):
    receipt['current_stage']=stage
    destination.write_text(json.dumps(receipt,indent=2)+'\n')
    with (e.HERE/(a.label+'-'+stage+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,str(e.HERE/script),*arguments],stdout=log,stderr=subprocess.STDOUT)
    receipt['stages'][stage]=dict(returncode=result.returncode,ended_at=datetime.datetime.now().astimezone().isoformat())
    destination.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(label=a.label,stage=stage,returncode=result.returncode)),flush=True)
    return result.returncode
if run('switch','switch_wrist_pd.py',[a.version]):
    raise SystemExit(1)
run('driver','empty_experiment.py',['empty',a.label,'--seed','0','--version',a.version,
    '--expected-controller-sha256',selected['sha256'],'--extended-proximal'])
d=e.HERE/a.label
if (d/'chain.jsonl').exists():
    # Only file analysis after the driver has shut down the official reset.
    e.wait_idle(e.ALGO);e.wait_idle(e.SIM)
    for script,out in [('analyze_reference.py','reference-analysis.json'),('analyze_pd.py','pd-audit.json')]:
        if script=='analyze_pd.py' and not selected['expect_pd']:continue
        e.command('docker','cp',str(e.HERE/script),e.SIM+':/tmp/'+script)
        remote='/tmp/'+a.label+'-'+out
        result=subprocess.run(['docker','exec',e.SIM,'python3','/tmp/'+script,
            '--trace','/tmp/'+a.label+'-chain.jsonl','--output',remote],capture_output=True,text=True)
        (d/(out+'.log')).write_text(result.stdout+result.stderr)
        receipt['stages'][script]=dict(returncode=result.returncode)
        if result.returncode==0:e.command('docker','cp',e.SIM+':'+remote,str(d/out))
    run('archive','archive_empty.py',[a.label])
run('summary','summarize_experiments.py',[])
receipt['current_stage']='done'
destination.write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(any(x['returncode'] for x in receipt['stages'].values()))
