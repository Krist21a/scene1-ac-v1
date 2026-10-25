"""Analyze a completed trace without retaining redundant large staging files."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import environment as e

ap=argparse.ArgumentParser();ap.add_argument('label');a=ap.parse_args()
e.wait_idle(e.ALGO);e.wait_idle(e.SIM)
d=e.HERE/a.label;derived=e.HERE/'analysis-full'/a.label;derived.mkdir(parents=True,exist_ok=True);assert json.loads((d/'experiment.json').read_text()).get('ended_at')
base='/tmp/scene1-integrated-analysis';where=base+'/'+a.label
e.command('docker','exec',e.SIM,'mkdir','-p',base+'/scene1',where)
for n in ['__init__.py','kinematics.py','motion.py']:
    e.command('docker','cp',str(e.OLD/'baseline/production/scene1'/n),e.SIM+':'+base+'/scene1/'+n)
for n in ['analyze_task_full.py','analyze_pd.py','analyze_release_events.py','analyze_chain_targets.py','analyze_reference.py']:
    e.command('docker','cp',str(e.HERE/n),e.SIM+':'+base+'/'+n)
trace='/tmp/'+a.label+'-chain.jsonl'
code='''import pathlib,hashlib,sys,json
p=pathlib.Path(sys.argv[1]);h=hashlib.sha256()
if not p.is_file():print('missing');sys.exit()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
print(h.hexdigest())
'''
h=hashlib.sha256()
with (d/'chain.jsonl').open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
expected=h.hexdigest()
found=e.command('docker','exec',e.SIM,'python3','-c',code,trace).strip()
if found=='missing':
    trace=where+'/chain.jsonl'
    e.command('docker','cp',str(d/'chain.jsonl'),e.SIM+':'+trace)
    found=e.command('docker','exec',e.SIM,'python3','-c',code,trace).strip()
assert found==expected
steps=where+'/steps.jsonl'
e.command('docker','cp',str(d/'output/steps.jsonl'),e.SIM+':'+steps)
urdf=json.loads((d/'runtime.json').read_text())['params']['/urdfFile']
out=e.command('docker','exec',e.SIM,'python3',base+'/analyze_task_full.py','--trace',trace,'--steps',steps,'--output',where+'/metrics-full.json','--project',base,'--urdf',urdf)
(derived/'analysis-full.log').write_text(out+'\n')
e.command('docker','cp',e.SIM+':'+where+'/metrics-full.json',str(derived/'metrics-full.json'))
for script,result in [('analyze_chain_targets.py','chain-targets.json'),('analyze_reference.py','reference-analysis.json')]:
    out=e.command('docker','exec',e.SIM,'python3',base+'/'+script,'--trace',trace,'--steps',steps,'--output',where+'/'+result)
    (derived/(result+'.log')).write_text(out+'\n')
    e.command('docker','cp',e.SIM+':'+where+'/'+result,str(derived/result))
out=e.command('docker','exec',e.SIM,'python3',base+'/analyze_release_events.py',
    '--steps',steps,'--output',where+'/release-transients.json','--project',base,'--urdf',urdf)
(derived/'release-analysis.log').write_text(out+'\n')
e.command('docker','cp',e.SIM+':'+where+'/release-transients.json',str(derived/'release-transients.json'))
subprocess.run(['python3',str(e.HERE/'summarize_visual.py'),str(d/'output/steps.jsonl'),'--output',str(derived/'visual-full.json')],check=True)
if json.loads((d/'experiment.json').read_text())['version'] in ['candidate','velocity_pd','mpc_q','proximal_kp','origin_pd','wrist_damping','arm_weight','pd_scale','pd_scale15','origin_mpc_q']:
    out=e.command('docker','exec',e.SIM,'python3',base+'/analyze_pd.py','--trace',trace,'--output',where+'/pd-audit.json')
    (derived/'pd-analysis.log').write_text(out+'\n')
    e.command('docker','cp',e.SIM+':'+where+'/pd-audit.json',str(derived/'pd-audit.json'))
if json.loads((d/'experiment.json').read_text())['version'] in ['velocity','velocity_pd']:
    subprocess.run(['python3',str(e.HERE/'analyze_velocity_reference.py'),'--trace',str(d/'chain.jsonl'),
                    '--steps',str(d/'output/steps.jsonl'),'--output',str(derived/'velocity-audit.json')],check=True)
# The copied input is redundant with d; leave original run /tmp trace for archive's
# separate byte-verified cleanup after its complete artifact manifest is written.
if trace.startswith(where+'/'):
    e.command('docker','exec',e.SIM,'python3','-c','import pathlib,sys;pathlib.Path(sys.argv[1]).unlink()',trace)
subprocess.run(['python3',str(e.HERE/'audit_joint_bounds.py'),a.label],check=True)
print(a.label+': analyzed with zero-truth FK and raw sensor force fields',flush=True)
