"""Reuse previously verified process helpers without changing historical artifacts."""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent/'tools'))
from evidence_storage import STORE, ARCHIVE_ROOT, require_storage
ROOT = Path('/home/krist/kuavo_course/scene1_act_optimization')
OLD = ROOT / 'diagnostics/mpc-wbc-tracking-20261022'
ASSETS = ROOT / 'diagnostics/path-tracking-continuation-20261023'
_spec = importlib.util.spec_from_file_location('tracking_previous_experiment', OLD / 'experiment.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ALGO, SIM, PROJECT, WS = _module.ALGO, _module.SIM, _module.PROJECT, _module.WS
command, wait_idle = _module.command, _module.wait_idle


def stop(container, suffix):
    """Bound policy shutdown even inside SDK initialization; retain other cleanup."""
    name = suffix.rsplit('/', 1)[-1]
    if name not in ('run_scene1.py', 'challenge_task.py', 'target_guard.py', 'robot_guard.py', 'record_chain.py'):
        return _module.stop(container, suffix)
    code = '''import pathlib,os,signal,time,json
name=TARGET_NAME
def identity(pid):
 p=pathlib.Path('/proc')/str(pid)
 try:
  fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return None if fields[0]=='Z' else fields[19]
 except (FileNotFoundError, ProcessLookupError):return None
matches=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:args=(p/'cmdline').read_bytes().split(b'\\0')
 except (FileNotFoundError, ProcessLookupError):continue
 if any(x.decode(errors='replace').rsplit('/',1)[-1]==name for x in args):
  token=identity(int(p.name))
  if token is not None:matches.append((int(p.name),token))
assert len(matches)<=1,matches
events=[]
for pid,token in matches:
 for sig,wait in [(signal.SIGINT,2. if name=='run_scene1.py' else 10.),(signal.SIGTERM,2. if name=='run_scene1.py' else 5.),(signal.SIGKILL,2.)]:
  if identity(pid)!=token:break
  try:os.kill(pid,sig)
  except ProcessLookupError:break
  events.append(dict(pid=pid,signal=int(sig)))
  until=time.monotonic()+wait
  while identity(pid)==token and time.monotonic()<until:time.sleep(.05)
 assert identity(pid)!=token,('Process survived SIGKILL',name,pid)
print(json.dumps(dict(kind='bounded_process_stop',process=name,events=events)))
'''
    code = code.replace('TARGET_NAME', repr(name))
    result = command('docker','exec',container,'python3','-c',code)
    print(result, flush=True)
    return result
BASE_SHA = 'e46e9a800011f8c5bf43e269cb2ba0ab7d543c27e09ac10deaea6ae06e1179e4'
PD_SHA = 'ab20325136810cc785bfdb4abd80c3ecc2b2e73627ebebea42b904bd7bb3f0ba'

HOST_PROJECT = HERE / 'candidate-policy'
PROJECT = '/root/scene1_mpcwbc_candidates/reference_hold'
