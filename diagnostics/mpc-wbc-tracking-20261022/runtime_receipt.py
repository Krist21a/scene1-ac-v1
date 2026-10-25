"""Read actual executable/library mappings and runtime configuration, no control."""
import argparse
import hashlib
import json
from pathlib import Path
import rospy

ap=argparse.ArgumentParser()
ap.add_argument('--output',required=True)
args=ap.parse_args()
rospy.init_node('tracking_runtime_receipt',anonymous=True)
keys=['/run_id','/robot_version','/taskFile','/referenceFile','/urdfFile','/mpc/mpcArmsDof',
      '/armRealDof','/is_rl_start','/use_external_mpc','/use_shm_communication',
      '/wbc_frequency','/sensor_frequency','/only_half_up_body','/use_sim_time']
r=dict(params={k:rospy.get_param(k,None) for k in keys},processes=[],files={})
for p in Path('/proc').glob('[0-9]*'):
    try:
        argv=(p/'cmdline').read_bytes().split(b'\0')
        exe=(p/'exe').resolve()
        if exe.name not in ['humanoid_VR_hand_control','humanoid_sqp_mpc','nodelet']:continue
        if exe.name=='nodelet' and b'manager' not in argv:continue
        libs=sorted({l.split()[-1] for l in (p/'maps').read_text().splitlines() if '.so' in l and any(s in l for s in ['humanoid','mujoco'])})
        r['processes'].append(dict(pid=int(p.name),argv=[x.decode() for x in argv if x],exe=str(exe),
                                  sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                                  libraries={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in libs}))
    except FileNotFoundError:pass
for key in ['/taskFile','/referenceFile','/urdfFile']:
    p=Path(r['params'][key]);r['files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
Path(args.output).write_text(json.dumps(r,indent=2)+'\n')
