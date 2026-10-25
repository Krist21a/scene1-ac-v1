"""One official reset per invocation; single strategy; preserve every outcome."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from clip_guard import ClipGuard, READ_NEW_STEPS

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SIM = 'kuavo_challenge_container_GPU_046e4301'
ALGO = 'kuavo-ros-noetic-s12-ready'
PROJECT = '/root/scene1_act_optimization'
WS = ROOT.parent/'workspace/leju-kuavo-challenge-cup-2026'


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def processes(container):
    return command('docker','exec',container,'ps','-eo','pid,args').splitlines()[1:]


def wait_idle(container):
    until=time.monotonic()+60
    while True:
        active=[l for l in processes(container) if not (l.strip().endswith(' zsh') or l.strip().endswith(' /bin/bash') or l.strip().endswith(' ps -eo pid,args'))]
        if not active:
            return
        if time.monotonic()>until:
            raise RuntimeError(('Container not idle',container,active))
        time.sleep(.5)


def stop(container, suffix):
    code="""import pathlib,os,signal
suffix=%r
matches=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:a=(p/'cmdline').read_bytes().split(b'\\0')
 except:continue
 if any(x.decode(errors='replace').rsplit('/',1)[-1] == suffix.rsplit('/',1)[-1] for x in a):matches.append(int(p.name))
assert len(matches)<=1,matches
for pid in matches:os.kill(pid,signal.SIGINT)
""" % suffix
    command('docker','exec',container,'python3','-c',code)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('kind',choices=['empty','task'])
    ap.add_argument('label')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--parcels',type=int,default=1)
    ap.add_argument('--max-steps',type=int,default=5000)
    ap.add_argument('--expected-controller-sha256')
    ap.add_argument('--clip-stop-steps',type=int,default=0,
                    help='For future task trials: SIGINT after N consecutive logged target clips; polling delay up to 2s. Zero retains historical manual-guard behavior.')
    args=ap.parse_args()
    d=HERE/args.label
    d.mkdir(exist_ok=False)
    state=dict(kind=args.kind,label=args.label,seed=args.seed,status='precheck',
               task_success=None, independent_full_success=False,
               started_at=datetime.datetime.now().astimezone().isoformat())
    def save():
        (d/'experiment.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    wait_idle(ALGO);wait_idle(SIM)
    assert shutil.disk_usage('/').free>2_000_000_000
    task=WS/'src/humanoid-control/humanoid_controllers/config/kuavo_v52/mpc/task.info'
    source=WS/'src/humanoid-control/humanoid_controllers/src/humanoidController.cpp'
    lib=WS/'devel/.private/humanoid_controllers/lib/libhumanoid_controllers.so'
    vr_source=WS/'src/humanoid-control/humanoid_interface_ros/src/HumanoidVRHandCommandNode.cpp'
    vr_exe=WS/'devel/.private/humanoid_interface_ros/lib/humanoid_interface_ros/humanoid_VR_hand_control'
    state['artifacts']={str(p.relative_to(WS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [task,source,lib,vr_source,vr_exe]}
    if args.expected_controller_sha256:
        actual=state['artifacts'][str(lib.relative_to(WS))]
        if actual!=args.expected_controller_sha256:
            state['status']='rejected_artifact_hash'
            state['expected_controller_sha256']=args.expected_controller_sha256
            save()
            raise RuntimeError(('Controller library differs from requested experiment',actual,args.expected_controller_sha256))
    state['outer_config_sha256']=hashlib.sha256((ROOT/'configs/visual_cycle_four_parcels.yaml').read_bytes()).hexdigest()
    for script in ['record_chain.py','empty_probe.py','analyze_empty.py','runtime_receipt.py']:
        command('docker','cp',str(HERE/script),SIM+':/tmp/'+script)
    sim=rec=controller=None
    try:
        sim_cmd='cd /root/kuavo_ws; source installed/setup.zsh; source devel/setup.zsh; export LD_LIBRARY_PATH=/opt/drake/lib:$LD_LIBRARY_PATH; exec rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed %d --no-timer-gui'%args.seed
        sim=subprocess.Popen(['docker','exec','-t','-e','DISPLAY=:0','-e','XAUTHORITY=/tmp/scene1-current-display.Xauthority',SIM,'zsh','-lc',sim_cmd],stdout=(d/'simulation.log').open('wb'),stderr=subprocess.STDOUT)
        deadline=time.monotonic()+120
        while '场景实例已初始化。' not in (d/'simulation.log').read_text(errors='replace'):
            if sim.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('Official reset did not initialize')
            time.sleep(1)
        command('docker','exec',SIM,'bash','-lc','source /root/kuavo_ws/devel/setup.bash; python3 /tmp/runtime_receipt.py --output /tmp/'+args.label+'-runtime.json')
        command('docker','cp',SIM+':/tmp/'+args.label+'-runtime.json',str(d/'runtime.json'))
        state['status']='recording';save()
        trace='/tmp/'+args.label+'-chain.jsonl'
        probe='/tmp/'+args.label+'-probe.jsonl'
        setup='source /root/kuavo_ws/devel/setup.bash; exec python3 '
        rec=subprocess.Popen(['docker','exec',SIM,'bash','-lc',setup+'/tmp/record_chain.py --duration %d --output %s'%(80 if args.kind=='empty' else 7200,trace)],stdout=(d/'recorder.log').open('wb'),stderr=subprocess.STDOUT)
        deadline=time.monotonic()+15
        while not (d/'recorder.log').stat().st_size:
            if rec.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Recorder startup failed')
            time.sleep(.2)
        meta=json.loads((d/'recorder.log').read_text().splitlines()[0]);state['ros_run_id']=meta['run_id']
        state['status']='running';save();print(json.dumps(state),flush=True)
        if args.kind=='empty':
            controller=subprocess.Popen(['docker','exec',SIM,'bash','-lc',setup+'/tmp/empty_probe.py --output '+probe],stdout=(d/'controller.log').open('wb'),stderr=subprocess.STDOUT)
            rc=controller.wait(timeout=70)
            state['controller_exit_code']=rc
            if rc:raise RuntimeError('Probe failed')
            time.sleep(1)
        else:
            before=set(command('docker','exec',ALGO,'ls',PROJECT+'/outputs').splitlines())
            controller=subprocess.Popen(['docker','exec',ALGO,'bash',PROJECT+'/run.sh','--config','configs/visual_cycle_four_parcels.yaml','--task','go_run','--mode','optimized','--stop-after-parcels',str(args.parcels),'--max-steps',str(args.max_steps)],stdout=(d/'controller.log').open('wb'),stderr=subprocess.STDOUT)
            guard=ClipGuard(args.clip_stop_steps) if args.clip_stop_steps else None
            guard_run=None;guard_offset=0
            while controller.poll() is None:
                if shutil.disk_usage('/').free<1_500_000_000:raise RuntimeError('Root disk reserve reached')
                if guard and 'clip_guard_stop' not in state:
                    if guard_run is None:
                        new=set(command('docker','exec',ALGO,'ls',PROJECT+'/outputs').splitlines())-before
                        if len(new)>1:raise RuntimeError(('Ambiguous strategy output',new))
                        if new:guard_run=new.pop()
                    if guard_run:
                        chunk=json.loads(command('docker','exec',ALGO,'python3','-c',READ_NEW_STEPS,
                                                 PROJECT+'/outputs/'+guard_run+'/steps.jsonl',str(guard_offset)))
                        guard_offset=chunk['offset'];trigger=guard.consume(chunk['rows'])
                        if trigger:
                            state['clip_guard_stop']=dict(**trigger,at=datetime.datetime.now().astimezone().isoformat())
                            save();stop(ALGO,'/run_scene1.py')
                time.sleep(2)
            state['controller_exit_code']=controller.returncode
            after=set(command('docker','exec',ALGO,'ls',PROJECT+'/outputs').splitlines())
            new=after-before
            assert len(new)==1,new
            state['run']=new.pop()
            command('docker','cp',ALGO+':'+PROJECT+'/outputs/'+state['run'],str(d/'output'))
            # Never delete original outputs automatically.
        stop(SIM,'/record_chain.py');rec.wait(timeout=15);rec=None
        command('docker','cp',SIM+':'+trace,str(d/'chain.jsonl'))
        if args.kind=='empty':
            command('docker','cp',SIM+':'+probe,str(d/'probe.jsonl'))
            metrics='/tmp/'+args.label+'-metrics.json'
            text=command('docker','exec',SIM,'python3','/tmp/analyze_empty.py','--trace',trace,'--probe',probe,'--output',metrics)
            (d/'analysis.log').write_text(text+'\n')
            command('docker','cp',SIM+':'+metrics,str(d/'metrics.json'))
        state['status']='completed'
    except BaseException as exc:
        state['status']='failed_or_interrupted';state['error']=repr(exc)
        raise
    finally:
        if controller is not None and controller.poll() is None:stop(SIM if args.kind=='empty' else ALGO,'/empty_probe.py' if args.kind=='empty' else '/run_scene1.py')
        if rec is not None and rec.poll() is None:stop(SIM,'/record_chain.py')
        if sim is not None and sim.poll() is None:stop(SIM,'/challenge_task.py')
        wait_idle(SIM);wait_idle(ALGO)
        state['ended_at']=datetime.datetime.now().astimezone().isoformat();save()
        print(json.dumps(state),flush=True)


if __name__=='__main__':
    main()
