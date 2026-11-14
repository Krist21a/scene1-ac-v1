"""One official reset, immutable outer policy, symmetric online stop rules.

Every outcome is retained, including preflight and instrumentation failures.
No strategy continuation or initialization is attempted on a held-object scene.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import tempfile

import environment as e
import usage_guard
from controller_variants import CHOICES, QR_RELATIVE, TASK_RELATIVE, VR_RELATIVE, variant


def now():
    return datetime.datetime.now().astimezone().isoformat()


def guard_events(path, cursor=0):
    with path.open('rb') as f:
        f.seek(cursor)
        data = f.read()
    end = data.rfind(b'\n')
    if end < 0:
        return [], cursor
    rows = []
    for line in data[:end+1].splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows, cursor + end + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('version', choices=CHOICES)
    ap.add_argument('label')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-steps', type=int, default=5000)
    ap.add_argument('--parcels', type=int, choices=[1,2,4], default=1)
    ap.add_argument('--policy', choices=['axis','axis_sweep','axis_feasible','executor','executor_geometry','executor_spatial','executor_release','executor_physical','executor_context','executor_hold','executor_open','executor_project','executor_obb','executor_fresh','executor_fast','executor_mpc_hold','executor_numerics','executor_prepare','executor_prepare_adapt','executor_grasp_frame','executor_grasp_gain','executor_grasp_settle','executor_support_heading','executor_support_gain','executor_pending_hold','executor_first_gain','executor_arm_prepare','executor_soft_open','executor_support_prepare','executor_insert_pitch','executor_support_recover','executor_placement_physical','executor_release_quiet','executor_first_transport','executor_first_pitch','executor_first_pickup','executor_first_select','executor_pickup_sweep','executor_first_clear','executor_first_reobserve','executor_pickup_hold_sweep','executor_first_bin','executor_bin_components','executor_bin_dense','executor_table_dense','executor_pad_dense'], default='axis_sweep')
    ap.add_argument('--speed-profile')
    ap.add_argument('--policy-root', type=Path, required=True)
    ap.add_argument('--record-video', action='store_true')
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--container-project', required=True)
    a = ap.parse_args()
    e.HOST_PROJECT = a.policy_root.resolve()
    e.PROJECT = a.container_project
    manifest_path = a.manifest.resolve()
    registration_path = e.HERE / 'registration.json'
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*',a.label):
        ap.error('label must contain only lowercase letters, digits, underscore and hyphen')
    d = e.HERE / a.label
    archive_root = e.ARCHIVE_ROOT
    e.require_storage()
    archive_root.mkdir(exist_ok=True)
    if shutil.disk_usage(archive_root).free < 5_000_000_000:
        raise RuntimeError('Insufficient evidence disk space')
    if d.exists() or d.is_symlink():
        raise RuntimeError('Experiment label already exists')
    if (archive_root/a.label).exists():
        raise RuntimeError('External experiment label already exists')
    # Native gzip keeps local staging bounded. Publish the complete run only
    # after analysis, so USB latency cannot stall initialization or guards.
    d.mkdir(exist_ok=False)
    # Read safety heartbeats from a local spool, independent of USB latency.
    live_logs = Path(tempfile.mkdtemp(prefix=a.label+'-', dir='/tmp'))
    video = None
    trace = '/tmp/' + a.label + '-chain.jsonl'
    selected = variant(a.version)
    expected = selected['sha256']
    cpu_layout = os.environ.get('SCENE1_CPU_LAYOUT', 'original')
    if cpu_layout not in ('original', 'p2-algo-p4e8-sim', 'p2-algo-p4e8-sim-mpc-pcores', 'p2-algo-p4e8-sim-critical-pcores', 'p2-algo-p4e8-sim-critical-pcores-io-reserve', 'p2e4-algo-p4e4-sim-critical-pcores-io-reserve', 'p2-algo-p4e8-sim-critical-pcores-performance-io-reserve', 'p2-algo-p4e8-sim-critical-pcores-phase-performance-io-reserve', 'p2-algo-p4e8-sim-critical-pcores-algo-performance-io-reserve', 'p2-algo-p4e8-sim-render-pcores', 'p2-algo-p4e8-sim-render-performance', 'p2-algo-p4e8-sim-render-performance-io-reserve'):
        raise ValueError('Unknown CPU layout')
    state = dict(cpu_layout=cpu_layout, version=a.version, label=a.label, seed=a.seed, status='preflight',
                 started_at=now(), task_success=None, independent_full_success=None,
                 expected_controller_sha256=expected, max_steps=a.max_steps, parcels=a.parcels, policy_root=e.PROJECT, policy_variant=a.policy,
                 speed_profile=a.speed_profile,
                 orchestrator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 registration_sha256=hashlib.sha256(registration_path.read_bytes()).hexdigest())
    state['registration_amendments'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(e.HERE.glob('full-validation-amendment-*.json'))}
    procs = {}
    before = None
    errors = []

    def save():
        # Guard-trigger state changes must never wait on the external disk.
        (live_logs/'experiment.json').write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')

    def launch(name, argv):
        if cpu_layout in ('p2-algo-p4e8-sim', 'p2-algo-p4e8-sim-mpc-pcores', 'p2-algo-p4e8-sim-critical-pcores', 'p2-algo-p4e8-sim-critical-pcores-io-reserve', 'p2e4-algo-p4e4-sim-critical-pcores-io-reserve', 'p2-algo-p4e8-sim-critical-pcores-performance-io-reserve', 'p2-algo-p4e8-sim-critical-pcores-phase-performance-io-reserve', 'p2-algo-p4e8-sim-critical-pcores-algo-performance-io-reserve', 'p2-algo-p4e8-sim-render-pcores', 'p2-algo-p4e8-sim-render-performance', 'p2-algo-p4e8-sim-render-performance-io-reserve'):
            container = e.ALGO if e.ALGO in argv else e.SIM
            position = argv.index(container)+1
            if cpu_layout == 'p2e4-algo-p4e4-sim-critical-pcores-io-reserve':
                cpus = '0-3,12-15' if container == e.ALGO else '4-11,16-19'
            else:
                cpus = '0-3' if container == e.ALGO else '4-19'
            argv = argv[:position]+['taskset', '-c', cpus]+argv[position:]
        # All live stdout must stay on the local spool. External-disk
        # backpressure must not stall policy commands or delay exit detection.
        # The finally block copies and hashes every log after process teardown.
        log_root = live_logs
        with (log_root/(name+'.log')).open('wb') as log:
            procs[name] = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        return procs[name]

    def stop_process(name, container, filename):
        p = procs.get(name)
        if p is not None and p.poll() is None:
            e.stop(container, filename)
            p.wait(timeout=20)

    save()
    source_snapshot=d/'orchestration'
    source_snapshot.mkdir()
    for filename in ['full_experiment.py','controller_variants.py','environment.py',
                     'runtime_receipt.py','robot_guard.py','target_guard.py','record_chain.py','usage_guard.py','critical_affinity.py','trace_io.py','window_video.py']:
        shutil.copy2(e.HERE/filename,source_snapshot/filename)
    shutil.copy2(manifest_path,d/'policy-manifest.json')
    shutil.copy2(selected['task_source'],d/'selected-task.info')
    shutil.copy2(selected['qr_source'],d/'selected-dynamic-qr.info')
    for filename in ('cpu_allocation.py', 'speed_trial.py', 'speed_acceptance.py', 'speed_matrix.py', 'evidence_storage.py'):
        shutil.copy2(e.HERE.parent.parent/'tools'/filename, source_snapshot/filename)
    state['orchestration_files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source_snapshot.iterdir()}
    save()
    try:
        state['usage_at_start'] = usage_guard.latest()
        if state['usage_at_start']['reached']:
            raise RuntimeError('Live quota threshold reached; no new experiment')
        e.wait_idle(e.ALGO)
        e.wait_idle(e.SIM)
        if shutil.disk_usage('/').free < 4_000_000_000:
            raise RuntimeError('Insufficient root disk reserve')
        state['cuda_preflight'] = e.command('docker','exec',e.ALGO,
            '/root/miniforge3/envs/kls_dev/bin/python','-c',
            'import torch; assert torch.cuda.is_available(); x=torch.ones(4,device="cuda"); print(torch.cuda.get_device_name(), x.sum().item())').strip()
        initial = json.loads((e.OLD/'initial-manifest.json').read_text())
        # Historical test assets were intentionally removed. Preserve the old
        # manifest and verify its unchanged runtime subset in a new receipt.
        production_runtime = {k: v for k, v in initial['production'].items()
                              if 'tests' not in Path(k).parts
                              and not Path(k).name.startswith('test_')}
        (d/'production-runtime-manifest.json').write_text(
            json.dumps(production_runtime, indent=2)+'\n')
        from verify_environment import check
        expected_official=dict(initial['baseline_official'])
        expected_official[TASK_RELATIVE]=selected['task_sha256']
        expected_official[VR_RELATIVE]=selected['vr_sha256']
        checks = dict(production_host=check(e.ROOT, production_runtime),
                      official=check(e.WS, expected_official))
        if any(x['missing'] or x['mismatch'] for x in checks.values()):
            raise RuntimeError(('Production/source/config differs from frozen baseline', checks))
        lib = e.WS/'devel/.private/humanoid_controllers/lib/libhumanoid_controllers.so'
        state['library_sha256'] = hashlib.sha256(lib.read_bytes()).hexdigest()
        if state['library_sha256'] != expected:
            raise RuntimeError('Wrong controller artifact; policy will not start')
        state['dynamic_qr_sha256']=hashlib.sha256((e.WS/QR_RELATIVE).read_bytes()).hexdigest()
        state['task_sha256']=hashlib.sha256((e.WS/TASK_RELATIVE).read_bytes()).hexdigest()
        state['vr_sha256']=hashlib.sha256((e.WS/VR_RELATIVE).read_bytes()).hexdigest()
        if state['vr_sha256']!=selected['vr_sha256']:
            raise RuntimeError('Wrong active arm-reference executable')
        if state['dynamic_qr_sha256']!=selected['qr_sha256']:
            raise RuntimeError('Wrong active dynamic QR configuration')
        # Check original production and separately frozen isolated policy before any run.
        candidate_manifest=json.loads(manifest_path.read_text())
        candidate_check=check(e.HOST_PROJECT,candidate_manifest)
        if candidate_check['missing'] or candidate_check['mismatch']:raise RuntimeError(('Candidate files differ',candidate_check))
        code = """import pathlib,json,hashlib,sys
root=pathlib.Path('/root/scene1_act_optimization');expected=json.load(sys.stdin)
bad=[k for k,v in expected.items() if not (root/k).is_file() or hashlib.sha256((root/k).read_bytes()).hexdigest()!=v]
print(json.dumps(bad));sys.exit(bool(bad))
"""
        subprocess.run(['docker','exec','-i',e.ALGO,'python3','-c',code],
                       input=json.dumps(production_runtime), text=True, check=True,
                       stdout=(d/'upper-hash-check.json').open('w'))
        candidate_code=code.replace('/root/scene1_act_optimization',e.PROJECT)
        subprocess.run(['docker','exec','-i',e.ALGO,'python3','-c',candidate_code],
                       input=json.dumps(candidate_manifest),text=True,check=True,
                       stdout=(d/'candidate-hash-check.json').open('w'))
        state['policy_manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        e.command('docker','exec',e.ALGO,'mkdir','-p',e.PROJECT+'/outputs')
        for name in ['record_chain.py','robot_guard.py','runtime_receipt.py']:
            e.command('docker','cp',str(e.HERE/name),e.SIM+':/tmp/'+name)
        for name in ['target_guard.py','clip_guard.py']:
            e.command('docker','cp',str(e.HERE/name),e.ALGO+':/tmp/'+name)
        cmd = ('cd /root/kuavo_ws; source installed/setup.zsh; source devel/setup.zsh; '
               'export LD_LIBRARY_PATH=/opt/drake/lib:$LD_LIBRARY_PATH; '
               'exec rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed %d --no-timer-gui') % a.seed
        sim = launch('simulation', ['docker','exec','-t','-e','DISPLAY=:0','-e',
                                   'XAUTHORITY=/tmp/scene1-current-display.Xauthority',e.SIM,'zsh','-lc',cmd])
        deadline = time.monotonic()+120
        while '场景实例已初始化。' not in (live_logs/'simulation.log').read_text(errors='replace'):
            if sim.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('Official scene reset failed')
            time.sleep(.5)
        setup = 'source /root/kuavo_ws/devel/setup.bash; exec python3 '
        runtime = '/tmp/'+a.label+'-runtime.json'
        e.command('docker','exec',e.SIM,'bash','-lc',setup+'/tmp/runtime_receipt.py --output '+runtime)
        e.command('docker','cp',e.SIM+':'+runtime,str(d/'runtime.json'))
        receipt = json.loads((d/'runtime.json').read_text())
        if receipt['files'][receipt['params']['/dynamicQrFile']]!=selected['qr_sha256']:
            raise RuntimeError('Runtime dynamic QR file differs from selected candidate')
        if receipt['files'][receipt['params']['/taskFile']]!=selected['task_sha256']:
            raise RuntimeError('Runtime WBC task file differs from selected candidate')
        mapped = [sha for p in receipt['processes'] for name,sha in p['libraries'].items()
                  if name.endswith('/libhumanoid_controllers.so')]
        if mapped != [expected]:
            raise RuntimeError(('Actual mapped controller differs from requested binary', mapped))
        vr_mapped=[p['sha256'] for p in receipt['processes']
                   if Path(p['exe']).name=='humanoid_VR_hand_control']
        if vr_mapped != [selected['vr_sha256']]:
            raise RuntimeError(('Actual arm-reference executable differs', vr_mapped))
        state['runtime_vr_sha256']=vr_mapped[0]
        state['runtime_controller_sha256'] = mapped[0]
        wbc_mapped = [sha for p in receipt['processes'] for name, sha in p['libraries'].items()
                      if name.endswith('/libhumanoid_wbc.so')]
        if a.version in ('wbc_warmstart','wbc_reduced'):
            registration = json.loads((e.HERE.parents[1]/('controllers/candidate-wbc-warmstart/registration.json' if a.version=='wbc_warmstart' else 'controllers/candidate-wbc-reduced/registration.json')).read_text())
            if wbc_mapped != [registration['baseline_wbc_library_sha256']]:
                raise RuntimeError('Underlying WBC task builder library changed')
        state['wbc_library_sha256'] = wbc_mapped[0] if len(wbc_mapped) == 1 else None
        if cpu_layout == 'p2-algo-p4e8-sim-mpc-pcores':
            affinity_code = '''import os,json
from pathlib import Path
matches=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  if (p/'exe').resolve().name=='humanoid_sqp_mpc':matches.append(int(p.name))
 except FileNotFoundError:pass
assert len(matches)==1,matches
pid=matches[0]
before=sorted(os.sched_getaffinity(pid))
assert set(range(4,12))<=set(before),before
os.sched_setaffinity(pid,set(range(4,12)))
applied=sorted(os.sched_getaffinity(pid))
assert applied==list(range(4,12)),applied
print(json.dumps(dict(pid=pid,before=before,applied=applied,scope='MPC main thread only')))
'''
            state['mpc_main_affinity'] = json.loads(e.command(
                'docker', 'exec', e.SIM, 'python3', '-c', affinity_code))
        if cpu_layout in ('p2-algo-p4e8-sim-critical-pcores', 'p2-algo-p4e8-sim-critical-pcores-io-reserve', 'p2e4-algo-p4e4-sim-critical-pcores-io-reserve', 'p2-algo-p4e8-sim-critical-pcores-performance-io-reserve', 'p2-algo-p4e8-sim-critical-pcores-phase-performance-io-reserve', 'p2-algo-p4e8-sim-critical-pcores-algo-performance-io-reserve', 'p2-algo-p4e8-sim-render-pcores', 'p2-algo-p4e8-sim-render-performance', 'p2-algo-p4e8-sim-render-performance-io-reserve'):
            from critical_affinity import apply
            state['critical_affinity'] = apply(e.SIM, pin_render='-render-' in cpu_layout,
                vision_workers=cpu_layout == 'p2e4-algo-p4e4-sim-critical-pcores-io-reserve')
        state['ros_run_id'] = receipt['params']['/run_id']
        state['independent_official_full_reset'] = True
        rec = launch('recorder',['docker','exec',e.SIM,'bash','-lc',
                                setup+'/tmp/record_chain.py --duration 3900 --output '+trace])
        launch('robot_guard',['docker','exec',e.SIM,'bash','-lc',setup+'/tmp/robot_guard.py'+
                              (' --expect-pd' if selected['expect_pd'] else '')+
                              (' --expect-velocity' if selected['expect_velocity'] else '')])
        before = set(e.command('docker','exec',e.ALGO,'ls',e.PROJECT+'/outputs').splitlines())
        launch('target_guard',['docker','exec',e.ALGO,'python3','/tmp/target_guard.py',
                               '--outputs',e.PROJECT+'/outputs','--before',json.dumps(sorted(before)),
                               '--threshold','5'])
        ready = set()
        deadline = time.monotonic()+15
        while ready != {'robot_guard','target_guard','recorder'}:
            for name in ['robot_guard','target_guard','recorder']:
                events,_ = guard_events(live_logs/(name+'.log'))
                if any(r.get('kind')=='trigger' for r in events):
                    raise RuntimeError(('Guard triggered before strategy', name, events[-1]))
                if any(r.get('kind')=='ready' for r in events) or (name=='recorder' and events):
                    ready.add(name)
                if procs[name].poll() is not None:
                    raise RuntimeError(('Observer exited before policy',name))
            if time.monotonic()>deadline:
                raise RuntimeError(('Observation readiness timeout',ready))
            time.sleep(.1)
        meta = json.loads((live_logs/'recorder.log').read_text().splitlines()[0])
        required = ['/sensors_data_raw','/joint_cmd','/humanoid_controller/optimizedState_wbc_mrt_origin',
                    '/humanoid_controller/optimizedInput_wbc_mrt_origin','/kuavo_arm_traj']
        if selected['expect_pd']:
            required.append('/scene1_tracking/wrist_pd')
        if a.version in ('wbc_warmstart','wbc_reduced'):
            required.append('/scene1_tracking/wbc_warmstart')
        if selected['expect_velocity']:
            required.append('/scene1_tracking/wrist_velocity_reference')
        if any(meta['topics'].get(k,{}).get('missing',True) for k in required
               if 'type' not in meta['topics'].get(k,{})):
            raise RuntimeError('Required trace topics missing')
        state['observers_ready_at'] = now()
        if a.record_video:
            from window_video import WindowVideo
            video = WindowVideo(d)
            state['video'] = video.receipt
        state['status'] = 'running'
        save()
        policy = launch('controller',['docker','exec',e.ALGO,'bash',e.PROJECT+'/run.sh',
                                      '--config','configs/visual_cycle_four_parcels.yaml','--task','go_run',
                                      '--mode','optimized','--stop-after-parcels',str(a.parcels),'--max-steps',str(a.max_steps)] +
                                      (['--speed-profile',a.speed_profile] if a.speed_profile else []))
        print(json.dumps(dict(label=a.label,status='running',run_id=state['ros_run_id'],sha256=mapped[0])),flush=True)
        cursors = dict(robot_guard=0,target_guard=0)
        last = {name:time.monotonic() for name in cursors}
        start = time.monotonic()
        usage_checked = 0.
        power_log_cursor = 0
        power_log_tail = ""
        while policy.poll() is None:
            if video is not None and video.process.poll() is not None:
                raise RuntimeError('Video recorder exited during policy')
            if (cpu_layout == 'p2-algo-p4e8-sim-critical-pcores-phase-performance-io-reserve'
                    and 'phase_power_transition' not in state):
                with (live_logs/'controller.log').open('r', errors='replace') as stream:
                    stream.seek(power_log_cursor)
                    chunk = stream.read(1024*1024)
                    power_log_cursor = stream.tell()
                power_log_tail = (power_log_tail+chunk)[-16384:]
                match = re.search(r'VISUAL_CYCLE_AFTER_REGRASP step=(\d+) parcel=1(?:\s|$)', power_log_tail)
                if match:
                    paths = sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*'))
                    transition = dict(trigger=match.group(0).strip(), step=int(match.group(1)),
                        observed_monotonic_s=time.monotonic(),
                        before={p.name:(p/'scaling_governor').read_text().strip() for p in paths})
                    state['phase_power_transition'] = transition
                    save()
                    for path in paths:
                        (path/'scaling_governor').write_text('performance')
                    transition['applied'] = {p.name:(p/'scaling_governor').read_text().strip() for p in paths}
                    transition['applied_monotonic_s'] = time.monotonic()
                    if not paths or set(transition['applied'].values()) != {'performance'}:
                        raise RuntimeError('Phase performance mode did not apply')
                    save()
            if time.monotonic()-usage_checked >= 5.:
                state['latest_usage'] = usage_guard.latest()
                usage_checked = time.monotonic()
                if state['latest_usage']['reached'] and 'stop_event' not in state:
                    state['stop_event'] = dict(source='user_usage_budget',received_at=now(),usage=state['latest_usage'])
                    state['stop_signal_sent_monotonic_s'] = time.monotonic()
                    save()
                    e.stop(e.ALGO,'/run_scene1.py')
            for name in cursors:
                events,cursors[name] = guard_events(live_logs/(name+'.log'),cursors[name])
                if events:
                    last[name] = time.monotonic()
                for event in events:
                    if event.get('kind')=='trigger' and 'stop_event' not in state:
                        state['stop_event'] = dict(source=name,received_at=now(),**event)
                        state['stop_signal_sent_monotonic_s'] = time.monotonic()
                        save()
                        e.stop(e.ALGO,'/run_scene1.py')
                if 'stop_event' not in state and (procs[name].poll() is not None or time.monotonic()-last[name]>2):
                    raise RuntimeError(('Guard process/heartbeat lost',name,
                                        procs[name].poll(),time.monotonic()-last[name]))
            if rec.poll() is not None or sim.poll() is not None:
                raise RuntimeError('Recorder or official simulation exited')
            if 'stop_event' in state and time.monotonic()-state['stop_signal_sent_monotonic_s']>20:
                raise RuntimeError('Policy did not stop after guard')
            if time.monotonic()-start > {1:1200,2:1800,4:3600}[a.parcels] or shutil.disk_usage('/').free<1_500_000_000:
                raise RuntimeError('Experiment time/disk reserve reached')
            time.sleep(.1)
        state['controller_exit_code'] = policy.returncode
        state['policy_exited_monotonic_s'] = time.monotonic()
        state['status'] = ('guard_stopped' if 'stop_event' in state else
                           'policy_finished' if policy.returncode == 0 else 'policy_error')
    except BaseException as exc:
        state['status'] = 'failed_or_interrupted'
        state['error'] = repr(exc)
        errors.append(repr(exc))
    finally:
        for name,container,filename in [('controller',e.ALGO,'run_scene1.py'),
                                        ('target_guard',e.ALGO,'target_guard.py'),
                                        ('robot_guard',e.SIM,'robot_guard.py'),
                                        ('recorder',e.SIM,'record_chain.py'),
                                        ('simulation',e.SIM,'challenge_task.py')]:
            if name == 'simulation' and video is not None:
                try: state['video'] = video.stop()
                except BaseException as exc: errors.append('finalize video: '+repr(exc))
            try:
                stop_process(name,container,filename)
            except BaseException as exc:
                errors.append('cleanup '+name+': '+repr(exc))
        # Retain partial outputs on every failure, before returning any error.
        try:
            for log in live_logs.glob('*.log'):
                shutil.copy2(log, d/log.name)
                if hashlib.sha256(log.read_bytes()).digest() != hashlib.sha256((d/log.name).read_bytes()).digest():
                    raise RuntimeError('Live log archival mismatch')
        except BaseException as exc:
            errors.append('preserve live logs: '+repr(exc))
        if before is not None:
            try:
                after = set(e.command('docker','exec',e.ALGO,'ls',e.PROJECT+'/outputs').splitlines())
                new = sorted(after-before)
                state['output_runs'] = new
                for run in new:
                    e.command('docker','cp',e.ALGO+':'+e.PROJECT+'/outputs/'+run,str(d/('output' if len(new)==1 else run)))
                if len(new)==1:
                    state['run']=new[0]
            except BaseException as exc:
                errors.append('preserve policy output: '+repr(exc))
        if 'recorder' in procs:
            try:
                from trace_io import collect
                state['trace_transport'] = collect(e.SIM, trace, d, e.command)
            except BaseException as exc:
                errors.append('preserve trace: '+repr(exc))
        for container in [e.ALGO,e.SIM]:
            try:
                e.wait_idle(container)
            except BaseException as exc:
                errors.append('idle check: '+repr(exc))
        summary_path = d/'output/summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            state['policy_summary'] = {k: summary.get(k) for k in
                ['reason','error','steps','visual_confirmed_parcels','task_success']}
            state['visual_completion_checks_passed'] = (
                summary.get('reason') == 'visual_cycle_target_reached' and
                summary.get('visual_confirmed_parcels') == a.parcels and
                state.get('controller_exit_code') == 0 and 'stop_event' not in state and not errors)
        else:
            state['visual_completion_checks_passed'] = False
        state['independent_four_parcel_visual_completion'] = a.parcels == 4 and state['visual_completion_checks_passed']
        state['errors']=errors
        state['ended_at']=now()
        save()
        shutil.copy2(live_logs/'experiment.json', d/'experiment.json')
        if hashlib.sha256((live_logs/'experiment.json').read_bytes()).digest() != hashlib.sha256((d/'experiment.json').read_bytes()).digest():
            raise RuntimeError('Experiment-state archive mismatch; local state retained')
        shutil.rmtree(live_logs)
        print(json.dumps(state,ensure_ascii=False),flush=True)
    if errors or not state.get('visual_completion_checks_passed'):
        raise SystemExit(1)


if __name__=='__main__':
    main()
