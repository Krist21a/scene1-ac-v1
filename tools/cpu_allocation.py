"""Reversible, recorded CPU placement for the original 13700H workstation."""
from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
from evidence_storage import STORE, require_storage
PROFILE = 'p2-algo-p4e8-sim'
MPC_PROFILE = PROFILE+'-mpc-pcores'
CRITICAL_PROFILE = PROFILE+'-critical-pcores'
RENDER_PROFILE = PROFILE+'-render-pcores'
PERFORMANCE_PROFILE = PROFILE+'-render-performance'
WRITEBACK_PROFILE = CRITICAL_PROFILE+'-io-reserve'
PERFORMANCE_WRITEBACK_PROFILE = PERFORMANCE_PROFILE+'-io-reserve'
CRITICAL_PERFORMANCE_WRITEBACK_PROFILE = CRITICAL_PROFILE+'-performance-io-reserve'
ALGO_POWER_PROFILE = CRITICAL_PROFILE+'-algo-performance-io-reserve'
PHASE_POWER_PROFILE = CRITICAL_PROFILE+'-phase-performance-io-reserve'
VISION_PROFILE = 'p2e4-algo-p4e4-sim-critical-pcores-io-reserve'
PROFILES = (ALGO_POWER_PROFILE, PHASE_POWER_PROFILE, VISION_PROFILE, PROFILE, MPC_PROFILE, CRITICAL_PROFILE, RENDER_PROFILE, PERFORMANCE_PROFILE, WRITEBACK_PROFILE, PERFORMANCE_WRITEBACK_PROFILE, CRITICAL_PERFORMANCE_WRITEBACK_PROFILE)
LAYOUT = {'kuavo-ros-noetic-s12-ready': '0-3',
          'kuavo_challenge_container_GPU_046e4301': '4-19'}


def layout_for(profile):
    if profile == VISION_PROFILE:
        return {'kuavo-ros-noetic-s12-ready': '0-3,12-15',
                'kuavo_challenge_container_GPU_046e4301': '4-11,16-19'}
    return LAYOUT


def read_placement(container):
    return subprocess.check_output(['docker', 'inspect', '--format',
                                   '{{.HostConfig.CpusetCpus}}', container], text=True).strip()


def topology():
    base = Path('/sys/devices/system/cpu')
    if (base/'online').read_text().strip() != '0-19':
        raise RuntimeError('This layout requires the original 20-logical-CPU workstation')
    result = {}
    for n in range(20):
        p = base/f'cpu{n}'
        result[str(n)] = dict(siblings=(p/'topology/thread_siblings_list').read_text().strip(),
                             max_khz=int((p/'cpufreq/cpuinfo_max_freq').read_text()),
                             governor=(p/'cpufreq/scaling_governor').read_text().strip())
    if result['0']['siblings'] != '0-1' or result['2']['siblings'] != '2-3':
        raise RuntimeError('Performance-core topology changed')
    if min(result[str(n)]['max_khz'] for n in range(4)) <= result['12']['max_khz']:
        raise RuntimeError('Reserved CPUs are not identified as performance cores')
    return result


@contextmanager
def allocation(profile, label):
    if profile == 'original':
        yield
        return
    if profile not in PROFILES or not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', label):
        raise ValueError('Invalid CPU profile or run label')
    sys.path.insert(0, str(ROOT/'diagnostics/path-tracking-continuation-20261023'))
    import environment
    import usage_guard
    if usage_guard.latest()['reached']:
        raise RuntimeError('Quota stop before CPU allocation')
    for c in LAYOUT:
        environment.wait_idle(c)
    receipt = dict(profile=profile, requested=layout_for(profile), topology=topology(),
                   original={c: read_placement(c) for c in LAYOUT},
                   started_at=datetime.datetime.now().astimezone().isoformat(),
                   restored=False)
    output = STORE/(label+'-cpu-allocation.json')
    if output.exists():
        raise RuntimeError('CPU receipt exists; use a fresh run label')
    local = ROOT/'diagnostics/path-tracking-continuation-20261023'/(label+'-allocation-live.json') if profile in (ALGO_POWER_PROFILE, PHASE_POWER_PROFILE, VISION_PROFILE, WRITEBACK_PROFILE, PERFORMANCE_WRITEBACK_PROFILE, CRITICAL_PERFORMANCE_WRITEBACK_PROFILE) else output
    def save():
        local.write_text(json.dumps(receipt, indent=2)+'\n')
    save()
    previous = os.environ.get('SCENE1_CPU_LAYOUT')
    power_paths = sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')) if profile in (ALGO_POWER_PROFILE, PHASE_POWER_PROFILE, PERFORMANCE_PROFILE, PERFORMANCE_WRITEBACK_PROFILE, CRITICAL_PERFORMANCE_WRITEBACK_PROFILE) else []
    power_original = {}
    writeback_original = {}
    bdi_path = None
    try:
        if profile in (ALGO_POWER_PROFILE, PHASE_POWER_PROFILE, VISION_PROFILE, WRITEBACK_PROFILE, PERFORMANCE_WRITEBACK_PROFILE, CRITICAL_PERFORMANCE_WRITEBACK_PROFILE):
            for name in ('dirty_ratio', 'dirty_bytes'):
                writeback_original[name] = (Path('/proc/sys/vm')/name).read_text().strip()
            receipt['writeback_original'] = writeback_original
            device = os.stat(STORE).st_dev
            bdi_path = Path('/sys/class/bdi')/f'{os.major(device)}:{os.minor(device)}'/'min_ratio'
            if not bdi_path.exists():
                # Partitions share their parent block-device writeback domain.
                block = (Path('/sys/dev/block')/f'{os.major(device)}:{os.minor(device)}').resolve().parent
                bdi_path = Path('/sys/class/bdi')/(block/'dev').read_text().strip()/'min_ratio'
            receipt['writeback_bdi'] = dict(path=str(bdi_path), original=bdi_path.read_text().strip(), requested='80')
            save()
            # Reserve RAM for slow external evidence writeback, so the global
            # dirty limit cannot stall local control logs. Background flushing
            # stays enabled and all original settings are restored afterward.
            Path('/proc/sys/vm/dirty_ratio').write_text('40')
            bdi_path.write_text('80')
            receipt['writeback_bdi']['applied'] = bdi_path.read_text().strip()
            receipt['writeback_applied'] = {n:(Path('/proc/sys/vm')/n).read_text().strip() for n in writeback_original}
            if receipt['writeback_applied'] != {'dirty_ratio':'40','dirty_bytes':'0'}:
                raise RuntimeError('Writeback reserve did not apply')
        for path in power_paths:
            if 'performance' not in (path/'scaling_available_governors').read_text().split():
                raise RuntimeError('Stock performance governor unavailable')
            power_original[path.name] = dict(governor=(path/'scaling_governor').read_text().strip(),
                preference=(path/'energy_performance_preference').read_text().strip())
        receipt['power_original'] = power_original
        save()  # Persist recovery settings before modifying the host.
        receipt['power_initial'] = {p.name: (p/'scaling_governor').read_text().strip() for p in power_paths}
        for path in ([] if profile == PHASE_POWER_PROFILE else power_paths):
            if profile == ALGO_POWER_PROFILE and int(path.name.removeprefix('policy')) >= 4:
                continue
            (path/'scaling_governor').write_text('performance')
            if (path/'scaling_governor').read_text().strip() != 'performance':
                raise RuntimeError('Performance governor did not apply')
        if power_paths and profile != PHASE_POWER_PROFILE:
            receipt['power_applied'] = {p.name: (p/'scaling_governor').read_text().strip() for p in power_paths}
        os.environ['SCENE1_CPU_LAYOUT'] = profile
        receipt['applied'] = layout_for(profile)
        receipt['mechanism'] = 'taskset on experiment child processes; no container update'
        if profile == MPC_PROFILE:
            receipt['mechanism'] += '; MPC main thread pinned to CPUs 4-11 before policy start'
        if profile in (ALGO_POWER_PROFILE, PHASE_POWER_PROFILE, VISION_PROFILE, CRITICAL_PROFILE, WRITEBACK_PROFILE, CRITICAL_PERFORMANCE_WRITEBACK_PROFILE):
            receipt['mechanism'] += '; pre-policy sampled MPC/WBC/physics on distinct P cores'
        if profile in (RENDER_PROFILE, PERFORMANCE_PROFILE, PERFORMANCE_WRITEBACK_PROFILE):
            receipt['mechanism'] += '; pre-policy sampled MPC/WBC/physics/render on distinct P cores'
        save()
        yield
    finally:
        power_errors = []
        if writeback_original:
            receipt['writeback_at_end'] = {n:(Path('/proc/sys/vm')/n).read_text().strip() for n in writeback_original}
            try:
                if bdi_path is not None and 'writeback_bdi' in receipt:
                    bdi = receipt['writeback_bdi']
                    bdi['at_end'] = bdi_path.read_text().strip()
                    bdi_path.write_text(bdi['original'])
                    bdi['restored'] = bdi_path.read_text().strip() == bdi['original']
                    if not bdi['restored']:
                        raise RuntimeError('Device writeback restoration mismatch')
                name = 'dirty_bytes' if int(writeback_original['dirty_bytes']) else 'dirty_ratio'
                (Path('/proc/sys/vm')/name).write_text(writeback_original[name])
                receipt['writeback_restored'] = all((Path('/proc/sys/vm')/n).read_text().strip()==v for n,v in writeback_original.items())
                if not receipt['writeback_restored']:
                    raise RuntimeError('Writeback restoration mismatch')
            except Exception as error:
                power_errors.append(str(error))
        if power_original:
            receipt['power_at_end'] = {p.name: (p/'scaling_governor').read_text().strip() for p in power_paths}
            for path in power_paths:
                if path.name not in power_original:
                    continue
                original = power_original[path.name]
                try:
                    (path/'scaling_governor').write_text(original['governor'])
                    (path/'energy_performance_preference').write_text(original['preference'])
                    if ((path/'scaling_governor').read_text().strip() != original['governor'] or
                            (path/'energy_performance_preference').read_text().strip() != original['preference']):
                        raise RuntimeError('Power settings restoration mismatch')
                except Exception as error:
                    power_errors.append(str(path)+': '+str(error))
            receipt['power_restored'] = not power_errors
        if previous is None:
            os.environ.pop('SCENE1_CPU_LAYOUT', None)
        else:
            os.environ['SCENE1_CPU_LAYOUT'] = previous
        receipt['container_settings_after'] = {c: read_placement(c) for c in LAYOUT}
        receipt['restored'] = receipt['container_settings_after'] == receipt['original'] and not power_errors
        receipt['restore_errors'] = power_errors + ([] if receipt['container_settings_after'] == receipt['original'] else ['Container settings changed unexpectedly'])
        receipt['ended_at'] = datetime.datetime.now().astimezone().isoformat()
        save()
        if local != output:
            output.write_bytes(local.read_bytes())
            if output.read_bytes() != local.read_bytes():
                raise RuntimeError('Allocation receipt copy mismatch; local receipt retained')
            local.unlink()
        if not receipt['restored']:
            raise RuntimeError(('CPU settings restoration failed', receipt['restore_errors']))
