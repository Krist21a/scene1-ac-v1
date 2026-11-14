"""Identify and place critical simulation threads before policy timing starts."""
import collections
import os
from pathlib import Path
import subprocess
import tempfile


def apply(container, pin_render=False, vision_workers=False):
    rows = subprocess.check_output(['docker', 'top', container, '-eo', 'pid,comm'], text=True)
    # comm can contain spaces; only PID is a separate field.
    processes = [line.split(maxsplit=1) for line in rows.splitlines()[1:] if line.strip()]
    mpc = [int(pid) for pid, name in processes if name == 'humanoid_sqp_mp']
    nodes = []
    for pid, name in processes:
        if name != 'nodelet':
            continue
        mapped = Path('/proc')/pid/'maps'
        libraries = mapped.read_text()
        if 'libhumanoid_wbc.so' in libraries and 'libmujoco.so' in libraries:
            nodes.append(int(pid))
    if len(mpc) != 1 or len(nodes) != 1:
        raise RuntimeError(('Ambiguous critical processes', mpc, nodes))
    node = nodes[0]
    with tempfile.TemporaryDirectory(prefix='scene1-affinity-') as temporary:
        profile = str(Path(temporary)/'perf.data')
        subprocess.run(['perf', 'record', '-q', '-F', '99', '-p', str(node),
                        '-o', profile, '--', 'sleep', '3'], check=True, capture_output=True)
        report = subprocess.check_output(['perf', 'report', '--stdio', '--no-children',
            '--call-graph', 'none', '-i', profile, '--sort', 'pid,dso',
            '--percent-limit', '0', '--field-separator', '|'], text=True)
    scores = {role: collections.Counter() for role in ('wbc', 'physics')}
    if pin_render:
        scores['render'] = collections.Counter()
    for line in report.splitlines():
        fields = [field.strip() for field in line.split('|')]
        if len(fields) != 3 or not fields[0].endswith('%'):
            continue
        weight = float(fields[0][:-1])
        tid = int(fields[1].split(':')[0])
        library = fields[2]
        if library.startswith(('libqpOASES.so', 'libhumanoid_wbc.so', 'libhumanoid_controllers.so')):
            scores['wbc'][tid] += weight
        if library.startswith('libmujoco.so'):
            scores['physics'][tid] += weight
        if pin_render and library.startswith(('libnvidia-glcore.so', 'libGLX_nvidia.so', 'libnvidia-eglcore.so')):
            scores['render'][tid] += weight
    roles = {'mpc': mpc[0]}
    for role, score in scores.items():
        ranked = score.most_common()
        if not ranked or ranked[0][1] < 1 or (len(ranked) > 1 and ranked[0][1] < 2*ranked[1][1]):
            raise RuntimeError(('Ambiguous critical thread profile', role, ranked))
        roles[role] = ranked[0][0]
    if len(set(roles.values())) != len(roles):
        raise RuntimeError(('Critical thread identities overlap', roles))
    node_threads = {int(p.name) for p in (Path('/proc')/str(node)/'task').iterdir()}
    if not {tid for role, tid in roles.items() if role != 'mpc'} <= node_threads:
        raise RuntimeError('Sampled threads exited before affinity placement')
    requested = {'mpc': [4, 5], 'wbc': [6, 7], 'physics': [8, 9]}
    if pin_render:
        requested['render'] = [10, 11]
    workers = [10, 11, 16, 17, 18, 19] if vision_workers else list(range(12 if pin_render else 10, 20))
    changes = []
    # Keep render/encode and MPC workers on the remaining P core and E cores.
    # No physics parameters, frequencies or container cgroups are modified.
    for pid in (mpc[0], node):
        for path in (Path('/proc')/str(pid)/'task').iterdir():
            tid = int(path.name)
            role = next((name for name, value in roles.items() if value == tid), 'worker')
            target = requested.get(role, workers)
            try:
                before = sorted(os.sched_getaffinity(tid))
                if not set(target) <= set(before):
                    raise RuntimeError(('Unexpected initial thread placement', tid, before))
                os.sched_setaffinity(tid, target)
                after = sorted(os.sched_getaffinity(tid))
            except ProcessLookupError:
                if role != 'worker':
                    raise
                continue
            if after != target:
                raise RuntimeError(('Thread affinity did not apply', tid, after, target))
            changes.append(dict(tid=tid, role=role, before=before, applied=after))
    return dict(roles=roles, requested=requested, changes=changes,
                scores={role: dict(score) for role, score in scores.items()},
                scope='Pre-policy profile; each identified critical thread on a distinct P core; process teardown restores placement')
