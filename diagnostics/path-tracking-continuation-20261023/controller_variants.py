"""Explicit isolated controller identities; no relabeling of research variants."""
import hashlib
import json
import environment as e

CHOICES = ['baseline', 'candidate', 'velocity', 'velocity_pd', 'mpc_q', 'proximal_kp', 'origin_pd', 'wrist_damping', 'arm_weight', 'pd_scale','pd_scale15','origin_mpc_q']
QR_RELATIVE = 'src/humanoid-control/humanoid_controllers/config/kuavo_v52/mpc/dynamic_qr.info'
TASK_RELATIVE = 'src/humanoid-control/humanoid_controllers/config/kuavo_v52/mpc/task.info'
VR_RELATIVE = 'devel/.private/humanoid_interface_ros/lib/humanoid_interface_ros/humanoid_VR_hand_control'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant(name):
    if name == 'baseline':
        source = e.OLD / 'baseline/controller-library/libhumanoid_controllers.so'
        expected, factor = e.BASE_SHA, 'original controller'
    elif name in ['candidate','mpc_q','proximal_kp','origin_pd','arm_weight','origin_mpc_q']:
        source = e.OLD / 'candidate-wrist-pd/libhumanoid_controllers.so'
        expected, factor = e.PD_SHA, 'C: bounded post-WBC wrist torque PD'
    elif name in ['wrist_damping','pd_scale','pd_scale15','origin_mpc_q']:
        d=e.HERE/('candidate-wrist-damping' if name=='wrist_damping' else 'candidate-pd-scale15' if name=='pd_scale15' else 'candidate-pd-scale')
        registration=json.loads((d/'registration.json').read_text())
        build=json.loads((d/'build-manifest.json').read_text())
        assert sha(d/'humanoidController.cpp')==registration['source_sha256']==build['source_sha256']
        source=d/'libhumanoid_controllers.so'
        expected=build['library_sha256']
        factor='G: C post-WBC wrist damping0.05to0.20; all bounds unchanged'
    elif name in ['velocity', 'velocity_pd']:
        d = e.HERE / ('candidate-velocity' if name == 'velocity' else 'candidate-velocity-pd')
        registration = json.loads((d / 'registration.json').read_text())
        build = json.loads((d / 'build-manifest.json').read_text())
        assert sha(d / 'humanoidController.cpp') == registration['source_sha256'] == build['source_sha256']
        assert sha(d / 'wrist_velocity_reference.h') == registration['helper_sha256']
        source = d / 'libhumanoid_controllers.so'
        expected = build['library_sha256']
        factor = 'D: normal-route wrist velocity reference' + (' plus unchanged C' if name == 'velocity_pd' else '')
    else:
        raise ValueError(name)
    assert sha(source) == expected, ('Candidate artifact hash mismatch', name)
    qr_source=e.HERE/'baseline-config/dynamic_qr.info'
    qr_expected=json.loads((e.HERE/'baseline-config/manifest.json').read_text())['sha256']
    if name in ['mpc_q','origin_mpc_q']:
        qr_source=e.HERE/'candidate-mpc-arm-q/dynamic_qr.info'
        qr_expected=json.loads((e.HERE/'candidate-mpc-arm-q/registration.json').read_text())['candidate_sha256']
        factor='E: C plus active stance MPC proximal arm Q x2'
    assert sha(qr_source)==qr_expected
    task_source=e.HERE/'baseline-task/task.info'
    task_expected=json.loads((e.HERE/'baseline-task/manifest.json').read_text())['sha256']
    if name=='proximal_kp':
        task_source=e.HERE/'candidate-proximal-kp/task.info'
        task_expected=json.loads((e.HERE/'candidate-proximal-kp/registration.json').read_text())['candidate_sha256']
        factor='F: unchanged C plus WBC proximal arm task Kp80to100; wrist gains, damping and MPC unchanged'
    if name=='arm_weight':
        task_source=e.HERE/'candidate-arm-weight/task.info'
        task_expected=json.loads((e.HERE/'candidate-arm-weight/registration.json').read_text())['candidate_sha256']
    assert sha(task_source)==task_expected
    vr_source=e.OLD/'baseline/vr-reference/humanoid_VR_hand_control'
    vr_expected='155191dc393e6c92552e57a2c4af6d52833283bc22660e7d8788ba65424ce293'
    if name in ['origin_pd','arm_weight','pd_scale','pd_scale15','origin_mpc_q']:
        vr_root=e.OLD/'candidate-reference-origin'
        build=json.loads((vr_root/'build-manifest.json').read_text())
        registration=json.loads((vr_root/'registration.json').read_text())
        assert sha(vr_root/'HumanoidVRHandCommandNode.cpp')==build['source_sha256']==registration['candidate_source_sha256']
        vr_source=vr_root/'humanoid_VR_hand_control'
        vr_expected=build['executable_sha256']
        factor='A+C: correct arm first-knot indexing before MPC plus unchanged bounded post-WBC wrist PD'
    if name=='arm_weight':
        factor='H: unchanged A+C plus WBC accArm task weight9to18; gains, bounds and MPC unchanged'
    if name=='pd_scale':
        factor='I: A plus C post-WBC PD amplitude x2; original added and total torque limits'
    if name=='pd_scale15':
        factor='J: A plus C post-WBC PD amplitude x1.5; original torque limits'
    if name=='origin_mpc_q':
        factor='K: unchanged A+C plus stance MPC proximal arm Q200to400; first-knot indexing corrected'
    assert sha(vr_source)==vr_expected
    return dict(source=source, sha256=expected, factor=factor,
                expect_pd=name in ['candidate', 'velocity_pd','mpc_q','proximal_kp','origin_pd','wrist_damping','arm_weight','pd_scale','pd_scale15','origin_mpc_q'],
                expect_velocity=name in ['velocity', 'velocity_pd'],
                qr_source=qr_source,qr_sha256=qr_expected,
                task_source=task_source,task_sha256=task_expected,
                vr_source=vr_source,vr_sha256=vr_expected)


def available():
    return {name: variant(name) for name in CHOICES if name in ['baseline', 'candidate','mpc_q','proximal_kp','origin_pd','wrist_damping','arm_weight','pd_scale','pd_scale15','origin_mpc_q'] or
            (e.HERE / ('candidate-velocity' if name == 'velocity' else 'candidate-velocity-pd') /
             'libhumanoid_controllers.so').exists()}
