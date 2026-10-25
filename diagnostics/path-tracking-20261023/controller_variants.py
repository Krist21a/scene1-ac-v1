"""Explicit isolated controller identities; no relabeling of research variants."""
import hashlib
import json
import environment as e

CHOICES = ['baseline', 'candidate', 'velocity', 'velocity_pd', 'mpc_q']
QR_RELATIVE = 'src/humanoid-control/humanoid_controllers/config/kuavo_v52/mpc/dynamic_qr.info'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant(name):
    if name == 'baseline':
        source = e.OLD / 'baseline/controller-library/libhumanoid_controllers.so'
        expected, factor = e.BASE_SHA, 'original controller'
    elif name in ['candidate','mpc_q']:
        source = e.OLD / 'candidate-wrist-pd/libhumanoid_controllers.so'
        expected, factor = e.PD_SHA, 'C: bounded post-WBC wrist torque PD'
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
    if name=='mpc_q':
        qr_source=e.HERE/'candidate-mpc-arm-q/dynamic_qr.info'
        qr_expected=json.loads((e.HERE/'candidate-mpc-arm-q/registration.json').read_text())['candidate_sha256']
        factor='E: C plus active stance MPC proximal arm Q x2'
    assert sha(qr_source)==qr_expected
    return dict(source=source, sha256=expected, factor=factor,
                expect_pd=name in ['candidate', 'velocity_pd','mpc_q'],
                expect_velocity=name in ['velocity', 'velocity_pd'],
                qr_source=qr_source,qr_sha256=qr_expected)


def available():
    return {name: variant(name) for name in CHOICES if name in ['baseline', 'candidate','mpc_q'] or
            (e.HERE / ('candidate-velocity' if name == 'velocity' else 'candidate-velocity-pd') /
             'libhumanoid_controllers.so').exists()}
