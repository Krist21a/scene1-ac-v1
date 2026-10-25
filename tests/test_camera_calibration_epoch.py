import copy
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

from scene1.rgbd import CameraCalibrationEpoch, RgbdScene


def transform():
    return SimpleNamespace(header=SimpleNamespace(frame_id='head_mount'), child_frame_id='optical',
        transform=SimpleNamespace(translation=SimpleNamespace(x=.01,y=.02,z=.03),
                                  rotation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.)))


class CameraCalibrationEpochTests(unittest.TestCase):
    def test_identical_extrinsic_accepts_new_stamps(self):
        original=transform();epoch=CameraCalibrationEpoch(original, 10.)
        new=copy.deepcopy(original);new.header.stamp=999.
        epoch.check_clock(10.);epoch.check_clock(11.);epoch.check_transform(new)
        self.assertFalse(epoch.invalid)

    def test_time_rollback_latches_invalid(self):
        epoch=CameraCalibrationEpoch(transform(), 10.)
        with self.assertRaises(RuntimeError):epoch.check_clock(9.99)
        with self.assertRaises(RuntimeError):epoch.check_clock(11.)
        with self.assertRaises(RuntimeError):epoch.check_transform(transform())

    def test_changed_values_or_frames_are_not_recalibrated(self):
        for field in ('translation','frame'):
            epoch=CameraCalibrationEpoch(transform(),10.);new=transform()
            if field=='translation':new.transform.translation.x+=1e-12
            else:new.child_frame_id='other'
            with self.assertRaises(RuntimeError):epoch.check_transform(new)
            with self.assertRaises(RuntimeError):epoch.check_transform(transform())

    def test_nonfinite_clock_or_transform_is_rejected(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):CameraCalibrationEpoch(transform(),value)
            epoch=CameraCalibrationEpoch(transform(),10.)
            with self.assertRaises(RuntimeError):epoch.check_clock(value)
            new=transform();new.transform.translation.x=value
            with self.assertRaises(ValueError):CameraCalibrationEpoch(new,10.)
            with self.assertRaises(RuntimeError):CameraCalibrationEpoch(transform(),10.).check_transform(new)

    def bootstrap(self, static=None, error=None):
        scene=object.__new__(RgbdScene);scene.calibration_epoch=None
        scene.config=dict(camera_mount_frame='head_mount',camera_frame='optical')
        scene.buffer=Mock();scene.buffer.lookup_transform.return_value=transform()
        scene.listener=Mock();scene.rospy=Mock()
        scene.rospy.Time.now.return_value.to_sec.return_value=12.
        temporary=Mock();scene.rospy.Subscriber.return_value=temporary
        buffer=Mock();buffer.lookup_transform.return_value=static or transform()
        if error:buffer.lookup_transform.side_effect=error
        tf=ModuleType('tf2_ros');tf.Buffer=lambda:buffer
        msg=ModuleType('tf2_msgs.msg');msg.TFMessage=object
        return scene,temporary,buffer,{'tf2_ros':tf,'tf2_msgs.msg':msg}

    def test_bootstrap_keeps_original_buffer_and_static_subscription(self):
        scene,temp,static,modules=self.bootstrap();original=scene.buffer
        with patch.dict('sys.modules',modules):scene.finish_camera_calibration()
        self.assertIs(scene.buffer,original)
        scene.listener.tf_sub.unregister.assert_called_once()
        scene.listener.tf_static_sub.unregister.assert_not_called()
        scene.listener.unregister.assert_not_called()
        temp.unregister.assert_called_once()
        self.assertIsNotNone(scene.calibration_epoch)
        scene.finish_camera_calibration()  # An idempotent finish cannot relatch TF.
        scene.listener.tf_sub.unregister.assert_called_once()

    def test_missing_static_calibration_keeps_dynamic_listener(self):
        scene,temp,static,modules=self.bootstrap(error=RuntimeError('no static chain'))
        with patch.dict('sys.modules',modules),self.assertRaises(RuntimeError):scene.finish_camera_calibration()
        self.assertIsNone(scene.calibration_epoch)
        scene.listener.tf_sub.unregister.assert_not_called();temp.unregister.assert_called_once()

    def test_static_calibration_difference_keeps_dynamic_listener(self):
        changed=transform();changed.transform.rotation.z=1e-6
        scene,temp,static,modules=self.bootstrap(static=changed)
        with patch.dict('sys.modules',modules),self.assertRaises(RuntimeError):scene.finish_camera_calibration()
        self.assertIsNone(scene.calibration_epoch)
        scene.listener.tf_sub.unregister.assert_not_called();temp.unregister.assert_called_once()
