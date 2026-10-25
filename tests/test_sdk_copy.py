import copy
from types import SimpleNamespace
import unittest
from scene1.sdk_copy import numeric_deepcopy, SensorNumericCopies


class SensorCopyTests(unittest.TestCase):
    def test_flat_numeric_tuple_keeps_deepcopy_identity(self):
        value = (1., -3., 0, True)
        self.assertIs(copy.deepcopy(value), value)
        self.assertIs(numeric_deepcopy(value), value)

    def test_list_retains_type_and_independent_storage(self):
        value = [1., 2.]
        result = numeric_deepcopy(value)
        self.assertEqual(result, copy.deepcopy(value))
        self.assertIs(type(result), list)
        value[0] = 9.
        self.assertEqual(result, [1., 2.])

    def test_nested_mutable_values_and_memo_follow_deepcopy(self):
        inner = [1.]
        value = (inner, inner)
        result = numeric_deepcopy(value)
        self.assertIs(result[0], result[1])
        self.assertIsNot(result[0], inner)
        memo = {id(inner): 'existing'}
        self.assertEqual(numeric_deepcopy(value, memo), ('existing', 'existing'))

    def test_only_callback_copy_binding_changes_and_close_restores(self):
        class State:
            def _sensors_data_raw_callback(self, msg):
                self.q = copy.deepcopy(msg.q)
                self.nested = copy.deepcopy(msg.nested)
            _sensors_data_raw_shm_callback = _sensors_data_raw_callback
        original = State._sensors_data_raw_callback
        stdlib_copy = copy.deepcopy
        patcher = SensorNumericCopies(State)
        try:
            self.assertIs(State._sensors_data_raw_callback.__code__, original.__code__)
            self.assertIs(copy.deepcopy, stdlib_copy)
            for name in patcher.names:
                state = State()
                msg = SimpleNamespace(q=[1., 2.], nested=[[3.]])
                getattr(state, name)(msg)
                msg.q[0] = 9.; msg.nested[0][0] = 9.
                self.assertEqual(state.q, [1., 2.])
                self.assertEqual(state.nested, [[3.]])
        finally:
            patcher.close()
        self.assertIs(State._sensors_data_raw_callback, original)

    def test_active_subscriptions_reject_late_patch(self):
        class State:
            _instance = SimpleNamespace(_initialized=True)
        with self.assertRaisesRegex(RuntimeError, 'precede SDK subscriptions'):
            SensorNumericCopies(State)
