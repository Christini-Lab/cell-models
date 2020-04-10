import unittest

import protocols


class ProtocolsTest(unittest.TestCase):

    def setUp(self):
        step_one = protocols.VoltageClampStep(duration=1.0, voltage=0.02)
        step_two = protocols.VoltageClampStep(duration=2.5, voltage=-0.03)
        step_three = protocols.VoltageClampStep(duration=0.3, voltage=0.05)
        steps = [step_one, step_two, step_three]
        self.voltage_protocol = protocols.VoltageClampProtocol(steps=steps)

    # Tests for IrregularPacingProtocol
    def test_init_protocol_raises_value_error(self):
        duration = 10
        stimulation_offsets = [1.6, 0.3, 5]

        with self.assertRaises(ValueError):
            protocols.IrregularPacingProtocol(
                duration=duration,
                stimulation_offsets=stimulation_offsets)

    def test_make_offset_generator(self):
        ip = protocols.IrregularPacingProtocol(
            duration=10,
            stimulation_offsets=[1.3, 0.3, 1])

        offset_generator = ip.make_offset_generator()
        expected_tuple_from_generator = (1.3, 0.3, 1)

        self.assertEqual(tuple(offset_generator), expected_tuple_from_generator)

    # Tests for VoltageClampProtocol
    def test_init_voltage_change_endpoints(self):
        endpoints = self.voltage_protocol.get_voltage_change_endpoints()

        expected_endpoints = [1, 2.0, 4.5, 4.8]

        self.assertListEqual(endpoints, expected_endpoints)

    def test_get_voltage_at_time_returns_successfully(self):
        self.assertEqual(
            -0.03,
            self.voltage_protocol.get_voltage_at_time(time=3.3))
        self.assertEqual(
            -0.08,
            self.voltage_protocol.get_voltage_at_time(time=0.01))

    def test_get_voltage_at_time_raises_value_error(self):
        self.assertRaises(
            ValueError,
            self.voltage_protocol.get_voltage_at_time,
            4.9)

    def test_voltage_clamp_init_sets_holding_step(self):
        self.assertEqual(
            self.voltage_protocol.steps[0],
            self.voltage_protocol.HOLDING_STEP)


if __name__ == '__main__':
    unittest.main()
