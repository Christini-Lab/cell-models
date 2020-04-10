import unittest

import pandas as pd

import ga_configs
import trace


class TraceFake:

    def __init__(self, t, y):
        self.t = t
        self.y = y


class TraceTest(unittest.TestCase):

    # Tests for IrregularPacingInfo
    def test_add_apd90(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.apd_90_end_voltage = 10
        pacing_info.add_apd_90(apd_90=1.5)

        self.assertEqual(pacing_info.apd_90_end_voltage, -1)

    def test_should_stimulate_returns_false(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.stimulations = [1.1, 1.3005, 1.7]

        test_times = [1.2, 1.0999, 1.8]
        for i in test_times:
            with self.subTest():
                self.assertFalse(pacing_info.should_stimulate(t=i))

    def test_should_stimulate_returns_true(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.stimulations = [1.1, 1.3005, 1.7]

        test_times = [1.10001, 1.3006, 1.7003]
        for i in test_times:
            with self.subTest():
                self.assertTrue(pacing_info.should_stimulate(t=i))

    def test_detect_peak_returns_false_too_close_to_past_peak(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.peaks.append(0.3)

        detected_peak = pacing_info.detect_peak(
            t=[.1, .2, .3],
            y_voltage=0.03,
            d_y_voltage=[1.5, -1.5])

        self.assertFalse(detected_peak)

    def test_detect_peak_returns_false_no_switch_in_d_y(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.peaks.append(0.1)

        detected_peak = pacing_info.detect_peak(
            t=[.1, .2, 3],
            y_voltage=0.03,
            d_y_voltage=[-1.5, -1.5])

        self.assertFalse(detected_peak)

    def test_detect_peak_returns_false_under_voltage_threshold(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.peaks.append(0.1)

        detected_peak = pacing_info.detect_peak(
            t=[.1, .2, 3],
            y_voltage=-0.01,
            d_y_voltage=[1.5, -1.5])

        self.assertFalse(detected_peak)

    def test_detect_peak_returns_true(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.peaks.append(0.1)

        detected_peak = pacing_info.detect_peak(
            t=[.1, .2, 3],
            y_voltage=0.03,
            d_y_voltage=[1.5, -1.5])

        self.assertTrue(detected_peak)

    def test_detect_apd90_returns_false_apd_end_not_set(self):
        pacing_info = trace.IrregularPacingInfo()

        detected_apd90 = pacing_info.detect_apd_90(y_voltage=10)

        self.assertFalse(detected_apd90)

    def test_detect_apd90_returns_false_different_y_voltage(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.apd_90_end_voltage = 5

        detected_apd90 = pacing_info.detect_apd_90(y_voltage=10)

        self.assertFalse(detected_apd90)

    def test_detect_apd90_returns_true(self):
        pacing_info = trace.IrregularPacingInfo()
        pacing_info.apd_90_end_voltage = 5

        detected_apd90 = pacing_info.detect_apd_90(y_voltage=5.0001)

        self.assertTrue(detected_apd90)

    def test_calculate_current_contribution(self):
        _ = self  # Silence pycharm warning.
        current_response_info = trace.CurrentResponseInfo(protocol=None)
        current_response_info.currents.append(
            [trace.Current(name='I_Na', value=1.0),
             trace.Current(name='I_Ca', value=2.0)])
        current_response_info.currents.append(
            [trace.Current(name='I_Na', value=3.0),
             trace.Current(name='I_Ca', value=-4.0)])
        current_response_info.currents.append(
            [trace.Current(name='I_Na', value=2.0),
             trace.Current(name='I_Ca', value=-6.0)])
        current_response_info.currents.append(
            [trace.Current(name='I_Na', value=5.0),
             trace.Current(name='I_Ca', value=-2.0)])
        current_response_info.currents.append(
            [trace.Current(name='I_Na', value=2.0),
             trace.Current(name='I_Ca', value=12.0)])
        current_response_info.currents.append(
            [trace.Current(name='I_Na', value=2.0),
             trace.Current(name='I_Ca', value=12.0)])
        current_response_info.currents.append(
            [trace.Current(name='I_Na', value=15.0),
             trace.Current(name='I_Ca', value=3.0)])
        test_trace = trace.Trace(
            t=[i for i in range(7)],
            y=[5 for _ in range(7)],
            current_response_info=current_response_info)

        contribs = test_trace.current_response_info.get_current_contributions(
            time=test_trace.t,
            window=2.0,
            step_size=1.0)

        expected_contribs = pd.DataFrame(data={
            'Time Start': [0.0, 1.0, 2.0, 3.0, 4.0],
            'Time End': [2.0, 3.0, 4.0, 5.0, 6.0],
            'I_Na': [0.333, 0.454, 0.310, 0.257, 0.413],
            'I_Ca': [0.666, 0.545, 0.689, 0.742, 0.586]})

        pd.testing.assert_frame_equal(
            contribs,
            expected_contribs,
            check_less_precise=2)


if __name__ == '__main__':
    unittest.main()
