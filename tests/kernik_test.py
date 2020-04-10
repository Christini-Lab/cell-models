import unittest
import kernik
import protocols
import numpy as np
import numpy.testing as tst
import pdb


class TestKernik(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        protocol = protocols.SpontaneousProtocol(1200)
        cls.baseline_model = kernik.KernikModel()
        cls.baseline_model.generate_response(protocol)

    def test_kernik_matches_original(self):
        expected_trace = np.load('./model_data/baseline_kernik.npy')
        expected_time = expected_trace[0,:]
        expected_V = expected_trace[1,:]

        tst.assert_array_equal(expected_time, self.baseline_model.t,
                          'The time array has changed from the original \
                          Kernik model')

        tst.assert_array_equal(expected_V, self.baseline_model.y_voltage, 
                          'The voltage array has changed from the original \
                          Kernik model')

    def test_generate_trace_SAP(self):
        self.assertTrue(len(self.baseline_model.t) > 100,
                'Kernik errored in less than .4s')
        self.assertTrue(min(self.baseline_model.y_voltage) < -10,
                'baseline Kernik min is greater than -.01')
        self.assertTrue(max(self.baseline_model.y_voltage) < 60,
                'baseline Kernik max is greater than .06')
    
    def test_no_ion_selective(self):
        protocol = protocols.SpontaneousProtocol(500)
        kernik_baseline = kernik.KernikModel()
        ion_selective_scales = {'I_CaL': .4, 'I_NaCa': .4}
        kernik_no_ion = kernik.KernikModel(
            no_ion_selective_dict=ion_selective_scales)

        baseline_trace = kernik_baseline.generate_response(protocol)
        baseline_no_ion_trace = kernik_no_ion.generate_response(protocol,
                                is_no_ion_selective=True)

        tst.assert_raises(AssertionError, tst.assert_array_equal,
                          baseline_trace.y, baseline_no_ion_trace.y,
                          'generate_no_ion_trace() does not update trace')

    def test_get_currents(self):
        protocol = protocols.SpontaneousProtocol(100)
        baseline_model = kernik.KernikModel()
        baseline_model.generate_response(protocol)
        start_currents = baseline_model.current_response_info        

        baseline_model.calc_currents()

        self.assertTrue(len(baseline_model.current_response_info.currents) > 100,
                '.get_currents() does not update the current parameter')
        self.assertTrue(start_currents is None,
                'model is saving to currents when it should not be')



#    def test_kernik_py_vs_mat(self):
#        mat_baseline = np.loadtxt('model_data/original_baseline_3000ms.csv')
#        # compare_voltage_plots(self.baseline_model, mat_baseline)
#
#        tst.assert_raises(AssertionError, tst.assert_array_equal,
#                          mat_baseline[:, 1], self.baseline_model.y_voltage,
#                          'updating parameters does not change trace')

def compare_voltage_plots(individual, original_matlab):
    import matplotlib.pyplot as plt
    plt.plot(original_matlab[:, 0], original_matlab[:, 1], label='Matlab')
    plt.plot(individual.t, individual.y_voltage, label='Python')
    axes = plt.gca()
    axes.set_xlabel('Time (ms)', fontsize=20)
    axes.set_ylabel('Voltage', fontsize=20)
    plt.legend(fontsize=16)
    plt.show()
    pdb.set_trace()


if __name__ == '__main__':
    unittest.main()
