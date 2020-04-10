import unittest
import paci_2018
import protocols
import numpy.testing as tst
import pdb
import ga_configs


class TestPaci2018(unittest.TestCase):
    def test_generate_trace_SAP(self):
        protocol = protocols.SpontaneousProtocol()

        baseline_model = paci_2018.PaciModel()
        baseline_model.generate_response(protocol)

        self.assertTrue(len(baseline_model.t) > 100,
                'Paci errored in less than .4s')
        self.assertTrue(min(baseline_model.y_voltage) < -.01,
                'baseline Paci min is greater than .01')
        self.assertTrue(max(baseline_model.y_voltage)<.06,
                'baseline Paci max is greater than .06')

    def test_update_parameters(self):
        protocol = protocols.SpontaneousProtocol()
        tunable_parameters = [
            ga_configs.Parameter(name='G_Na', value=1),
            ga_configs.Parameter(name='G_F', value=1),
            ga_configs.Parameter(name='G_Ks', value=1),
            ga_configs.Parameter(name='G_Kr', value=1),
            ga_configs.Parameter(name='G_K1', value=1),
            ga_configs.Parameter(name='G_bNa', value=1),
            ga_configs.Parameter(name='G_NaL', value=1),
            ga_configs.Parameter(name='G_CaL', value=1),
            ga_configs.Parameter(name='G_pCa', value=1),
            ga_configs.Parameter(name='G_bCa', value=1)]
        new_params = [.7, .7, .7, .7, .7, .7, .7, .7, .7, .7]

        baseline_trace = paci_2018.generate_trace(protocol)
        new_trace = paci_2018.generate_trace(protocol,
                                             tunable_parameters, new_params)

        tst.assert_raises(AssertionError, tst.assert_array_equal,
                          baseline_trace.y, new_trace.y,
                          'updating parameters does not change trace')

    def test_no_ion_selective(self):
        protocol = protocols.SpontaneousProtocol()
        paci_baseline = paci_2018.PaciModel()
        ion_selective_scales = {'I_CaL': .4, 'I_NaCa': .4}
        paci_no_ion = paci_2018.PaciModel(
            no_ion_selective_dict=ion_selective_scales)

        baseline_trace = paci_baseline.generate_response(protocol)
        baseline_no_ion_trace = paci_no_ion.generate_response(protocol,
                is_no_ion_selective=True)

        tst.assert_raises(AssertionError, tst.assert_array_equal,
                          baseline_trace.y, baseline_no_ion_trace.y,
                          'generate_no_ion_trace() does not update trace')

    def test_no_ion_selective_setter(self):
        protocol = protocols.SpontaneousProtocol()
        paci_baseline = paci_2018.PaciModel()
        ion_selective_scales = {'I_CaL': .4, 'I_NaCa': .4}

        baseline_trace = paci_baseline.generate_response(protocol)
        paci_baseline.no_ion_selective = ion_selective_scales
        baseline_no_ion_trace = paci_baseline.generate_response(
            protocol, is_no_ion_selective=True)

        tst.assert_raises(AssertionError, tst.assert_array_equal,
                          baseline_trace.y, baseline_no_ion_trace.y,
                          'generate_no_ion_trace() does not update trace')

    def test_no_ion_selective_IK1_no_effect(self):
        protocol = protocols.SpontaneousProtocol()
        paci_k1_increase = paci_2018.PaciModel(updated_parameters=
                                               {'G_K1':1.4})
        ion_selective_scales = {'I_K1': .4}
        paci_k1_no_ion = paci_2018.PaciModel(no_ion_selective_dict=
                                             ion_selective_scales)

        trace_ion_selective = paci_k1_increase.generate_response(protocol)
        trace_no_ion_selective = paci_k1_no_ion.generate_response(protocol, is_no_ion_selective=True)

        tst.assert_raises(AssertionError, tst.assert_array_equal,
                          trace_ion_selective.y, trace_no_ion_selective.y,
                          'generate_no_ion_trace() does not update trace')

    def test_voltage_protocol(self):
        pass

    def test_stim_protocol(self):
        pass


if __name__ == '__main__':
    unittest.main()
