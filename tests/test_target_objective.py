import context

import unittest
import matplotlib.pyplot as plt
import numpy as np
import pickle

from cell_models import protocols, paci_2018, kernik
from cell_models.ga.target_objective import TargetObjective, create_target_from_protocol



class TestTargetObjective(unittest.TestCase):
    """Test to make sure that target objectives can be created correctly"""

    def test_create_and_run_target(self):
        """
        Test if the baseline model generates a valid response
        """
        kernik_baseline = kernik.KernikModel(updated_parameters={'G_Na':.5})
        kernik_g_k1 = kernik.KernikModel(no_ion_selective_dict={'I_K1_Ishi': .5})
        aperiodic_proto = protocols.AperiodicPacingProtocol('Kernik')

        cell_path = 'input_data/cell_9_July'
        target_file = 'recording_dc_irregular_trial_2_t_146-156.pkl'

        cell_data = pickle.load(open(f'{cell_path}/targets/{target_file}', 'rb'))

        new_target = TargetObjective(
                cell_data.time,
                cell_data.voltage,
                cell_data.current,
                cell_data.cm,
                cell_data.protocol_type,
                is_exp=True,
                target_protocol=aperiodic_proto,
                target_meta=cell_data.target_meta,
                times_to_compare=None,
                g_ishi=cell_data.g_ishi)

        tr = kernik_baseline.generate_response(new_target,
                is_no_ion_selective=True)
        tr_gk1 = kernik_g_k1.generate_response(aperiodic_proto,
                is_no_ion_selective=True)
        
        #plotted example of test
        #plt.plot(tr.t, tr.y)
        #plt.plot(tr_gk1.t, tr_gk1.y)
        #plt.show()

        self.assertTrue((np.array_equal(tr.t, tr_gk1.t) and
            np.array_equal(tr.y, tr_gk1.y)), 
            "The solution to a Target Objective and protocol input with the same parameters is not equal")


    def test_get_error_from_target(self):
        """
        Test if the baseline model generates a valid response
        """
        paci_baseline = paci_2018.PaciModel(
                no_ion_selective_dict={'I_K1_Ishi': .5})
        aperiodic_proto = protocols.AperiodicPacingProtocol('Paci')
        tr_baseline = paci_baseline.generate_response(aperiodic_proto,
                is_no_ion_selective=True)

        cell_path = 'input_data/cell_9_July'
        target_file = 'recording_dc_irregular_trial_2_t_146-156.pkl'
        cell_data = pickle.load(open(f'{cell_path}/targets/{target_file}', 'rb'))

        target_with_model_data = create_target_from_protocol(paci_baseline,
                aperiodic_proto, g_ishi=cell_data.g_ishi)

        target_with_exp_data = TargetObjective(
                cell_data.time,
                cell_data.voltage,
                cell_data.current,
                cell_data.cm,
                cell_data.protocol_type,
                model_name='Experimental',
                target_meta=cell_data.target_meta,
                times_to_compare=None,
                g_ishi=cell_data.g_ishi)

        error_with_model = target_with_model_data.compare_individual(tr_baseline)
        error_with_exp = target_with_exp_data.compare_individual(tr_baseline)

        #plt.plot(target_with_exp_data.time, target_with_exp_data.voltage, label='Exp')
        #plt.plot(target_with_model_data.time, target_with_model_data.voltage, label='Model Target')
        #plt.plot(tr_baseline.t * 1000, tr_baseline.y * 1000, label='New trace')
        #plt.legend()
        #plt.show()

        self.assertTrue(np.log10(error_with_model) < (
            np.log10(error_with_exp) - 1),
            "The solution to a Target Objective and protocol input with the same parameters is not equal")
        



if __name__ == '__main__':
    unittest.main()
