import context

from cell_models.ga import mp_parameter_tuning
from cell_models.ga import ga_configs
from cell_models import protocols
from cell_models.ga.target_objective import TargetObjective

import unittest
import pickle

class TestGA(unittest.TestCase):
    """Basic test cases."""
    def test_fit_to_target_model(self):
        cell_path = 'input_data/cell_9_July'
        target_file = 'recording_dc_irregular_trial_2_t_146-156.pkl'
        cell_data = pickle.load(open(f'{cell_path}/targets/{target_file}', 'rb'))

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

        targets = {'targets': target_with_exp_data}

        KERNIK_PARAMETERS = [
            ga_configs.Parameter(name='G_Na', default_value=1),
            ga_configs.Parameter(name='G_F', default_value=1),
            ga_configs.Parameter(name='G_Ks', default_value=1),
            ga_configs.Parameter(name='G_Kr', default_value=1),
            ga_configs.Parameter(name='G_K1', default_value=1),
            ga_configs.Parameter(name='P_CaL', default_value=1),
            ga_configs.Parameter(name='G_to', default_value=1)]

        vc_config = ga_configs.ParameterTuningConfig(
            population_size=6,
            max_generations=3,
            targets=targets,
            tunable_parameters=KERNIK_PARAMETERS,
            params_lower_bound=0.1,
            params_upper_bound=10,
            mate_probability=0.9,
            mutate_probability=0.9,
            gene_swap_probability=0.2,
            gene_mutation_probability=0.2,
            tournament_size=2,
            with_exp_artefact=False,
            model_name='Kernik')

        final_population = mp_parameter_tuning.start_ga(vc_config,
                                                        is_baseline=False)




    def test_fit_to_target_exp(self):
        paci_baseline = paci_2018.PaciModel(
                no_ion_selective_dict={'I_K1_Ishi': .5})
        aperiodic_proto = protocols.AperiodicPacingProtocol('Paci')
        tr_baseline = paci_baseline.generate_response(aperiodic_proto,
                is_no_ion_selective=True)

        target_with_model_data = create_target_from_protocol(paci_baseline,
                    aperiodic_proto, g_ishi=cell_data.g_ishi)

    def test_baseline_parameter_tuning(self):
        """
        Test if the baseline model generates a valid response
        """
        final_population = mp_parameter_tuning.start_ga(self.kernik_protocol,
                self.vc_config, is_baseline=False)

        
        self.assertIsNotNone(res_kernik.target.t,
                        "There was an error when initializing the target")

if __name__ == '__main__':
    unittest.main()

