import context

from cell_models.ga.parameter_tuning import ParameterTuningGeneticAlgorithm
from cell_models.ga import ga_configs
from cell_models import protocols

import unittest

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""
    def setUp(self):
        KERNIK_PARAMETERS = [
             ga_configs.Parameter(name='G_Na', default_value=1),
             ga_configs.Parameter(name='G_F', default_value=1),
             ga_configs.Parameter(name='G_Ks', default_value=1),
             ga_configs.Parameter(name='G_Kr', default_value=1),
             ga_configs.Parameter(name='G_K1', default_value=1),
             ga_configs.Parameter(name='G_b_Na', default_value=1),
             ga_configs.Parameter(name='P_CaL', default_value=1),
             ga_configs.Parameter(name='G_PCa', default_value=1),
             ga_configs.Parameter(name='G_b_Ca', default_value=1),
             ga_configs.Parameter(name='K_NaCa', default_value=1)
         ]

        self.kernik_protocol = protocols.VoltageClampProtocol()

        self.vc_config = ga_configs.ParameterTuningConfig(
            population_size=30,
            max_generations=10,
            protocol=self.kernik_protocol,
            tunable_parameters=KERNIK_PARAMETERS,
            params_lower_bound=0.1,
            params_upper_bound=10,
            mate_probability=0.9,
            mutate_probability=0.9,
            gene_swap_probability=0.2,
            gene_mutation_probability=0.2,
            tournament_size=4)

    #def test_baseline_parameter_tuning(self):
    #    res_kernik = ParameterTuningGeneticAlgorithm('Kernik',
    #                                                 self.vc_config,
    #                                                 self.kernik_protocol)
    #    
    #    assert True

    def test_random_parameter_tuning(self):
        res_kernik = ParameterTuningGeneticAlgorithm('Kernik',
                                                     self.vc_config,
                                                     self.kernik_protocol,
                                                     is_target_baseline=False)
        
        assert True

if __name__ == '__main__':
    unittest.main()