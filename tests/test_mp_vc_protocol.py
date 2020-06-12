import context

from cell_models.ga import ga_configs, mp_voltage_clamp_optimization
from cell_models import protocols, kernik

import unittest
import copy
import time


class TestGA(unittest.TestCase):
    """Basic test cases."""

    def test_vco_ga(self):
        """
        Test if the baseline model generates a valid response
        """
        VCO_CONFIG_KERNIK = ga_configs.VoltageOptimizationConfig(
            window=10,
            step_size=5,
            steps_in_protocol=3,
            step_duration_bounds=(25, 2500),
            step_voltage_bounds=(-200, 100),
            target_current='I_Na',
            population_size=40,
            max_generations=6,
            mate_probability=0.9,
            mutate_probability=0.9,
            gene_swap_probability=0.2,
            gene_mutation_probability=0.1,
            tournament_size=2,
            step_types=['step'])

        COMBINED_VC_CONFIG = ga_configs.CombinedVCConfig(
            currents=['I_Na', 'I_K1', 'I_To', 'I_CaL', 'I_Kr', 'I_Ks'],
            step_range=range(2, 3, 1),
            adequate_fitness_threshold=0.95,
            ga_config=VCO_CONFIG_KERNIK)


        result = mp_voltage_clamp_optimization.start_ga(COMBINED_VC_CONFIG)

        best_ind = get_high_fitness(result)

        vc_protocol = best_ind.protocol

        baseline_kernik = kernik.KernikModel()
        i_trace = baseline_kernik.generate_response(vc_protocol)

        max_currents = i_trace.current_response_info.get_max_current_contributions(
                i_trace.t, window=10, step_size=5)

        print(max_currents)
        print(best_ind)


def get_high_fitness(ga_result):
    best_individual = ga_result.generations[0][0]

    for i, gen in enumerate(ga_result.generations):
        best_in_gen = ga_result.get_high_fitness_individual(i)
        print(best_in_gen)
        if best_in_gen.fitness > best_individual.fitness:
            best_individual = best_in_gen

    return best_individual



if __name__ == '__main__':
    unittest.main()
