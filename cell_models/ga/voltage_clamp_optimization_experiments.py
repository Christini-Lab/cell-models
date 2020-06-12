"""Contains functions to run VC optimization genetic algorithm experiments."""

import copy
import os
from typing import List

from cell_models.ga import ga_configs, genetic_algorithm_results, voltage_clamp_optimization
from cell_models import protocols


def get_highest_fitness_individual_overall(
        result: genetic_algorithm_results.GAResultVoltageClampOptimization
) -> genetic_algorithm_results.VCOptimizationIndividual:
    """Gets the highest fitness individual across all generations."""
    highest_fitness = 0
    highest_fitness_individual = None
    for i in range(len(result.generations)):
        curr_individual = result.get_high_fitness_individual(generation=i)
        if curr_individual.fitness > highest_fitness:
            highest_fitness = curr_individual.fitness
            highest_fitness_individual = curr_individual
    return highest_fitness_individual


def run_voltage_clamp_experiment(
        config: ga_configs.VoltageOptimizationConfig,
        full_output: bool=False
) -> genetic_algorithm_results.GAResultVoltageClampOptimization:
    """Runs a voltage clamp experiment with output if specified."""
    result = voltage_clamp_optimization.VCOGeneticAlgorithm(config=config).run()

    if full_output:
        result.generate_heatmap()
        result.graph_fitness_over_generation(with_scatter=False)

        random_0 = result.get_random_individual(generation=0)
        worst_0 = result.get_low_fitness_individual(generation=0)
        best_0 = result.get_high_fitness_individual(generation=0)
        best_middle = result.get_high_fitness_individual(
            generation=config.max_generations // 2)
        best_end = result.get_high_fitness_individual(
            generation=config.max_generations - 1)
        best_all_around = get_highest_fitness_individual_overall(result=result)
        print('Best protocol: {}'.format(best_all_around.protocol))
        print('Best protocol\'s fitness: {}'.format(best_all_around.fitness))

        genetic_algorithm_results.graph_combined_current_contributions(
            protocol=best_end.protocol,
            config=result.config,
            title='Single VC Optimization/Best individual currents, generation'
                  ' {}'.format(config.max_generations - 1))

        genetic_algorithm_results.graph_combined_current_contributions(
            protocol=best_all_around.protocol,
            config=result.config,
            title='Single VC Optimization/Best individual currents, all'
                  ' generations')

        genetic_algorithm_results.graph_vc_protocol(
            protocol=random_0.protocol,
            title='Random individual, generation 0')

        genetic_algorithm_results.graph_vc_protocol(
            protocol=worst_0.protocol,
            title='Worst individual, generation 0')

        genetic_algorithm_results.graph_vc_protocol(
            protocol=best_0.protocol,
            title='Best individual, generation 0')

        genetic_algorithm_results.graph_vc_protocol(
            protocol=best_middle.protocol,
            title='Best individual, generation {}'.format(
                config.max_generations // 2))

        genetic_algorithm_results.graph_vc_protocol(
            protocol=best_end.protocol,
            title='Best individual, generation {}'.format(
                config.max_generations - 1))

        genetic_algorithm_results.graph_vc_protocol(
            protocol=best_all_around.protocol,
            title='Best individual, all generations')
    return result


def construct_optimal_protocol(
        vc_protocol_optimization_config: ga_configs.CombinedVCConfig,
        with_output: bool=False,
) -> protocols.VoltageClampProtocol:
    """Constructs the optimal VC protocol to isolate the provided currents.

    Attempts to optimize voltage clamp protocols for a single current and then
    combines them together with a holding current in between.
    """
    optimal_protocols = {}
    for i in vc_protocol_optimization_config.currents:
        print('Optimizing current: {}'.format(i))
        optimal_protocols[i] = find_single_current_optimal_protocol(
            current=i,
            vc_opt_config=vc_protocol_optimization_config)
    optimal_protocol = combine_protocols(list(optimal_protocols.values()))

    if with_output:
        # Create the appropriate directory, if one does not exist.
        if not os.path.exists('figures/Voltage Clamp Figure/'
                              'Full VC Optimization'):
            os.makedirs('figures/Voltage Clamp Figure/Full VC Optimization')
        try:
            genetic_algorithm_results.graph_optimized_vc_protocol_full_figure(
                single_current_protocols=optimal_protocols,
                combined_protocol=optimal_protocol,
                config=vc_protocol_optimization_config.ga_config)
        except:
            import pdb
            pdb.set_trace()
    return optimal_protocol


def find_single_current_optimal_protocol(
        current: str,
        vc_opt_config: ga_configs.CombinedVCConfig,
) -> protocols.VoltageClampProtocol:
    """Runs genetic algorithm to find optimal VC protocol for a single current.

    Protocols of varying step sizes will be generated. The first protocol to
    meet the adequate fitness threshold set in the config parameter will be
    returned. If no such protocol exists, the highest fitness protocol will be
    returned.
    """
    best_individuals = []
    for i in vc_opt_config.step_range:
        print('Trying to optimize with {} steps.'.format(i))
        new_ga_config = copy.deepcopy(vc_opt_config.ga_config)
        new_ga_config.steps_in_protocol = i
        new_ga_config.target_currents = [current]
        result = run_voltage_clamp_experiment(config=new_ga_config)
        best_individual = get_highest_fitness_individual_overall(result=result)

        best_individuals.append(best_individual)
        if best_individual.fitness > vc_opt_config.adequate_fitness_threshold:
            break

    best_individuals.sort()
    return best_individuals[-1].protocol


def combine_protocols(
        optimal_protocols: List[protocols.VoltageClampProtocol]
) -> protocols.VoltageClampProtocol:
    """Combines protocols together."""
    combined_protocol = protocols.VoltageClampProtocol()
    for i in optimal_protocols:
        combined_protocol.steps.extend(i.steps)
    return combined_protocol

