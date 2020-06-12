"""Runs a genetic algorithm for parameter tuning on specified target objective.

Example usage:
    config = <GENERATE CONFIG OBJECT>
    ga_instance = ParameterTuningGeneticAlgorithm(config)
    ga_instance.run()
"""

import random
from typing import List
from math import log10

from deap import base, creator, tools
import numpy as np
from scipy.interpolate import interp1d
from pickle import dump, HIGHEST_PROTOCOL, load
import time
import copy
from multiprocessing import Pool
from os import environ
import pkg_resources
import matplotlib.pyplot as plt

from cell_models.kernik import KernikModel
from cell_models import kernik
from cell_models import protocols
from cell_models import trace

from cell_models.ga import target_objective
from cell_models.ga import ga_configs
from cell_models.ga import genetic_algorithm_results


class GAParams():
    """
    Initializes and runs a parameter tuning GA

    Parameters
    ----------
    model_name: String
        The name can be any model in cell_models. Currently: 'Kernik', 'Paci',
        or 'Ohara-Rudy'
    vc_config: cell_models.ga.ga_configs.ParameterTuningConfig() object
        Contains all of the GA configuration data for a particular parameter
        tuning GA
    protocol: cell_models.protocols type object
    is_parameter_recovery: boolean
        Default is True. This part of the code needs to be edited when we want 
        to fit to experimental data.
    is_target_baseline : boolean
        If this is true, then the target objective will be the baseline model.
        If this is false, the target objective will be a random model selected
        as index 5 from models_at_ss.npy
    """

    def __init__(self, model_name, vc_config,
                 protocol, is_parameter_recovery=True,
                 is_target_baseline=True):
        """
        Initialize the class
        """
        if model_name == "Kernik":
            self.cell_model = KernikModel

        self.vc_config = vc_config
        self.protocol = protocol
        self.previous_population = None

def initialize_target(ga_params, is_baseline=True, updated_parameters=None):
    """
    Initialize the target objective. This will produce a random trace
    to fit to.

    Parameters
    ----------
    updated_parameters : dict of conductance values
    The default, None, will produce a random individual
    """
    if is_baseline and (updated_parameters is not None):
        print(
            """InputError: ParameterTuningGeneticAlgorithm.initialize_target()
           was given incompatible inputs. You should not set is_baseline to
           True and add a value for updated_parameters""")


    if not is_baseline:
        print("Taking random trace with index 5")
        random_ss = np.load(pkg_resources.resource_stream(
            __name__, "models_at_ss.npy"), allow_pickle=True)

        index = 5

        target_cell = ga_params.cell_model(
            updated_parameters=random_ss[index][0])

        target_cell.y_initial = random_ss[index][1]
        target_cell.y_ss = random_ss[index][1]

    else:
        try:
            tr = load(pkg_resources.resource_stream(
                __name__, "baseline_trace.npy"))
            tr.is_interpolated = False
            return tr
        except:
            target_cell = ga_params.cell_model()
            baseline_y_ss = np.load(pkg_resources.resource_stream(
                __name__, "baseline_ss.npy"), allow_pickle=True)
            target_cell.y_ss = baseline_y_ss[1]
            target_cell.y_initial = baseline_y_ss[1]


    tr = get_model_response(target_cell, ga_params.protocol)

    return tr

def run_ga(ga_params, toolbox):
    """
    Runs an instance of the genetic algorithm.

    Returns
    -------
    final_population : List[model obects]
    """
    print('Evaluating initial population.')

    population = toolbox.population(ga_params.vc_config.population_size)

    protocols = [copy.deepcopy(ga_params.vc_config.protocol) for i in range(
        0, ga_params.vc_config.population_size)]

    eval_input = np.transpose([population, protocols])

    fitnesses = toolbox.map(toolbox.evaluate, eval_input)
   
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = [fit]

    # Store initial population details for result processing.
    initial_population = []
    for i in range(len(population)):
        initial_population.append(
            genetic_algorithm_results.ParameterTuningIndividual(
                parameters=population[i][0].default_parameters,
                fitness=population[i].fitness.values[0]))

    final_population = [initial_population]

    for generation in range(1, ga_params.vc_config.max_generations):
        print('Generation {}'.format(generation))
        # Offspring are chosen through tournament selection. They are then
        # cloned, because they will be modified in-place later on.
        ga_params.previous_population = population

        selected_offspring = toolbox.select(population, len(population))

        offspring = [toolbox.clone(i) for i in selected_offspring]

        for i_one, i_two in zip(offspring[::2], offspring[1::2]):
            if random.random() < ga_params.vc_config.mate_probability:
                toolbox.mate(i_one, i_two)
                del i_one.fitness.values
                del i_two.fitness.values

        for i in offspring:
            if random.random() < ga_params.vc_config.mutate_probability:
                toolbox.mutate(i)
                del i.fitness.values

        # All individuals who were updated, either through crossover or
        # mutation, will be re-evaluated.
        updated_individuals = [i for i in offspring if not i.fitness.values]

        protocols = [copy.deepcopy(ga_params.vc_config.protocol) for i in range(
            0, len(updated_individuals))]

        eval_input = np.transpose([updated_individuals, protocols])

        fitnesses = toolbox.map(toolbox.evaluate, eval_input)

        for ind, fit in zip(updated_individuals, fitnesses):
            ind.fitness.values = [fit]

        population = offspring

        # Store intermediate population details for result processing.
        intermediate_population = []
        for i in range(len(population)):
            intermediate_population.append(
                genetic_algorithm_results.ParameterTuningIndividual(
                    parameters=population[i][0].default_parameters,
                    fitness=population[i].fitness.values[0]))

        final_population.append(intermediate_population)

        generate_statistics(population)
    
    final_ga_results = genetic_algorithm_results.GAResultParameterTuning(
            'kernik', TARGET, final_population, GA_PARAMS.vc_config)

    return final_ga_results

def get_model_response(model, protocol, prestep=10000.0):
    """
    Parameters
    ----------
    model : CellModel
        This can be a Kernik, Paci, or OR model instance
    protocol : VoltageClampProtocol
        This can be any VoltageClampProtocol

    Returns
    -------
    trace : Trace
        Trace object with the current and voltage data during the protocol

    Accepts a model object, applies  a -80mV holding prestep, and then 
    applies the protocol. The function returns a trace object with the 
    recording during the input protocol.
    """
    prestep_protocol = protocols.VoltageClampProtocol(
        [protocols.VoltageClampStep(voltage=-80.0,
                                    duration=prestep)])

    model.generate_response(prestep_protocol)

    model.y_ss = model.y[:, -1]

    response_trace = model.generate_response(protocol)

    return response_trace

def _initialize_individuals(vc_config, cell_model):
    """
    Creates the initial population of individuals. The initial 
    population 

    Returns:
        A model instance with a new set of parameters
    """
    # Builds a list of parameters using random upper and lower bounds.
    lower_exp = log10(vc_config.params_lower_bound)
    upper_exp = log10(vc_config.params_upper_bound)
    initial_params = [10**random.uniform(lower_exp, upper_exp)
                      for i in range(0, len(
                          vc_config.tunable_parameters))]

    keys = [val.name for val in vc_config.tunable_parameters]

    return cell_model(
        updated_parameters=dict(zip(keys, initial_params)))

def _evaluate_performance(eval_input):
    """
    Evaluates performance of an individual compared to the target obj.

        Returns
        -------
            error: Number
                The error between the trace generated by the individual's
                parameter set and the baseline target objective.
    """
    t_start = time.time()
    individual, protocol = eval_input
    individual = individual[0]

    primary_trace = None

    while primary_trace is None:
        primary_trace = get_model_response(individual,
                                           protocol)
        if primary_trace is None:
            print("Individual errored. Returning 10,000")
            return 10000

    error = compare_individual(primary_trace)
    
    return error

def _mate(i_one, i_two):
    """Performs crossover between two individuals.

    There may be a possibility no parameters are swapped. This probability
    is controlled by `self.config.gene_swap_probability`. Modifies
    both individuals in-place.

    Args:
        i_one: An individual in a population.
        i_two: Another individual in the population.
    """
    for key, val in i_one[0].default_parameters.items():
        if random.random() < GA_PARAMS.vc_config.gene_swap_probability:
            i_one[0].default_parameters[key],\
                i_two[0].default_parameters[key] = (
                    i_two[0].default_parameters[key],
                    i_one[0].default_parameters[key])

def _mutate(individual):
    """Performs a mutation on an individual in the population.

    Chooses random parameter values from the normal distribution centered
    around each of the original parameter values. Modifies individual
    in-place.

    Args:
        individual: An individual to be mutated.
    """
    keys = [p.name for p in GA_PARAMS.vc_config.tunable_parameters]

    for key in keys:
        if random.random() < GA_PARAMS.vc_config.gene_mutation_probability:
            new_param = -1

            while ((new_param < GA_PARAMS.vc_config.params_lower_bound) or
                   (new_param > GA_PARAMS.vc_config.params_upper_bound)):
                new_param = np.random.normal(
                        individual[0].default_parameters[key],
                        individual[0].default_parameters[key] * .1)

            individual[0].default_parameters[key] = new_param

def generate_statistics(population: List[List[List[float]]]) -> None:
    fitness_values = [i.fitness.values[0] for i in population]
    print('  Min fitness: {}'.format(min(fitness_values)))
    print('  Max fitness: {}'.format(max(fitness_values)))

    print('  Average fitness: {}'.format(np.mean(fitness_values)))
    print('  Standard deviation: {}'.format(np.std(fitness_values)))

def compare_individual(individual):
    if individual is None:
        return 4500 
    if not TARGET.is_interpolated:
        TARGET.interp_time = np.linspace(TARGET.t[0], TARGET.t[-1], int(
            TARGET.t[-1] - TARGET.t[0]))

        f = interp1d(TARGET.t,
                     TARGET.current_response_info.get_current_summed())

        TARGET.interp_current = f(TARGET.interp_time)

        TARGET.is_interpolated = True

    f = interp1d(individual.t,
                 individual.current_response_info.get_current_summed())

    individual_current = f(TARGET.interp_time)

    error = sum(abs(TARGET.interp_current - individual_current))

    return error


creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

def start_ga(kernik_protocol, vc_config, is_baseline=True):
    global GA_PARAMS
    global TARGET

    GA_PARAMS = GAParams('Kernik', vc_config, kernik_protocol)
    TARGET = initialize_target(GA_PARAMS, is_baseline=is_baseline)

    toolbox = base.Toolbox()
    toolbox.register('init_param',
                     _initialize_individuals,
                     GA_PARAMS.vc_config,
                     GA_PARAMS.cell_model)
    toolbox.register('individual',
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.init_param,
                     n=1)
    toolbox.register('population',
                     tools.initRepeat,
                     list,
                     toolbox.individual)
    toolbox.register('evaluate', _evaluate_performance)
    toolbox.register('select',
                     tools.selTournament,
                     tournsize=GA_PARAMS.vc_config.tournament_size)
    toolbox.register('mate', _mate)
    toolbox.register('mutate', _mutate)

    p = Pool()
    toolbox.register("map", p.map)

    final_population = run_ga(GA_PARAMS, toolbox)

    return final_population
