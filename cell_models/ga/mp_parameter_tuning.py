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
from cell_models.paci_2018 import PaciModel
from cell_models import kernik
from cell_models import protocols
from cell_models import trace

from cell_models.rtxi.target_objective import TargetObjective
from cell_models.ga import ga_configs
from cell_models.ga import genetic_algorithm_results
from cell_models.ga.model_target_objective import ModelTarget


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
                 protocols, is_parameter_recovery=True,
                 is_target_baseline=True):
        """
        Initialize the class
        """
        if model_name == "Kernik":
            self.cell_model = KernikModel
        elif model_name == "Paci":
            self.cell_model = PaciModel

        self.vc_config = vc_config
        self.protocols = protocols
        self.previous_population = None
        self.target_params = None

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
        if ga_params.vc_config.with_exp_artefact:
            random_ss = np.load(pkg_resources.resource_stream(
                __name__, "models_at_ss_artefact.npy"), allow_pickle=True)
        else:
            random_ss = np.load(pkg_resources.resource_stream(
                __name__, "models_at_ss.npy"), allow_pickle=True)

        index = random.randint(0, 8)

        print(f"Taking random trace with index {index}")

        target_cell = ga_params.cell_model(
            updated_parameters=random_ss[index][0],
            is_exp_artefact=ga_params.vc_config.with_exp_artefact)

        GA_PARAMS.target_params = random_ss[index][0]

        target_cell.y_initial = random_ss[index][1]
        target_cell.y_ss = random_ss[index][1]

    else:
        target_cell = ga_params.cell_model(
            is_exp_artefact=ga_params.vc_config.with_exp_artefact)
        baseline_y_ss = np.load(pkg_resources.resource_stream(
                __name__, "baseline_ss.npy"), allow_pickle=True)

        if ga_params.vc_config.with_exp_artefact:
            target_cell.y_initial[0:23] = baseline_y_ss[1]
            target_cell.y_ss = target_cell.y_initial
        else:
            target_cell.y_ss = baseline_y_ss[1]
            target_cell.y_initial = baseline_y_ss[1]

    traces = {}

    for current, protocol in ga_params.protocols.items():
        traces[current] = get_model_response(target_cell, protocol)

    return traces

def run_ga(ga_params, toolbox):
    """
    Runs an instance of the genetic algorithm.

    Returns
    -------
    final_population : List[model obects]
    """
    print('Evaluating initial population.')

    population = toolbox.population(ga_params.vc_config.population_size)

    targets = [copy.deepcopy(ga_params.vc_config.targets) for i in range(
        0, ga_params.vc_config.population_size)]

    eval_input = np.transpose([population, targets])

    fitnesses = toolbox.map(toolbox.evaluate, eval_input)
   
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Store initial population details for result processing.
    initial_population = []
    for i in range(len(population)):
        initial_population.append(
            genetic_algorithm_results.ParameterTuningIndividual(
                parameters=population[i][0].default_parameters,
                fitness=population[i].fitness.values))

    final_population = [initial_population]

    avg_fitness = []

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

        targets = [copy.deepcopy(ga_params.vc_config.targets) for i in
                     range(0, len(updated_individuals))]

        eval_input = np.transpose([updated_individuals, targets])

        fitnesses = toolbox.map(toolbox.evaluate, eval_input)

        for ind, fit in zip(updated_individuals, fitnesses):
            ind.fitness.values = fit

        population = offspring

        # Store intermediate population details for result processing.
        intermediate_population = []
        for i in range(len(population)):
            intermediate_population.append(
                genetic_algorithm_results.ParameterTuningIndividual(
                    parameters=population[i][0].default_parameters,
                    fitness=population[i].fitness.values))

        final_population.append(intermediate_population)

        generate_statistics(population)

        fitness_values = [i.fitness.values for i in population]

        #TODO: create exit condition for all multi-objective
        #if len(avg_fitness) > 3:
        #    if len(fitness_values) > 1:
        #        print('multiobjective')
        #    if np.mean(fitness_values) >= max(avg_fitness[-3:]):
        #        break

        #avg_fitness.append(np.mean(fitness_values))

    
    final_ga_results = genetic_algorithm_results.GAResultParameterTuning(
            'kernik', TARGETS, GA_PARAMS.target_params,
            final_population, GA_PARAMS.vc_config,
            )

    return final_ga_results


def get_model_response(model, command, prestep=5000.0, is_command_prestep=True):
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
    if is_command_prestep:
        prestep_protocol = protocols.VoltageClampProtocol(
            [protocols.VoltageClampStep(voltage=-80.0,
                                        duration=prestep)])
    else:
        prestep_protocol = command

    if isinstance(command, TargetObjective):
        if command.protocol_type == 'Dynamic Clamp':
            #TODO: Aperiodic Pacing Protocol
            prestep_protocol = protocols.AperiodicPacingProtocol(
                GA_PARAMS.vc_config.model_name)
            command = prestep_protocol
            model.generate_response(prestep_protocol,
                        is_no_ion_selective=True)
            response_trace = model.generate_response(command,
                    is_no_ion_selective=True)
        else:
            model.generate_response(prestep_protocol,
                        is_no_ion_selective=False)
            model.y_ss = model.y[:, -1]
            response_trace = model.generate_response(command.protocol,
                    is_no_ion_selective=False)
    else:
        model.generate_response(prestep_protocol,
                    is_no_ion_selective=False)

        model.y_ss = model.y[:, -1]

        response_trace = model.generate_response(command,
                is_no_ion_selective=False)

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
        updated_parameters=dict(zip(keys, initial_params)),
        is_exp_artefact=vc_config.with_exp_artefact)


def _evaluate_recovery(eval_input):
    """
    Evaluates performance of an individual compared to the target obj.

        Returns
        -------
            error: Number
                The error between the trace generated by the individual's
                parameter set and the baseline target objective.
    """
    individual_model, input_commands = eval_input
    individual_model = individual_model[0]
    y_initial = individual_model.y_initial

    errors = []

    for current_target, command in input_commands.items():
        individual_model.y_ss = None
        individual_model.y_initial = y_initial

        updated_parameters = individual_model.default_parameters

        try:
            if command.protocol_type == 'Dynamic Clamp':
                if isinstance(command, TargetObjective):
                    g_k1_ishi = command.g_ishi
                    individual_model = GA_PARAMS.cell_model(
                            updated_parameters=updated_parameters,
                            no_ion_selective_dict={'I_K1_Ishi': g_k1_ishi},
                            is_exp_artefact=False)
                else:
                    individual_model = GA_PARAMS.cell_model(
                            updated_parameters=updated_parameters,
                            is_exp_artefact=False)

                new_trace = get_model_response(individual_model,
                                               command,
                                               is_command_prestep=False)
            else:
                individual_model = GA_PARAMS.cell_model(
                        updated_parameters=updated_parameters, 
                        is_exp_artefact=GA_PARAMS.vc_config.with_exp_artefact)

                new_trace = get_model_response(individual_model, command)

            target = TARGETS[current_target]

            error = target.compare_individual(new_trace)
        except:
            error = 10E8

        errors.append(log10(error))

    return errors

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
    #for index, current in enumerate(list(GA_PARAMS.vc_config.protocols.keys())):
    for index in range(0, len(population[0].fitness.values)):
        fitness_values = [i.fitness.values[index] for i in population]
        #print(f'Details for: {current}')
        print('\t\tMin fitness: {}'.format(min(fitness_values)))
        print('\t\tMax fitness: {}'.format(max(fitness_values)))
        print('\t\tAverage fitness: {}'.format(np.mean(fitness_values)))
        print('\t\tStandard deviation: {}'.format(np.std(fitness_values)))

creator.create('FitnessMulti', base.Fitness, weights=(-1.0, -1.0, -1.0,
                                                    -1.0, -1.0, -1.0))

creator.create('Individual', list, fitness=creator.FitnessMulti)

def start_ga(target_objectives, vc_config, is_baseline=True):
    global GA_PARAMS
    global TARGETS

    GA_PARAMS = GAParams(vc_config.model_name, vc_config, target_objectives)

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

    TARGETS = target_objectives

    toolbox.register('evaluate', _evaluate_recovery)
    toolbox.register('select',
                     tools.selTournament,
                     tournsize=GA_PARAMS.vc_config.tournament_size)

    toolbox.register('mate', _mate)
    toolbox.register('mutate', _mutate)

    p = Pool()
    toolbox.register("map", p.map)

    final_population = run_ga(GA_PARAMS, toolbox)

    return final_population
