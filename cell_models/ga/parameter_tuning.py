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
import multiprocessing
from os import environ
import pkg_resources

from cell_models.kernik import KernikModel
from cell_models import protocols
from cell_models import trace

from cell_models.ga import target_objective
from cell_models.ga import ga_configs
from cell_models.ga import genetic_algorithm_results


class ParameterTuningGeneticAlgorithm():
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
        self.toolbox = self.initialize_toolbox()
        self.previous_population = None

        if is_parameter_recovery:
            self.target = self.initialize_target(
                is_baseline=is_target_baseline)

    def run_ga(self):
        """
        Runs an instance of the genetic algorithm.

        Returns
        -------
        final_population : List[model obects]
        """
        print('Evaluating initial population.')

        population = self.toolbox.population(self.vc_config.population_size)

        fitnesses = self.toolbox.map(self.toolbox.evaluate, population)
       
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

        for generation in range(1, self.vc_config.max_generations):
            print('Generation {}'.format(generation))
            # Offspring are chosen through tournament selection. They are then
            # cloned, because they will be modified in-place later on.
            self.previous_population = population

            selected_offspring = self.toolbox.select(population, len(population))

            offspring = [self.toolbox.clone(i) for i in selected_offspring]

            for i_one, i_two in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.vc_config.mate_probability:
                    self.toolbox.mate(i_one, i_two)
                    del i_one.fitness.values
                    del i_two.fitness.values

            for i in offspring:
                if random.random() < self.vc_config.mutate_probability:
                    self.toolbox.mutate(i)
                    del i.fitness.values

            # All individuals who were updated, either through crossover or
            # mutation, will be re-evaluated.
            updated_individuals = [i for i in offspring if not i.fitness.values]

            num_inputs = len(updated_individuals)
            fitnesses = self.toolbox.map(self.toolbox.evaluate, updated_individuals)

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
            import pdb
            pdb.set_trace()

        return final_population

    def initialize_target(self, is_baseline=True, updated_parameters=None):
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
            try:
                tr = load(pkg_resources.resource_stream(
                    __name__, "random5_trace.npy"))
                tr.is_interpolated = False
                return tr
            except:
                random_ss = np.load(pkg_resources.resource_stream(
                    __name__, "models_at_ss.npy"), allow_pickle=True)

                index = 5

                target_cell = self.cell_model(
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
                target_cell = self.cell_model()
                baseline_y_ss = np.load(pkg_resources.resource_stream(
                    __name__, "baseline_ss.npy"), allow_pickle=True)
                target_cell.y_ss = baseline_y_ss[1]
                target_cell.y_initial = baseline_y_ss[1]

    
        tr = self.get_model_response(target_cell, self.protocol)

        return tr

    def get_model_response(self, model, protocol, prestep=10000.0):
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

    def initialize_toolbox(self):
        """
        Initializes the DEAP toolbox
        """
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register('init_param',
                         self._initialize_individuals,
                         self.vc_config,
                         self.cell_model)
        toolbox.register('individual',
                         tools.initRepeat,
                         creator.Individual,
                         toolbox.init_param,
                         n=1)
        toolbox.register('population',
                         tools.initRepeat,
                         list,
                         toolbox.individual)
        toolbox.register('evaluate', self._evaluate_performance)
        toolbox.register('select',
                         tools.selTournament,
                         tournsize=self.vc_config.tournament_size)
        toolbox.register('mate', self._mate)
        toolbox.register('mutate', self._mutate)

        toolbox.register("map", parmap)

        return toolbox

    def _initialize_individuals(self, vc_config, cell_model):
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

    def _evaluate_performance(self, individual):
        """
        Evaluates performance of an individual compared to the target obj.

            Returns
            -------
                error: Number
                    The error between the trace generated by the individual's
                    parameter set and the baseline target objective.
        """
        individual = individual[0]
        
        primary_trace = None

        i = 1
        while primary_trace is None:
            primary_trace = self.get_model_response(individual,
                                                    self.protocol)
            if primary_trace is None:
                print("Individual errored. Returning 5")
                return 5
                print(f"Individual errored {i} times while generating current response")
                if self.previous_population is None:
                    return 5
                if i > 2:
                    individual = random.choice(self.previous_population)[0]
                else:
                    i += 1
                    primary_trace = None
                    new_individual = self.toolbox.select( self.previous_population, 1)
                    individual = new_individual[0][0]

        error = self.target.compare_individual(primary_trace)

        return error

    def _mate(self, i_one, i_two):
        """Performs crossover between two individuals.

        There may be a possibility no parameters are swapped. This probability
        is controlled by `self.config.gene_swap_probability`. Modifies
        both individuals in-place.

        Args:
            i_one: An individual in a population.
            i_two: Another individual in the population.
        """
        for key, val in i_one[0].default_parameters.items():
            if random.random() < self.vc_config.gene_swap_probability:
                i_one[0].default_parameters[key],\
                    i_two[0].default_parameters[key] = (
                        i_two[0].default_parameters[key],
                        i_one[0].default_parameters[key])

    def _mutate(self, individual):
        """Performs a mutation on an individual in the population.

        Chooses random parameter values from the normal distribution centered
        around each of the original parameter values. Modifies individual
        in-place.

        Args:
            individual: An individual to be mutated.
        """
        for key, val in individual[0].default_parameters.items():
            if random.random() < self.vc_config.gene_mutation_probability:
                new_param = -1

                while ((new_param < self.vc_config.params_lower_bound) and
                       (new_param < self.vc_config.params_upper_bound)):
                    new_param = individual[0].default_parameters[key] + \
                        np.random.normal(
                            individual[0].default_parameters[key] * .3)

                individual[0].default_parameters[key] = new_param

def generate_statistics(population: List[List[List[float]]]) -> None:
    fitness_values = [i.fitness.values[0] for i in population]
    print('  Min fitness: {}'.format(min(fitness_values)))
    print('  Max fitness: {}'.format(max(fitness_values)))

    print('  Average fitness: {}'.format(np.mean(fitness_values)))
    print('  Standard deviation: {}'.format(np.std(fitness_values)))

def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]
