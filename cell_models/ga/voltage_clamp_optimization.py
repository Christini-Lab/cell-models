"""Module level comment."""

import copy
import random
from typing import List

import numpy as np

from cell_models.ga import ga_configs, genetic_algorithm_results
from cell_models import protocols


class VCOGeneticAlgorithm:

    def __init__(self, config: ga_configs.VoltageOptimizationConfig):
        self.config = config

    def run(self):
        population = self._init_population()

        ga_result = genetic_algorithm_results.GAResultVoltageClampOptimization(
            config=self.config)
        ga_result.generations.append(population)

        print('\tEvaluating initial population.')
        for individual in population:
            individual.fitness = self._evaluate(individual=individual)
        
        for generation in range(1, self.config.max_generations):
            print('\tGeneration {}'.format(generation))
            i_max = max(population)
            i_max_index = population.index(i_max)
            if i_max_index == 0:
                i_rand = population[1]
            else:
                i_rand = population[0]

            population.remove(i_max)
            population.remove(i_rand)

            population = self._select(population=population)

            for i_one, i_two in zip(population[::2], population[1::2]):
                if random.random() < self.config.mate_probability:
                    self._mate(i_one=i_one, i_two=i_two)

            for individual in population:
                if random.random() < self.config.mutate_probability:
                    self._mutate(individual=individual)

            population.append(i_max)
            population.append(i_rand)

            # Update fitness of all individuals in population.
            for individual in population:
                individual.fitness = self._evaluate(individual=individual)

            ga_result.generations.append(population)
            generate_statistics(population)

        return ga_result

    def _evaluate(
            self,
            individual: genetic_algorithm_results.VCOptimizationIndividual
    ) -> int:
        """Evaluates the fitness of an individual.

        Fitness is determined by how well the voltage clamp protocol isolates
        individual ionic currents.
        """
        return individual.evaluate(config=self.config)

    def _mate(
            self,
            i_one: genetic_algorithm_results.VCOptimizationIndividual,
            i_two: genetic_algorithm_results.VCOptimizationIndividual) -> None:
        """Mates two individuals, modifies them in-place."""
        if len(i_one.protocol.steps) != len(i_two.protocol.steps):
            raise ValueError('Individuals do not have the same num of steps.')

        for i in range(len(i_one.protocol.steps)):
            # Do not mutate first step, which is the holding step.
            if i == 0:
                continue

            if random.random() < self.config.gene_swap_probability:
                i_one.protocol.steps[i], i_two.protocol.steps[i] = (
                    i_two.protocol.steps[i], i_one.protocol.steps[i])

    def _mutate(
            self,
            individual: genetic_algorithm_results.VCOptimizationIndividual
    ) -> None:
        """Mutates an individual by choosing a number for norm. distribution."""
        for i in range(len(individual.protocol.steps)):
            # Do not mutate first step, which is the holding step.
            if i == 0:
                continue

            if random.random() < self.config.gene_mutation_probability:
                # If np.random.normal exceeds the bounds set in config, then
                # set them to the bounds.
                v_bounds = self.config.step_voltage_bounds
                d_bounds = self.config.step_duration_bounds

                # Standard deviation of normal distribution is set to the range
                # of possible values / 3.
                new_voltage_offset = np.random.normal(
                    loc=0,
                    scale=abs(v_bounds[0] - v_bounds[1]) / 3)

                individual.protocol.steps[i].voltage += new_voltage_offset
                while ((individual.protocol.steps[i].voltage > v_bounds[1]) or
                       (individual.protocol.steps[i].voltage < v_bounds[0])):
                    new_voltage_offset = np.random.normal(
                        loc=0,
                        scale=abs(v_bounds[0] - v_bounds[1]) / 3)
                    individual.protocol.steps[i].voltage += new_voltage_offset

                new_duration_offset = np.random.normal(
                    loc=0,
                    scale=abs(d_bounds[0] - d_bounds[1]) / 3)
                individual.protocol.steps[i].duration += new_duration_offset
                while ((individual.protocol.steps[i].duration > d_bounds[1]) or
                       (individual.protocol.steps[i].duration < d_bounds[0])):
                    new_duration_offset = np.random.normal(
                        loc=0,
                        scale=abs(d_bounds[0] - d_bounds[1]) / 3)
                    individual.protocol.steps[i].duration += new_duration_offset

    def _select(
            self,
            population: List[
                genetic_algorithm_results.VCOptimizationIndividual]
    ) -> List[genetic_algorithm_results.VCOptimizationIndividual]:
        """Selects a list of individuals using tournament selection."""
        new_population = []
        for i in range(len(population)):
            tournament = random.sample(
                population,
                k=self.config.tournament_size)
            best_individual = max(tournament, key=lambda j: j.fitness)
            new_population.append(copy.deepcopy(best_individual))
        return new_population

    def _init_individual(self):
        """Initializes a individual with a randomized protocol."""
        steps = []
        for i in range(self.config.steps_in_protocol):
            random_step = protocols.VoltageClampStep(
                voltage=random.uniform(*self.config.step_voltage_bounds),
                duration=random.uniform(*self.config.step_duration_bounds))
            steps.append(random_step)
        return genetic_algorithm_results.VCOptimizationIndividual(
            protocol=protocols.VoltageClampProtocol(steps=steps),
            fitness=0)

    def _init_population(self):
        return [
            self._init_individual() for _ in range(self.config.population_size)
        ]


def generate_statistics(population):
    fitness_values = [i.fitness for i in population]
    print('\t\tMin fitness: {}'.format(min(fitness_values)))
    print('\t\tMax fitness: {}'.format(max(fitness_values)))
    print('\t\tAverage fitness: {}'.format(np.mean(fitness_values)))
    print('\t\tStandard deviation: {}'.format(np.std(fitness_values)))


