"""Contains classes used to configure the genetic algorithm."""
from typing import List, Tuple

from cell_models import protocols


class Parameter:
    """Represents a parameter in the model.

    Attributes:
        name: Name of parameter.
        default_value: Default value of parameter.
    """

    def __init__(self, name: str, default_value: float) -> None:
        self.name = name
        # Do not change default value once set during init.
        self.default_value = default_value

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: 'Parameter') -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__


class GeneticAlgorithmConfig:
    """Contains hyperparameters for configuring a genetic algorithm.

    Attributes:
        population_size: Size of the population in each generation.
        max_generations: Max number of generations to run the algorithm for.
        mate_probability: The probability two individuals will `mate`.
        gene_swap_probability: The probability a parameter, or `gene`, will
            be swapped between a pair of `mated` individuals.
        gene_mutation_probability: Probability a certain gene will be mutated:
            replaced with a random number from a normal distribution centered
            around the value of the gene.
        tournament_size: Number of individuals chosen during each round of
            tournament selection.
    """

    def __init__(self,
                 population_size: int,
                 max_generations: int,
                 mate_probability: float,
                 mutate_probability: float,
                 gene_swap_probability: float,
                 gene_mutation_probability: float,
                 tournament_size: int) -> None:
        self.population_size = population_size
        self.max_generations = max_generations
        self.mate_probability = mate_probability
        self.mutate_probability = mutate_probability
        self.gene_swap_probability = gene_swap_probability
        self.gene_mutation_probability = gene_mutation_probability
        self.tournament_size = tournament_size

    def has_equal_hyperparameters(self,
                                  other: 'GeneticAlgorithmConfig') -> bool:
        return (self.population_size == other.population_size and
                self.max_generations == other.max_generations and
                self.mate_probability == other.mate_probability and
                self.mutate_probability == other.mutate_probability and
                self.gene_swap_probability == other.gene_swap_probability and
                self.gene_mutation_probability ==
                other.gene_mutation_probability and
                self.tournament_size == other.tournament_size)


class ParameterTuningConfig(GeneticAlgorithmConfig):
    """Config for a parameter tuning genetic algorithm.

    Attributes:
        protocol: Object representing the specific target objective of the
            genetic algorithm.
        tunable_parameters: List of strings representing the names of parameters
            that will be tuned.
        params_lower_bound: A float representing the lower bound a randomized
            parameter value can be for any individual in the population. For
            example, if the default parameter value is 100, and
            params_lower_bound is 0.1, than 10 is smallest value that parameter
            can be set to.
        params_upper_bound: A float representing the upper bound a randomized
            parameter value can be for any individual in the population. See
            the description of `params_lower_bound` for more info.
        secondary_protocol: A secondary protocol used for a combined protocol.
    """

    # If a model with an individual's parameter set fails to generate a trace,
    # the individual will have it's fitness set to one of the following,
    # according to the protocol.
    SAP_MAX_ERROR = 100
    IP_MAX_ERROR = 130
    VC_MAX_ERROR = 130

    def __init__(self,
                 targets: dict,
                 model_name: str,
                 cell_model,
                 params_lower_bound: float,
                 params_upper_bound: float,
                 tunable_parameters: List[Parameter],
                 population_size: int,
                 max_generations: int,
                 mate_probability: float,
                 mutate_probability: float,
                 gene_swap_probability: float,
                 gene_mutation_probability: float,
                 tournament_size: int,
                 secondary_protocol: protocols.PROTOCOL_TYPE=None,
                 target_params=None,
                 with_exp_artefact=False,
                 ) -> None:
        super().__init__(
            population_size=population_size,
            max_generations=max_generations,
            mate_probability=mate_probability,
            mutate_probability=mutate_probability,
            gene_swap_probability=gene_swap_probability,
            gene_mutation_probability=gene_mutation_probability,
            tournament_size=tournament_size)
        self.targets = targets 
        self.params_lower_bound = params_lower_bound
        self.params_upper_bound = params_upper_bound
        self.tunable_parameters = tunable_parameters
        self.secondary_protocol = secondary_protocol
        self.target_params = target_params
        self.with_exp_artefact = with_exp_artefact
        self.model_name = model_name
        self.cell_model = cell_model

    def has_equal_hyperparameters(self, other: 'ParameterTuningConfig') -> bool:
        return (super().has_equal_hyperparameters(other=other) and
                self.params_lower_bound == other.params_lower_bound and
                self.params_upper_bound == other.params_upper_bound)


def get_appropriate_max_error(protocol: protocols.PROTOCOL_TYPE) -> int:
    if isinstance(protocol, protocols.SingleActionPotentialProtocol):
        return ParameterTuningConfig.SAP_MAX_ERROR
    elif isinstance(protocol, protocols.IrregularPacingProtocol):
        return ParameterTuningConfig.IP_MAX_ERROR
    elif isinstance(protocol, protocols.VoltageClampProtocol):
        return ParameterTuningConfig.VC_MAX_ERROR


class VoltageOptimizationConfig(GeneticAlgorithmConfig):
    """Config for a voltage optimization genetic algorithm.

    Attributes:
        window: Window of time over which the fraction contribution
            of each channel is calculated.
        step_size: Step size when calculating windows over which the fraction
            contribution of each channel is calculated.
        steps_in_protocol: Locked number of steps in a generated voltage clamp
            protocol.
        step_duration_bounds: The bounds from which the duration of a step can
            be randomly initialized.
        step_voltage_bounds: The bounds from which the voltage of a step can be
            randomly initialized.
    """

    def __init__(self,
                 window: float,
                 step_size: float,
                 steps_in_protocol: int,
                 step_duration_bounds: Tuple[float, float],
                 step_voltage_bounds: Tuple[float, float],
                 population_size: int,
                 max_generations: int,
                 mate_probability: float,
                 mutate_probability: float,
                 gene_swap_probability: float,
                 gene_mutation_probability: float,
                 tournament_size: int,
                 target_current: str = None,
                 step_types = ["step", "ramp", "sinusoid"],
                 with_artefact=False,
                 model_name='Kernik'):
        super().__init__(
            population_size=population_size,
            max_generations=max_generations,
            mate_probability=mate_probability,
            mutate_probability=mutate_probability,
            gene_swap_probability=gene_swap_probability,
            gene_mutation_probability=gene_mutation_probability,
            tournament_size=tournament_size)
        self.window = window
        self.step_size = step_size
        self.steps_in_protocol = steps_in_protocol
        self.step_duration_bounds = step_duration_bounds
        self.step_voltage_bounds = step_voltage_bounds
        self.target_current = target_current
        self.step_types = step_types
        self.with_artefact = with_artefact
        self.model_name = model_name


class CombinedVCConfig:
    """Config for building a VC protocol from smaller VC protocols.

    Attributes:
        currents: A list of currents to use during optimization.
        step_range: When building smaller VC protocols, vary steps between
            `step_range` when running the genetic algorithm.
        adequate_fitness_threshold: When building smaller VC protocols, return
            the protocol if its fitness is greater than this, else continue
            using GA with larger number of steps.
        ga_config: A config file storing genetic algorithm hyperparameters.
    """

    def __init__(self,
                 currents: List[str],
                 step_range: range,
                 adequate_fitness_threshold: float,
                 ga_config: VoltageOptimizationConfig) -> None:
        self.currents = currents
        self.step_range = step_range
        self.adequate_fitness_threshold = adequate_fitness_threshold
        self.ga_config = ga_config
