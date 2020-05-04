"""Main driver for program. Before running, make sure to have a directory
called `figures` for matplotlib pictures to be stored in."""

import copy
import time
import matplotlib.pyplot as plt

from cell_models import protocols
from cell_models.protocols import VoltageClampStep

from cell_models.kernik import KernikModel
from cell_models.paci_2018 import PaciModel
from cell_models.ohara_rudy import OharaRudyModel

# Spontaneous / Stimulated
def spontaneous_example():
    """
    Plots spontaneous Kernik/Paci and stimulated O'Hara Rudy
    """
    KERNIK_PROTOCOL = protocols.SpontaneousProtocol(2000)
    kernik_baseline = KernikModel()
    tr_b = kernik_baseline.generate_response(KERNIK_PROTOCOL)
    plt.plot(tr_b.t, tr_b.y)
    plt.show()

    PACI_PROTOCOL = protocols.SpontaneousProtocol(2)
    paci_baseline = PaciModel()
    tr_bp = paci_baseline.generate_response(PACI_PROTOCOL)
    plt.plot(tr_bp.t, tr_bp.y)
    plt.show()

    OHARA_RUDY = protocols.PacedProtocol(model_name="OR")
    or_baseline = OharaRudyModel()
    tr = or_baseline.generate_response(OHARA_RUDY)
    plt.plot(tr.t, tr.y)
    plt.show()

# Update Parameters
def example_update_params():
    """
    Plots baseline Kernik vs updated. You can see the names of all updateable
    parameters in the KernikModel() __init__.
    """
    KERNIK_PROTOCOL = protocols.SpontaneousProtocol(2000)
    
    kernik_baseline = KernikModel()
    kernik_updated = KernikModel(
            updated_parameters={'G_K1': 1.2, 'G_Kr': 0.8, 'G_Na':2.2})
    tr_baseline = kernik_baseline.generate_response(KERNIK_PROTOCOL)
    tr_updated =  kernik_updated.generate_response(KERNIK_PROTOCOL)
    plt.plot(tr_baseline.t, tr_baseline.y, label="Baseline")
    plt.plot(tr_updated.t, tr_updated.y, label="Updated")
    plt.legend()
    plt.show()

# Run parameter tuning experiment
def run_parameter_tuning():
    from cell_models.ga.parameter_tuning import ParameterTuningGeneticAlgorithm
    from cell_models.ga import ga_configs

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

    KERNIK_PROTOCOL = protocols.VoltageClampProtocol()

    VC_CONFIG = ga_configs.ParameterTuningConfig(
        population_size=30,
        max_generations=10,
        protocol=KERNIK_PROTOCOL,
        tunable_parameters=KERNIK_PARAMETERS,
        params_lower_bound=0.1,
        params_upper_bound=10,
        mate_probability=0.9,
        mutate_probability=0.9,
        gene_swap_probability=0.2,
        gene_mutation_probability=0.2,
        tournament_size=4)

    res_kernik = ParameterTuningGeneticAlgorithm('Kernik',
                                                 VC_CONFIG,
                                                 KERNIK_PROTOCOL)


def main():
    run_parameter_tuning()

if __name__ == '__main__':
    main()
