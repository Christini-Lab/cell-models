"""Main driver for program. Before running, make sure to have a directory
called `figures` for matplotlib pictures to be stored in."""

import copy
import time
import matplotlib.pyplot as plt

import protocols
from protocols import VoltageClampStep

from kernik import KernikModel
from paci_2018 import PaciModel
from ohara_rudy import OharaRudyModel

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


def main():
    example_update_params()


if __name__ == '__main__':
    main()

