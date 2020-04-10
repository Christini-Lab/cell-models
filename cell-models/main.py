"""Main driver for program. Before running, make sure to have a directory
called `figures` for matplotlib pictures to be stored in."""

import copy
import time
import matplotlib.pyplot as plt

import protocols
from protocols import VoltageClampStep

from kernik import KernikModel
from paci_2018 import PaciModel


KERNIK_PROTOCOL = protocols.SpontaneousProtocol(2000)
PACI_PROTOCOL = protocols.SpontaneousProtocol(2)

def main():
    """Run parameter tuning or voltage clamp protocol experiments here
    """

    """Configures toolbox functions."""
    kernik_baseline = KernikModel()
    tr_b = kernik_baseline.generate_response(KERNIK_PROTOCOL)
    plt.plot(tr_b.t, tr_b.y)
    plt.show()

    paci_baseline = PaciModel()
    tr_bp = paci_baseline.generate_response(PACI_PROTOCOL)
    plt.plot(tr_bp.t, tr_bp.y)
    plt.show()





if __name__ == '__main__':
    main()
