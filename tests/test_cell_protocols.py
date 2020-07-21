import context

from cell_models.ga import ga_configs, mp_voltage_clamp_optimization
from cell_models import protocols
from cell_models import kernik

import matplotlib.pyplot as plt

import unittest
import copy
import time


class TestCellProtocols(unittest.TestCase):
    """Basic test cases."""

    def test_spontaneous(self):
        """
        Test if the baseline model generates a valid response
        """
        protocol = protocols.SpontaneousProtocol(4000)
        
        baseline = kernik.KernikModel()
        tr = baseline.generate_response(protocol)
        data_baseline = tr.get_last_ap(is_peak=True)
        data_baseline.t = data_baseline.t - min(data_baseline.t)

        for current in ["G_Na", "P_CaL", "G_Kr", "G_Ks", "G_K1", "G_to"]:
            model = kernik.KernikModel(updated_parameters={current: 1.4})
            tr = model.generate_response(protocol)
            try:
                data = tr.get_last_ap(is_peak=True)
                data.t = data.t - min(data.t)
                plt.plot(data.t, data.V, label=current)
                plt.plot(data_baseline.t, data_baseline.V, label="Baseline")
                plt.legend()
                plt.show()
            except:
                import pdb
                pdb.set_trace()
        
        
        plt.legend()


if __name__ == '__main__':
    unittest.main()
