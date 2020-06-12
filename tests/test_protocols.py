import context

from cell_models.ga import ga_configs, mp_voltage_clamp_optimization
from cell_models import protocols
from cell_models import kernik

import unittest
import copy
import time


class TestGA(unittest.TestCase):
    """Basic test cases."""

    def test_vco_ga(self):
        """
        Test if the baseline model generates a valid response
        """
        voltage_bounds = (-120, 60)
        duration_bounds = (25, 1500)
        step_1 = protocols.VoltageClampStep()
        step_1.set_to_random_step(voltage_bounds, duration_bounds)
        step_2 = protocols.VoltageClampRamp()
        step_2.set_to_random_step(voltage_bounds, duration_bounds)
        step_3 = protocols.VoltageClampSinusoid()
        step_3.set_to_random_step(voltage_bounds, duration_bounds)

        protocol = protocols.VoltageClampProtocol([step_1, step_2, step_3])
        model = kernik.KernikModel()
        i_trace = model.generate_response(protocol)

        #Can plot with:
        #i_trace.plot...

        



if __name__ == '__main__':
    unittest.main()
