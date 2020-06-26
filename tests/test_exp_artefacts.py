import context

from cell_models import protocols
from cell_models import kernik

import unittest
import matplotlib.pyplot as plt


class TestExpArtefact(unittest.TestCase):
    """Basic test cases."""

    def test_artefact(self):
        """
        Test if the baseline model generates a valid response
        """

        baseline_model = kernik.KernikModel(is_exp_artefact=True)
        simple_protocol = protocols.VoltageClampProtocol([
            protocols.VoltageClampStep(voltage=-80, duration=100),
            protocols.VoltageClampStep(voltage=1, duration=100)])
            #protocols.VoltageClampStep(voltage=-80, duration=500),
            #protocols.VoltageClampStep(voltage=80, duration=500)])

        tr = baseline_model.generate_response(simple_protocol)

        #plt.plot(baseline_model.t, baseline_model.y[24,:])
        #plt.show()

        #tr.plot_with_individual_currents(['I_Cm', 'I_Cp'], with_artefacts=True)
        tr.plot_with_individual_currents(['I_Na'], with_artefacts=True)

        #plt.plot(baseline_model.t, baseline_model.y[0, :])
        #plt.plot(baseline_model.t, baseline_model.y[26, :])
        #plt.show()

        plt.show()

if __name__ == '__main__':
    unittest.main()
