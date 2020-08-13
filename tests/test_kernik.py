import context

from cell_models import protocols
from cell_models import kernik

import unittest
import matplotlib.pyplot as plt


class TestKernik(unittest.TestCase):
    """Basic test cases."""

    def test_baseline_artefact(self):
        """
        Test if the baseline model generates a valid response
        """
        model_baseline = kernik.KernikModel()
        model_baseline_artefact = kernik.KernikModel(
            updated_parameters={"I_Na": 2},
            is_exp_artefact=True)
        model_new_leak = kernik.KernikModel(
            updated_parameters={"I_Na": 2, "G_seal_leak": 6},
            is_exp_artefact=True)
        model_new_v_off = kernik.KernikModel(
            updated_parameters={"I_Na": 2, "V_off": 3},
            is_exp_artefact=True)


        spontaneous_protocol = protocols.SpontaneousProtocol(1500)

        simple_protocol = protocols.VoltageClampProtocol([
            protocols.VoltageClampStep(voltage=-80, duration=1000),
            protocols.VoltageClampStep(voltage=1, duration=1000)])

        tr_baseline = model_baseline.generate_response(spontaneous_protocol)
        tr_base = model_baseline_artefact.generate_response(simple_protocol)
        tr_leak = model_new_leak.generate_response(simple_protocol)
        tr_v_off = model_new_v_off.generate_response(simple_protocol)

        for tr in [tr_base, tr_leak, tr_v_off]:
            plt.plot(tr.t, tr.y)

        plt.show()

        plt.plot(tr_base.t, tr_base.current_response_info.get_current_summed())
        plt.plot(tr_leak.t, tr_leak.current_response_info.get_current_summed())
        plt.plot(tr_v_off.t, tr_v_off.current_response_info.get_current_summed())

        plt.show()


if __name__ == '__main__':
    unittest.main()
