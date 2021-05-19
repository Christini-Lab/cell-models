import context

from cell_models import protocols
from cell_models import kernik, paci_2018

import unittest
import matplotlib.pyplot as plt


class TestExpArtefact(unittest.TestCase):
    """Basic test cases."""

    def test_artefact_paci(self):
        """
        Test if the baseline model generates a valid response
        """

        simple_protocol = protocols.VoltageClampProtocol([
            protocols.VoltageClampStep(voltage=-80, duration=1000),
            protocols.VoltageClampStep(voltage=1, duration=1000)])

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

        for i, alpha in enumerate([0, .4, .7, .9]):
            baseline_model = paci_2018.PaciModel(is_exp_artefact=True, exp_artefact_params={'alpha': alpha})
            tr = baseline_model.generate_response(simple_protocol,
                    is_no_ion_selective=False)
            if i == 0:
                axs[0].plot(tr.t, tr.command_voltages, 'k')
            axs[0].plot(tr.t, tr.y)
            axs[1].plot(tr.t, tr.current_response_info.get_current_summed())
            axs[2].plot(tr.t, tr.current_response_info.get_current('I_Na'), label=f'alpha={alpha}')

        plt.legend()
        plt.show()

        #plt.plot(baseline_model.t, baseline_model.y[0, :])
        #plt.plot(baseline_model.t, baseline_model.y[26, :])
        #plt.show()

        plt.show()

    def test_artefact_kernik(self):
        """
        Test if the baseline model generates a valid response
        """

        simple_protocol = protocols.VoltageClampProtocol([
            protocols.VoltageClampStep(voltage=-80, duration=1000),
            protocols.VoltageClampStep(voltage=-40, duration=1000)])

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 8))

        for i, alpha in enumerate([0, .4, .7, .9]):
            if i == 0:
                baseline_model = kernik.KernikModel()
            else:
                baseline_model = kernik.KernikModel(is_exp_artefact=True, exp_artefact_params={'alpha': alpha})
            tr = baseline_model.generate_response(simple_protocol,
                    is_no_ion_selective=False)
            if i == 0:
                axs[0].plot(tr.t, tr.command_voltages, 'k')
            axs[0].plot(tr.t, tr.y)
            axs[1].plot(tr.t, tr.current_response_info.get_current_summed())
            axs[2].plot(tr.t, tr.current_response_info.get_current('I_Na'), label=f'alpha={alpha}')
            if i != 0:
                axs[3].plot(tr.t, tr.current_response_info.get_current('I_seal_leak'), label=f'alpha={alpha}')

        axs[2].legend()
        plt.show()

        #plt.plot(baseline_model.t, baseline_model.y[0, :])
        #plt.plot(baseline_model.t, baseline_model.y[26, :])
        #plt.show()

        plt.show()

    def test_kernik_paci_iv(self):
        """
        Test if the baseline model generates a valid response
        """
        proto = []
        for st in range(-7, 9):
            st += .1
            v = st*10
            proto.append(protocols.VoltageClampStep(voltage=-80.0, duration=300.0))
            proto.append(protocols.VoltageClampStep(voltage=v, duration=50.0))

        simple_protocol = protocols.VoltageClampProtocol(proto)

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 8))

        for i, alpha in enumerate([0, .9]):
            if i == 0:
                baseline_model = paci_2018.PaciModel()
            else:
                baseline_model = paci_2018.PaciModel(is_exp_artefact=True, exp_artefact_params={'alpha': alpha})
            tr = baseline_model.generate_response(simple_protocol,
                    is_no_ion_selective=False)
            if i == 0:
                axs[1].plot(tr.t, tr.current_response_info.get_current_summed())
            else:
                axs[1].plot(tr.t, tr.current_response_info.get_current('I_out'))
                axs[0].plot(baseline_model.t, baseline_model.y[24, :], label="V_P")
                axs[0].plot(baseline_model.t, baseline_model.y[25, :], label="V_clamp")
                axs[0].plot(baseline_model.t, baseline_model.y[27, :], label="V_cmd")
                axs[0].plot(baseline_model.t, baseline_model.y[28, :], label="V_est")
                axs[0].plot(tr.t, tr.y, label="Vm")

            axs[2].plot(tr.t, tr.current_response_info.get_current('I_Na'), label=f'alpha={alpha}')
            if i != 0:
                axs[3].plot(tr.t, tr.current_response_info.get_current('I_seal_leak'))

        axs[0].plot(tr.t, tr.command_voltages, 'k', label='Commands')

        axs[0].set_ylabel('Vm')
        axs[1].set_ylabel('I_out')
        axs[2].set_ylabel('INa')
        axs[3].set_ylabel('I_leak')
        axs[0].legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
