import context

from cell_models import protocols
from cell_models import kernik, paci_2018

import unittest
import matplotlib.pyplot as plt
from computational_methods.plot_figures import plt_mult_frame


class TestKernik(unittest.TestCase):
    """Basic test cases."""

    def test_baseline_vs_avg(self):
        """
        Test if the baseline model generates a valid response
        """
        model_baseline = kernik.KernikModel()
        model_average = kernik.KernikModel(model_kinetics_type='Average', model_conductances_type='Average')

        spontaneous_protocol = protocols.SpontaneousProtocol(1000)

        tr_baseline = model_baseline.generate_response(spontaneous_protocol, is_no_ion_selective=False)
        tr_average = model_average.generate_response(spontaneous_protocol,
                is_no_ion_selective=False)

        for tr in [tr_baseline, tr_average]:
            plt.plot(tr.t, tr.y)

        plt.show()

    def test_baseline_vs_avg_vc(self):
        """
        Test if the baseline model generates a valid response
        """
        simple_protocol = protocols.VoltageClampProtocol([
            protocols.VoltageClampStep(voltage=-80, duration=5000),
            protocols.VoltageClampStep(voltage=40, duration=8000),
            protocols.VoltageClampStep(voltage=-80, duration=4000),
            protocols.VoltageClampStep(voltage=100, duration=8000)
            ])

        model_baseline = kernik.KernikModel()
        model_average = kernik.KernikModel(model_kinetics_type='Average', model_conductances_type='Average')

        print(model_baseline.conductances)
        print(model_average.conductances)

        tr_baseline = model_baseline.generate_response(simple_protocol, is_no_ion_selective=False)
        tr_average = model_average.generate_response(simple_protocol,
                is_no_ion_selective=False)

        fig, axs = plt.subplots(2, 1, sharex=True)
        labels = ['Baseline', 'Average']
        for i, tr in enumerate([tr_baseline, tr_average]):
            axs[0].plot(tr.t, tr.y)
            axs[1].plot(tr.t, tr.current_response_info.get_current_summed(), label=labels[i])
            
        plt.legend()
        plt.show()

    def test_baseline_vs_rand(self):
        rand = kernik.KernikModel(model_kinetics_type='Random',
                model_conductances_type='Random',
                is_exp_artefact=True)
        baseline = kernik.KernikModel(is_exp_artefact=True)

        proto = protocols.VoltageClampProtocol([
            protocols.VoltageClampStep(voltage=-80, duration=5000),
            protocols.VoltageClampStep(voltage=40, duration=8000),
            protocols.VoltageClampStep(voltage=-80, duration=4000),
            protocols.VoltageClampStep(voltage=100, duration=8000)
            ])

        tr_baseline = baseline.generate_response(proto, is_no_ion_selective=False)
        tr_rand = rand.generate_response(proto, is_no_ion_selective=False)

        fig, axs = plt.subplots(2, 1, sharex=True)
        labels = ['Baseline', 'Random']

        for i, tr in enumerate([tr_baseline, tr_rand]):
            axs[0].plot(tr.t, tr.y)
            axs[1].plot(tr.t, tr.current_response_info.get_current_summed(),
                    label=labels[i])

        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
