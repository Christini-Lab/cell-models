import context

from cell_models.ga import mp_parameter_tuning
from cell_models.ga import ga_configs
from cell_models import protocols
import pickle
import matplotlib.pyplot as plt

import unittest

class TestGAResults(unittest.TestCase):
    """Basic test cases."""
    def setUp(self):
        self.ga_results = pickle.load(open("ga_results_example", "rb"))

    def test_ga_results(self):
        """
        Test if the baseline model generates a valid response

        """
        pass
        #individual = self.ga_results.all_individuals[0][0]
        #self.ga_results.graph_individual(individual)
        #plt.show()

    def test_plot_params(self):
        """
        Test if the baseline model generates a valid response
        """
        individual = self.ga_results.all_individuals[0][0]
        self.ga_results.graph_individual_param_set(individual)
        plt.show()

if __name__ == '__main__':
    unittest.main()
