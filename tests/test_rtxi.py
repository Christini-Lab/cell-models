import context

from cell_models.rtxi.create_metadata_object import RTXIMeta
from cell_models.rtxi.target_objective import create_target_objective

import matplotlib.pyplot as plt

import unittest
import pickle
from os import listdir, mkdir


class TestRTXIExploration(unittest.TestCase):

    def test_data_exploration(self):
        """Simply test whether I can explore a cell dataset"""

        file_path = 'results/cell_2_072720.h5'
        explore_data(file_path)


class TestRTXITargets(unittest.TestCase):

    def test_rtxi_metadata(self):
        """Simply test whether I can explore a cell dataset"""

        file_path = 'results/cell_2_072720'
        trial = 3
        t_range = [10.6, 23.375]
        protocol_type = 'Voltage Clamp'
        cm = 65.2E-9

        meta_vc = RTXIMeta(file_path=f'{file_path}/raw_cell_data.h5',
                           trial=trial,
                           t_range=t_range,
                           protocol_type=protocol_type,
                           mem_capacitance=cm)

        trial = 1
        t_range = [120, 140]
        protocol_type = 'Dynamic Clamp'

        meta_dc = RTXIMeta(file_path=f'{file_path}/raw_cell_data.h5',
                           trial=trial,
                           t_range=t_range,
                           protocol_type=protocol_type,
                           mem_capacitance=cm)

        save_rtxi_metadata(file_path, [meta_vc, meta_dc])

    def test_rtxi_targets(self):
        """Simply test whether I can explore a cell dataset"""
        meta_file_location = 'results/cell_2_072720/meta_targets/meta_trial_3_t_10.6-23.375.pkl'

        target_meta = pickle.load(open(meta_file_location, 'rb'))

        target = create_target_objective(target_meta)

        target.plot_data()


def save_rtxi_metadata(cell_path, rtxi_meta_list):
    if not 'meta_targets' in listdir(cell_path):
        mkdir(f'{cell_path}/meta_targets')

    for meta in rtxi_meta_list:
        f = open(f'{cell_path}/meta_targets/meta_trial_{meta.trial}_t_{meta.t_range[0]}-{meta.t_range[1]}.pkl', 'wb')
        pickle.dump(meta, f)

         



if __name__ == '__main__':
    unittest.main()
