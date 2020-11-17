import context

from cell_models import kernik, paci_2018, protocols
from cell_models.ga import target_objective
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle
from random import choice
from string import ascii_uppercase
from os import listdir
import csv
import numpy as np


def plot_baseline_vs_average():
    baseline = kernik.KernikModel(is_exp_artefact=True)
    average = kernik.KernikModel(model_kinetics_type='Average',
            is_exp_artefact=True)

    proto = pickle.load(open('../run_vc_ga/results/trial_steps_ramps_200_50_4_-120_60/shortened_trial_steps_ramps_200_50_4_-120_60_500_artefact_True_short.pkl', 'rb'))

    tr_baseline = baseline.generate_response(proto, is_no_ion_selective=False)
    tr_avg = average.generate_response(proto, is_no_ion_selective=False)

    fig, axs = plt.subplots(2, 1, sharex=True)
    labels = ['Baseline', 'Average']

    for i, tr in enumerate([tr_baseline, tr_avg]):
        axs[0].plot(tr.t, tr.y)
        axs[1].plot(tr.t, tr.current_response_info.get_current_summed(),
                label=labels[i])

    plt.legend()
    plt.show()


def plot_baseline_vs_random():
    baseline = kernik.KernikModel(is_exp_artefact=True)
    rand = kernik.KernikModel(model_kinetics_type='Random',
            model_conductances_type='Random',
            is_exp_artefact=True)

    proto = pickle.load(open('../run_vc_ga/results/trial_steps_ramps_200_50_4_-120_60/shortened_trial_steps_ramps_200_50_4_-120_60_500_artefact_True_short.pkl', 'rb'))

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


def compare_random_kernik_to_paci(path_to_res):
    # Setup target
    target_ranges = {'ALL': [0, 9060],
                     'I_Kr': [1255, 1275],
                     'I_CaL': [1965, 1985],
                     'I_Na': [2750, 2780],
                     'I_To': [3620, 3650],
                     'I_K1': [4280, 4310],
                     'I_F': [5810, 5850],
                     'I_Ks': [7500, 9060]}
    proto = pickle.load(open('shortened_trial_steps_ramps_200_50_4_-120_60_500_artefact_True_short.pkl', 'rb'))
    paci_baseline = paci_2018.PaciModel(is_exp_artefact=True)

    if 'paci_baseline_target.pkl' not in listdir('./'):
        paci_baseline.find_steady_state(max_iters=20)
        paci_target = target_objective.create_target_from_protocol(paci_baseline,
                proto, times_to_compare=target_ranges)
    else:
        paci_target = pickle.load(open('./paci_baseline_target.pkl', 'rb'))

    pickle.dump(paci_target, open('paci_baseline_target.pkl', 'wb'))

    rand_individual = kernik.KernikModel(model_kinetics_type='Random',
            model_conductances_type='Random',
            is_exp_artefact=True)

    try:
        ind_errors = paci_target.compare_individual(rand_individual,
                prestep=10000,
                return_all_errors=True)
    except:
        ind_errors = 0

    rand_file_name = ''.join(choice(ascii_uppercase) for i in range(16))

    if ind_errors == 0:
        rand_file_name = rand_file_name + '_FAILED'

    if 'targets.pkl' not in listdir(path_to_res):
        with open(f'{path_to_res}/target.csv', 'w') as f:
            for key in target_ranges.keys():
                f.write("%s,%s,%s\n"%(key, target_ranges[key][0],
                    target_ranges[key][1]))

    with open(f'{path_to_res}/conductances/{rand_file_name}.csv', 'w') as f:
        for key in rand_individual.default_parameters.keys():
            f.write("%s,%s\n"%(key, rand_individual.default_parameters[key]))

    if ind_errors == 0:
        ind_errors = np.asarray([ind_errors])
    else:
        ind_errors = np.asarray(ind_errors)
        
    np.savetxt(f'{path_to_res}/target_errors/{rand_file_name}.csv',
                ind_errors)
    np.savetxt(f'{path_to_res}/kinetics/{rand_file_name}.csv',
                rand_individual.kinetics)
    

def compare_to_paci_keep_trace():
    prestep = 5000
    prestep_proto = protocols.VoltageClampProtocol([protocols.VoltageClampStep(voltage=-80, duration=prestep)])
    rand_individual.generate_response(prestep_proto, is_no_ion_selective=False)
    rand_individual.y_ss = rand_individual.y[:, -1]

    individual_tr = rand_individual.generate_response(
                        proto, is_no_ion_selective=False)

    scale = 1

    ind_time = individual_tr.t * scale
    ind_current = individual_tr.current_response_info.get_current_summed()

    max_simulated_t = ind_time.max()
    freq = 10
    max_exp_index = int(round(freq * max_simulated_t)) - 1

    t_interp = paci_target.time[0:max_exp_index]
    f = interp1d(ind_time, ind_current)

    ind_interp_current = f(t_interp)

    errors = paci_target.calc_errors_in_ranges(ind_interp_current,
        paci_target.current[0:max_exp_index],
        return_all_errors=True)


def main():
    #plot_baseline_vs_average()
    #plot_baseline_vs_random()
    path_to_results = 'results/trial_1'
    for i in range(0, 10):
        compare_random_kernik_to_paci(path_to_results)

if __name__ == '__main__':
    main()
