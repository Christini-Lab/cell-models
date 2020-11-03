import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import pandas as pd
import os
from scipy import signal
import random
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import re
import math
import datetime


def extract_channel_data(data_h5, trial_number):
    trial_str = f'Trial{trial_number}'
    data = data_h5[trial_str]['Synchronous Data']['Channel Data'][()]
    return data


def plot_V_and_I(data, t_range, title, col=None):
    if col is None:
        col = 'b'
    if t_range is not None:
        idx_start = (data['Time (s)']-t_range[0]).abs().idxmin()
        idx_end = (data['Time (s)']-t_range[1]).abs().idxmin()
        data = data.copy().iloc[idx_start:idx_end, :]

    fig, axes = plt.subplots(2, 1, figsize=(10,8), sharex=True)
    if title is not None:
        fig.suptitle(title, fontsize=24)
    data['Voltage (V)'] = data['Voltage (V)'] * 1000

    axes[0].set_ylabel('Voltage (mV)', fontsize=20)
    axes[0].plot(data['Time (s)'], data['Voltage (V)'])
    axes[0].tick_params(labelsize=14)

    axes[1].set_ylabel('Current (pA/pF)', fontsize=20)
    axes[1].set_xlabel('Time (s)', fontsize=20)
    axes[1].plot(data['Time (s)'], data['Current'], col)
    axes[1].tick_params(labelsize=14)

    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    #axes[1].set_ylim([-.5, .5])
    plt.show()


def get_time_data(data_h5, trial_number):
    total_time, period = get_time_and_period(data_h5, trial_number)
    ch_data = extract_channel_data(data_h5, trial_number)

    time_array = np.arange(0, len(ch_data[:,0])) * period

    return time_array


def get_time_and_period(data_h5, trial_number):
    start_time, end_time = start_end_time(data_h5, trial_number)
    trial_str = f'Trial{trial_number}'
    total_time = (end_time - start_time) / 1E9
    period = data_h5[trial_str]['Period (ns)'][()] / 1E9

    return total_time, period


def start_end_time(data_h5, trial_number):
    trial_str = f'Trial{trial_number}'
    start_time = data_h5[trial_str]['Timestamp Start (ns)'][()]
    end_time = data_h5[trial_str]['Timestamp Stop (ns)'][()]
    return start_time, end_time


def get_current_and_voltage(f, trial):
    ch_data = f[f'Trial{trial}']['Synchronous Data']['Channel Data'][()]

    channel_1 = ch_data[:, 0]
    channel_2 = ch_data[:, 1]

    channel_1_test = channel_1[np.logical_not(np.isnan(channel_1))]
    channel_2_test = channel_2[np.logical_not(np.isnan(channel_2))]

    if np.abs(channel_1_test).mean() == 0:
        current = channel_1
        voltage = channel_2

    if np.abs(channel_1_test).std() < np.abs(channel_2_test).mean():
        current = channel_1
        voltage = channel_2
    else:
        current = channel_2
        voltage = channel_1

    is_current_clamp = False

    for k, v in f[f'Trial{trial}']['Parameters'].items():
        parameter_values = v.value

        for p in parameter_values:
            if 'Pace' in k:
                if p[1] != 0:
                    is_current_clamp = True
    
    is_voltage_max_in_range = (voltage.max() > .085) and (voltage.max() < .11)
    is_voltage_min_in_range = (voltage.min() > -.21) and (voltage.min() < -.19)

    if is_voltage_max_in_range  and is_voltage_min_in_range:
        is_current_clamp = False

    if is_current_clamp:
        current = -current

    return current, voltage


def get_exp_as_df(data_h5, trial_number, cm):
    """I was going to save the time, voltage and current as a csv,
    but decided not to, because there can be >3million points in 
    the h5 dataset. If you want to make comparisons between trials or
    experiments, call this multiple times.
    """
    current, voltage = get_current_and_voltage(data_h5, trial_number)

    t_data = get_time_data(data_h5, trial_number)
    d_as_frame = pd.DataFrame({'Time (s)': t_data,
                               'Voltage (V)': voltage,
                               'Current': current / cm})

    return d_as_frame


def plot_recorded_data(recorded_data, trial_number, does_plot=False, t_range=None, title=None, col=None):

    if does_plot:
        plot_V_and_I(recorded_data, t_range, title=title, col=col)

    return recorded_data


def get_tags(f, trial_number):
    tags = []
    
    print(f'Trial {trial_number} tags:')

    for tag in f['Tags'].keys():
        raw_tag = f['Tags'][tag].value[0]
        date, tag_text = str(raw_tag).replace("'", "").replace('b',
                                                            '').split(',', 1)
        trial_date = str(f[f'Trial{trial_number}']['Date'].value).replace(
            "'", '').replace('b', '')
        trial_length = f[f'Trial{trial_number}']['Trial Length (ns)'].value / 1E9



        acquisition_date = datetime.datetime.fromtimestamp(int(date)/1E9)
        acquisition_delta = datetime.timedelta(days=18470, seconds=329)
        tag_datetime = acquisition_date + acquisition_delta

        trial_datetime = datetime.datetime(int(trial_date[0:4]),
                                           int(trial_date[5:7]),
                                           int(trial_date[8:10]),
                                           int(trial_date[11:13]),
                                           int(trial_date[14:16]),
                                           int(trial_date[17:19]))

        tag_time_after_trial_start = (tag_datetime - trial_datetime
                                      ).total_seconds()

        if tag_time_after_trial_start < 0:
            continue
        if tag_time_after_trial_start < trial_length:
            tags.append([tag_time_after_trial_start, tag_text])

            print(f'\tAt {tag_time_after_trial_start}: {tag_text}')


    if len(tags) == 0:
        print('No Tags')

    return tags


def print_parameters(f, trial_number):
    parameters = {}

    sampling_frequency = 1 / (f[f'Trial{trial_number}']['Period (ns)'].value / 1E9)

    initial_conditions = []
    added_conditions = {}

    for k, v in f[f'Trial{trial_number}']['Parameters'].items():
        parameter_values = v.value

        for p in parameter_values:
            try:
                if p[0] == 0:
                    initial_conditions.append(f'{k} equal to {p[1]}')
                else:
                    if k not in added_conditions.keys():
                        added_conditions[k] = []
                        added_conditions[k].append(
                                f'Equal to {p[1]} at {p[0]/sampling_frequency}.')
            except:
                continue
    
    print(f'Trial {trial_number} Initial Conditions')


    for val in initial_conditions:
        print(f'\t{val}')

    print(f'Trial {trial_number} Condition Changes')

    for k, v in added_conditions.items():
        print(f'\t{k} set to: ')
        for change in v:
            print(f'\t\t{change}')


def explore_data(file_path, col=None):
    f = h5py.File(file_path, 'r')
    does_plot = True

    trial_names = []
    is_tags = False

    for k, v in f.items():
        if 'Trial' in k:
            trial_names.append(k)
        if k == 'Tags':
            print('There are tags')
            is_tags = True

    cm_key = [key for key in f['Trial1']['Parameters'].keys() if 'Cm' in key][0]
    cm = f['Trial1']['Parameters'][cm_key].value[0][1] * 1E-12

    print(trial_names)

    trial_number = input(f"Which trial number would you like to view? Type a number between 1 and {len(trial_names)}. Type 'all' if you want to view each one in succession. ")
    if trial_number == 'all':
        trial_range = range(1, len(trial_names) + 1)
    else:
        trial_range = range(int(trial_number), int(trial_number) + 1)

    for trial in trial_range:
        print_parameters(f, trial)

        if is_tags:
            get_tags(f, trial)

        recorded_data = get_exp_as_df(f, trial, cm)
        
        plot_recorded_data(recorded_data, trial, does_plot, title=f'Trial {trial}', col=col)

