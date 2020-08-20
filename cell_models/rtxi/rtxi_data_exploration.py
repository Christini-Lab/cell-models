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


def extract_channel_data(data_h5, trial_number):
    trial_str = f'Trial{trial_number}'
    data = data_h5[trial_str]['Synchronous Data']['Channel Data'][()]
    return data


def plot_V_and_I(data, t_range, title):
    if t_range is not None:
        idx_start = (data['Time (s)']-t_range[0]).abs().idxmin()
        idx_end = (data['Time (s)']-t_range[1]).abs().idxmin()
        data = data.copy().iloc[idx_start:idx_end, :]

    fig, axes = plt.subplots(2, 1, figsize=(10,8), sharex=True)
    if title is not None:
        fig.suptitle(title, fontsize=24)
    data['Current'][data['Current'].abs() > .25E-9] = 0
    data['Voltage (V)'] = data['Voltage (V)'] * 1000

    axes[0].set_ylabel('Voltage (mV)', fontsize=20)
    axes[0].plot(data['Time (s)'], data['Voltage (V)'])
    axes[0].tick_params(labelsize=14)

    axes[1].set_ylabel('Current (A)', fontsize=20)
    axes[1].set_xlabel('Time (s)', fontsize=20)
    axes[1].plot(data['Time (s)'], data['Current'])
    axes[1].tick_params(labelsize=14)
    #axes[1].set_ylim([-.5, .5])
    plt.show()


def get_time_data(data_h5, trial_number):
    total_time, period = get_time_and_period(data_h5, trial_number)
    ch_data = extract_channel_data(data_h5, trial_number)

    V = get_channel_data(ch_data, 1)
    time_array = np.arange(0, len(V)) * period

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


def get_channel_data(ch_data, channel_V):
    return ch_data[:, channel_V - 1]


def get_exp_as_df(data_h5, trial_number):
    """I was going to save the time, voltage and current as a csv,
    but decided not to, because there can be >3million points in 
    the h5 dataset. If you want to make comparisons between trials or
    experiments, call this multiple times.
    """
    ch_data = extract_channel_data(data_h5, trial_number)
    ch_metadata = [key for key in
                   data_h5[f'Trial{trial_number}']['Synchronous Data'].keys()]

    output_channel = int([s for s in ch_metadata if 'Output' in s][0][0])
    input_channel = int([s for s in ch_metadata if 'Input' in s][0][0])

    output_data = get_channel_data(ch_data, output_channel)
    input_data = get_channel_data(ch_data, input_channel)

    if np.mean(np.abs(output_data)) < 1E-6:
        current_data = output_data
        voltage_data = input_data
    else:
        current_data = input_data
        voltage_data = output_data

    t_data = get_time_data(data_h5, trial_number)
    d_as_frame = pd.DataFrame({'Time (s)': t_data,
                               'Voltage (V)': voltage_data,
                               'Current': current_data})
    return d_as_frame


def plot_recorded_data(recorded_data, trial_number, does_plot=False, no_tags=True, t_range=None, title=None):
    if (no_tags==False):
        tags = get_tags(f, trial_number)

    if does_plot:
        if (no_tags==False):
            print(tags)
        plot_V_and_I(recorded_data, t_range, title=title)

    return recorded_data


def explore_data(file_path):
    f = h5py.File(file_path, 'r')
    does_plot = True
    no_tag = True
    trials = f.keys()

    print(trials)

    trial_number = input(f"Which trial number would you like to view? Type a number between 1 and {len(trials)}. Type 'all' if you want to view each one in succession. ")
    if trial_number == 'all':
        trial_range = range(1, len(trials) + 1)
    else:
        trial_range = range(int(trial_number), int(trial_number) + 1)

    for trial in trial_range:
        recorded_data = get_exp_as_df(f, trial)

        plot_recorded_data(recorded_data, trial, does_plot, no_tag, title=f'Trial {trial}')











