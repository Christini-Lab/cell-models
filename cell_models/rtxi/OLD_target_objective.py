import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from math import floor, log, ceil

from cell_models import protocols, kernik, paci_2018


class TargetObjective():
    def __init__(self, capacitance, file_name='./model_data/exp_vc_trial2.csv',
        protocol=protocols.VoltageClampProtocol):
        self.target = pd.read_csv(file_name)
        self.target.columns = ['time', 'voltage', 'current']
        self.protocol = protocol
        self.cm = capacitance

        self.filter_signal()

    def filter_signal(self):
        fft = scipy.fft(self.target.current)

        bp=fft[:]
        for i in range(len(bp)): # (H-red)
            if ((i>=200 and i<=57000) or (i > 60000)):
                bp[i]=0
        ibp=scipy.ifft(bp)

        window = 4
        start_indices = np.array(range(0,floor(len(ibp)/window)))*window

        new_currents = []
        new_times = []
        new_voltages = []
        for start in start_indices:
            mid_index = floor(start + window/2)
            new_times.append(self.target.time[mid_index])
            new_voltages.append(self.target.voltage[mid_index]) 
            
            new_currents.append(np.mean(ibp[start:(start+window)].real))

        self.target = pd.DataFrame({'t': np.array(new_times)*1000.0,
                                    'I': np.array(new_currents)*1E12/self.cm,
                                    'V': new_voltages})

    def compare_individual(self, times, I):
        i = 0
        error = 0

        prev_idx = 0

        sampling_diff = self.target.t[2] - self.target.t[1]
        max_diff_model = np.diff(times).max()*2
        index_max = ceil(max_diff_model/sampling_diff)

        for t in times:
            curr_idx = (self.target.t[prev_idx:(prev_idx+index_max)] -t).abs().idxmin()

            error = error + np.abs(self.target.I.loc[curr_idx] - I[i])

            prev_idx = curr_idx
            i += 1

        norm_error = error
        print(norm_error)

        return norm_error

