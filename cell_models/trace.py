"""Contains three classes containing information about a trace."""
import collections
from typing import List

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import argrelextrema
import protocols


class IrregularPacingInfo:
    """Contains information regarding irregular pacing.

    Attributes:
        peaks: Times when a AP reaches its peak.
        stimulations: Times when cell is stimulated.
        diastole_starts: Times when the diastolic period begins.
        apd_90_end_voltage: The voltage at next APD 90. Is set to -1 to indicate
            voltage has not yet been calculated.
        apd_90s: Times of APD 90s.
    """

    _STIMULATION_DURATION = 0.005
    _PEAK_DETECTION_THRESHOLD = 0.0
    _MIN_VOLT_DIFF = 0.00001
    # TODO Changed peak min distance to find peaks next to each other.
    _PEAK_MIN_DIS = 0.0001
    AVG_AP_START_VOLTAGE = -0.075

    def __init__(self) -> None:
        self.peaks = []
        self.stimulations = []
        self.diastole_starts = []

        # Set to -1 to indicate it has not yet been set.
        self.apd_90_end_voltage = -1
        self.apd_90s = []

    def add_apd_90(self, apd_90: float) -> None:
        self.apd_90s.append(apd_90)
        self.apd_90_end_voltage = -1

    def should_stimulate(self, t: float) -> bool:
        """Checks whether stimulation should occur given a time point."""
        for i in range(len(self.stimulations)):
            distance_from_stimulation = t - self.stimulations[i]
            if 0 < distance_from_stimulation < self._STIMULATION_DURATION:
                return True
        return False

    def plot_stimulations(self, trace: 'Trace') -> None:
        stimulation_y_values = _find_trace_y_values(
            trace=trace,
            timings=self.stimulations)

        sti = plt.scatter(self.stimulations, stimulation_y_values, c='red')
        plt.legend((sti,), ('Stimulation',), loc='upper right')

    def plot_peaks_and_apd_ends(self, trace: 'Trace') -> None:
        peak_y_values = _find_trace_y_values(
            trace=trace,
            timings=self.peaks)
        apd_end_y_values = _find_trace_y_values(
            trace=trace,
            timings=self.apd_90s)

        peaks = plt.scatter(
            [i * 1000 for i in self.peaks],
            [i * 1000 for i in peak_y_values], c='red')
        apd_end = plt.scatter(
            [i * 1000 for i in self.apd_90s],
            [i * 1000 for i in apd_end_y_values],
            c='orange')
        plt.legend(
            (peaks, apd_end),
            ('Peaks', 'APD 90'),
            loc='upper right',
            bbox_to_anchor=(1, 1.1))

    def detect_peak(self,
                    t: List[float],
                    y_voltage: float,
                    d_y_voltage: List[float]) -> bool:
        # Skip check on first few points.
        if len(t) < 2:
            return False

        if y_voltage < self._PEAK_DETECTION_THRESHOLD:
            return False
        if d_y_voltage[-1] <= 0 < d_y_voltage[-2]:
            # TODO edit so that successive peaks are discovered. Decrease peak
            # TODO mean distance.
            if not (self.peaks and t[-1] - self.peaks[-1] < self._PEAK_MIN_DIS):
                return True
        return False

    def detect_apd_90(self, y_voltage: float) -> bool:
        return self.apd_90_end_voltage != -1 and abs(
            self.apd_90_end_voltage - y_voltage) < 0.001


def _find_trace_y_values(trace, timings):
    """Given a trace, finds the y values of the timings provided."""
    y_values = []
    for i in timings:
        array = np.asarray(trace.t)
        index = find_closest_index(array, i)
        y_values.append(trace.y[index])
    return y_values


class Current:
    """Encapsulates a current at a single time step."""

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def __str__(self):
        return '{}: {}'.format(self.name, self.value)

    def __repr__(self):
        return '{}: {}'.format(self.name, self.value)


class CurrentResponseInfo:
    """Contains info of currents in response to voltage clamp protocol.

    Attributes:
        protocol: Specifies voltage clamp protocol which created the current
            response.
        currents: A list of current timesteps.

    """

    def __init__(self, protocol: protocols.VoltageClampProtocol=None) -> None:
        self.protocol = protocol
        self.currents = []

    def get_current_summed(self):
        current = []
        for i in self.currents:
            current.append(sum([j.value for j in i]))

        #current = [i / 100 for i in current]
        #median_current = np.median(current)
        #for i in range(len(current)):
        #    if abs(current[i] - median_current) > 0.1:
        #        current[i] = 0
        return current

    def get_max_current_contributions(self,
                                      time: List[float],
                                      window: float,
                                      step_size: float) -> pd.DataFrame:
        """Finds the max contribution given contributions of currents.

        Args:
            time: The time stamps of the trace.
            window: A window of time, in seconds, over which current
                contributions are calculated. For example, if window was 1.0
                seconds and the total trace was 10 seconds, 10 current
                contributions would be recorded.
            step_size: The time between windows. For example, if step_size was
                equal to `window`, there would be no overlap when calculating
                current contributions. The smaller the step size, the increased
                computation required. Step size cannot be 0.

        Returns:
            A pd.DataFrame containing the max current contribution for each
            current. Here is an example:

            Index  Time Start  Time End  Contribution  Current

            0      0.1         0.6       0.50          I_Na
            1      0.2         0.7       0.98          I_K1
            2      0.0         0.5       0.64          I_Kr
        """
        contributions = self.get_current_contributions(
            time=time,
            window=window,
            step_size=step_size)
        max_contributions = collections.defaultdict(list)
        for i in list(contributions.columns.values):
            if i in ('Time Start', 'Time End'):
                continue
            max_contrib_window = contributions.loc[contributions[i].idxmax()]
            max_contributions['Current'].append(i)
            max_contributions['Contribution'].append(max_contrib_window[i])
            max_contributions['Time Start'].append(
                max_contrib_window['Time Start'])
            max_contributions['Time End'].append(max_contrib_window['Time End'])
        return pd.DataFrame(data=max_contributions)

    def get_current_contributions(self,
                                  time: List[float],
                                  window: float,
                                  step_size: float) -> pd.DataFrame:
        """Calculates each current contribution over a window of time.

        Args:
            time: The time stamps of the trace.
            window: A window of time, in seconds, over which current
                contributions are calculated. For example, if window was 1.0
                seconds and the total trace was 10 seconds, 10 current
                contributions would be recorded.
            step_size: The time between windows. For example, if step_size was
                equal to `window`, there would be no overlap when calculating
                current contributions. The smaller the step size, the increased
                computation required. Step size cannot be 0.

        Returns:
            A pd.DataFrame containing the fraction contribution of each current
            at each window. Here is an example:

            Index  Time Start  Time End  I_Na  I_K1  I_Kr

            0      0.0         0.5       0.12  0.24  0.64
            1      0.1         0.6       0.50  0.25  0.25
            2      0.2         0.7       0.01  0.98  0.01
            3      0.3         0.8       0.2   0.3   0.5
        """
        if not self.currents:
            raise ValueError('No current response recorded.')

        current_contributions = collections.defaultdict(list)
        i = 0
        while i <= time[-1] - window:
            start_index = find_closest_index(time, i)
            end_index = find_closest_index(time, i + window)
            currents_in_window = self.currents[start_index: end_index + 1]
            window_current_contributions = calculate_current_contributions(
                currents=currents_in_window)

            if window_current_contributions:
                # Append results from current window to overall contributions
                # dict.
                current_contributions['Time Start'].append(i)
                current_contributions['Time End'].append(i + window)

                for key, val in window_current_contributions.items():
                    current_contributions[key].append(val)
            i += step_size

        return pd.DataFrame(data=current_contributions)

def find_closest_index(array, t):
    """Given an array, return the index with the value closest to t."""
    return (np.abs(np.array(array) - t)).argmin()

def calculate_current_contributions(currents: List[List[Current]]):
    """Calculates the contributions of a list of a list current time steps."""
    current_sums = {}
    total_sum = 0
    for time_steps in currents:
        for current in time_steps:
            if current.name in current_sums:
                current_sums[current.name] += abs(current.value)
            else:
                current_sums[current.name] = abs(current.value)
            total_sum += abs(current.value)

    current_contributions = {}
    for key, val in current_sums.items():
        current_contributions[key] = val / total_sum

    return current_contributions


class Trace:
    """Represents a spontaneous or probed response from cell.

    Attributes:
        t: Timestamps of the response.
        y: The membrane voltage, in volts, at a point in time.
        pacing_info: Contains additional information about cell pacing. Will be
            None if no pacing has occurred.
        current_response_info: Contains information about individual currents
            in the cell. Will be set to None if the voltage clamp protocol was
            not used.
    """

    def __init__(self,
                 t: List[float],
                 y: List[float],
                 pacing_info: IrregularPacingInfo=None,
                 current_response_info: CurrentResponseInfo=None) -> None:
        self.t = np.array(t)
        self.y = np.array(y)
        self.pacing_info = pacing_info
        self.current_response_info = current_response_info
        self.last_ap = None

    def get_cl(self):
        if self.last_ap is None:
            self.get_last_ap()

        return self.last_ap.t.max() - self.last_ap.t.min()

    def get_di(self):
        pass

    def get_apd_90(self):
        apd_90_v = self.last_ap.V.max() - .9*(self.last_ap.V.max()-self.last_ap.V.min())
        max_v_idx = self.last_ap.V.idxmax()

        idx = (self.last_ap.V - apd_90_v).abs().argsort()
        idx = idx[idx > max_v_idx].reset_index().V[0]

        apd_90_t = self.last_ap.iloc[idx].t

        dv_dt_max_t = self.get_dv_dt_max_time()

        return apd_90_t - dv_dt_max_t

    def get_dv_dt_max_time(self):
        dv_dt = self.last_ap.diff().abs()
        dv_dt_diff = dv_dt.V/dv_dt.t

        return [self.last_ap.t.iloc[dv_dt_diff.idxmax()], dv_dt_diff.idxmax()]

    def get_last_ap(self):
        inds = argrelextrema(self.y, np.greater)
        bounds = inds[0][-3:-1]
        
        start_idx = np.abs(self.t - (self.t[bounds[0]] - 30.0)).argmin()
        end_idx = np.abs(self.t - (self.t[bounds[1]] - 30.0)).argmin()

        self.last_ap = pd.DataFrame({'t': self.t[start_idx:end_idx],
                                    'V': self.y[start_idx:end_idx]})

    def plot(self):
        plt.figure(figsize=(10, 5))
        ax = plt.subplot()
        plt.plot(
            [i * 1000 for i in self.t],
            [i * 1000 for i in self.y],
            color='b')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel('Time (ms)')
        plt.ylabel(r'$V_m$ (mV)')

    def plot_just_currents(self, title=None):
        plt.plot(
            [1000 * i for i in self.t],
            [i * 1000 for i in self.current_response_info.get_current_summed()],
            'r--',
            label='Current')
        plt.ylabel(r'$I_m$ (pA/pF)')

        ax_1.spines['top'].set_visible(False)
        plt.show()

    def plot_with_currents(self, title=None):
        if not self.current_response_info:
            return ValueError('Trace does not have current info stored. Trace '
                              'was not generated with voltage clamp protocol.')
        fig = plt.figure(figsize=(10, 5))

        ax_1 = fig.add_subplot(111)
        ax_1.plot(
            [1000 * i for i in self.t],
            [i * 1000 for i in self.y],
            'b',
            label='Voltage')
        plt.xlabel('Time (ms)')
        plt.ylabel(r'$V_m$ (mV)')

        ax_2 = fig.add_subplot(111, sharex=ax_1, frameon=False)
        ax_2.plot(
            [1000 * i for i in self.t],
            [i * 1000 for i in self.current_response_info.get_current_summed()],
            'r--',
            label='Current')
        ax_2.yaxis.tick_right()
        ax_2.yaxis.set_label_position("right")
        plt.ylabel(r'$I_m$ (nA/nF)')

        ax_1.spines['top'].set_visible(False)
        plt.show()
        if title:
            plt.title(r'{}'.format(title))

    def plot_only_currents(self, label="None", time_conversion=1.0):
        if not self.current_response_info:
            return ValueError('Trace does not have current info stored. Trace '
                              'was not generated with voltage clamp protocol.')

        plt.plot(self.t*time_conversion, self.current_response_info.get_current_summed(), label=label)
