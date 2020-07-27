"""Contains three classes containing information about a trace."""
import collections
from typing import List

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from scipy.signal import argrelextrema
from cell_models import protocols


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

    def get_current(self, names):
        currents = []
        for i in self.currents:
            currents.append([current.value for current in i if current.name in names])

        currents = np.array(currents)

        if len(names) == 1:
            return currents[:, 0]

        return currents

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
            if i in ('Time Start', 'Time End', 'Time Mid'):
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
                current_contributions['Time Mid'].append((2*i + window)/2)

                for key, val in window_current_contributions.items():
                    current_contributions[key].append(val)
            i += step_size

        return pd.DataFrame(data=current_contributions)

def find_closest_index(array, t):
    """Given an array, return the index with the value closest to t."""
    return (np.abs(np.array(array) - t)).argmin()

def calculate_current_contributions(currents: List[List[Current]]):
    """Calculates the contributions of a list of a list current time steps."""
    current_contributions = {}

    for time_steps in currents:
        total_curr = sum([abs(curr.value) for curr in time_steps if curr.name not in ["I_out", "I_ion"]])
        for current in time_steps:
            if current.name in current_contributions:
                current_contributions[current.name].append(
                        abs(current.value) / total_curr)
            else:
                current_contributions[current.name] = [
                        abs(current.value) / total_curr]

    for key, val in current_contributions.items():
        current_contributions[key] = sum(val)/len(val)

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
                 command_voltages=None,
                 pacing_info: IrregularPacingInfo=None,
                 current_response_info: CurrentResponseInfo=None) -> None:

        self.is_interpolated = False
        self.t = np.array(t)
        self.y = np.array(y)
        self.pacing_info = pacing_info
        self.current_response_info = current_response_info
        self.last_ap = None
        self.command_voltages = command_voltages

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
        dv_dt = np.diff(self.y) / np.diff(self.t)

        dv_dt_inds = argrelextrema(dv_dt, np.greater, order=450)
        bounds = dv_dt_inds[0][-4:-2]

        #plt.plot(self.t, self.y)
        #plt.plot(self.t[dv_dt_inds], self.y[dv_dt_inds])

        cycle = self.t[bounds[1]] - self.t[bounds[0]]
        cycle_25p = cycle *.25
        start_time = self.t[bounds[0]] - cycle_25p
        end_time = start_time + cycle

        start_idx = np.abs(self.t - start_time).argmin()
        end_idx = np.abs(self.t - end_time).argmin()

        self.last_ap = pd.DataFrame({'t': self.t[start_idx:end_idx] - self.t[bounds[0]],
                                    'V': self.y[start_idx:end_idx]})

        return self.last_ap

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

    def plot_with_currents(self, title=None, label=None):
        if not self.current_response_info:
            return ValueError('Trace does not have current info stored. Trace '
                              'was not generated with voltage clamp protocol.')
        fig, (ax_1, ax_2) = plt.subplots(1, 2, num=1)

        ax_1.plot(
            [i for i in self.t],
            [i for i in self.y],
            'b',
            label='Voltage')
        plt.xlabel('Time (ms)')
        plt.ylabel(r'$V_m$ (mV)')
        
        plt.legend()

        ax_2.plot(
            [i for i in self.t],
            [i for i in self.current_response_info.get_current_summed()],
            '--',
            label=label)
        ax_2.yaxis.tick_right()
        ax_2.yaxis.set_label_position("right")
        plt.ylabel(r'$I_m$ (nA/nF)')

        ax_1.spines['top'].set_visible(False)
        if title:
            plt.title(r'{}'.format(title))

    def plot_only_currents(self, label="None", time_conversion=1.0):
        if not self.current_response_info:
            return ValueError('Trace does not have current info stored. Trace '
                              'was not generated with voltage clamp protocol.')

        plt.plot(self.t*time_conversion, self.current_response_info.get_current_summed(), label=label)

    def adjust_sampling(self, frequency=1000):
        """
        The purpose of this function is to return the specified parameter with
        a different sampling rate. It makes it easier to find an error in a GA
        because the traces will be sampled at the same timepoints.

        Parameters
        ----------
        frequency : integer
            The number of samples per millisecond to resample the trace
            from starting point to end point.
        """
        return

    def compare_individual(self, individual):
        if individual is None:
            print("Returning 100")
            return 100
        if not self.is_interpolated:
            self.interp_time = np.linspace(self.t[0], self.t[-1], int(
                self.t[-1] - self.t[0]))

            f = interp1d(self.t,
                         self.current_response_info.get_current_summed())

            self.interp_current = f(self.interp_time)

            self.is_interpolated = True

        f = interp1d(individual.t,
                     individual.current_response_info.get_current_summed())

        individual_current = f(self.interp_time)

        error = sum(abs(self.interp_current - individual_current))

        return error

    def plot_currents_contribution(self, current, window=75, step_size=5, 
            title=None, saved_to=None, voltage_bounds=None):
        current_contributions = self.current_response_info.\
            get_current_contributions(
                time=self.t,
                window=window,
                step_size=step_size)

        total_current = [i for i in 
                self.current_response_info.get_current_summed()]
        c = []
        for t in self.t:
            #for each timepoint, find the closest times in 
            #current_contributions
            idx = current_contributions['Time Mid'].sub(t).abs().idxmin()
            c.append(current_contributions[current].loc[idx])


        fig, (ax_1, ax_2) = plt.subplots(2, 1, num=1, sharex=True, figsize=(12, 8))


        ax_1.plot(
            [i for i in self.t],
            [i for i in self.y],
            'b',
            label='Voltage')
        plt.xlabel('Time (ms)', fontsize=18)
        plt.ylabel(r'$V_m$ (mV)', fontsize=18)

        if voltage_bounds is not None:
            ax_1.set_ylim(voltage_bounds[0], voltage_bounds[1])
        
        ax_im = ax_2.scatter(self.t, total_current, c=c, cmap=cm.copper, vmin=0, vmax=1)
        plt.ylabel(r'$I_m$ (nA/nF)', fontsize=18)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(ax_im, cax=cbar_ax)

        ax_1.spines['top'].set_visible(False)

        if title is not None:
            fig.suptitle(title)

        for ax in [ax_1, ax_2]:
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)


        if saved_to is None:
            plt.show()
        else: 
            plt.savefig(saved_to)

    def plot_with_individual_currents(self, currents, with_artefacts=False):
        """
        Plots the voltage on tope, then the current response of each
        input current.
        """
        num_subplots = len(currents) + 2 # +2 for Vm and Total current
        #create subplots
        fig, axs = plt.subplots(num_subplots, 1, sharex=True)

        axs[0].plot(
            [i for i in self.t],
            [i for i in self.y],
            label=r'$V_m$')
        axs[0].set_ylabel('Voltage (mV)')


        if with_artefacts:
            axs[0].plot(
                [i for i in self.t],
                [i for i in self.command_voltages],
                label=r'$V_{cmd}$')

            axs[0].legend()

        if with_artefacts:
            i_ion = self.current_response_info.get_current(["I_ion"])
            i_out = self.current_response_info.get_current(["I_out"])
            axs[1].plot([i for i in self.t], [i for i in i_ion],
                        label=r'$I_{ion}$')
            axs[1].plot([i for i in self.t], [i for i in i_out], '--',
                        label=r'$I_{out}$')
            axs[1].set_ylabel(r'$I_{total}$ (nA/nF)')
            axs[1].legend()
        else:
            axs[1].plot(
                [i for i in self.t],
                [i for i in self.current_response_info.get_current_summed()],
                '--',
                label=r'$I_{ion}$')
            axs[1].set_ylabel(r'$I_m$ (nA/nF)')

        for i, current in enumerate(currents):
            i_curr = self.current_response_info.get_current([current])
            current_ax = 2 + i
            axs[current_ax].plot([i for i in self.t], [i for i in i_curr],
            label=current)
            axs[current_ax].set_ylabel(f'{current} (pA/pF)')
            axs[current_ax].legend()

        axs[-1].set_xlabel("Time (ms)")
        plt.show() 
