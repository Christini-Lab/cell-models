from math import log, sqrt, floor
from typing import List

import numpy as np
from scipy import integrate
from scipy.signal import argrelextrema

from cell_models import protocols
from cell_models import trace

class CellModel:
    """An implementation a general cell model

    Attributes:
        default_parameters: A dict containing tunable parameters
        updated_parameters: A dict containing all parameters that are being
            tuned.
    """

    def __init__(self, concentration_indices, y_initial=[], 
                 default_parameters=None, updated_parameters=None,
                 no_ion_selective_dict=None, default_time_unit='s',
                 default_voltage_unit='V', default_voltage_position=0,
                 y_ss=None):
        self.y_initial = y_initial
        self.default_parameters = default_parameters
        self.no_ion_selective = {}
        self.is_no_ion_selective = False
        self.default_voltage_position = default_voltage_position
        self.y_ss = y_ss
        self.concentration_indices = concentration_indices
        self.i_stimulation = 0

        if updated_parameters:
            self.default_parameters.update(updated_parameters)
        if no_ion_selective_dict:
            self.no_ion_selective = no_ion_selective_dict
            self.is_no_ion_selective = True

        if default_time_unit == 's':
            self.time_conversion = 1.0
        else:
            self.time_conversion = 1000.0

        if default_voltage_unit == 'V':
            self.voltage_conversion = 1
        else:
            self.voltage_conversion = 1000

        self.t = []
        self.y_voltage = []
        self.d_y_voltage = []
        self.current_response_info = None
        self.full_y = []

    @property
    def no_ion_selective(self):
        return self.__no_ion_selective

    @no_ion_selective.setter
    def no_ion_selective(self, no_ion_selective):
        self.__no_ion_selective = no_ion_selective

    def calc_currents(self):
        self.current_response_info = trace.CurrentResponseInfo()      
        if len(self.y) < 200:
            list(map(self.action_potential_diff_eq, self.t, self.y.transpose()))
        else:
            list(map(self.action_potential_diff_eq, self.t, self.y))

    def generate_response(self, protocol, is_no_ion_selective=False):
        """Returns a trace based on the specified target objective.

        Args:
            protocol: An object of a specified protocol.

        Returns:
            A Trace object representing the change in membrane potential over
            time.
        """
        # Reset instance variables when model is run again.
        self.t = []
        self.y_voltage = []
        self.d_y_voltage = []
        self.full_y = []
        self.current_response_info = None

        self.is_no_ion_selective = is_no_ion_selective


        if isinstance(protocol, protocols.SpontaneousProtocol):
            return self.generate_spontaneous_response(protocol)
        elif isinstance(protocol, protocols.IrregularPacingProtocol):
            return self.generate_irregular_pacing_response(protocol)
        elif isinstance(protocol, protocols.VoltageClampProtocol):
            return self.generate_VC_protocol_response(protocol)
        elif isinstance(protocol, protocols.PacedProtocol):
            return self.generate_pacing_response(protocol)

    def find_steady_state(self, ss_type=None, from_peak=False, time_unit='ms', tol = 1E-3, max_iters=140):
        """
        Finds the steady state conditions for a spontaneous or stimulated
        (in the case of OR) AP
        """
        if self.y_ss is not None:
            return

        if (ss_type is None) and (self.time_conversion == 1000.0):
            protocol = protocols.VoltageClampProtocol(
                [protocols.VoltageClampStep(voltage=-80.0, duration=10000)])

        if (ss_type is None) and (self.time_conversion == 1.0):
            protocol = protocols.VoltageClampProtocol(
                [protocols.VoltageClampStep(voltage=-.080, duration=10)])

        concentration_indices = list(self.concentration_indices.values())

        is_err = True
        i = 0
        y_values = []

        import time
        outer_time = time.time()

        print("Starting to find steady-state")
        while is_err:
            init_t = time.time()

            tr = self.generate_response(protocol)

            if isinstance(protocol, protocols.VoltageClampProtocol):
                y_val = self.y[:, -1]
            else:
                y_val = self.get_last_min_max(from_peak)
            self.y_initial = self.y[:, -1]
            y_values.append(y_val)
            y_percent = []

            if len(y_values) > 2:
                y_percent = np.abs((y_values[i][concentration_indices] -
                                    y_values[i - 1][concentration_indices]) / (
                    y_values[i][concentration_indices]))
                is_below_tol = (y_percent < tol)
                is_err = not is_below_tol.all()

            if i > max_iters:
                print("Did not reach steady state")
                self.y_ss = np.ones(23)
                return

            i = i + 1
            print(
                f'Iteration {i}; {time.time() - init_t} seconds; {y_percent}')

        self.y_ss = y_values[-1]
        print(f'Total Time: {time.time() - outer_time}')
        return [tr, i]

    def get_last_min_max(self, from_peak):
        if from_peak:
            inds = argrelextrema(self.y_voltage, np.less)
            last_peak_time = self.t[inds[0][-2]]
            ss_time = last_peak_time - .04*self.time_conversion
            y_val_idx = np.abs(self.t - ss_time).argmin()
        else:
            inds = argrelextrema(self.y_voltage, np.less)
            y_val_idx = inds[0][-2]
        try:
            y_val = self.y[:,y_val_idx]
        except:
            y_val = self.y[y_val_idx,:]

        return y_val

    def generate_spontaneous_function(self):
        def spontaneous(t, y):
            return self.action_potential_diff_eq(t, y)

        return spontaneous

    def generate_spontaneous_response(self, protocol):
        """
        Args:
            protocol: An object of a specified protocol.

        Returns:
            A single action potential trace
        """
        if self.y_ss is not None:
            y_init = self.y_ss
        else:
            y_init = self.y_initial

        try:
            solution = integrate.solve_ivp(
                self.generate_spontaneous_function(),
                [0, protocol.duration],
                y_init,
                method='BDF',
                max_step=1e-3*self.time_conversion)

            self.t = solution.t
            self.y = solution.y.transpose()
            self.y_initial = self.y[-1]
            self.y_voltage = solution.y[self.default_voltage_position,:]

            self.calc_currents()

        except ValueError:
            print('Model could not produce trace.')
            return None

        return trace.Trace(self.t,
                           self.y_voltage,
                           current_response_info=self.current_response_info)

    def generate_irregular_pacing_response(self, protocol):
        """
        Args:
            protocol: An irregular pacing protocol 
        Returns:
            A irregular pacing trace
        """
        if self.y_ss is not None:
            y_init = self.y_ss
        else:
            y_init = self.y_initial

        pacing_info = trace.IrregularPacingInfo()

        try:
            solution = integrate.solve_ivp(self.generate_irregular_pacing_function(
                protocol, pacing_info), [0, protocol.duration],
                                y_init,
                                method='BDF',
                                max_step=1e-3*self.time_conversion)

            self.t = solution.t
            self.y = solution.y
            self.y_initial = self.y[-1]
            self.y_voltage = solution.y[self.default_voltage_position,:]


            self.calc_currents()

        except ValueError:
            return None
        return trace.Trace(self.t, self.y_voltage, pacing_info=pacing_info)

    def generate_irregular_pacing_function(self, protocol, pacing_info):
        offset_times = protocol.make_offset_generator()

        def irregular_pacing(t, y):
            d_y = self.action_potential_diff_eq(t, y)

            if pacing_info.detect_peak(self.t, y[0], self.d_y_voltage):
                pacing_info.peaks.append(t)
                voltage_diff = abs(pacing_info.AVG_AP_START_VOLTAGE - y[0])
                pacing_info.apd_90_end_voltage = y[0] - voltage_diff * 0.9

            if pacing_info.detect_apd_90(y[0]):
                try:
                    pacing_info.add_apd_90(t)
                    pacing_info.stimulations.append(t + next(offset_times))
                except StopIteration:
                    pass

            if pacing_info.should_stimulate(t):
                i_stimulation = protocol.STIM_AMPLITUDE_AMPS / self.cm_farad
            else:
                i_stimulation = 0.0

            d_y[0] += i_stimulation
            return d_y

        return irregular_pacing

    def generate_VC_protocol_response(self, protocol):
        """
        Args:
            protocol: A voltage clamp protocol
        Returns:
            A Trace object for a voltage clamp protocol
        """
        if self.y_ss is not None:
            y_init = self.y_ss
        else:
            y_init = self.y_initial

        self.current_response_info = trace.CurrentResponseInfo(
            protocol=protocol)

        try:
            solution = integrate.solve_ivp(
                self.generate_voltage_clamp_function(protocol),
                [0, protocol.get_voltage_change_endpoints()[-1]],
                y_init,
                method='BDF',
                max_step=1e-3*self.time_conversion)

            self.t = solution.t
            self.y = solution.y
            #self.y_initial = self.y[:,-1]
            self.y_voltage = solution.y[self.default_voltage_position,:]

            self.calc_currents()
        except:
            print("There was an error")
            return None



        return trace.Trace(self.t,
                           self.y_voltage,
                           current_response_info=self.current_response_info)

    def generate_voltage_clamp_function(self, protocol):
        def voltage_clamp(t, y):
            y[self.default_voltage_position] = protocol.get_voltage_at_time(t)
            return self.action_potential_diff_eq(t, y)

        return voltage_clamp

    def generate_pacing_response(self, protocol):
        """
        Args:
            protocol: A pacing protocol
        Returns:
            A pacing trace
        """
        if self.y_ss is not None:
            y_init = self.y_ss
        else:
            y_init = self.y_initial

        pacing_info = trace.IrregularPacingInfo()

        solution = integrate.solve_ivp(self.generate_pacing_function(
            protocol), [0, protocol.stim_end],
                            y_init,
                            method='LSODA',
                            max_step=8e-4*self.time_conversion)

        self.t = solution.t
        self.y = solution.y
        self.y_initial = self.y[:,-1]
        self.y_voltage = solution.y[self.default_voltage_position,:]

        self.calc_currents()

        return trace.Trace(self.t, self.y_voltage, pacing_info=pacing_info)

    def generate_pacing_function(self, protocol):
        def pacing(t, y):
            #i_stim_period = floor(self.time_conversion / protocol.pace)
            i_stim_period = self.time_conversion / protocol.pace
            if self.time_conversion == 1:
                denom = 1E12
            else:
                denom = 1
            
            self.i_stimulation = (protocol.stim_amplitude if t - protocol.stim_start -\
                i_stim_period*floor((t - protocol.stim_start)/i_stim_period) <=\
                protocol.stim_duration and t <= protocol.stim_end and t >= protocol.stim_start else\
                0) / self.cm_farad/denom

            d_y = self.action_potential_diff_eq(t, y)

            return d_y

        return pacing
