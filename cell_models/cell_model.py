from math import log, sqrt, floor
from typing import List

import numpy as np
from scipy import integrate
from scipy.signal import argrelextrema

from cell_models import protocols, trace
from cell_models.current_models import ExperimentalArtefacts


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
                 y_ss=None, is_exp_artefact=False, exp_artefact_params=None):
        self.y_initial = y_initial
        self.default_parameters = default_parameters
        self.no_ion_selective = {}
        self.is_no_ion_selective = False
        self.default_voltage_position = default_voltage_position
        self.y_ss = y_ss
        self.concentration_indices = concentration_indices
        self.i_stimulation = 0
        self.is_exp_artefact = is_exp_artefact
        

        if updated_parameters:
            self.default_parameters.update(updated_parameters)

        if no_ion_selective_dict:
            self.no_ion_selective = no_ion_selective_dict
            self.is_no_ion_selective = True

        if default_time_unit == 's':
            self.time_conversion = 1.0
            self.default_unit = 'standard'
        else:
            self.time_conversion = 1000.0
            self.default_unit = 'milli'

        if default_voltage_unit == 'V':
            self.voltage_conversion = 1
        else:
            self.voltage_conversion = 1000

        self.t = []
        self.y_voltage = []
        self.d_y_voltage = []
        self.current_response_info = None
        self.full_y = []


        self.exp_artefacts = ExperimentalArtefacts()

        if exp_artefact_params is not None:
            for k, v in exp_artefact_params.items():
                setattr(self.exp_artefacts, k, v)

        self.exp_artefacts.g_leak *= default_parameters['G_seal_leak']

        v_off_shift = np.log10(default_parameters['V_off']) * 2
        v_off = -2.8 + v_off_shift  #mV
        self.exp_artefacts.v_off += v_off_shift

        v_p_initial = -80 #mV
        v_clamp_initial = -80 #mV
        i_out_initial = 1
        v_cmd_initial = -80 #mV

        if is_exp_artefact:
            """
            differential equations for Kernik iPSC-CM model
            solved by ODE15s in main_ipsc.m

            # State variable definitions:
            # 0: Vm (millivolt)

            # Ionic Flux: ---------------------------------------------------------
            # 1: Ca_SR (millimolar)
            # 2: Cai (millimolar)
            # 3: Nai (millimolar)
            # 4: Ki (millimolar)

            # Current Gating (dimensionless):--------------------------------------
            # 5: y1    (I_K1 Ishihara)
            # 6: d     (activation in i_CaL)
            # 7: f1    (inactivation in i_CaL)
            # 8: fCa   (calcium-dependent inactivation in i_CaL)
            # 9: Xr1   (activation in i_Kr)
            # 10: Xr2  (inactivation in i_Kr
            # 11: Xs   (activation in i_Ks)
            # 12: h    (inactivation in i_Na)
            # 13: j    (slow inactivation in i_Na)
            # 14: m    (activation in i_Na)
            # 15: Xf   (inactivation in i_f)
            # 16: s    (inactivation in i_to)
            # 17: r    (activation in i_to)
            # 18: dCaT (activation in i_CaT)
            # 19: fCaT (inactivation in i_CaT)
            # 20: R (in Irel)
            # 21: O (in Irel)
            # 22: I (in Irel)

            # With experimental artefact --------------------------------------
            # 23: Vp (millivolt)
            # 24: Vclamp (millivolt)
            # 25: Iout (nA)
            # 26: Vcmd (millivolt)
            """
            if default_voltage_unit == 'V':
                conversion = 1000
            else:
                conversion = 1
            self.y_initial = np.append(self.y_initial, v_p_initial/conversion)
            self.y_initial = np.append(self.y_initial, v_clamp_initial/conversion)
            self.y_initial = np.append(self.y_initial, i_out_initial)
            self.y_initial = np.append(self.y_initial, v_cmd_initial/conversion)

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

        return trace.Trace(protocol,
                           self.default_parameters,
                self.t, self.y_voltage, pacing_info=pacing_info,
                current_response_info=self.current_response_info,
                default_unit=self.default_unit)

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

    def generate_aperiodic_pacing_response(self, protocol):
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

        solution = integrate.solve_ivp(self.generate_aperiodic_pacing_function(
            protocol), [0, protocol.duration / 1E3 * self.time_conversion],
                            y_init,
                            method='BDF',
                            max_step=1e-3*self.time_conversion)

        self.t = solution.t
        self.y = solution.y
        self.y_initial = self.y[:,-1]
        self.y_voltage = solution.y[self.default_voltage_position,:]

        self.calc_currents()

        return trace.Trace(protocol, self.default_parameters, self.t,
                self.y_voltage, current_response_info=self.current_response_info,
                default_unit=self.default_unit)

    def generate_aperiodic_pacing_function(self, protocol):
        def pacing(t, y):
            for t_start in protocol.stim_starts:
                t_start = t_start / 1000 * self.time_conversion
                t_end = t_start + (protocol.stim_duration /
                        1000 * self.time_conversion)

                if (t > t_start) and (t < t_end):
                    self.i_stimulation = protocol.stim_amplitude
                    break 
                else:
                    self.i_stimulation = 0

            d_y = self.action_potential_diff_eq(t, y)

            return d_y

        return pacing

    def generate_exp_voltage_clamp(self, exp_target):
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
            protocol=exp_target)

        solution = integrate.solve_ivp(
            self.generate_voltage_clamp_function(exp_target),
            [0, floor(exp_target.time.max()) /
                1E3 * self.time_conversion],
            y_init,
            method='BDF',
            max_step=1e-3*self.time_conversion,
            atol=1E-2, rtol=1E-4)

        self.t = solution.t
        self.y = solution.y

        command_voltages = [exp_target.get_voltage_at_time(t *
            1E3 / self.time_conversion) / 1E3 * self.time_conversion
            for t in self.t]
        self.command_voltages = command_voltages

        if self.is_exp_artefact:
            self.y_voltages = self.y[0, :]
        else:
            self.y_voltages = command_voltages

        self.calc_currents()



        #import matplotlib.pyplot as plt
        #plt.plot(self.t, self.command_voltages)
        #plt.plot(self.t, self.y_voltages)
        #plt.show()

        return trace.Trace(exp_target, self.default_parameters, self.t,
                           command_voltages=self.command_voltages,
                           y=self.y_voltages,
                           current_response_info=self.current_response_info,
                           default_unit=self.default_unit)

    def generate_exp_voltage_clamp_function(self, exp_target):
        def voltage_clamp(t, y):
            if self.is_exp_artefact:
                try:
                    y[26] = exp_target.get_voltage_at_time(t * 1e3 / self.time_conversion)
                except:
                    y[26] = 20000

                y[26] /= (1E3 / self.time_conversion)
            else:
                try:
                    y[self.default_voltage_position] = exp_target.get_voltage_at_time(t * 1E3 / self.time_conversion)
                except:
                    y[self.default_voltage_position] = 2000

            y[self.default_voltage_position] /= (1E3 / self.time_conversion)

            return self.action_potential_diff_eq(t, y)

        return voltage_clamp

    def generate_exp_current_clamp(self, exp_target):
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
            protocol=exp_target)

        solution = integrate.solve_ivp(
            self.generate_exp_dynamic_clamp_function(exp_target),
            [0, floor(exp_target.time.max()) /
                1E3 * self.time_conversion],
            y_init,
            method='BDF',
            max_step=1e-3*self.time_conversion)

        self.t = solution.t
        self.y = solution.y

        self.y_voltages = self.y[0, :]

        self.calc_currents(exp_target)

        voltages_offset_added = (self.y_voltages +
                self.exp_artefacts['v_off'] / 1000 * self.time_conversion)

        return trace.Trace(exp_target, self.t,
                           y=voltages_offset_added,
                           current_response_info=self.current_response_info,
                           voltages_with_offset=self.y_voltages,
                           default_unit=self.default_unit)

    def generate_exp_dynamic_clamp_function(self, exp_target):
        def dynamic_clamp(t, y):
            r_access = self.exp_artefacts['r_access']
            r_seal = 1 / self.exp_artefacts['g_leak']

            i_access_proportion = r_seal / (r_seal + r_access)

            self.i_stimulation = (-exp_target.get_current_at_time(t * 1000 /
                    self.time_conversion) *
                        i_access_proportion)

            d_y = self.action_potential_diff_eq(t, y)

            return d_y

        return dynamic_clamp
