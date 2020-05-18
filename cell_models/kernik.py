from math import log, sqrt 
from typing import List

from cell_models.cell_model import CellModel
from cell_models.current_models import KernikCurrents, Ishi

import numpy as np
from scipy import integrate

from cell_models import protocols
from cell_models import trace
from cell_models.model_initial import kernik_model_initial 
from math import log, exp


class KernikModel(CellModel):
    """An implementation of the Kernik model by Kernik et al.

    Attributes:
        default_parameters: A dict containing tunable parameters along with
            their default values as specified in Kernik et al.
        updated_parameters: A dict containing all parameters that are being
            tuned.
    """
    cm_farad = 60

    # Constants
    t_kelvin = 310.0  
    r_joule_per_mole_kelvin = 8.314472   
    f_coulomb_per_mmole = 96.4853415  

    Ko = 5.4  # millimolar (in model_parameters)
    Cao = 1.8  # millimolar (in model_parameters
    Nao = 140.0  # millimolar (in model_parameters)

    def __init__(self, default_parameters=None,
                 updated_parameters=None,
                 no_ion_selective_dict=None,
                 default_time_unit='ms', 
                 default_voltage_unit='mV',
                 concentration_indices={'Ca_SR': 1, 'Cai': 2,
                                        'Nai': 3, 'Ki': 4}
                 ):

        self.kernik_currents = KernikCurrents(self.t_kelvin,
                                              self.f_coulomb_per_mmole, 
                                              self.r_joule_per_mole_kelvin)

        default_parameters = {
            'G_K1': 1,
            'G_K1_Ishi': 0,
            'G_Kr': 1,
            'G_Ks': 1,
            'G_to': 1,
            'P_CaL': 1,
            'G_CaT': 1,
            'G_Na': 1,
            'G_F': 1,
            'K_NaCa': 1,
            'P_NaK': 1,
            'VmaxUp': 1,
            'V_leak': 1,
            'ks': 1,
            'G_b_Na': 1,
            'G_b_Ca': 1,
            'G_PCa': 1
        }

        y_initial = kernik_model_initial()

        super().__init__(concentration_indices,
                         y_initial, default_parameters,
                         updated_parameters,
                         no_ion_selective_dict,
                         default_time_unit,
                         default_voltage_unit)

    def action_potential_diff_eq(self, t, y):
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
        """

        d_y = np.zeros(23)


        # --------------------------------------------------------------------
        # Reversal Potentials:
        E_Ca = 0.5 * self.r_joule_per_mole_kelvin * self.t_kelvin / self.f_coulomb_per_mmole * log(self.Cao / y[2])  # millivolt
        E_Na = self.r_joule_per_mole_kelvin * self.t_kelvin / self.f_coulomb_per_mmole * log(self.Nao / y[3])  # millivolt
        E_K = self.r_joule_per_mole_kelvin * self.t_kelvin / self.f_coulomb_per_mmole * log(self.Ko / y[4])  # millivolt

        # --------------------------------------------------------------------
        # Currents:
        i_K1 = self.kernik_currents.i_K1(y[0], E_K, self.default_parameters['G_K1'])
        i_K1_ishi, d_y[5] = Ishi.I_K1(y[0], E_K, y[5], self.Ko, self.default_parameters['G_K1_Ishi']) 

        d_y[9], d_y[10], i_Kr = self.kernik_currents.i_Kr(
                y[0], E_K, y[9], y[10], self.default_parameters['G_Kr'])

        d_y[11], i_Ks = self.kernik_currents.i_Ks(
                y[0], E_K, y[11], self.default_parameters['G_Ks'])

        d_y[16], d_y[17], i_to = self.kernik_currents.i_to(
                y[0], E_K, y[16], y[17], self.default_parameters['G_to'])

        d_y[6], d_y[7], d_y[8], i_CaL, i_CaL_Ca, i_CaL_Na, i_CaL_K = \
                self.kernik_currents.i_CaL(
                y[0], y[6], y[7], y[8], y[2], y[3], y[4], 
                self.default_parameters['P_CaL'])

        d_y[18], d_y[19], i_CaT = self.kernik_currents.i_CaT(y[0], E_Ca, y[18], y[19],
                self.default_parameters['G_CaT'])

        d_y[12], d_y[13], d_y[14], i_Na = self.kernik_currents.i_Na(y[0], E_Na,
                y[12], y[13], y[14], self.default_parameters['G_Na'])


        d_y[15], i_f, i_fNa, i_fK = self.kernik_currents.i_f(y[0], E_K, E_Na, y[15], 
                self.default_parameters['G_F'])

        i_NaCa = self.kernik_currents.i_NaCa(y[0], y[2], y[3], self.default_parameters['K_NaCa'])

        i_NaK = self.kernik_currents.i_NaK(y[0], y[3], self.default_parameters['P_NaK'])
        
        i_up = self.kernik_currents.i_up(y[2], self.default_parameters['VmaxUp'])

        i_leak = self.kernik_currents.i_leak(y[1], y[2], self.default_parameters['V_leak'])

        d_y[20], d_y[21], d_y[22], i_rel = self.kernik_currents.i_rel(y[1], y[2], 
                y[20], y[21], y[22], self.default_parameters['ks'])

        i_b_Na = self.kernik_currents.i_b_Na(y[0], E_Na,
                self.default_parameters['G_b_Na'])

        i_b_Ca = self.kernik_currents.i_b_Ca(y[0], E_Ca, 
                self.default_parameters['G_b_Ca'])

        i_PCa = self.kernik_currents.i_PCa(y[2], self.default_parameters['G_PCa'])


        # --------------------------------------------------------------------
        # Concentration Changes:
        d_y[1] = self.kernik_currents.Ca_SR_conc(y[1], i_up, i_rel, i_leak)

        d_y[2] = self.kernik_currents.Cai_conc(y[2], i_leak, i_up, i_rel, d_y[5],
                                         i_CaL_Ca, i_CaT, i_b_Ca,
                                         i_PCa, i_NaCa, self.cm_farad)

        d_y[3] = self.kernik_currents.Nai_conc(i_Na, i_b_Na, i_fNa, i_NaK, i_NaCa, 
                                         i_CaL_Na, self.cm_farad, t)

        #d_y[4] = self.kernik_currents.Ki_conc(i_K1, i_to, i_Kr, i_Ks, i_fK, 
        #                                i_NaK, i_CaL_K, self.cm_farad)
        d_y[4] = -d_y[3]

        # --------------------------------------------------------------------
        # Handling i_no_ion 
        i_no_ion = 0
        if self.is_no_ion_selective:
            current_dictionary = {
                'I_K1':    i_K1,
                'I_To':    i_to,
                'I_Kr':    i_Kr,
                'I_Ks':    i_Ks,
                'I_CaL':   i_CaL_Ca,
                'I_NaK':   i_NaK,
                'I_Na':    i_Na,
                'I_NaCa':  i_NaCa,
                'I_pCa':   i_PCa,
                'I_F':     i_f,
                'I_bNa':   i_b_Na,
                'I_bCa':   i_b_Ca,
                'I_CaT':   i_CaT,
                'I_up':    i_up,
                'I_leak':  i_leak
            }
            for curr_name, scale in self.no_ion_selective.items():
                if curr_name == 'I_K1':
                    i_no_ion += scale * i_K1_ishi
                else:
                    i_no_ion += scale * current_dictionary[curr_name]


        # --------------------------------------------------------------------
        # Calculate change in Voltage and Save currents 
        d_y[0] = -(i_K1+i_to+i_Kr+i_Ks+i_CaL+i_CaT+i_NaK+i_Na+i_NaCa +
                   i_PCa+i_f+i_b_Na+i_b_Ca + i_K1_ishi + i_no_ion) + self.i_stimulation

        if self.current_response_info:
            current_timestep = [
                trace.Current(name='I_K1', value=i_K1),
                trace.Current(name='I_To', value=i_to),
                trace.Current(name='I_Kr', value=i_Kr),
                trace.Current(name='I_Ks', value=i_Ks),
                trace.Current(name='I_CaL', value=i_CaL_Ca),
                trace.Current(name='I_NaK', value=i_NaK),
                trace.Current(name='I_Na', value=i_Na),
                trace.Current(name='I_NaCa', value=i_NaCa),
                trace.Current(name='I_pCa', value=i_PCa),
                trace.Current(name='I_F', value=i_f),
                trace.Current(name='I_bNa', value=i_b_Na),
                trace.Current(name='I_bCa', value=i_b_Ca),
                trace.Current(name='I_CaT', value=i_CaT),
                trace.Current(name='I_up', value=i_up),
                trace.Current(name='I_leak', value=i_leak)
            ]
            self.current_response_info.currents.append(current_timestep)

        return d_y


def generate_trace(protocol, tunable_parameters=None, params=None):
    """Generates a trace.

    Leave `params` argument empty if generating baseline trace with
    default parameter values.

    Args:
        tunable_parameters: List of tunable parameters.
        protocol: A protocol object used to generate the trace.
        params: A set of parameter values (where order must match with ordered
            labels in `tunable_parameters`).

    Returns:
        A Trace object.
    """
    new_params = dict()
    if params and tunable_parameters:
        for i in range(len(tunable_parameters)):
            new_params[tunable_parameters[i].name] = params[
                    tunable_parameters[i].name]

    return KernikModel(updated_parameters=new_params).generate_response(protocol)
