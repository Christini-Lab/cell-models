from cell_models import kernik, paci_2018, protocols
from scipy.interpolate import interp1d
import numpy as np


class ModelTarget():
    """
    protocol – type from protocols
    model – initialize Paci or Kernik model
    target_type – Spontaneous, Voltage clamp, paced, SAP
    tr – trace object
    """

    def __init__(self, protocol, model, protocol_type, tr):
        self.protocol = protocol
        self.model = model
        self.protocol_type = protocol_type
        self.tr = tr

    def compare_individual(self, individual_tr):
        if isinstance(self.protocol, protocols.SpontaneousProtocol):
            return self.get_sap_error(individual_tr)
        elif isinstance(self.protocol, protocols.VoltageClampProtocol):
            return self.get_vc_error(individual_tr)
        else:
            return self.get_current_clamp_error(individual_tr)

    def get_sap_error(self, individual_tr):
        target_ap, target_bounds, max_t = self.tr.get_last_ap()
        individual_ap, ind_bounds, max_t_ind = individual_tr.get_last_ap()

        if self.tr.default_unit == 'milli':
            target_conversion = 1
        else:
            target_conversion = 1000
            target_ap.t = target_ap.t * target_conversion
            target_ap.V = target_ap.V * target_conversion

        if individual_tr.default_unit == 'milli':
            ind_conversion = 1
        else:
            ind_conversion = 1000
            individual_ap.t = individual_ap.t * ind_conversion
            individual_ap.V = individual_ap.V * ind_conversion
        
        #import matplotlib.pyplot as plt
        #plt.plot(individual_ap.t, individual_ap.V)
        #plt.plot(target_ap.t, target_ap.V)


        #interp_time = np.linspace(target_ap.t[0], target_ap.t.values[-1], 3000)

        #f = interp1d(target_ap.t, target_ap.V)

        #interp_V = f(interp_time)

        errors = []

        for i, t in enumerate(target_ap.t):
            curr_idx = (individual_ap.t - t).abs().idxmin()

            diff = abs(target_ap.V[i] - individual_ap.V[curr_idx])

            t_diff = abs(individual_ap.t[curr_idx] - t)
            
            #penalty for large CL difference
            if t_diff > 10:
                errors.append(diff*3)
            else:
                errors.append(diff)

        return sum(errors)


