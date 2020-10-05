from h5py import File

class RTXIMeta():
    """
    Holds all information for extracting data from an h5 file for processing and input into a GA.
    """
    def __init__(self, file_path, trial, t_range, protocol_type, max_current_ranges=None):
        """
        Parameters:
            file_path (str) – path to the h5 files
            trial (int) – trial number for the data of interest
            t_range ([int]) – range of times in the given trial
            protocol_type (str) – one of: Spontaneous, Dynamic Clamp,
                or Voltage Clamp
            mem_capacitance (num) – cell membrane capacitance
        """
        self.file_path = file_path
        self.trial = trial
        self.t_range = t_range
        self.protocol_type = protocol_type
        self.mem_capacitance = get_cell_capacitance(self.file_path)
        self.max_current_ranges = max_current_ranges

def get_cell_capacitance(h5_file):
    f = File(h5_file, 'r')

    cm = 0

    for k in f.keys():
        if 'Trial' in k:
            for k, v in f[k]['Parameters'].items():
                if 'Cm' in k:
                    cm_temp = f['Trial1']['Parameters'][k].value[0][1] * 1E-12
                    if cm_temp > cm:
                        cm = cm_temp

    return cm
