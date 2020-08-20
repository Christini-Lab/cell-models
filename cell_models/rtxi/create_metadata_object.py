from h5py import File

class RTXIMeta():
    """
    Holds all information for extracting data from an h5 file for processing and input into a GA.
    """
    def __init__(self, file_path, trial, t_range, protocol_type):
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

def get_cell_capacitance(h5_file):
    f = raw_exp_file = File(h5_file, 'r')

    cm_key = [key for key in f['Trial1']['Parameters'].keys() if 'Cm' in key][0]
    
    cm = f['Trial1']['Parameters'][cm_key].value[0][1] * 1E-12

    return cm
