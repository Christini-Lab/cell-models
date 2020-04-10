"""Contains protocols to act in silico to probe cellular mechanics."""

import bisect
from typing import List, Union

class SpontaneousProtocol:
    """Encapsulates state and behavior of a single action potential protocol."""

    def __init__(self, duration: float=1.8):
        self.duration = duration

class PacedProtocol:
    """Encapsulates state and behavior of a paced protocol"""
    def __init__(self, model_name, stim_end=6000,
                 stim_start=10, pace=1):
        """
        model_name: "Paci", "Kernik", "OR"

        """
        if (model_name == "Kernik"):
            self.stim_amplitude = 180
            self.stim_duration = 5
        elif model_name == "OR":
            self.stim_amplitude = 80
            self.stim_duration = 1
        elif (model_name == "Paci"):
            self.stim_amplitude = 220 
            self.stim_duration = 5/1000

        self.pace = pace
        self.stim_end = stim_end
        self.stim_start = stim_start

class IrregularPacingProtocol:
    """Encapsulates state and behavior of a irregular pacing protocol.

    Attributes:
        duration: Duration of integration.
        stimulation_offsets: Each offset corresponds to the
            seconds after diastole begins that stimulation will
            occur. Cannot exceed `max_stim_interval_duration`, which is the
            time between beats when cell is pacing naturally.
    """

    # The start of a diastole must be below this voltage, in Vm.
    DIAS_THRESHOLD_VOLTAGE = -0.06

    # Set to time between naturally occurring spontaneous beats.
    _MAX_STIM_INTERVAL = 1.55

    STIM_AMPLITUDE_AMPS = 7.5e-10
    STIM_DURATION_SECS = 0.005

    def __init__(self, duration: int, stimulation_offsets: List[float]) -> None:
        self.duration = duration
        self.stimulation_offsets = stimulation_offsets
        self.all_stimulation_times = []

    @property
    def stimulation_offsets(self):
        return self._stimulation_offsets

    @stimulation_offsets.setter
    def stimulation_offsets(self, offsets):
        for i in offsets:
            if i > self._MAX_STIM_INTERVAL:
                raise ValueError(
                    'Stimulation offsets from diastolic start cannot be '
                    'greater than `self.max_stim_interval_duration` because '
                    'the cell will have started to spontaneously beat.')
        self._stimulation_offsets = offsets

    def make_offset_generator(self):
        return (i for i in self._stimulation_offsets)

class VoltageClampStep:
    """A step in a voltage clamp protocol."""

    def __init__(self, voltage: float, duration: float) -> None:
        self.voltage = voltage
        self.duration = duration

    def __str__(self):
        return 'Voltage: {}, Duration: {}'.format(self.voltage, self.duration)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (abs(self.voltage - other.voltage) < 0.001 and
                abs(self.duration - other.duration) < 0.001)

class VoltageClampProtocol:
    """Encapsulates state and behavior of a voltage clamp protocol."""

    HOLDING_STEP = VoltageClampStep(voltage=-80.0, duration=1.0)

    def __init__(self, steps: List[VoltageClampStep]=[
                 VoltageClampStep(duration=50.0, voltage=-80.0),
                 VoltageClampStep(duration=50.0, voltage=-120.0),
                 VoltageClampStep(duration=500.0, voltage=-57.0),
                 VoltageClampStep(duration=25.0, voltage=-40.0),
                 VoltageClampStep(duration=75.0, voltage=20),
                 VoltageClampStep(duration=25.0, voltage=-80.0),
                 VoltageClampStep(duration=250.0, voltage=40.0),
                 VoltageClampStep(duration=1900.0, voltage=-30.0),
                 VoltageClampStep(duration=750.0, voltage=40.0),
                 VoltageClampStep(duration=1725.0, voltage=-30.0),
                 VoltageClampStep(duration=650.0, voltage=-80.0)]):
        if steps:
            self.steps = steps
        else:
            self.steps = []

        #[VoltageClampStep(duration=.050, voltage=-.080),
        # VoltageClampStep(duration=.050, voltage=-.120),
        # VoltageClampStep(duration=.500, voltage=-.057),
        # VoltageClampStep(duration=.025, voltage=-.040),
        # VoltageClampStep(duration=.075, voltage=.020),
        # VoltageClampStep(duration=.025, voltage=-.080),
        # VoltageClampStep(duration=.250, voltage=.040),
        # VoltageClampStep(duration=1.900, voltage=-.030),
        # VoltageClampStep(duration=.750, voltage=.040),
        # VoltageClampStep(duration=1.725, voltage=-.030),
        # VoltageClampStep(duration=.650, voltage=-.080)]

    def __eq__(self, other):

        if not isinstance(other, self.__class__):
            return False

        if len(other.steps) != len(self.steps):
            return False

        for i in range(len(other.steps)):
            if other.steps[i] != self.steps[i]:
                return False
        return True

    def __str__(self):
        return ' | '.join([i.__str__() for i in self.steps])

    def __repr__(self):
        return self.__str__()

    def get_voltage_change_endpoints(self) -> List[float]:
        """Initializes voltage change endpoints based on the steps provided.

        For example, if the steps provided are:
            VoltageClampStep(voltage=1, duration=1),
            VoltageClampStep(voltage=2, duration=0.5),
        the voltage change points would be at 1 second and 1.5 seconds.

        Returns:
            A list of voltage change endpoints.
        """

        voltage_change_endpoints = []
        cumulative_time = 0
        for i in self.steps:
            cumulative_time += i.duration
            voltage_change_endpoints.append(cumulative_time)
        return voltage_change_endpoints

    def get_voltage_at_time(self, time: float) -> float:
        """Gets the voltage based on provided steps for the specified time."""
        step_index = bisect.bisect_left(
            self.get_voltage_change_endpoints(),
            time)
        if step_index != len(self.get_voltage_change_endpoints()):
            return self.steps[step_index].voltage
        raise ValueError('End of voltage protocol.')


PROTOCOL_TYPE = Union[
    SpontaneousProtocol,
    IrregularPacingProtocol,
    VoltageClampProtocol,
    PacedProtocol
]
