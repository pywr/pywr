from ..core import Storage
from ..parameters import InterpolatedLevelParameter
from ..parameters.groundwater import KeatingStreamFlowParameter


class KeatingAquifer(Storage):
    def __init__(self, model, name, levels, volumes, stream_flow_levels, transmissivity, coefficient=1.0, **kwargs):
        super(KeatingAquifer, self).__init__(model, name, **kwargs)

        self.max_volume = max(volumes)
        self.min_volume = min(volumes)
        self.level = InterpolatedLevelParameter(volumes, levels)
        # Make the first StorageInput node the aquifer outflow
        inpt = self.inputs[0]
        param = KeatingStreamFlowParameter(self, stream_flow_levels, transmissivity, coefficient=coefficient)
        # Ensure the StorageInput flows out the prescribed amount.
        inpt.max_flow = param
        inpt.min_flow = param