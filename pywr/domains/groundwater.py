from ..core import Storage
from ..parameters import InterpolatedLevelParameter
from ..parameters.groundwater import KeatingStreamFlowParameter


class KeatingAquifer(Storage):
    def __init__(self, model, name,
                 num_streams, num_additional_inputs,
                 levels, volumes,
                 stream_flow_levels, transmissivity, coefficient,
                 **kwargs):
        """Storage node with one or more Keating outflows

        Parameters
        ----------
        model : pywr.core.Model
            The Pywr Model.
        name : string
            A unique name for the node in the model.
        num_streams : integer
            Number of keating outflows.
        num_additional_inputs : integer
            Number of additional outflows (e.g. for direct abstraction or
            discharge from the aquifer).
        levels : list of floats
            A list of levels for the level-volume relationship. The length
            should be greater than 1.
        volumes : list of floats
            A list of volumes for the level-volume relationship. The length
            should be the same as `levels`.
        stream_flow_levels : list of list of floats
            For each stream a list of levels to pass to the keating streamflow
            parameter.
        transmissivity : list of floats
            The transmissivity for each stream flow level.
        coefficient : list of floats
            The coefficient for each stream flow level.

        See also documentation for the `KeatingStreamFlowParameter`.
        """
        super(KeatingAquifer, self).__init__(model, name,
            num_inputs=(num_streams + num_additional_inputs), **kwargs)

        self.max_volume = max(volumes)
        self.min_volume = min(volumes)
        self.level = InterpolatedLevelParameter(volumes, levels)

        # initialise streamflow parameters
        for n, node in enumerate(self.inputs[0:num_streams]):
            parameter = KeatingStreamFlowParameter(self, stream_flow_levels[n],
                                                   transmissivity,
                                                   coefficient)
            node.max_flow = parameter
            node.min_flow = parameter
