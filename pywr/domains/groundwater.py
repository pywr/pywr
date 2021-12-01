from ..core import Storage
from ..parameters import InterpolatedVolumeParameter
from ..parameters.groundwater import KeatingStreamFlowParameter

import numbers
from scipy.interpolate import interp1d


class KeatingAquifer(Storage):
    def __init__(
        self,
        model,
        name,
        num_streams,
        num_additional_inputs,
        stream_flow_levels,
        transmissivity,
        coefficient,
        levels,
        volumes=None,
        area=None,
        storativity=None,
        **kwargs,
    ):
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
        stream_flow_levels : list of list of floats
            For each stream a list of levels to pass to the keating streamflow
            parameter.
        transmissivity : list of floats
            The transmissivity for each stream flow level.
        coefficient : list of floats
            The coefficient for each stream flow level.
        levels : list of floats
            A list of levels for the level-volume relationship. The length
            should be greater than 1.
        volumes : list of floats (optional)
            A list of volumes for the level-volume relationship. The length
            should be the same as `levels`.
        area : float (optional)
            Area of the aquifer in m2.
        storativity : list of floats (optional)
            Storativity of the aquifer as a factor (e.g. 0.05). This defines
            part of the volume-level relationship. The length should be one
            less than `levels`.

        Either supply the `volumes` argument or both the `area` and
        `storativity` arguments.

        See also documentation for the `KeatingStreamFlowParameter`.
        """
        super(KeatingAquifer, self).__init__(
            model, name, inputs=(num_streams + num_additional_inputs), **kwargs
        )

        if not (num_streams > 0):
            raise ValueError("Keating aquifer must have at least one stream outflow")
        if len(stream_flow_levels) != num_streams:
            raise ValueError("Stream flow levels must have `num_streams` items")
        for i in stream_flow_levels:
            if len(i) != len(transmissivity):
                raise ValueError(
                    "Items in stream flow levels should have the same length as transmissivity"
                )
        if not isinstance(coefficient, numbers.Number):
            raise ValueError("Coefficient must be a scalar")

        if volumes is None:
            if not isinstance(area, numbers.Number):
                raise ValueError("Area must be a scalar")
            if len(storativity) != (len(levels) - 1):
                raise ValueError("Storativity must have one less item than levels")
            heights = [levels[n + 1] - levels[n] for n in range(0, len(levels) - 1)]
            volumes = [0.0]
            for n, (s, h) in enumerate(zip(storativity, heights)):
                volumes.append(volumes[-1] + area * s * h * 0.001)
        else:
            # check volumes
            if len(volumes) != len(levels):
                raise ValueError("Volumes must have the same length as levels")

        self.area = area

        if len(levels) != len(volumes):
            raise ValueError("Levels and volumes must have the same length")

        self._volumes = volumes
        self._levels = levels
        self._level_to_volume = interp1d(levels, volumes)

        self.max_volume = max(volumes)
        self.min_volume = min(volumes)

        self.level = InterpolatedVolumeParameter(model, self, volumes, levels)

        # initialise streamflow parameters
        for n, node in enumerate(self.inputs[0:num_streams]):
            parameter = KeatingStreamFlowParameter(
                model, self, stream_flow_levels[n], transmissivity, coefficient
            )
            node.max_flow = parameter
            node.min_flow = parameter

    def initial_level():
        def fget(self):
            # get the initial level from the volume
            return self.level.interp(self.initial_volume)

        def fset(self, value):
            # actually sets the initial volume
            volume = self._level_to_volume(value)
            self.initial_volume = volume

        return locals()

    initial_level = property(**initial_level())
