# -*- coding: utf-8 -*-
from typing_extensions import TYPE_CHECKING
from typing import Iterable, Optional

if TYPE_CHECKING:
    from ..model import Model
    from ..core import ScenarioIndex, Timestep
    from ..nodes import Storage

from ._parameters import Parameter
from ..core import Storage
import numpy as np


class KeatingStreamFlowParameter(Parameter):
    """This parameter returns the stream flow from an Aquifer based on the groundwater level.

    The approach is based on the lumped parameter model by [Keating (1982)](doi.com/10.1111/j.1745-6584.1982.tb02763.x).
    This Parameter calculates the perennial and winterbourne stream flow components based on the Storage node.
    It requires the Storage node to have a valid level Parameter.

    In contrast to the Keating paper, a general coefficient is provided for calibration. The Keating approach
    utilised a coefficient of `B/L` where "`B` and `L` are the dimensions of the aquifer block parallel
    and perpendicular to the stream."

        Keating, T. (1982), A Lumped Parameter Model of a Chalk Aquifer-Stream System in Hampshire,
        United Kingdom. Ground Water, 20: 430–436. doi:10.1111/j.1745-6584.1982.tb02763.x

    Attributes
    ----------
    model : Model
        The model instance.
    storage_node : Storage
        The storage node to use to model the aquifer.
    levels : Iterable[float]
        The list of levels.
    transmissivity : Iterable[float]
        The transmissivity coefficients for each level.
    coefficient : float
        The calibration coefficient.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """

    def __init__(
        self,
        model: "Model",
        storage_node: "Storage",
        levels: list[float],
        transmissivity: list[float],
        coefficient: Optional[float] = 1.0,
        **kwargs,
    ):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        storage_node : Storage
            The storage node to use to model the aquifer.
        levels : Iterable[float]
            The list of levels.
        transmissivity : Iterable[float]
            The transmissivity coefficients for each level.
        coefficient : Optional[float], default=1.0
            The calibration coefficient.

        Other Parameters
        ----------------
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.

        Raises
        ------
        ValueError
            If the number of transmissivity values does not equal the number of levels.
        """
        super(KeatingStreamFlowParameter, self).__init__(model, **kwargs)
        self.storage_node = storage_node

        if len(levels) != len(transmissivity):
            raise ValueError(
                "The number of transmissivity values must equal the number of levels."
            )

        self.levels = np.array(levels)
        self.transmissivity = np.array(transmissivity)
        self.coefficient = coefficient

    def value(self, ts: "Timestep", scenario_index: "ScenarioIndex") -> float:
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.

        Returns
        -------
        float
            The parameter value.
        """
        # Get the current level of the aquifer
        # TODO: this is a HACK - we can't use get_level/get_value as there is
        #       no way to define the parent/child relationship
        level = self.storage_node.level.value(ts, scenario_index)

        # Coefficient
        C = self.coefficient

        # Calculate flow at each stream level
        Q = 0.0
        for n, stream_flow_level in enumerate(self.levels):
            T = self.transmissivity[n]
            if level > stream_flow_level:
                Q += 2 * T * C * (level - stream_flow_level)

        return Q
