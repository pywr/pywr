"""
This module contains a set of pywr._core.Parameter subclasses for defining control curve based parameters.
"""

from ._control_curves import BaseControlCurveParameter, ControlCurveInterpolatedParameter


class ControlCurvePiecewiseParameter(BaseControlCurveParameter):
    """ A control curve Parameter that returns one of two values depending on whether the current
    volume is above or below the control curve.

    """
    def __init__(self, control_curve, above_curve_value, below_curve_value, storage_node=None):
        """

        :param control_curve: Parameter to use as the control curve
        :param above_curve_value: Value to return when storage >= control_curve
        :param below_curve_value: Value to return when storage < control_curve
        :param storage_node: Optional Storage node to compare with the control_curve. If None
            self.node is used (i.e. the Node this object is attached to).
        """
        super(ControlCurvePiecewiseParameter, self).__init__(control_curve, storage_node=storage_node)

        self.above_curve_value = above_curve_value
        self.below_curve_value = below_curve_value

    def value(self, ts, scenario_indices=[0]):
        i = self.node.model.scenarios.ravel_indices(scenario_indices)
        control_curve = self.control_curve.value(ts, scenario_indices)
        node = self.node if self.storage_node is None else self.storage_node
        # If level above control curve then return above_curve_cost
        if node.current_pc[i] >= control_curve:
            return self.above_curve_value
        return self.below_curve_value