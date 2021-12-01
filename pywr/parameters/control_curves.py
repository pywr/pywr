"""
This module contains a set of pywr._core.Parameter subclasses for defining control curve based parameters.
"""
from ._control_curves import (
    ControlCurveParameter,
    BaseControlCurveParameter,
    ControlCurveInterpolatedParameter,
    ControlCurveIndexParameter,
    ControlCurvePiecewiseInterpolatedParameter,
)
