import os
import datetime
from ._parameters import (
    Parameter, parameter_registry, ConstantParameter,
    ConstantScenarioParameter, AnnualHarmonicSeriesParameter,
    ArrayIndexedParameter, ConstantScenarioParameter,
    ArrayIndexedScenarioMonthlyFactorsParameter, TablesArrayParameter,
    DailyProfileParameter, MonthlyProfileParameter,
    ArrayIndexedScenarioParameter, ScenarioMonthlyProfileParameter,
    AggregatedParameter, AggregatedIndexParameter,
    align_and_resample_dataframe, DataFrameParameter,
    IndexParameter, AggregatedParameter, AggregatedIndexParameter,
    load_parameter, load_parameter_values, load_dataframe)
from ._polynomial import Polynomial1DParameter, Polynomial2DStorageParameter
from ._thresholds import StorageThresholdParameter, RecorderThresholdParameter
from past.builtins import basestring
import numpy as np
import pandas


class FunctionParameter(Parameter):
    def __init__(self, model, parent, func, *args, **kwargs):
        super(FunctionParameter, self).__init__(model, *args, **kwargs)
        self._parent = parent
        self._func = func

    def value(self, ts, scenario_index):
        return self._func(self._parent, ts, scenario_index)
FunctionParameter.register()


class ScaledProfileParameter(Parameter):
    def __init__(self, model, scale, profile, *args, **kwargs):
        super(ScaledProfileParameter, self).__init__(model, *args, **kwargs)
        self.scale = scale

        profile.parents.add(self)
        self.profile = profile

    @classmethod
    def load(cls, model, data):
        scale = float(data.pop("scale"))
        profile = load_parameter(model, data.pop("profile"))
        return cls(model, scale, profile, **data)

    def value(self, ts, si):
        p = self.profile.get_value(si)
        return self.scale * p
ScaledProfileParameter.register()


class InterpolatedLevelParameter(Parameter):
    """
    Level parameter calculated by interpolation from current volume
    """
    def __init__(self, model, node, volumes, levels, kind='linear', **kwargs):
        super(InterpolatedLevelParameter, self).__init__(model, **kwargs)
        from scipy.interpolate import interp1d
        # Create level interpolator
        self.interp = interp1d(volumes, levels, bounds_error=True, kind=kind)
        self._node = node

    def value(self, ts, scenario_index):
        # Return interpolated value from current volume
        v = self._node.volume[scenario_index.global_id]
        level = self.interp(v)
        return level


def pop_kwarg_parameter(kwargs, key, default):
    """Pop a parameter from the keyword arguments dictionary

    Parameters
    ----------
    kwargs : dict
        A keyword arguments dictionary
    key : string
        The argument name, e.g. 'flow'
    default : object
        The default value to use if the dictionary does not have that key

    Returns a Parameter
    """
    value = kwargs.pop(key, default)
    if isinstance(value, Parameter):
        return value
    elif callable(value):
        # TODO this is broken?
        return FunctionParameter(self, value)
    else:
        return value


class PropertiesDict(dict):
    def __setitem__(self, key, value):
        if not isinstance(value, Property):
            value = ConstantParameter(value)
        dict.__setitem__(self, key, value)
