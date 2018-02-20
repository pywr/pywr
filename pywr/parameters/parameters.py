import os
import datetime
from ..parameter_property import parameter_property
from ._parameters import (
    Parameter, parameter_registry, UnutilisedDataWarning, ConstantParameter,
    ConstantScenarioParameter, AnnualHarmonicSeriesParameter,
    ArrayIndexedParameter, ConstantScenarioParameter, IndexedArrayParameter,
    ArrayIndexedScenarioMonthlyFactorsParameter, TablesArrayParameter,
    DailyProfileParameter, MonthlyProfileParameter, WeeklyProfileParameter,
    ArrayIndexedScenarioParameter, ScenarioMonthlyProfileParameter,
    align_and_resample_dataframe, DataFrameParameter,
    IndexParameter, AggregatedParameter, AggregatedIndexParameter,
    NegativeParameter, MaxParameter, NegativeMaxParameter, MinParameter, NegativeMinParameter,
    DeficitParameter, load_parameter, load_parameter_values, load_dataframe)
from . import licenses
from ._polynomial import Polynomial1DParameter, Polynomial2DStorageParameter
from ._thresholds import StorageThresholdParameter, RecorderThresholdParameter
from past.builtins import basestring
import numpy as np
from scipy.interpolate import interp1d
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


class AbstractInterpolatedParameter(Parameter):
    def __init__(self, model, x, y, interp_kwargs=None, **kwargs):
        super(AbstractInterpolatedParameter, self).__init__(model, **kwargs)
        self.x = x
        self.y = y
        self.interp = None
        default_interp_kwargs = dict(kind='linear', bounds_error=True)
        if interp_kwargs is not None:
            # Overwrite or add to defaults with given values
            default_interp_kwargs.update(interp_kwargs)
        self.interp_kwargs = default_interp_kwargs

    def _value_to_interpolate(self, ts, scenario_index):
        raise NotImplementedError()

    def setup(self):
        super(AbstractInterpolatedParameter, self).setup()
        self.interp = interp1d(self.x, self.y, **self.interp_kwargs)

    def value(self, ts, scenario_index):
        v = self._value_to_interpolate(ts, scenario_index)
        return self.interp(v)


class InterpolatedParameter(AbstractInterpolatedParameter):
    """
    Parameter value is equal to the interpolation of another parameter

    Example
    -------
    >>> x = [0, 5, 10, 20]
    >>> y = [0, 10, 30, -5]
    >>> p1 = ConstantParameter(model, 9.3) # or something more interesting
    >>> p2 = InterpolatedParameter(model, x, y, interp_kwargs={"kind": "linear"})
    """
    def __init__(self, model, parameter, x, y, interp_kwargs=None, **kwargs):
        super(InterpolatedParameter, self).__init__(model, x, y, interp_kwargs, **kwargs)
        self._parameter = None
        self.parameter = parameter

    parameter = parameter_property("_parameter")

    def _value_to_interpolate(self, ts, scenario_index):
        return self._parameter.get_value(scenario_index)


class InterpolatedVolumeParameter(AbstractInterpolatedParameter):
    """
    Generic interpolation parameter calculated from current volume
    """
    def __init__(self, model, node, x, y, interp_kwargs=None, **kwargs):
        super(InterpolatedVolumeParameter, self).__init__(model, x, y, interp_kwargs, **kwargs)
        self._node = node

    def _value_to_interpolate(self, ts, scenario_index):
        return self._node.volume[scenario_index.global_id]

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        volumes = np.array(data.pop("volumes"))
        values = np.array(data.pop("values"))
        kind = data.pop("kind", "linear")
        return cls(model, node, volumes, values, interp_kwargs={'kind': kind})
InterpolatedVolumeParameter.register()


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
