import os
import datetime
from ._parameters import (
    Parameter as BaseParameter, parameter_registry,
    ConstantScenarioParameter,
    ArrayIndexedParameter, ConstantScenarioParameter,
    ArrayIndexedScenarioMonthlyFactorsParameter,
    DailyProfileParameter, ArrayIndexedScenarioParameter,
    load_parameter, load_dataframe)
import numpy as np
import pandas

class Parameter(BaseParameter):
    def value(self, ts, scenario_index):
        raise NotImplementedError()

class ParameterCollection(Parameter):
    """A collection of Parameters

    This object behaves like a set. Licenses can be added to or removed from it.

    """
    def __init__(self, parameters=None):
        super(ParameterCollection, self).__init__()
        if parameters is None:
            self._parameters = set()
        else:
            self._parameters = set(parameters)
            for param in self._parameters:
                param.parent = self

    def add(self, parameter):
        self._parameters.add(parameter)
        parameter.parent = self

    def remove(self, parameter):
        self._parameters.remove(parameter)
        parameter.parent = None

    def __len__(self):
        return len(self._parameters)

    def value(self, timestep, scenario_index):
        raise NotImplementedError()

    def setup(self, model):
        for parameter in self._parameters:
            parameter.setup(model)

    def after(self, timestep):
        for parameter in self._parameters:
            parameter.after(timestep)

    def reset(self):
        for parameter in self._parameters:
            parameter.reset()
parameter_registry.add(ParameterCollection)


class MinimumParameterCollection(ParameterCollection):
    def value(self, timestep, scenario_index):
        min_available = float('inf')
        for parameter in self._parameters:
            min_available = min(parameter.value(timestep, scenario_index), min_available)
        return min_available
parameter_registry.add(MinimumParameterCollection)


class MaximumParameterCollection(ParameterCollection):
    def value(self, timestep, scenario_index):
        max_available = -float('inf')
        for parameter in self._parameters:
            max_available = max(parameter.value(timestep, scenario_index), max_available)
        return max_available
parameter_registry.add(MaximumParameterCollection)


class ConstantParameter(Parameter):
    def __init__(self, value=None, lower_bounds=0.0, upper_bounds=np.inf):
        super(ConstantParameter, self).__init__()
        self._value = value
        self.size = 1
        self._lower_bounds = np.ones(self.size) * lower_bounds
        self._upper_bounds = np.ones(self.size) * upper_bounds

    def value(self, ts, scenario_index):
        return self._value

    def update(self, values):
        self._value = values[0]

    def lower_bounds(self):
        return self._lower_bounds

    def upper_bounds(self):
        return self._upper_bounds
parameter_registry.add(ConstantParameter)

class FunctionParameter(Parameter):
    def __init__(self, parent, func):
        super(FunctionParameter, self).__init__()
        self._parent = parent
        self._func = func

    def value(self, ts, scenario_index):
        return self._func(self._parent, ts, scenario_index)
parameter_registry.add(FunctionParameter)


class MonthlyProfileParameter(Parameter):
    def __init__(self, values, lower_bounds=0.0, upper_bounds=np.inf):
        super(MonthlyProfileParameter, self).__init__()
        self.size = 12
        if len(values) != self.size:
            raise ValueError("12 values must be given for a monthly profile.")
        self._values = np.array(values)
        self._lower_bounds = np.ones(self.size)*lower_bounds
        self._upper_bounds = np.ones(self.size)*upper_bounds

    def value(self, ts, scenario_index):
        return self._values[ts.datetime.month-1]

    def update(self, values):
        self._values[...] = values

    def lower_bounds(self):
        return self._lower_bounds

    def upper_bounds(self):
        return self._upper_bounds
parameter_registry.add(MonthlyProfileParameter)


class AnnualHarmonicSeriesParameter(Parameter):
    def __init__(self, mean, amplitudes, phases, **kwargs):
        super(AnnualHarmonicSeriesParameter, self).__init__()
        if len(amplitudes) != len(phases):
            raise ValueError("The number  of amplitudes and phases must be the same.")
        n = len(amplitudes)
        self.size = 1 + 2*n
        self.mean = mean
        self.amplitudes = np.array(amplitudes)
        self.phases = np.array(phases)

        self._mean_lower_bounds = kwargs.pop('mean_lower_bounds', 0.0)
        self._mean_upper_bounds = kwargs.pop('mean_upper_bounds', np.inf)
        self._amplitude_lower_bounds = np.ones(n)*kwargs.pop('amplitude_lower_bounds', 0.0)
        self._amplitude_upper_bounds = np.ones(n)*kwargs.pop('amplitude_upper_bounds', np.inf)
        self._phase_lower_bounds = np.ones(n)*kwargs.pop('phase_lower_bounds', 0.0)
        self._phase_upper_bounds = np.ones(n)*kwargs.pop('phase_upper_bounds', np.pi*2)

    def value(self, ts, scenario_index):
        doy = ts.datetime.dayofyear
        n = len(self.amplitudes)
        return self.mean + sum(self.amplitudes[i]*np.cos(doy*(i+1)*np.pi*2/365 + self.phases[i]) for i in range(n))

    def update(self, values):
        n = len(self.amplitudes)
        self.mean = values[0]
        self.amplitudes[...] = values[1:n+1]
        self.phases[...] = values[n+1:]

    def lower_bounds(self):
        return np.r_[self._mean_lower_bounds, self._amplitude_lower_bounds, self._phase_lower_bounds]

    def upper_bounds(self):
        return np.r_[self._mean_upper_bounds, self._amplitude_upper_bounds, self._phase_upper_bounds]
parameter_registry.add(AnnualHarmonicSeriesParameter)


def align_and_resample_dataframe(df, datetime_index):
    # Must resample and align the DataFrame to the model.
    start = datetime_index[0]
    end = datetime_index[-1]

    df_index = df.index

    if df_index[0] > start:
        raise ValueError('DataFrame data begins after the index start date.')
    if df_index[-1] < end:
        raise ValueError('DataFrame data ends before the index end date.')

    # Downsampling (i.e. from high freq to lower model freq)
    if datetime_index.freq >= df_index.freq:
        # Slice to required dates
        df = df[start:end]
        if df.index[0] != start:
            raise ValueError('Start date of DataFrame can not be aligned with the desired index start date.')
        # Take mean at the model's frequency
        df = df.resample(datetime_index.freq).mean()
    else:
        raise NotImplementedError('Upsampling DataFrame not implemented.')

    return df


class DataFrameParameter(Parameter):
    def __init__(self, df, scenario=None, metadata=None):
        super(DataFrameParameter, self).__init__()
        self.df = df
        if metadata is None:
            metadata = {}
        self.metadata = metadata
        self.scenario = scenario
        self._param = None

    @classmethod
    def load(cls, model, data):
        scenario = data.pop('scenario', None)
        if scenario is not None:
            raise NotImplementedError('Loading Scenarios not implemented in JSON.')
        df = load_dataframe(model, data)
        return cls(df, scenario=scenario)

    def setup(self, model):

        df = align_and_resample_dataframe(self.df, model.timestepper.datetime_index)

        if df.ndim == 1:
            # Single timeseries for the entire run
            param = ArrayIndexedParameter(df.values.astype(dtype=np.float64))
        elif df.shape[1] == 1:
            # DataFrame with one column for the entire run
            param = ArrayIndexedParameter(df.values[:, 0].astype(dtype=np.float64))
        else:
            if self.scenario is None:
                raise ValueError("Scenario must be given for a DataFrame input with multiple columns.")
            if self.scenario.size != df.shape[1]:
                raise ValueError("Scenario size ({}) is different to the number of columns ({}) "
                                 "in the DataFrame input.".format(self.scenario.size, df.shape[1]))
            # We assume the columns are in the correct order for the scenario.
            param = ArrayIndexedScenarioParameter(self.scenario, df.values.astype(dtype=np.float64))

        param.parent = self
        self._param = param

    def value(self, ts, scenario_index):
        return self._param.value(ts, scenario_index)
parameter_registry.add(DataFrameParameter)


class InterpolatedLevelParameter(Parameter):
    """
    Level parameter calculated by interpolation from current volume
    """
    def __init__(self, volumes, levels, kind='linear'):
        from scipy.interpolate import interp1d
        # Create level interpolator
        self.interp = interp1d(volumes, levels, bounds_error=True, kind=kind)

    def value(self, ts, scenario_index):
        # Return interpolated value from current volume
        v = self.node.volume[scenario_index.global_id]
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
