import os
import datetime
from ._parameters import (
    Parameter as BaseParameter, parameter_registry, ConstantParameter,
    ConstantScenarioParameter, AnnualHarmonicSeriesParameter,
    ArrayIndexedParameter, ConstantScenarioParameter,
    ArrayIndexedScenarioMonthlyFactorsParameter,
    DailyProfileParameter, ArrayIndexedScenarioParameter,
    IndexParameter, CachedParameter, RecorderThresholdParameter,
    AggregatedParameter, AggregatedIndexParameter,
    load_parameter, load_parameter_values, load_dataframe)
from past.builtins import basestring
import numpy as np
import pandas
from pywr.h5tools import H5Store

class Parameter(BaseParameter):
    def value(self, ts, scenario_index):
        raise NotImplementedError()


class FunctionParameter(Parameter):
    def __init__(self, parent, func, *args, **kwargs):
        super(FunctionParameter, self).__init__(*args, **kwargs)
        self._parent = parent
        self._func = func

    def value(self, ts, scenario_index):
        return self._func(self._parent, ts, scenario_index)
FunctionParameter.register()


class MonthlyProfileParameter(Parameter):
    def __init__(self, values, lower_bounds=0.0, upper_bounds=np.inf, **kwargs):
        super(MonthlyProfileParameter, self).__init__(**kwargs)
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
MonthlyProfileParameter.register()


class ScaledProfileParameter(Parameter):
    def __init__(self, scale, profile, *args, **kwargs):
        super(ScaledProfileParameter, self).__init__(*args, **kwargs)
        self.scale = scale

        profile.parents.add(self)
        self.profile = profile

    @classmethod
    def load(cls, model, data):
        scale = float(data.pop("scale"))
        profile = load_parameter(model, data.pop("profile"))
        return cls(scale, profile, **data)

    def value(self, ts, si):
        p = self.profile.value(ts, si)
        return self.scale * p
ScaledProfileParameter.register()


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
    def __init__(self, df, scenario=None, metadata=None, **kwargs):
        super(DataFrameParameter, self).__init__(**kwargs)
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
            scenario = model.scenarios[scenario]
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

        param.parents.add(self)
        self._param = param

    def value(self, ts, scenario_index):
        return self._param.value(ts, scenario_index)
DataFrameParameter.register()


class TablesArrayParameter(IndexParameter):
    def __init__(self, h5file, node, where='/', scenario=None, **kwargs):
        """
        This Parameter reads array data from a PyTables HDF database.

        The parameter reads data using the PyTables array interface and therefore
        does not require loading the entire dataset in to memory. This is useful
        for large model runs.

        Parameters
        ----------
        h5file : tables.File or filename
            The tables file handle or filename to attach the CArray objects to. If a
            filename is given the object will open and close the file handles.
        node : string
            Name of the node in the tables database to read data from
        where : string
            Path to read the node from.
        scenario : Scenario
            Scenario to use as the second index in the array.
        """
        super(TablesArrayParameter, self).__init__(**kwargs)

        self.h5file = h5file
        self.h5store = None
        self.node = node
        self.where = where
        self.scenario = scenario

        # Private attributes, initialised during reset()
        self._node = None
        self._scenario_index = None

    def setup(self, model):
        self.model = model
        self._scenario_index = None
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        if self.scenario is not None:
            self._scenario_index = model.scenarios.get_scenario_index(self.scenario)

    def reset(self):
        self.h5store = H5Store(self.h5file, None, "r")
        self._node = self.h5store.file.get_node(self.where, self.node)
        if self.scenario is not None:
            if self._node.shape[1] != self.scenario.size:
                raise RuntimeError("The second length of the dimension of the tables Node should the same as the size of the specified Scenario.")
        if self._node.shape[0] < len(self.model.timestepper):
            raise IndexError("The length of the first dimension of the tables Node should be equal to or greater than the number of timesteps.")

    def value(self, ts, scenario_index):
        i = ts.index

        # Support 1D and 2D indexing when scenario is or is not given.
        if self._scenario_index is None:
            return self._node[i]
        else:
            j = scenario_index.indices[self._scenario_index]
            return self._node[i, j]

    def index(self, ts, scenario_index):
        return self.value(ts, scenario_index)

    def finish(self):
        self.h5store = None

    @classmethod
    def load(cls, model, data):
        scenario = data.pop('scenario', None)
        if scenario is not None:
            scenario = model.scenarios[scenario]

        url = data.pop('url')
        if not os.path.isabs(url) and model.path is not None:
            url = os.path.join(model.path, url)
        node = data.pop('node')
        where = data.pop('where', '/')

        return cls(url, node, where=where, scenario=scenario)
TablesArrayParameter.register()


class InterpolatedLevelParameter(Parameter):
    """
    Level parameter calculated by interpolation from current volume
    """
    def __init__(self, node, volumes, levels, kind='linear', **kwargs):
        super(InterpolatedLevelParameter, self).__init__(**kwargs)
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
