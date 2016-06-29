import os
import numpy as np
cimport numpy as np
import pandas

from past.builtins import basestring

parameter_registry = set()

class PairedSet(set):
    def __init__(self, obj, *args, **kwargs):
        set.__init__(self)
        self.obj = obj

    def add(self, item):
        set.add(self, item)
        if(self is self.obj.parents):
            set.add(item.children, self.obj)
        else:
            set.add(item.parents, self.obj)

    def remove(self, item):
        set.remove(self, item)
        if(self is self.obj.parents):
            set.remove(item.children, self.obj)
        else:
            set.remove(item.parents, self.obj)

    def clear(self):
        if(self is self.obj.parents):
            for parent in list(self):
                set.remove(parent.children, self.obj)
        else:
            for child in list(self):
                set.remove(child.parents, self.obj)
        set.clear(self)

cdef class Parameter:
    def __init__(self):
        self.parents = PairedSet(self)
        self.children = PairedSet(self)

    cpdef setup(self, model):
        cdef Parameter child
        for child in self.children:
            child.setup(model)

    cpdef reset(self):
        cdef Parameter child
        for child in self.children:
            child.reset()

    cpdef before(self, Timestep ts):
        cdef Parameter child
        for child in self.children:
            child.before(ts)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return 0

    cpdef after(self, Timestep ts):
        cdef Parameter child
        for child in self.children:
            child.after(ts)

    cpdef update(self, double[:] values):
        raise NotImplementedError()

    cpdef double[:] lower_bounds(self):
        raise NotImplementedError()

    cpdef double[:] upper_bounds(self):
        raise NotImplementedError()

    property size:
        def __get__(self):
            return self._size

        def __set__(self, value):
            self._size = value

    property is_variable:
        def __get__(self):
            return self._is_variable

        def __set__(self, value):
            self._is_variable = value

    property variables:
        def __get__(self):
            cdef Parameter var
            vars = []
            if self._is_variable:
                vars.append(self)
            for var in self.children:
                vars.extend(var.variables)
            return vars

    @classmethod
    def load(cls, model, data):
        values = load_parameter_values(model, data)
        return cls(values)

parameter_registry.add(Parameter)

cdef class CachedParameter(IndexParameter):
    """Cache a parameter which varies in both time and by scenario"""
    def __init__(self, parameter):
        super(IndexParameter, self).__init__()
        self.parameter = parameter
        self.timestep = None
        self.scenario_index = ScenarioIndex(-1, np.array([], dtype=int))

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # refresh cache if timestep or scenario has changed
        if not (timestep is self.timestep) or not (scenario_index.global_id == self.scenario_index.global_id):
            self.cached_value = self.parameter.value(timestep, scenario_index)
            self.timestep = timestep
            self.scenario_index = scenario_index
        return self.cached_value

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # refresh cache if timestep or scenario has changed
        if not (timestep is self.timestep) or not (scenario_index.global_id == self.scenario_index.global_id):
            self.cached_index = self.parameter.index(timestep, scenario_index)
            self.timestep = timestep
            self.scenario_index = scenario_index
        return self.cached_index

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data["parameter"])
        return cls(parameter)
parameter_registry.add(CachedParameter)

cdef class CachedTimeParameter(CachedParameter):
    """Cache a parameter which varies in time but not by scenario"""
    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # refresh cache if timestep has changed
        if not timestep is self.timestep:
            self.cached_value = self.parameter.value(timestep, scenario_index)
            self.timestep = timestep
        return self.cached_value
    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # refresh cache if timestep has changed
        if not timestep is self.timestep:
            self.cached_index = self.parameter.index(timestep, scenario_index)
            self.timestep = timestep
        return self.cached_index
parameter_registry.add(CachedTimeParameter)

cdef class ArrayIndexedParameter(Parameter):
    """Time varying parameter using an array and Timestep._index

    The values in this parameter are constant across all scenarios.
    """
    def __init__(self, double[:] values):
        super(ArrayIndexedParameter, self).__init__()
        self.values = values

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Returns the value of the parameter at a given timestep
        """
        return self.values[ts._index]
parameter_registry.add(ArrayIndexedParameter)


cdef class ArrayIndexedScenarioParameter(Parameter):
    """A Scenario varying Parameter

    The values in this parameter are vary in time based on index and vary within a single Scenario.
    """
    def __init__(self, Scenario scenario, double[:, :] values):
        """
        values should be an iterable that is the same length as scenario.size
        """
        super(ArrayIndexedScenarioParameter, self).__init__()
        cdef int i
        if scenario._size != values.shape[1]:
            raise ValueError("The size of the second dimension of values must equal the size of the scenario.")
        self.values = values
        self._scenario = scenario

    cpdef setup(self, model):
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self.values[ts._index, scenario_index._indices[self._scenario_index]]


cdef class ConstantScenarioParameter(Parameter):
    """A Scenario varying Parameter

    The values in this parameter are constant in time, but vary within a single Scenario.
    """
    def __init__(self, Scenario scenario, values):
        """
        values should be an iterable that is the same length as scenario.size
        """
        super(ConstantScenarioParameter, self).__init__()
        cdef int i
        if scenario._size != len(values):
            raise ValueError("The number of values must equal the size of the scenario.")
        self._values = np.empty(scenario._size)
        for i in range(scenario._size):
            self._values[i] = values[i]
        self._scenario = scenario

    cpdef setup(self, model):
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self._values[scenario_index._indices[self._scenario_index]]
parameter_registry.add(ConstantScenarioParameter)


cdef class ArrayIndexedScenarioMonthlyFactorsParameter(Parameter):
    """Time varying parameter using an array and Timestep._index with
    multiplicative factors per Scenario
    """
    def __init__(self, Scenario scenario, double[:] values, double[:, :] factors):
        """
        values is the baseline timeseries data that is perturbed by a factor. The
        factor is taken from factors which is shape (scenario.size, 12). Therefore
        factors vary with the individual scenarios in scenario and month.
        """
        super(ArrayIndexedScenarioMonthlyFactorsParameter, self).__init__()
        if scenario._size != factors.shape[0]:
            raise ValueError("First dimension of factors must be the same size as scenario.")
        if factors.shape[1] != 12:
            raise ValueError("Second dimension of factors must be 12.")
        self._scenario = scenario
        self._values = values
        self._factors = factors

    cpdef setup(self, model):
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        cdef int imth = ts.datetime.month-1
        return self._values[ts._index]*self._factors[scenario_index._indices[self._scenario_index], imth]
parameter_registry.add(ArrayIndexedScenarioMonthlyFactorsParameter)


cdef class DailyProfileParameter(Parameter):
    def __init__(self, values):
        super(DailyProfileParameter, self).__init__()
        v = np.squeeze(np.array(values))
        if v.ndim != 1:
            raise ValueError("values must be 1-dimensional.")
        if len(values) != 366:
            raise ValueError("366 values must be given for a daily profile.")
        self._values = v

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int i = ts.datetime.dayofyear-1
        return self._values[i]
parameter_registry.add(DailyProfileParameter)

cdef class IndexParameter(Parameter):
    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # return index as a float
        return float(self.index(timestep, scenario_index))

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # return index as an integer
        return 0
parameter_registry.add(IndexParameter)

cdef class IndexedArrayParameter(Parameter):
    def __init__(self, index_parameter=None, params=None, **kwargs):
        super(IndexedArrayParameter, self).__init__(**kwargs)
        assert(isinstance(index_parameter, IndexParameter))
        self.index_parameter = index_parameter
        self.params = params

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        cdef int index
        index = self.index_parameter.index(timestep, scenario_index)
        parameter = self.params[index]
        if isinstance(parameter, Parameter):
            value = parameter.value(timestep, scenario_index)
        else:
            value = parameter
        return value

    @classmethod
    def load(cls, model, data):
        index_parameter = load_parameter(model, data["index_parameter"])
        params = [load_parameter(model, data) for data in data["params"]]
        return cls(index_parameter, params)
parameter_registry.add(IndexedArrayParameter)

def load_parameter(model, data):
    """Load a parameter from a dict"""
    parameter_name = None
    if isinstance(data, basestring):
        # parameter is a reference
        try:
            parameter = model._parameters[data]
        except KeyError:
            parameter = None
        if parameter is None:
            if hasattr(model, "_parameters_to_load"):
                # we're still in the process of loading data from JSON and
                # the parameter requested hasn't been loaded yet - do it now
                name = data
                try:
                    data = model._parameters_to_load[name]
                except KeyError:
                    raise KeyError("Unknown parameter: '{}'".format(data))
                parameter = load_parameter(model, data)
            else:
                raise KeyError("Unknown parameter: '{}'".format(data))
    elif isinstance(data, (float, int)) or data is None:
        # parameter is a constant
        parameter = data
    elif "cached" in data.keys() and data["cached"]:
        # cached parameter wrapper
        cache_type = data["cached"]
        if cache_type == "time":
            cache_cls = CachedTimeParameter
        elif cache_type == "scenario":
            raise NotImplementedError("Scenario-varying cache has not been implemented (yet).")
        elif cache_type == "both" or cache_type is True:
            cache_cls = CachedParameter
        else:
            raise ValueError("""Unrecognised cache type "{}". Must be "time", "scenario" or "both".""".format(cache_type))
        del(data["cached"])
        if "name" in data:
            parameter_name = data["name"]
            del(data["name"])
        param = load_parameter(model, data)
        parameter = cache_cls(param)
    else:
        # parameter is dynamic
        parameter_type = data['type']
        try:
            parameter_name = data["name"]
        except:
            pass

        cls = None
        name2 = parameter_type.lower().replace('parameter', '')
        for parameter_class in parameter_registry:
            name1 = parameter_class.__name__.lower().replace('parameter', '')
            if name1 == name2:
                cls = parameter_class

        if cls is None:
            raise TypeError('Unknown parameter type: "{}"'.format(parameter_type))

        kwargs = dict([(k,v) for k,v in data.items()])
        del(kwargs["type"])
        if "name" in kwargs:
            del(kwargs["name"])
        parameter = cls.load(model, kwargs)

    if parameter_name is not None:
        # TODO FIXME: memory leak if parameter is subsequently removed from the model
        parameter.name = parameter_name
        model._parameters[parameter_name] = parameter

    return parameter


def load_parameter_values(model, data, values_key='values'):
    """ Function to load values from a data dictionary.

    This function tries to load values in to a `np.ndarray` if 'values_key' is
    in 'data'. Otherwise it tries to `load_dataframe` from a 'url' key.

    Parameters
    ----------
    model - `Model` instance
    data - dict
    values_key - str
        Key in data to load values directly to a `np.ndarray`

    """
    if values_key in data:
        # values are given as an array
        values = np.array(data[values_key], np.float64)
    else:
        url = data['url']
        df = load_dataframe(model, data)
        values = np.squeeze(df.values.astype(np.float64))
    return values


def load_dataframe(model, data):

    # values reference data in an external file
    url = data.pop('url')
    if not os.path.isabs(url) and model.path is not None:
        url = os.path.join(model.path, url)
    try:
        filetype = data['filetype']
    except KeyError:
        # guess file type based on extension
        if url.endswith(('.xls', '.xlsx')):
            filetype = "excel"
        elif url.endswith(('.csv', '.gz')):
            filetype = "csv"
        elif url.endswith(('.hdf', '.hdf5', '.h5')):
            filetype = "hdf"
        else:
            raise NotImplementedError('Unknown file extension: "{}"'.format(url))

    column = data.pop("column", None)

    if filetype == "csv":
        if hasattr(data, "index_col"):
            data["parse_dates"] = True
            if "dayfirst" not in data.keys():
                data["dayfirst"] = True # we're bias towards non-American dates here
        df = pandas.read_csv(url, **data) # automatically decompressed gzipped data!
    elif filetype == "excel":
        df = pandas.read_excel(url, **data)
    elif filetype == "hdf":
        df = pandas.read_hdf(url, columns=[column], **data)

    # if column is not specified, use the whole dataframe
    if column is not None:
        try:
            df = df[column]
        except KeyError:
            raise KeyError('Column "{}" not found in dataset "{}"'.format(column, url))

    try:
        freq = df.index.inferred_freq
    except AttributeError:
        pass
    else:
        # Convert to regular frequency
        freq = df.index.inferred_freq
        if freq is None:
            raise IndexError("Failed to identify frequency of dataset \"{}\"".format(url))
        df = df.asfreq(freq)
    return df
