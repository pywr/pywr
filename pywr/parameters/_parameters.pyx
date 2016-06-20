import os
import numpy as np
cimport numpy as np
import pandas

parameter_registry = set()

cdef class Parameter:
    def __init__(self):
        self._children = set()

    cpdef setup(self, model):
        pass

    cpdef reset(self):
        pass

    cpdef before(self, Timestep ts):
        pass

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return 0

    cpdef after(self, Timestep ts):
        pass

    cpdef update(self, double[:] values):
        raise NotImplementedError()

    cpdef double[:] lower_bounds(self):
        raise NotImplementedError()

    cpdef double[:] upper_bounds(self):
        raise NotImplementedError()

    property node:
        def __get__(self):
            if self._parent is not None:
                return self._parent.node
            return self._node
        def __set__(self, value):
            self._node = value

    property parent:
        """The parent Parameter of this object.
        """
        def __get__(self):
            return self._parent

        def __set__(self, Parameter value):
            if self._parent is not None:
                # If we have a current parent remove ourselves as a child
                self._parent._children.remove(self)
            # Update parent value
            self._parent = value
            if self._parent is not None:
                # If we have a new parent add ourselves as a child
                self._parent._children.add(self)

    property children:
        def __get__(self):
            return self._children

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
            for var in self._children:
                vars.extend(var.variables)
            return vars

    @classmethod
    def load(cls, model, data):
        values = load_parameter_values(model, data)
        return cls(values)

parameter_registry.add(Parameter)


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

def load_parameter(model, data):
    """Load a parameter from a dict"""
    if isinstance(data, str):
        # parameter is a reference
        parameter = model._parameters[data]
    elif isinstance(data, (float, int)) or data is None:
        # parameter is a constant
        parameter = data
    else:
        # parameter is dynamic
        parameter_type = data['type']

        cls = None
        name2 = parameter_type.lower().replace('parameter', '')
        for parameter_class in parameter_registry:
            name1 = parameter_class.__name__.lower().replace('parameter', '')
            if name1 == name2:
                cls = parameter_class

        if cls is None:
            raise TypeError('Unknown parameter type: "{}"'.format(parameter_type))

        del(data["type"])
        parameter = cls.load(model, data)
    
    return parameter


def load_parameter_values(model, data):
    if 'values' in data:
        # values are given as an array
        values = np.array(data['values'], np.float64)
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

    # Convert to regular frequency
    freq = df.index.inferred_freq
    if freq is None:
        raise IndexError("Failed to identify frequency of dataset \"{}\"".format(url))
    df = df.asfreq(freq)
    return df
