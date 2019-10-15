import os
import numpy as np
cimport numpy as np
import pandas
from libc.math cimport cos, M_PI
from libc.limits cimport INT_MIN, INT_MAX
from past.builtins import basestring
from pywr.h5tools import H5Store
from pywr.hashes import check_hash
from ..dataframe_tools import align_and_resample_dataframe, load_dataframe, read_dataframe
import warnings


parameter_registry = {}


class UnutilisedDataWarning(Warning):
    """ Simple warning to indicate that not all data has been used. """
    pass


cdef class Parameter(Component):
    def __init__(self, *args, is_variable=False, **kwargs):
        super(Parameter, self).__init__(*args, **kwargs)
        self.is_variable = is_variable
        self.double_size = 0
        self.integer_size = 0

    @classmethod
    def register(cls):
        parameter_registry[cls.__name__.lower()] = cls

    @classmethod
    def unregister(cls):
        del(parameter_registry[cls.__name__.lower()])

    cpdef setup(self):
        super(Parameter, self).setup()
        cdef int num_comb
        if self.model.scenarios.combinations:
            num_comb = len(self.model.scenarios.combinations)
        else:
            num_comb = 1
        self.__values = np.empty([num_comb], np.float64)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        raise NotImplementedError("Parameter must be subclassed")

    cdef calc_values(self, Timestep timestep):
        # default implementation calls Parameter.value in loop
        cdef ScenarioIndex scenario_index
        cdef ScenarioCollection scenario_collection = self.model.scenarios
        for scenario_index in scenario_collection.combinations:
            self.__values[<int>(scenario_index.global_id)] = self.value(timestep, scenario_index)

    cpdef double get_value(self, ScenarioIndex scenario_index):
        return self.__values[<int>(scenario_index.global_id)]

    cpdef double[:] get_all_values(self):
        return self.__values

    cpdef set_double_variables(self, double[:] values):
        raise NotImplementedError()

    cpdef double[:] get_double_variables(self):
        raise NotImplementedError()

    cpdef double[:] get_double_lower_bounds(self):
        raise NotImplementedError()

    cpdef double[:] get_double_upper_bounds(self):
        raise NotImplementedError()

    cpdef set_integer_variables(self, int[:] values):
        raise NotImplementedError()

    cpdef int[:] get_integer_variables(self):
        raise NotImplementedError()

    cpdef int[:] get_integer_lower_bounds(self):
        raise NotImplementedError()

    cpdef int[:] get_integer_upper_bounds(self):
        raise NotImplementedError()

    property size:
        def __get__(self):
            warnings.warn("Use of the `size` property on Parameters has been deprecated."
                          "Please use either `double_size` or `integer_size` instead.", DeprecationWarning)
            return self.double_size

        def __set__(self, value):
            warnings.warn("Use of the `size` property on Parameters has been deprecated."
                          "Please use either `double_size` or `integer_size` instead.", DeprecationWarning)
            self.double_size = value

    @classmethod
    def load(cls, model, data):
        # If a scenario is given don't pass this to the load values methods
        scenario = data.pop('scenario', None)

        values = load_parameter_values(model, data)
        data.pop("values", None)
        data.pop("url", None)
        name = data.pop("name", None)
        comment = data.pop("comment", None)

        if scenario is not None:
            scenario = model.scenarios[scenario]
            # Only pass scenario object if one provided; most Parameter subclasses
            # do not accept a scenario argument.
            return cls(model, scenario=scenario, values=values, name=name, comment=None, **data)
        else:
            return cls(model, values=values, name=name, comment=None, **data)
Parameter.register()


cdef class ConstantParameter(Parameter):
    def __init__(self, model, value, lower_bounds=0.0, upper_bounds=np.inf, scale=1.0, offset=0.0, **kwargs):
        super(ConstantParameter, self).__init__(model, **kwargs)
        self._value = value
        self.scale = scale
        self.offset = offset
        self.double_size = 1
        self.integer_size = 0
        self._lower_bounds = np.ones(self.double_size) * lower_bounds
        self._upper_bounds = np.ones(self.double_size) * upper_bounds

    cdef calc_values(self, Timestep timestep):
        # constant parameter can just set the entire array to one value
        self.__values[...] = self.offset + self._value * self.scale

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return self._value

    cpdef set_double_variables(self, double[:] values):
        self._value = values[0]

    cpdef double[:] get_double_variables(self):
        return np.array([self._value, ], dtype=np.float64)

    cpdef double[:] get_double_lower_bounds(self):
        return self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        return self._upper_bounds

    @classmethod
    def load(cls, model, data):
        if "value" in data:
            value = data.pop("value")
        else:
            value = load_parameter_values(model, data)
        parameter = cls(model, value, **data)
        return parameter

ConstantParameter.register()


cdef class DataFrameParameter(Parameter):
    """Timeseries parameter with automatic alignment and resampling

    Parameters
    ----------
    model : pywr.model.Model
    dataframe : pandas.DataFrame or pandas.Series
    scenario: pywr._core.Scenario (optional)
    """
    def __init__(self, model, dataframe, scenario=None, **kwargs):
        super(DataFrameParameter, self).__init__(model, *kwargs)
        self.dataframe = dataframe
        self.scenario = scenario

    cpdef setup(self):
        super(DataFrameParameter, self).setup()
        # align and resample the dataframe
        dataframe_resampled = align_and_resample_dataframe(self.dataframe, self.model.timestepper.datetime_index)
        if dataframe_resampled.ndim == 1:
            dataframe_resampled = pandas.DataFrame(dataframe_resampled)
        # dataframe should now have the correct number of timesteps for the model
        if len(dataframe_resampled) != len(self.model.timestepper):
            raise ValueError("Aligning DataFrame failed with a different length compared with model timesteps.")
        # check that if a 2D DataFrame is given that we also have a scenario assigned with it
        if dataframe_resampled.ndim == 2 and dataframe_resampled.shape[1] > 1:
            if self.scenario is None:
                raise ValueError("Scenario must be given for a DataFrame input with multiple columns.")
            if self.scenario.size != dataframe_resampled.shape[1]:
                raise ValueError("Scenario size ({}) is different to the number of columns ({}) "
                                 "in the DataFrame input.".format(self.scenario.size, dataframe_resampled.shape[1]))
        self._values = dataframe_resampled.values.astype(np.float64)
        if self.scenario is not None:
            self._scenario_index = self.model.scenarios.get_scenario_index(self.scenario)

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        cdef double value
        if self.scenario is not None:
            value = self._values[<int>(timestep.index), <int>(scenario_index._indices[self._scenario_index])]
        else:
            value = self._values[<int>(timestep.index), 0]
        return value

    @classmethod
    def load(cls, model, data):
        scenario = data.pop('scenario', None)
        if scenario is not None:
            scenario = model.scenarios[scenario]
        df = load_dataframe(model, data)
        return cls(model, df, scenario=scenario, **data)

DataFrameParameter.register()

cdef class ArrayIndexedParameter(Parameter):
    """Time varying parameter using an array and Timestep.index

    The values in this parameter are constant across all scenarios.
    """
    def __init__(self, model, values, *args, **kwargs):
        super(ArrayIndexedParameter, self).__init__(model, *args, **kwargs)
        self.values = np.asarray(values, dtype=np.float64)

    cdef calc_values(self, Timestep ts):
        # constant parameter can just set the entire array to one value
        self.__values[...] = self.values[ts.index]

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Returns the value of the parameter at a given timestep
        """
        return self.values[ts.index]
ArrayIndexedParameter.register()


cdef class ArrayIndexedScenarioParameter(Parameter):
    """A Scenario varying Parameter

    The values in this parameter are vary in time based on index and vary within a single Scenario.
    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        """
        values should be an iterable that is the same length as scenario.size
        """
        super(ArrayIndexedScenarioParameter, self).__init__(model, *args, **kwargs)
        cdef int i
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Values must be two dimensional.")
        if scenario._size != values.shape[1]:
            raise ValueError("The size of the second dimension of values must equal the size of the scenario.")
        self.values = values
        self._scenario = scenario

    cpdef setup(self):
        super(ArrayIndexedScenarioParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self.values[ts.index, scenario_index._indices[self._scenario_index]]


cdef class TablesArrayParameter(IndexParameter):
    def __init__(self, model, h5file, node, where='/', scenario=None, **kwargs):
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
        super(TablesArrayParameter, self).__init__(model, **kwargs)

        self.h5file = h5file
        self.h5store = None
        self.node = node
        self.where = where
        self.scenario = scenario

        # Private attributes, initialised during setup()
        self._values_dbl = None  # Stores the loaded data if float
        self._values_int = None  # Stores the loaded data if integer
        # If a scenario is present this is the index in the model list of scenarios
        self._scenario_index = -1
        self._scenario_ids = None  # Lookup of scenario index to the loaded data index

    cpdef setup(self):
        cdef Py_ssize_t n, i

        super(TablesArrayParameter, self).setup()
        self._scenario_index = -1
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        if self.scenario is not None:
            self._scenario_index = self.model.scenarios.get_scenario_index(self.scenario)

        self.h5store = H5Store(self.h5file, None, "r")
        node = self.h5store.file.get_node(self.where, self.node)
        if not node.dtype in (np.float32, np.float64, np.int8, np.int16, np.int32):
            raise TypeError("Unexpected dtype in array: {}".format(node.dtype))

        # check the shape of the data is valid
        if self.scenario is not None:
            if node.shape[1] < self.scenario.size:
                raise IndexError('The length of the second dimension ({:d}) of the tables node ({}:{}) '
                                 'should be the same as the size of the specified Scenario ({:d}).'
                                 .format(node.shape[1], node._v_file.filename, node._v_pathname, self.scenario.size))
            elif node.shape[1] > self.scenario.size:
                warnings.warn('The length of the second dimension ({:d}) of the tables node ({}:{}) '
                              'is greater than the size of the specified Scenario ({:d}). '
                              'Not all data is being used!'.format(node.shape[1], node._v_file.filename, node._v_pathname, self.scenario.size),
                              UnutilisedDataWarning)
        if node.shape[0] < len(self.model.timestepper):
            raise IndexError('The length of the first dimension ({:d}) of the tables node ({}:{}) '
                             'should be equal to or greater than the number of timesteps.'
                             .format(node.shape[0], node._v_file.filename, node._v_pathname, len(self.model.timestepper)))
        elif node.shape[0] > len(self.model.timestepper):
            warnings.warn('The length of the first dimension ({:d}) of the tables node ({}:{}) '
                          'is greater than the number of timesteps. Not all data is being used!'
                          .format(node.shape[0], node._v_file.filename, node._v_pathname, len(self.model.timestepper)),
                          UnutilisedDataWarning)

        # detect data type and read into memoryview
        self._values_dbl = None
        self._values_int = None
        if self.scenario:
            # if possible, only load the data required
            scenario_indices = None
            # Default to index that is just out of bounds to cause IndexError if something goes wrong
            self._scenario_ids = np.ones(self.scenario.size, dtype=np.int32) * self.scenario.size

            # Calculate the scenario indices to load dependning on how scenario combinations are defined.
            if self.model.scenarios.user_combinations:
                scenario_indices = set()
                for user_combination in self.model.scenarios.user_combinations:
                    scenario_indices.add(user_combination[self._scenario_index])
                scenario_indices = sorted(list(scenario_indices))
            elif self.scenario.slice:
                scenario_indices = range(*self.scenario.slice.indices(self.scenario.slice.stop))
            else:
                # scenario is defined, but all data required
                self._scenario_ids = None

            if scenario_indices is not None:
                # Now load only the required data
                for n, i in enumerate(scenario_indices):
                    self._scenario_ids[i] = n

                if node.dtype in (np.float32, np.float64):
                    self._values_dbl = node[:len(self.model.timestepper), scenario_indices].astype(np.float64)
                else:
                    self._values_int = node[:len(self.model.timestepper), scenario_indices].astype(np.int32)

        if node.dtype in (np.float32, np.float64):
            if self._values_dbl is None:
                self._values_dbl = node.read().astype(np.float64)
            # negative values are often erroneous
            if np.min(self._values_dbl) < 0.0:
                warnings.warn('Negative values in input file "{}" from node: {}'.format(self.h5file, self.node))
            if not np.all(np.isfinite(self._values_dbl)):
                raise ValueError('Non-finite values in input file "{}" from node: {}'.format(self.h5file, self.node))
        else:
            if self._values_int is None:
                self._values_int = node.read().astype(np.int32)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef Py_ssize_t i = ts.index
        cdef Py_ssize_t j
        if self._values_dbl is None:
            return float(self.index(ts, scenario_index))
        # Support 1D and 2D indexing when scenario is or is not given.
        if self._scenario_index == -1:
            return self._values_dbl[i, 0]
        else:
            j = scenario_index._indices[self._scenario_index]
            if self._scenario_ids is not None:
                j = self._scenario_ids[j]
            return self._values_dbl[i, j]

    cpdef int index(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef Py_ssize_t i = ts.index
        cdef Py_ssize_t j
        if self._values_int is None:
            return int(self.value(ts, scenario_index))
        # Support 1D and 2D indexing when scenario is or is not given.
        if self._scenario_index == -1:
            return self._values_int[i, 0]
        else:
            j = scenario_index._indices[self._scenario_index]
            if self._scenario_ids is not None:
                j = self._scenario_ids[j]
            return self._values_int[i, j]

    cpdef finish(self):
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

        # Check hashes if given before reading the data
        checksums = data.pop('checksum', {})
        for algo, hash in checksums.items():
            check_hash(url, hash, algorithm=algo)

        return cls(model, url, node, where=where, scenario=scenario)
TablesArrayParameter.register()


cdef class ConstantScenarioParameter(Parameter):
    """A Scenario varying Parameter

    The values in this parameter are constant in time, but vary within a single Scenario.
    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        """
        values should be an iterable that is the same length as scenario.size
        """
        super(ConstantScenarioParameter, self).__init__(model, *args, **kwargs)
        cdef int i
        if scenario._size != len(values):
            raise ValueError("The number of values must equal the size of the scenario.")
        self._values = np.empty(scenario._size)
        for i in range(scenario._size):
            self._values[i] = values[i]
        self._scenario = scenario

    cpdef setup(self):
        super(ConstantScenarioParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self._values[scenario_index._indices[self._scenario_index]]
ConstantScenarioParameter.register()


cdef class ArrayIndexedScenarioMonthlyFactorsParameter(Parameter):
    """Time varying parameter using an array and Timestep.index with
    multiplicative factors per Scenario
    """
    def __init__(self, model, Scenario scenario, values, factors, *args, **kwargs):
        """
        values is the baseline timeseries data that is perturbed by a factor. The
        factor is taken from factors which is shape (scenario.size, 12). Therefore
        factors vary with the individual scenarios in scenario and month.
        """
        super(ArrayIndexedScenarioMonthlyFactorsParameter, self).__init__(model, *args, **kwargs)

        values = np.asarray(values, dtype=np.float64)
        factors = np.asarray(factors, dtype=np.float64)
        if factors.ndim != 2:
            raise ValueError("Factors must be two dimensional.")

        if factors.shape[0] != scenario._size:
            raise ValueError("First dimension of factors must be the same size as scenario.")
        if factors.shape[1] != 12:
            raise ValueError("Second dimension of factors must be 12.")
        self._scenario = scenario
        self._values = values
        self._factors = factors

    cpdef setup(self):
        super(ArrayIndexedScenarioMonthlyFactorsParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        cdef int imth = ts.month-1
        cdef int i = scenario_index._indices[self._scenario_index]
        return self._values[ts.index]*self._factors[i, imth]

    @classmethod
    def load(cls, model, data):
        scenario = data.pop("scenario", None)
        if scenario is not None:
            scenario = model.scenarios[scenario]

        if isinstance(data["values"], list):
            values = np.asarray(data["values"], np.float64)
        elif isinstance(data["values"], dict):
            values = load_parameter_values(model, data["values"])
        else:
            raise TypeError("Unexpected type for \"values\" in {}".format(cls.__name__))

        if isinstance(data["factors"], list):
            factors = np.asarray(data["factors"], np.float64)
        elif isinstance(data["factors"], dict):
            factors = load_parameter_values(model, data["factors"])
        else:
            raise TypeError("Unexpected type for \"factors\" in {}".format(cls.__name__))

        return cls(model, scenario, values, factors)

ArrayIndexedScenarioMonthlyFactorsParameter.register()


cdef inline bint is_leap_year(int year):
    # http://stackoverflow.com/a/11595914/1300519
    return ((year & 3) == 0 and ((year % 25) != 0 or (year & 15) == 0))


cdef class DailyProfileParameter(Parameter):
    """ An annual profile consisting of daily values.

    This parameter provides a repeating annual profile with a daily resolution. A total of 366 values
    must be provided. These values are coerced to a `numpy.array` internally.

    Parameters
    ----------
    values : iterable, array
        The 366 values that represent the daily profile.

    """
    def __init__(self, model, values, *args, **kwargs):
        super(DailyProfileParameter, self).__init__(model, *args, **kwargs)
        v = np.squeeze(np.array(values))
        if v.ndim != 1:
            raise ValueError("values must be 1-dimensional.")
        if len(values) != 366:
            raise ValueError("366 values must be given for a daily profile.")
        self._values = v

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int i = ts.dayofyear - 1
        if not is_leap_year(<int>(ts.year)):
            if i > 58: # 28th Feb
                i += 1
        return self._values[i]
DailyProfileParameter.register()

cdef class WeeklyProfileParameter(Parameter):
    """Weekly profile (52-week year)

    The last week of the year will have more than 7 days, as 365 / 7 is not whole.
    """
    def __init__(self, model, values, *args, **kwargs):
        super(WeeklyProfileParameter, self).__init__(model, *args, **kwargs)
        v = np.squeeze(np.array(values))
        if v.ndim != 1:
            raise ValueError("values must be 1-dimensional.")
        if len(values) == 53:
            values = values[:52]
            warnings.warn("Truncating 53 week profile to 52 weeks.")
        if len(values) != 52:
            raise ValueError("52 values must be given for a weekly profile.")
        self._values = v

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int i = ts.dayofyear - 1
        if not is_leap_year(<int>(ts.datetime.year)):
            if i > 58: # 28th Feb
                i += 1
        cdef Py_ssize_t week
        if i >= 364:
            # last week of year is slightly longer than 7 days
            week = 51
        else:
            week = i // 7
        return self._values[week]
WeeklyProfileParameter.register()


cdef class MonthlyProfileParameter(Parameter):
    """ Parameter which provides a monthly profile

    A monthly profile is a static profile that returns a different
    value based on the current time-step.

    See also
    --------
    ScenarioMonthlyProfileParameter
    ArrayIndexedScenarioMonthlyFactorsParameter
    """
    def __init__(self, model, values, lower_bounds=0.0, upper_bounds=np.inf, **kwargs):
        super(MonthlyProfileParameter, self).__init__(model, **kwargs)
        self.double_size = 12
        self.integer_size = 0
        if len(values) != self.double_size:
            raise ValueError("12 values must be given for a monthly profile.")
        self._values = np.array(values)
        self._lower_bounds = np.ones(self.double_size)*lower_bounds
        self._upper_bounds = np.ones(self.double_size)*upper_bounds

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return self._values[ts.month-1]

    cpdef set_double_variables(self, double[:] values):
        self._values[...] = values

    cpdef double[:] get_double_variables(self):
        # Make sure we return a copy of the data instead of a view.
        return np.array(self._values).copy()

    cpdef double[:] get_double_lower_bounds(self):
        return self._lower_bounds

    cpdef double[:] get_double_upper_bounds(self):
        return self._upper_bounds
MonthlyProfileParameter.register()


cdef class ScenarioMonthlyProfileParameter(Parameter):
    """ Parameter which provides a monthly profile per scenario

    Behaviour is the same as `MonthlyProfileParameter` except a different
    profile is returned for each ensemble in a given scenario.

    See also
    --------
    MonthlyProfileParameter
    ArrayIndexedScenarioMonthlyFactorsParameter
    """
    def __init__(self, model, Scenario scenario, values, **kwargs):
        super(ScenarioMonthlyProfileParameter, self).__init__(model, **kwargs)

        if values.ndim != 2:
            raise ValueError("Factors must be two dimensional.")

        if scenario._size != values.shape[0]:
            raise ValueError("First dimension of factors must be the same size as scenario.")
        if values.shape[1] != 12:
            raise ValueError("Second dimension of factors must be 12.")
        self._scenario = scenario
        self._values = np.array(values)

    cpdef setup(self):
        super(ScenarioMonthlyProfileParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        return self._values[scenario_index._indices[self._scenario_index], ts.month-1]

ScenarioMonthlyProfileParameter.register()


cdef class IndexParameter(Parameter):
    """Base parameter providing an `index` method

    See also
    --------
    IndexedArrayParameter
    ControlCurveIndexParameter
    """
    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns the current index as a float"""
        # return index as a float
        return float(self.get_index(scenario_index))

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns the current index"""
        # return index as an integer
        return 0

    cpdef setup(self):
        super(IndexParameter, self).setup()
        cdef int num_comb
        if self.model.scenarios.combinations:
            num_comb = len(self.model.scenarios.combinations)
        else:
            num_comb = 1
        self.__indices = np.empty([num_comb], np.int32)

    cdef calc_values(self, Timestep timestep):
        cdef ScenarioIndex scenario_index
        cdef ScenarioCollection scenario_collection = self.model.scenarios
        for scenario_index in scenario_collection.combinations:
            self.__indices[<int>(scenario_index.global_id)] = self.index(timestep, scenario_index)
            self.__values[<int>(scenario_index.global_id)] = self.value(timestep, scenario_index)

    cpdef int get_index(self, ScenarioIndex scenario_index):
        return self.__indices[<int>(scenario_index.global_id)]

    cpdef int[:] get_all_indices(self):
        return self.__indices
IndexParameter.register()


cdef class ConstantScenarioIndexParameter(IndexParameter):
    """A Scenario varying IndexParameter

    The values in this parameter are constant in time, but vary within a single Scenario.
    """
    def __init__(self, model, Scenario scenario, values, *args, **kwargs):
        """
        values should be an iterable that is the same length as scenario.size
        """
        super(ConstantScenarioIndexParameter, self).__init__(model, *args, **kwargs)
        cdef int i
        if scenario._size != len(values):
            raise ValueError("The number of values must equal the size of the scenario.")
        self._values = np.empty(scenario._size, dtype=np.int32)
        for i in range(scenario._size):
            self._values[i] = values[i]
        self._scenario = scenario

    cpdef setup(self):
        super(ConstantScenarioIndexParameter, self).setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self._scenario)

    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self._values[scenario_index._indices[self._scenario_index]]
ConstantScenarioIndexParameter.register()


cdef class IndexedArrayParameter(Parameter):
    """Parameter which uses an IndexParameter to index an array of Parameters

    An example use of this parameter is to return a demand saving factor (as
    a float) based on the current demand saving level (calculated by an
    `IndexParameter`).

    Parameters
    ----------
    index_parameter : `IndexParameter`
    params : iterable of `Parameters` or floats


    Notes
    -----
    Float arguments `params` are converted to `ConstantParameter`
    """
    def __init__(self, model, index_parameter, params, **kwargs):
        super(IndexedArrayParameter, self).__init__(model, **kwargs)
        assert(isinstance(index_parameter, IndexParameter))
        self.index_parameter = index_parameter
        self.children.add(index_parameter)

        self.params = []
        for p in params:
            if not isinstance(p, Parameter):
                from pywr.parameters import ConstantParameter
                p = ConstantParameter(model, p)
            self.params.append(p)

        for param in self.params:
            self.children.add(param)
        self.children.add(index_parameter)

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        """Returns the value of the Parameter at the current index"""
        cdef int index
        index = self.index_parameter.get_index(scenario_index)
        cdef Parameter parameter = self.params[index]
        return parameter.get_value(scenario_index)

    @classmethod
    def load(cls, model, data):
        index_parameter = load_parameter(model, data.pop("index_parameter"))
        try:
            parameters = data.pop("params")
        except KeyError:
            parameters = data.pop("parameters")
        parameters = [load_parameter(model, parameter_data) for parameter_data in parameters]
        return cls(model, index_parameter, parameters, **data)
IndexedArrayParameter.register()


cdef class AnnualHarmonicSeriesParameter(Parameter):
    """ A `Parameter` which returns the value from an annual harmonic series

    This `Parameter` comprises a series N cosine function with a period of 365
     days. The calculation is performed using the Julien day of the year minus 1
     This causes a small discontinuity in non-leap years.

    .. math:: f(t) = A + \sum_{n=1}^N A_n\cdot \cos((2\pi nt)/365+\phi_n)

    Parameters
    ----------

    mean : float
        Mean value for the series (i.e. the position of zeroth harmonic)
    amplitudes : array_like
        The amplitudes for the N harmonic cosine functions. Must be the same
        length as phases.
    phases : array_like
        The phase shift of the N harmonic cosine functions. Must be the same
        length as amplitudes.

    """
    def __init__(self, model, mean, amplitudes, phases, *args, **kwargs):
        if len(amplitudes) != len(phases):
            raise ValueError("The number  of amplitudes and phases must be the same.")
        n = len(amplitudes)
        self.mean = mean
        self._amplitudes = np.array(amplitudes)
        self._phases = np.array(phases)

        self._mean_lower_bounds = kwargs.pop('mean_lower_bounds', 0.0)
        self._mean_upper_bounds = kwargs.pop('mean_upper_bounds', np.inf)
        self._amplitude_lower_bounds = np.ones(n)*kwargs.pop('amplitude_lower_bounds', 0.0)
        self._amplitude_upper_bounds = np.ones(n)*kwargs.pop('amplitude_upper_bounds', np.inf)
        self._phase_lower_bounds = np.ones(n)*kwargs.pop('phase_lower_bounds', 0.0)
        self._phase_upper_bounds = np.ones(n)*kwargs.pop('phase_upper_bounds', np.pi*2)
        super(AnnualHarmonicSeriesParameter, self).__init__(model, *args, **kwargs)
        # Size must be set after call to super.
        self.double_size = 1 + 2*n
        self._value_cache = 0.0
        self._ts_index_cache = -1

    @classmethod
    def load(cls, model, data):
        mean = data.pop('mean')
        amplitudes = data.pop('amplitudes')
        phases = data.pop('phases')

        return cls(model, mean, amplitudes, phases, **data)

    property amplitudes:
        def __get__(self):
            return np.asarray(self._amplitudes)

    property phases:
        def __get__(self):
            return np.asarray(self._phases)

    cpdef reset(self):
        Parameter.reset(self)
        self._value_cache = 0.0
        self._ts_index_cache = -1

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        cdef int ts_index = timestep.index
        cdef int doy = timestep.dayofyear - 1
        cdef int n = self._amplitudes.shape[0]
        cdef int i
        cdef double val
        if ts_index == self._ts_index_cache:
            val = self._value_cache
        else:
            val = self.mean
            for i in range(n):
                val += self._amplitudes[i]*cos(doy*(i+1)*M_PI*2/365 + self._phases[i])
            self._value_cache = val
            self._ts_index_cache = ts_index
        return val

    cpdef set_double_variables(self, double[:] values):
        n = len(self.amplitudes)
        self.mean = values[0]
        self._amplitudes[...] = values[1:n+1]
        self._phases[...] = values[n+1:]

    cpdef double[:] get_double_variables(self):
        return np.r_[np.array([self.mean, ]), np.array(self.amplitudes), np.array(self.phases)]

    cpdef double[:] get_double_lower_bounds(self):
        return np.r_[self._mean_lower_bounds, self._amplitude_lower_bounds, self._phase_lower_bounds]

    cpdef double[:] get_double_upper_bounds(self):
        return np.r_[self._mean_upper_bounds, self._amplitude_upper_bounds, self._phase_upper_bounds]
AnnualHarmonicSeriesParameter.register()

cdef enum AggFuncs:
    SUM = 0
    MIN = 1
    MAX = 2
    MEAN = 3
    PRODUCT = 4
    CUSTOM = 5
    ANY = 6
    ALL = 7
    MEDIAN = 8
_agg_func_lookup = {
    "sum": AggFuncs.SUM,
    "min": AggFuncs.MIN,
    "max": AggFuncs.MAX,
    "mean": AggFuncs.MEAN,
    "product": AggFuncs.PRODUCT,
    "custom": AggFuncs.CUSTOM,
    "any": AggFuncs.ANY,
    "all": AggFuncs.ALL,
    "median": AggFuncs.MEDIAN,
}
_agg_func_lookup_reverse = {v: k for k, v in _agg_func_lookup.items()}

def wrap_const(model, value):
    if isinstance(value, (int, float)):
        value = ConstantParameter(model, value)
    return value


cdef class AggregatedParameter(Parameter):
    """A collection of IndexParameters

    This class behaves like a set. Parameters can be added or removed from it.
    It's value is the value of it's child parameters aggregated using a
    aggregating function (e.g. sum).

    Parameters
    ----------
    parameters : iterable of `IndexParameter`
        The parameters to aggregate
    agg_func : callable or str
        The aggregation function. Must be one of {"sum", "min", "max", "mean",
        "product"}, or a callable function which accepts a list of values.
    """
    def __init__(self, model, parameters, agg_func=None, **kwargs):
        super(AggregatedParameter, self).__init__(model, **kwargs)
        self.agg_func = agg_func
        self.parameters = list(parameters)
        for parameter in self.parameters:
            self.children.add(parameter)

    @classmethod
    def load(cls, model, data):
        parameters_data = data.pop("parameters")
        parameters = []
        for pdata in parameters_data:
            parameter = load_parameter(model, pdata)
            parameters.append(wrap_const(model, parameter))

        agg_func = data.pop("agg_func", None)
        return cls(model, parameters=parameters, agg_func=agg_func, **data)

    property agg_func:
        def __get__(self):
            if self._agg_func == AggFuncs.CUSTOM:
                return self._agg_user_func
            return _agg_func_lookup_reverse[self._agg_func]
        def __set__(self, agg_func):
            self._agg_user_func = None
            if isinstance(agg_func, basestring):
                agg_func = _agg_func_lookup[agg_func.lower()]
            elif callable(agg_func):
                self._agg_user_func = agg_func
                agg_func = AggFuncs.CUSTOM
            else:
                raise ValueError("Unrecognised aggregation function: \"{}\".".format(agg_func))
            self._agg_func = agg_func

    cpdef add(self, Parameter parameter):
        self.parameters.append(parameter)
        parameter.parents.add(self)

    cpdef remove(self, Parameter parameter):
        self.parameters.remove(parameter)
        parameter.parent.remove(self)

    def __len__(self):
        return len(self.parameters)

    cpdef setup(self):
        super(AggregatedParameter, self).setup()
        assert(len(self.parameters))

    cdef calc_values(self, Timestep timestep):
        cdef Parameter parameter
        cdef double[:] accum = self.__values  # View of the underlying location for the data
        cdef double[:] values
        cdef int i
        cdef int nparam
        cdef int n = accum.shape[0]
        cdef ScenarioIndex scenario_index

        if self._agg_func == AggFuncs.PRODUCT:
            accum[...] = 1.0
            for parameter in self.parameters:
                values = parameter.get_all_values()
                for i in range(n):
                    accum[i] *= values[i]
        elif self._agg_func == AggFuncs.SUM:
            accum[...] = 0.0
            for parameter in self.parameters:
                values = parameter.get_all_values()
                for i in range(n):
                    accum[i] += values[i]
        elif self._agg_func == AggFuncs.MAX:
            accum[...] = np.NINF
            for parameter in self.parameters:
                values = parameter.get_all_values()
                for i in range(n):
                    if values[i] > accum[i]:
                        accum[i] = values[i]
        elif self._agg_func == AggFuncs.MIN:
            accum[...] = np.PINF
            for parameter in self.parameters:
                values = parameter.get_all_values()
                for i in range(n):
                    if values[i] < accum[i]:
                        accum[i] = values[i]
        elif self._agg_func == AggFuncs.MEAN:
            accum[...] = 0.0
            for parameter in self.parameters:
                values = parameter.get_all_values()
                for i in range(n):
                    accum[i] += values[i]

            nparam = len(self.parameters)
            for i in range(n):
                accum[i] /= nparam

        elif self._agg_func == AggFuncs.MEDIAN:
            for i, scenario_index in enumerate(self.model.scenarios.combinations):
                accum[i] = np.median([parameter.get_value(scenario_index) for parameter in self.parameters])
        elif self._agg_func == AggFuncs.CUSTOM:
            for i, scenario_index in enumerate(self.model.scenarios.combinations):
                accum[i] = self._agg_user_func([parameter.get_value(scenario_index) for parameter in self.parameters])
        else:
            raise ValueError("Unsupported aggregation function.")
AggregatedParameter.register()

cdef class AggregatedIndexParameter(IndexParameter):
    """A collection of IndexParameters

    This class behaves like a set. Parameters can be added or removed from it.
    Its index is the index of it's child parameters aggregated using a
    aggregating function (e.g. sum).

    Parameters
    ----------
    parameters : iterable of `IndexParameter`
        The parameters to aggregate
    agg_func : callable or str
        The aggregation function. Must be one of {"sum", "min", "max", "any",
        "all", "product"}, or a callable function which accepts a list of values.
    """
    def __init__(self, model, parameters, agg_func=None, **kwargs):
        super(AggregatedIndexParameter, self).__init__(model, **kwargs)
        self.agg_func = agg_func
        self.parameters = list(parameters)
        for parameter in self.parameters:
            self.children.add(parameter)

    @classmethod
    def load(cls, model, data):
        parameters_data = data.pop("parameters")
        parameters = list()
        for pdata in parameters_data:
            parameter = load_parameter(model, pdata)
            parameters.append(wrap_const(model, parameter))

        agg_func = data.pop("agg_func", None)
        return cls(model, parameters=parameters, agg_func=agg_func, **data)

    property agg_func:
        def __get__(self):
            if self._agg_func == AggFuncs.CUSTOM:
               return self._agg_user_func
            return _agg_func_lookup_reverse[self._agg_func]
        def __set__(self, agg_func):
            self._agg_user_func = None
            if isinstance(agg_func, basestring):
                agg_func = _agg_func_lookup[agg_func.lower()]
            elif callable(agg_func):
                self._agg_user_func = agg_func
                agg_func = AggFuncs.CUSTOM
            else:
                raise ValueError("Unrecognised aggregation function: \"{}\".".format(agg_func))
            self._agg_func = agg_func

    cpdef add(self, Parameter parameter):
        self.parameters.append(parameter)
        parameter.parents.add(self)

    cpdef remove(self, Parameter parameter):
        self.parameters.remove(parameter)
        parameter.parent.remove(self)

    def __len__(self):
        return len(self.parameters)

    cpdef setup(self):
        super(AggregatedIndexParameter, self).setup()
        assert len(self.parameters)
        assert all([isinstance(parameter, IndexParameter) for parameter in self.parameters])

    cdef calc_values(self, Timestep timestep):
        cdef IndexParameter parameter
        cdef int[:] accum = self.__indices  # View of the underlying location for the data
        cdef int[:] values
        cdef int i
        cdef int nparam
        cdef int n = accum.shape[0]
        cdef ScenarioIndex scenario_index

        if self._agg_func == AggFuncs.PRODUCT:
            accum[...] = 1
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    accum[i] *= values[i]
        elif self._agg_func == AggFuncs.SUM:
            accum[...] = 0
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    accum[i] += values[i]
        elif self._agg_func == AggFuncs.MAX:
            accum[...] = INT_MIN
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    if values[i] > accum[i]:
                        accum[i] = values[i]
        elif self._agg_func == AggFuncs.MIN:
            accum[...] = INT_MAX
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    if values[i] < accum[i]:
                        accum[i] = values[i]
        elif self._agg_func == AggFuncs.ANY:
            accum[...] = 0
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    if values[i]:
                        accum[i] = 1
        elif self._agg_func == AggFuncs.ALL:
            accum[...] = 1
            for parameter in self.parameters:
                values = parameter.get_all_indices()
                for i in range(n):
                    if not values[i]:
                        accum[i] = 0

        elif self._agg_func == AggFuncs.CUSTOM:
            for i, scenario_index in enumerate(self.model.scenarios.combinations):
                accum[i] = self._agg_user_func([parameter.get_index(scenario_index) for parameter in self.parameters])
        else:
            raise ValueError("Unsupported aggregation function.")

        # Finally set the float values
        for i in range(n):
            self.__values[i] = accum[i]


AggregatedIndexParameter.register()


cdef class DivisionParameter(Parameter):
    """ Parameter that divides one `Parameter` by another.

    Parameters
    ----------
    denominator : `Parameter`
        The parameter to use as the denominator (or divisor).
    numerator : `Parameter`
        The parameter to use as the numerator (or dividend).
    """
    def __init__(self, model, numerator, denominator, **kwargs):
        super().__init__(model, **kwargs)
        self._numerator = None
        self._denominator = None
        self.numerator = numerator
        self.denominator = denominator

    property numerator:
        def __get__(self):
            return self._numerator
        def __set__(self, parameter):
            # remove any existing parameter
            if self._numerator is not None:
                self._numerator.parents.remove(self)

            self._numerator = parameter
            self.children.add(parameter)
            
    property denominator:
        def __get__(self):
            return self._denominator
        def __set__(self, parameter):
            # remove any existing parameter
            if self._denominator is not None:
                self._denominator.parents.remove(self)

            self._denominator = parameter
            self.children.add(parameter)            

    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = self._numerator.__values[i] / self._denominator.__values[i]

    @classmethod
    def load(cls, model, data):
        numerator = load_parameter(model, data.pop("numerator"))
        denominator = load_parameter(model, data.pop("denominator"))
        return cls(model, numerator, denominator, **data)
DivisionParameter.register()


cdef class NegativeParameter(Parameter):
    """ Parameter that takes negative of another `Parameter`

    Parameters
    ----------
    parameter : `Parameter`
        The parameter to to compare with the float.
    """
    def __init__(self, model, parameter, *args, **kwargs):
        super(NegativeParameter, self).__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)

    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = -self.parameter.__values[i]

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)
NegativeParameter.register()


cdef class MaxParameter(Parameter):
    """ Parameter that takes maximum of another `Parameter` and constant value (threshold)

    This class is a more efficient version of `AggregatedParameter` where
    a single `Parameter` is compared to constant value.

    Parameters
    ----------
    parameter : `Parameter`
        The parameter to to compare with the float.
    threshold : float (default=0.0)
        The threshold value to compare with the given parameter.
    """
    def __init__(self, model, parameter, threshold=0.0, *args, **kwargs):
        super(MaxParameter, self).__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.threshold = threshold

    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = max(self.parameter.__values[i], self.threshold)

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)
MaxParameter.register()


cdef class NegativeMaxParameter(MaxParameter):
    """ Parameter that takes maximum of the negative of a `Parameter` and constant value (threshold) """
    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = max(-self.parameter.__values[i], self.threshold)

NegativeMaxParameter.register()


cdef class MinParameter(Parameter):
    """ Parameter that takes minimum of another `Parameter` and constant value (threshold)

    This class is a more efficient version of `AggregatedParameter` where
    a single `Parameter` is compared to constant value.

    Parameters
    ----------
    parameter : `Parameter`
        The parameter to to compare with the float.
    threshold : float (default=0.0)
        The threshold value to compare with the given parameter.
    """
    def __init__(self, model, parameter, threshold=0.0, *args, **kwargs):
        super(MinParameter, self).__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.threshold = threshold

    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = min(self.parameter.__values[i], self.threshold)

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)
MinParameter.register()


cdef class NegativeMinParameter(MinParameter):
    """ Parameter that takes minimum of the negative of a `Parameter` and constant value (threshold) """
    cdef calc_values(self, Timestep timestep):
        cdef int i
        cdef int n = self.__values.shape[0]

        for i in range(n):
            self.__values[i] = min(-self.parameter.__values[i], self.threshold)
NegativeMinParameter.register()


cdef class DeficitParameter(Parameter):
    """Parameter track the deficit (max_flow - actual flow) of a Node

    Parameters
    ----------
    model : pywr.model.Model
    node : Node
      The node that will have it's deficit tracked

    Notes
    -----
    This parameter is a little unusual in that it's value is calculated during
    the after method, not calc_values. It is intended to be used in combination
    with a recorder (e.g. NumpyArrayNodeRecorder) to record the deficit (
    defined as requested - actual flow) at a node. Note that this means
    recording this parameter does *not* give you the value that was used by
    the solver in this timestep. Alternatively, this parameter can be used
    in the model by other parameters and will evaluate to *yesterdays* deficit,
    where the deficit in the zeroth timestep is zero.
    """
    def __init__(self, model, node, *args, **kwargs):
        super(DeficitParameter, self).__init__(model, *args, **kwargs)
        self.node = node

    cpdef reset(self):
        self.__values[...] = 0.0

    cdef calc_values(self, Timestep timestep):
        pass # calculation done in after

    cpdef after(self):
        cdef double[:] max_flow
        cdef int i
        if self.node._max_flow_param is None:
            for i in range(0, self.node._flow.shape[0]):
                self.__values[i] = self.node._max_flow - self.node._flow[i]
        else:
            max_flow = self.node._max_flow_param.get_all_values()
            for i in range(0, self.node._flow.shape[0]):
                self.__values[i] = max_flow[i] - self.node._flow[i]

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        return cls(model, node=node, **data)

DeficitParameter.register()


cdef class FlowParameter(Parameter):
    """Parameter that provides the flow from a node from the previous time-step.

    Parameters
    ----------
    model : pywr.model.Model
    node : Node
      The node that will have its flow tracked
    initial_value : float (default=0.0)
      The value to return on the first  time-step before the node has any past flow.

    Notes
    -----
    This parameter keeps track of the previous time step's flow on the given node. These
    values can be used in calculations for the current timestep as though this was any
    other parameter.
    """
    def __init__(self, model, node, *args, **kwargs):
        self.initial_value = kwargs.pop('initial_value', 0)
        super().__init__(model, *args, **kwargs)
        self.node = node

    cpdef setup(self):
        super(FlowParameter, self).setup()
        cdef int num_comb
        if self.model.scenarios.combinations:
            num_comb = len(self.model.scenarios.combinations)
        else:
            num_comb = 1
        self.__next_values = np.empty([num_comb], np.float64)

    cpdef reset(self):
        self.__next_values[...] = self.initial_value
        self.__values[...] = 0.0

    cdef calc_values(self, Timestep timestep):
        cdef int i
        for i in range(self.__values.shape[0]):
            self.__values[i] = self.__next_values[i]

    cpdef after(self):
        cdef int i
        for i in range(self.node._flow.shape[0]):
            self.__next_values[i] = self.node._flow[i]

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        return cls(model, node=node, **data)
FlowParameter.register()


cdef class PiecewiseIntegralParameter(Parameter):
    """Parameter that integrates a piecewise function.

    This parameter calculates the integral of a piecewise function. The
    piecewise function is given as two arrays (`x` and `y`) and is assumed to
    start from (0, 0). The values of `x` should be monotonically increasing
    and greater than zero.

    Parameters
    ----------
    parameter : `Parameter`
        The parameter the defines the right hand bounds of the integration.
    x : iterable of doubles
    y : iterable of doubles

    """
    def __init__(self, model, parameter, x, y, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.parameter = parameter
        self.children.add(parameter)
        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)

    cpdef setup(self):
        super(PiecewiseIntegralParameter, self).setup()

        if len(self.x) != len(self.y):
            raise ValueError('The length of `x` and `y` should be the same.')

        if np.any(np.diff(self.x) < 0):
            raise ValueError('The array `x` should be monotonically increasing.')

    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1:
        cdef double integral = 0.0
        cdef double x = self.parameter.get_value(scenario_index)
        cdef int i
        cdef double dx, prev_x

        prev_x = 0
        for i in range(self.x.shape[0]):
            if x < self.x[i]:
                dx = x - prev_x
            else:
                dx = self.x[i] - prev_x

            if dx < 0.0:
                break
            else:
                integral += dx * self.y[i]
            prev_x = self.x[i]
        return integral

    @classmethod
    def load(cls, model, data):
        parameter = load_parameter(model, data.pop('parameter'))
        return cls(model, parameter, **data)
PiecewiseIntegralParameter.register()


def get_parameter_from_registry(parameter_type):
    key = parameter_type.lower()
    try:
        return parameter_registry[key]
    except KeyError:
        pass
    if key.endswith("parameter"):
        key.replace("parameter", "")
    else:
        key = key + "parameter"
    try:
        return parameter_registry[key]
    except KeyError:
        raise TypeError('Unknown parameter type: "{}"'.format(parameter_type))


def load_parameter(model, data, parameter_name=None):
    """Load a parameter from a dict"""
    if isinstance(data, basestring):
        # parameter is a reference
        try:
            parameter = model.parameters[data]
        except KeyError:
            parameter = None
        if parameter is None:
            if hasattr(model, "_parameters_to_load"):
                # we're still in the process of loading data from JSON and
                # the parameter requested hasn't been loaded yet - do it now
                name = data
                try:
                    data = model._parameters_to_load.pop(name)
                except KeyError:
                    raise KeyError("Unknown parameter: '{}'".format(data))
                parameter = load_parameter(model, data)
            else:
                raise KeyError("Unknown parameter: '{}'".format(data))
    elif isinstance(data, (float, int)) or data is None:
        # parameter is a constant
        parameter = data
    else:
        # parameter is dynamic
        parameter_type = data['type']
        try:
            parameter_name = data["name"]
        except:
            pass

        cls = get_parameter_from_registry(parameter_type)

        kwargs = dict([(k,v) for k,v in data.items()])
        del(kwargs["type"])
        if "name" in kwargs:
            del(kwargs["name"])
        parameter = cls.load(model, kwargs)

    if parameter_name is not None:
        # TODO FIXME: memory leak if parameter is subsequently removed from the model
        parameter.name = parameter_name
        model.parameters[parameter_name] = parameter

    return parameter


def load_parameter_values(model, data, values_key='values', url_key='url',
                          table_key='table'):
    """ Function to load values from a data dictionary.

    This function tries to load values in to a `np.ndarray` if 'values_key' is
    in 'data'. Otherwise it tries to `load_dataframe` from a 'url' key.

    Parameters
    ----------
    model - `Model` instance
    data - dict
    values_key - str
        Key in data to load values directly to a `np.ndarray`
    url_key - str
        Key in data to load values directly from an external file reference (using pandas)
    table_key - str
        Key in data to load values directly from an external file reference (using pandas)
    """
    if values_key in data:
        # values are given as an array
        values = np.array(data.pop(values_key), np.float64)
    elif url_key in data or table_key in data:
        df = load_dataframe(model, data)
        try:
            # If it's a DataFrame we coerce to a numpy array
            values = df.values
        except AttributeError:
            values = df
        values = np.squeeze(values.astype(np.float64))
    else:
        # Try to get some useful information about the parameter for the error message
        name = data.get('name', None)
        ptype = data.get('type', None)
        raise ValueError("Parameter ('{name}' of type '{ptype}' is missing a valid key to load its values. "
                         "Please provide either a '{}', '{}' or '{}' entry.".format(values_key, url_key, table_key, name=name, ptype=ptype))
    return values



