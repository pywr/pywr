
import numpy as np
cimport numpy as np

cdef class Parameter:
    cpdef setup(self, model):
        pass

    cpdef reset(self):
        pass

    cpdef before(self, Timestep ts):
        pass

    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1:
        return 0
    cpdef after(self, Timestep ts):
        pass

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

        def __set__(self, value):
            self._parent = value

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

cdef class ParameterArrayIndexed(Parameter):
    """Time varying parameter using an array and Timestep._index

    The values in this parameter are constant across all scenarios.
    """
    def __init__(self, double[:] values):
        self.values = values

    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1:
        """Returns the value of the parameter at a given timestep
        """
        return self.values[ts._index]

cdef class ParameterConstantScenario(Parameter):
    """A Scenario varying Parameter

    The values in this parameter are constant in time, but vary within a single Scenario.
    """
    def __init__(self, Scenario scenario, values):
        """
        values should be an iterable that is the same length as scenario.size
        """
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

    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        return self._values[scenario_indices[self._scenario_index]]


cdef class ParameterArrayIndexedScenarioMonthlyFactors(Parameter):
    """Time varying parameter using an array and Timestep._index with
    multiplicative factors per Scenario
    """
    def __init__(self, Scenario scenario, double[:] values, double[:, :] factors):
        """
        values is the baseline timeseries data that is perturbed by a factor. The
        factor is taken from factors which is shape (scenario.size, 12). Therefore
        factors vary with the individual scenarios in scenario and month.
        """
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

    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1:
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        cdef int imth = ts.datetime.month-1
        return self._values[ts._index]*self._factors[scenario_indices[self._scenario_index], imth]
