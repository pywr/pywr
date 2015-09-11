# cython: profile=False
from pywr._core cimport *

import numpy as np
cimport numpy as np

cdef double inf = float('inf')


cdef cartesian(sizes, out=None):
    """
    Generate a cartesian product of input sizes.

    Adapted from Stackoverflow answer (user: pv):
    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Parameters
    ----------
    sizes : array-like of the number of entries in each combination
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(sizes)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian([3, 2, 2])
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [1, 0, 0],
           [1, 0, 1],
           [1, 1, 0],
           [1, 1, 1],
           [2, 0, 0],
           [2, 0, 1],
           [2, 1, 0],
           [2, 1, 1]])

    """
    cdef int j
    dtype = np.int32
    array0 = np.arange(sizes[0], dtype=dtype)

    n = np.prod([x for x in sizes])
    if out is None:
        out = np.zeros([n, len(sizes)], dtype=dtype)

    m = n / sizes[0]
    out[:,0] = np.repeat(array0, m)
    if sizes[1:]:
        cartesian(sizes[1:], out=out[0:m,1:])
        for j in range(1, array0.size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

cdef class Scenario:
    def __init__(self, str name, int size):
        self._name = name
        self._size = size


cdef class ScenarioCollection:
    def __init__(self, ):
        self._scenarios = []

    cpdef get_scenario_index(self, Scenario sc):
        """Return the index of Scenario in this controller."""
        return self.scenarios.index(sc)

    cpdef add_scenario(self, Scenario sc):
        if sc in self._scenarios:
            raise ValueError("The same scenario can not be added twice.")
        self._scenarios.append(sc)

    cpdef int get_number_of_scenarios(self):
        return len(self._scenarios)

    cpdef int get_number_of_combinations(self):
        """Return the total number of combinations of Scenarios

        This is essentially a product of the sizes of the Scenarios. When
        there are no Scenarios 1 is returned.
        """
        cdef Scenario sc
        if len(self._scenarios) > 0:
            return np.prod([sc._size for sc in self._scenarios])
        return 1

    cpdef int[:, :] get_combinations(self, ):
        """ Return an 2D array of the all of the scenario combinations.

        Each corresponds to a particular combination's scenario index. For example
        a Scenario with size 2 will correspond to entries of 0 and 1 to indicate
        which of the two possibilities to refer to in a given combination.
        """
        cdef Scenario sc
        return cartesian([sc._size for sc in self._scenarios])


cdef class Timestep:
    def __init__(self, object datetime, int index, double days):
        self._datetime = datetime
        self._index = index
        self._days = days

    property datetime:
        """Timestep representation as a `datetime.datetime` object"""
        def __get__(self, ):
            return self._datetime

    property index:
        """The index of the timestep for use in arrays"""
        def __get__(self, ):
            return self._index

    property days:
        def __get__(self, ):
            return self._days

cdef class Recorder:
    cpdef setup(self, model):
        pass

    cpdef reset(self):
        pass

    cpdef int commit(self, Timestep ts, int scenario_index, double value) except -1:
        return 0

    cpdef int commit_all(self, Timestep ts, double[:] value) except -1:
        return 0


cdef class NumpyArrayRecorder(Recorder):
    cdef double[:, :] _data
    cpdef setup(self, model):
        cdef int ncomb = model.number_of_scenario_combinations
        cdef int nts = model.number_of_timesteps
        self._data = np.zeros((nts, ncomb))

    cpdef reset(self):
        self._data[:, :] = 0.0

    cpdef int commit(self, Timestep ts, int scenario_index, double value) except -1:
        self._data[ts._index, scenario_index] = value

    cpdef int commit_all(self, Timestep ts, double[:] value) except -1:
        cdef int i
        for i in range(self._data.shape[1]):
            self._data[ts._index,i] = value[i]

    property data:
        def __get__(self, ):
            return np.array(self._data)


cdef class Parameter:
    cpdef setup(self, model):
        pass
    cpdef double value(self, Timestep ts) except? -1:
        return 0

cdef class ParameterArrayIndexed(Parameter):
    """Time varying parameter using an array and Timestep._index
    """
    def __cinit__(self, double[:] values):
        self.values = values

    cpdef double value(self, Timestep ts) except? -1:
        """Returns the value of the parameter at a given timestep
        """
        return self.values[ts._index]

cdef class ParameterConstantScenario(Parameter):
    """A Scenario varying Parameter"""
    cdef Scenario _scenario
    cdef double[:] _values
    cdef int _scenario_index
    def __init__(self, Scenario scenario, values):
        cdef int i
        if scenario._size != len(values):
            raise ValueError("The number of values must equal the size of the scenario.")
        self._values = np.empty(scenario._size)
        for i in range(scenario._size):
            self._values[i] = values[i]

    cpdef setup(self, model):
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = model.scenarios.get_scenario_index(self._scenario)

    cpdef double value(self, Timestep ts) except? -1:
        return self._values[0]


cdef class Node:
    """Node class from which all others inherit
    """
    def __cinit__(self):
        """Initialise the node attributes
        """
        # Initialised attributes to zero
        self._min_flow = 0.0
        self._max_flow = 0.0
        self._cost = 0.0
        # Conversion is default to unity so that there is no loss
        self._conversion_factor = 1.0
        # Parameters are initialised to None which corresponds to
        # a static value
        self._min_flow_param = None
        self._max_flow_param = None
        self._cost_param = None
        self._conversion_factor_param = None
        self._recorder = None

    property prev_flow:
        """Total flow via this node in the previous timestep
        """
        def __get__(self):
            return np.array(self._prev_flow)

    property flow:
        """Total flow via this node in the current timestep
        """
        def __get__(self):
            return np.array(self._flow)

    property min_flow:
        """The minimum flow constraint on the node

        The minimum flow may be set to either a constant (i.e. a float) or a
        Parameter.
        """
        def __get__(self):
            return self._min_flow_param

        def __set__(self, value):
            if isinstance(value, Parameter):
                self._min_flow_param = value
            else:
                self._min_flow_param = None
                self._min_flow = value

    cpdef get_min_flow(self, Timestep ts):
        """Get the minimum flow at a given timestep
        """
        if self._min_flow_param is None:
            return self._min_flow
        return self._min_flow_param.value(ts)

    property max_flow:
        """The maximum flow constraint on the node

        The maximum flow may be set to either a constant (i.e. a float) or a
        Parameter.
        """
        def __set__(self, value):
            if value is None:
                self._max_flow = inf
            elif isinstance(value, Parameter):
                self._max_flow_param = value
            else:
                self._max_flow_param = None
                self._max_flow = value

    cpdef get_max_flow(self, Timestep ts):
        """Get the maximum flow at a given timestep
        """
        if self._max_flow_param is None:
            return self._max_flow
        return self._max_flow_param.value(ts)

    property cost:
        """The cost per unit flow via the node

        The cost may be set to either a constant (i.e. a float) or a Parameter.

        The value returned can be positive (i.e. a cost), negative (i.e. a
        benefit) or netural. Typically supply nodes will have an associated
        cost and demands will provide a benefit.
        """
        def __set__(self, value):
            if isinstance(value, Parameter):
                self._cost_param = value
            else:
                self._cost_param = None
                self._cost = value

    cpdef get_cost(self, Timestep ts):
        """Get the cost per unit flow at a given timestep
        """
        if self._cost_param is None:
            return self._cost
        return self._cost_param.value(ts)

    property conversion_factor:
        """The conversion between inflow and outflow for the node

        The conversion factor may be set to either a constant (i.e. a float) or
        a Parameter.
        """
        def __set__(self, value):
            self._conversion_factor_param = None
            if isinstance(value, Parameter):
                raise ValueError("Conversion factor can not be a Parameter.")
            else:
                self._conversion_factor = value

    cpdef get_conversion_factor(self):
        """Get the conversion factor

        Note: the conversion factor must be a constant.
        """
        return self._conversion_factor

    cdef set_parameters(self, Timestep ts):
        """Update the constant attributes by evaluating any Parameter objects

        This is useful when the `get_` functions need to be accessed multiple
        times and there is a benefit to caching the values.
        """
        if self._min_flow_param is not None:
            self._min_flow = self._min_flow_param.value(ts)
        if self._max_flow_param is not None:
            self._max_flow = self._max_flow_param.value(ts)
        if self._cost_param is not None:
            self._cost = self._cost_param.value(ts)

    property recorder:
        """The recorder for the node, e.g. a NumpyArrayRecorder
        """
        def __get__(self):
            return self._recorder

        def __set__(self, value):
            self._recorder = value

    cpdef setup(self, model):
        """Called before the first run of the model"""
        cdef int ncomb = model.number_of_scenario_combinations
        self._flow = np.empty(ncomb, dtype=np.float64)
        self._prev_flow = np.empty(ncomb, dtype=np.float64)
        if self._recorder is not None:
            self._recorder.setup(model)

    cpdef reset(self):
        """Called at the beginning of a run"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] = 0.0
        if self._recorder is not None:
            self._recorder.reset()

    cpdef before(self, Timestep ts):
        """Called at the beginning of the timestep"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] = 0.0

    cpdef commit(self, int scenario_index, double value):
        """Called once for each route the node is a member of"""
        self._flow[scenario_index] += value

    cpdef commit_all(self, double[:] value):
        """Called once for each route the node is a member of"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] += value[i]

    cpdef after(self, Timestep ts):
        """Called at the end of the timestep"""
        self._prev_flow[:] = self._flow[:]
        if self._recorder is not None:
            self._recorder.commit_all(ts, self._flow)

cdef class Storage:
    def __cinit__(self, ):
        self._initial_volume = 0.0
        self._min_volume = 0.0
        self._max_volume = 0.0
        self._cost = 0.0

        self._min_volume_param = None
        self._max_volume_param = None
        self._cost_param = None

    property volume:
        def __get__(self, ):
            return self._volume[0]

        def __set__(self, value):
            # This actually sets the initial volume
            # TODO provide a method to set the _volume for a given scenario(s).
            import warnings
            warnings.warn("Setting volume property directly is not supported. This method is updating the initial storage volume.")
            self._initial_volume = value

    property initial_volume:
        def __get__(self, ):
            return self._initial_volume

        def __set__(self, value):
            self._initial_volume

    property min_volume:
        def __set__(self, value):
            self._min_volume_param = None
            if isinstance(value, Parameter):
                self._min_volume_param = value
            else:
                self._min_volume = value

    cpdef get_min_volume(self, Timestep ts):
        if self._min_volume_param is None:
            return self._min_volume
        return self._min_volume_param.value(ts)

    property max_volume:
        def __set__(self, value):
            self._max_volume_param = None
            if isinstance(value, Parameter):
                self._max_volume_param = value
            else:
                self._max_volume = value

    cpdef get_max_volume(self, Timestep ts):
        if self._max_volume_param is None:
            return self._max_volume
        return self._max_volume_param.value(ts)

    property cost:
        def __set__(self, value):
            self._cost_param = None
            if isinstance(value, Parameter):
                self._cost_param = value
            else:
                self._cost = value

    cpdef get_cost(self, Timestep ts):
        if self._cost_param is None:
            return self._cost
        return self._cost_param.value(ts)

    property current_pc:
        " Current percentage full "
        def __get__(self, ):
            return np.array(self._volume[:]) / self._max_volume

    property recorder:
        def __get__(self, ):
            return self._recorder

        def __set__(self, value):
            self._recorder = value

    cpdef setup(self, model):
        """Called before the first run of the model"""
        cdef int ncomb = model.number_of_scenario_combinations
        self._flow = np.empty(ncomb, dtype=np.float64)
        self._volume = np.empty(ncomb, dtype=np.float64)
        if self._recorder is not None:
            self._recorder.setup(model)

    cpdef reset(self):
        """Called at the beginning of a run"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] = 0.0
            self._volume[i] = self._initial_volume
        if self._recorder is not None:
            self._recorder.reset()

    cpdef before(self, Timestep ts):
        """Called at the beginning of the timestep"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] = 0.0

    cpdef commit(self, int scenario_index, double value):
        """Called once for each route the node is a member of"""
        self._flow[scenario_index] += value

    cpdef commit_all(self, double[:] value):
        """Called once for each route the node is a member of"""
        cdef int i
        for i in range(self._flow.shape[0]):
            self._flow[i] += value[i]

    cpdef after(self, Timestep ts):
        # Update storage
        cdef int i
        for i in range(self._flow.shape[0]):
            self._volume[i] += self._flow[i]*ts._days
        if self._recorder is not None:
            self._recorder.commit_all(ts, self._volume)
