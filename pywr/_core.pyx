# cython: profile=False
from pywr._core cimport *

import numpy as np
cimport numpy as np

cdef double inf = float('inf')

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
    cpdef setup(self, int ntimesteps):
        pass

    cpdef int commit(self, Timestep ts, double value) except -1:
        return 0


cdef class NumpyArrayRecorder(Recorder):
    cdef double[:] _data
    def __init__(self, int size):
        self._data = np.zeros(size)

    cpdef int commit(self, Timestep ts, double value) except -1:
        self._data[ts._index] = value

    property data:
        def __get__(self, ):
            return np.array(self._data)


cdef class Parameter:
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


cdef class Node:
    """Node class from which all others inherit
    """
    def __cinit__(self):
        """Initialise the node attributes
        """
        self._prev_flow = 0.0
        self._flow = 0.0
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
            return self._prev_flow

    property flow:
        """Total flow via this node in the current timestep
        """
        def __get__(self):
            return self._flow

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
                self._conversion_factor_param = value
            else:
                self._conversion_factor = value

    cpdef get_conversion_factor(self, Timestep ts):
        """Get the conversion factor at a given timestep
        """
        if self._conversion_factor_param is None:
            return self._conversion_factor
        return self._conversion_factor_param.value(ts)

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

    cpdef before(self, Timestep ts):
        """Called at the beginning of the timestep"""
        self._flow = 0.0

    cpdef commit(self, double value):
        """Called once for each route the node is a member of"""
        self._flow += value

    cpdef after(self, Timestep ts):
        """Called at the end of the timestep"""
        self._prev_flow = self._flow
        if self._recorder is not None:
            self._recorder.commit(ts, self._flow)

cdef class Storage:
    def __cinit__(self, ):
        self._flow = 0.0
        self._volume = 0.0
        self._min_volume = 0.0
        self._max_volume = 0.0
        self._cost = 0.0

        self._min_volume_param = None
        self._max_volume_param = None
        self._cost_param = None

    property volume:
        def __get__(self, ):
            return self._volume

        def __set__(self, double value):
            self._volume = value

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
            return self._volume / self._max_volume

    property recorder:
        def __get__(self, ):
            return self._recorder

        def __set__(self, value):
            self._recorder = value

    cpdef before(self, Timestep ts):
        self._flow = 0.0

    cpdef commit(self, double value):
        self._flow += value

    cpdef after(self, Timestep ts):
        # Update storage
        self._volume += self._flow*ts._days
        if self._recorder is not None:
            self._recorder.commit(ts, self._volume)
