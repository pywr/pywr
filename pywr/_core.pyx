
import numpy as np
cimport numpy as np

cdef class Timestep:
    cdef object _datetime
    cdef int _index
    cdef double _days

    def __init__(self, object datetime, int index, double days):
        self._datetime = datetime
        self._index = index
        self._days = days

    property datetime:
        def __get__(self, ):
            return self._datetime

    property index:
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


cdef class Node:
    cdef double _prev_flow
    cdef double _flow
    cdef double _min_flow
    cdef double _max_flow
    cdef double _cost
    cdef double _conversion_factor
    cdef object _min_flow_param
    cdef Parameter _max_flow_param
    cdef Parameter _cost_param
    cdef Parameter _conversion_factor_param
    cdef Recorder _recorder

    def __cinit__(self, ):
        self._prev_flow = 0.0
        self._flow = 0.0
        # Initialised attributes to zero
        self._min_flow = 0.0
        self._max_flow = 0.0
        self._cost = 0.0
        # Conversion is default to unity some there is no loss
        self._conversion_factor = 1.0
        # Parameters are initialised to None which corresponds to
        # a static value
        self._min_flow_param = None
        self._max_flow_param = None
        self._cost_param = None
        self._conversion_factor_param = None
        self._recorder = None

    property prev_flow:
        def __get__(self, ):
            return self._prev_flow

    property flow:
        def __get__(self, ):
            return self._flow

    property min_flow:
        def __get__(self, ):
            return self._min_flow_param

        def __set__(self, value):
            self._min_flow_param = None
            if isinstance(value, Parameter):
                self._min_flow_param = value
            else:
                self._min_flow = value

    cpdef get_min_flow(self, Timestep ts):
        if self._min_flow_param is None:
            return self._min_flow
        return self._min_flow_param.value(ts)

    property max_flow:
        def __set__(self, value):
            self._max_flow_param = None
            if value is None:
                self._max_flow = float('inf')
            elif isinstance(value, Parameter):
                self._max_flow_param = value
            else:
                self._max_flow = value

    cpdef get_max_flow(self, Timestep ts):
        if self._max_flow_param is None:
            return self._max_flow
        return self._max_flow_param.value(ts)

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

    property conversion_factor:
        def __set__(self, value):
            self._conversion_factor_param = None
            if isinstance(value, Parameter):
                self._conversion_factor_param = value
            else:
                self._conversion_factor = value

    cpdef get_conversion_factor(self, Timestep ts):
        if self._conversion_factor_param is None:
            return self._conversion_factor
        return self._conversion_factor_param.value(ts)

    cdef set_parameters(self, Timestep ts):
        "Update the attributes by evaluating any Parameter objects"
        if self._min_flow_param is not None:
            self._min_flow = self._min_flow_param.value(ts)
        if self._max_flow_param is not None:
            self._max_flow = self._max_flow_param.value(ts)
        if self._cost_param is not None:
            self._cost = self._cost_param.value(ts)

    property recorder:
        def __get__(self, ):
            return self._recorder

        def __set__(self, value):
            self._recorder = value

    cpdef before(self, ):
        self._flow = 0.0

    cpdef commit(self, double value):
        self._flow += value

    cpdef after(self, Timestep ts):
        self._prev_flow = self._flow
        if self._recorder is not None:
            self._recorder.commit(ts, self._flow)

cdef class Storage:
    cdef double _flow
    cdef double _volume

    cdef double _min_volume
    cdef double _max_volume
    cdef double _cost
    cdef Parameter _min_volume_param
    cdef Parameter _max_volume_param
    cdef Parameter _cost_param
    cdef Recorder _recorder

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

    cpdef before(self, ):
        self._flow = 0.0

    cpdef commit(self, double value):
        self._flow += value

    cpdef after(self, Timestep ts):
        # Update storage
        self._volume += self._flow*ts._days
        if self._recorder is not None:
            self._recorder.commit(ts, self._volume)
