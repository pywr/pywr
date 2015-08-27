cdef class Timestep:
    cdef object _datetime
    cdef int _index
    cdef double _days

cdef class Parameter:
    cpdef double value(self, Timestep ts) except? -1

cdef class Recorder:
    cpdef setup(self, int ntimesteps)
    cpdef int commit(self, Timestep ts, double value) except -1

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

    cpdef get_min_flow(self, Timestep ts)
    cpdef get_max_flow(self, Timestep ts)
    cpdef get_cost(self, Timestep ts)
    cpdef get_conversion_factor(self, Timestep ts)
    cdef set_parameters(self, Timestep ts)
    
    cpdef before(self)
    cpdef commit(self, double value)
    cpdef after(self, Timestep ts)

cdef class Storage:
    cdef double _flow
    cdef public double _volume

    cdef double _min_volume
    cdef double _max_volume
    cdef double _cost
    cdef Parameter _min_volume_param
    cdef Parameter _max_volume_param
    cdef Parameter _cost_param
    cdef Recorder _recorder

    cpdef get_min_volume(self, Timestep ts)
    cpdef get_max_volume(self, Timestep ts)
    cpdef get_cost(self, Timestep ts)
    
    cpdef before(self)
    cpdef commit(self, double value)
    cpdef after(self, Timestep ts)
