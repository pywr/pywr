cdef class Recorder

from _core cimport Timestep

cdef class Recorder:
    cpdef setup(self, model)
    cpdef reset(self)
    cpdef int commit(self, Timestep ts, int scenario_index, double value) except -1
    cpdef int commit_all(self, Timestep ts, double[:] value) except -1
