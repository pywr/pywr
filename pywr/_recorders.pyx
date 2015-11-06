import numpy as np
cimport numpy as np

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
        cdef int ncomb = len(model.scenarios.combinations)
        cdef int nts = len(model.timestepper)
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
