import numpy as np
cimport numpy as np

cdef class Recorder:
    def __init__(self, model):
        self._model = model
        model.recorders.append(self)

    cpdef setup(self):
        pass

    cpdef reset(self):
        pass

    cpdef int save(self) except -1:
        return 0

    cpdef finish(self):
        pass

    property model:
        def __get__(self, ):
            return self._model


cdef class NodeRecorder(Recorder):
    def __init__(self, model, Node node):
        Recorder.__init__(self, model)
        self._node = node
        node._recorders.append(self)


cdef class StorageRecorder(Recorder):
    def __init__(self, model, Storage node):
        Recorder.__init__(self, model)
        self._node = node
        node._recorders.append(self)


cdef class NumpyArrayNodeRecorder(NodeRecorder):
    cpdef setup(self):
        cdef int ncomb = len(self._model.scenarios.combinations)
        cdef int nts = len(self._model.timestepper)
        self._data = np.zeros((nts, ncomb))

    cpdef reset(self):
        self._data[:, :] = 0.0

    cpdef int save(self) except -1:
        cdef int i
        cdef Timestep ts = self._model.timestepper.current
        for i in range(self._data.shape[1]):
            self._data[ts._index,i] = self._node._flow[i]
        return 0

    property data:
        def __get__(self, ):
            return np.array(self._data)


cdef class NumpyArrayStorageRecorder(StorageRecorder):
    cpdef setup(self):
        cdef int ncomb = len(self._model.scenarios.combinations)
        cdef int nts = len(self._model.timestepper)
        self._data = np.zeros((nts, ncomb))

    cpdef reset(self):
        self._data[:, :] = 0.0

    cpdef int save(self) except -1:
        cdef int i
        cdef Timestep ts = self._model.timestepper.current
        for i in range(self._data.shape[1]):
            self._data[ts._index,i] = self._node._volume[i]
        return 0

    property data:
        def __get__(self, ):
            return np.array(self._data)