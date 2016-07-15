cdef class Recorder

from _core cimport Timestep, AbstractNode, Storage, ScenarioIndex
from .parameters._parameters cimport Parameter, IndexParameter

cdef class Recorder:
    cdef bint _is_objective
    cdef object _name
    cdef object _model
    cpdef setup(self)
    cpdef reset(self)
    cpdef int save(self) except -1
    cpdef finish(self)
    cpdef value(self)

cdef class NodeRecorder(Recorder):
    cdef AbstractNode _node

cdef class StorageRecorder(Recorder):
    cdef Storage _node

cdef class ParameterRecorder(Recorder):
    cdef Parameter _param

cdef class IndexParameterRecorder(Recorder):
    cdef IndexParameter _param

cdef class NumpyArrayNodeRecorder(NodeRecorder):
    cdef double[:, :] _data

cdef class NumpyArrayStorageRecorder(StorageRecorder):
    cdef double[:, :] _data

cdef class NumpyArrayLevelRecorder(StorageRecorder):
    cdef double[:, :] _data

cdef class NumpyArrayParameterRecorder(ParameterRecorder):
    cdef double[:, :] _data

cdef class NumpyArrayIndexParameterRecorder(IndexParameterRecorder):
    cdef int[:, :] _data
