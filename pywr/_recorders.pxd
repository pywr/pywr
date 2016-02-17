cdef class Recorder

from _core cimport Timestep, AbstractNode, Storage

cdef class Recorder:
    cdef bint _is_objective
    cdef object _model
    cpdef setup(self)
    cpdef reset(self)
    cpdef int save(self) except -1
    cpdef finish(self)

cdef class NodeRecorder(Recorder):
    cdef AbstractNode _node


cdef class StorageRecorder(Recorder):
    cdef Storage _node

cdef class NumpyArrayNodeRecorder(NodeRecorder):
    cdef double[:, :] _data

cdef class NumpyArrayStorageRecorder(StorageRecorder):
    cdef double[:, :] _data