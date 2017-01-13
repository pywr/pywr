cdef class Recorder

from _core cimport Timestep, AbstractNode, Node, Storage, ScenarioIndex
from .parameters._parameters cimport Parameter, IndexParameter

cdef class Recorder:
    cdef bint _is_objective
    cdef public double epsilon
    cdef object _name
    cdef object _model
    cdef public basestring comment
    cdef object agg_func
    cdef int _agg_func
    cpdef setup(self)
    cpdef reset(self)
    cpdef int save(self) except -1
    cpdef finish(self)
    cpdef double aggregated_value(self) except? -1
    cpdef double[:] values(self)

cdef class AggregatedRecorder(Recorder):
    cdef object recorder_agg_func
    cdef int _recorder_agg_func
    cdef public list recorders

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

cdef class MeanParameterRecorder(ParameterRecorder):
    cdef public int timesteps
    cdef int position
    cdef double[:, :] _memory
    cdef double[:, :] _data

cdef class MeanFlowRecorder(NodeRecorder):
    cdef int position
    cdef public int timesteps
    cdef public int days
    cdef double[:, :] _memory
    cdef double[:, :] _data

cdef class BaseConstantNodeRecorder(NodeRecorder):
    cdef double[:] _values

cdef class TotalDeficitNodeRecorder(BaseConstantNodeRecorder):
    pass

cdef class TotalFlowNodeRecorder(BaseConstantNodeRecorder):
    cdef public double factor

cdef class DeficitFrequencyNodeRecorder(BaseConstantNodeRecorder):
    pass

cdef class BaseConstantStorageRecorder(StorageRecorder):
    cdef double[:] _values

cdef class MinimumVolumeStorageRecorder(BaseConstantStorageRecorder):
    pass