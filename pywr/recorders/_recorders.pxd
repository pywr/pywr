from pywr._component cimport Component
from pywr._core cimport Timestep, AbstractNode, Storage, ScenarioIndex, Scenario
from pywr.parameters._parameters cimport Parameter, IndexParameter


cdef class Recorder(Component):
    cdef int _is_objective
    cdef public bint is_constraint
    cdef public double epsilon
    cdef public bint ignore_nan
    cdef object _agg_user_func
    cdef int _agg_func
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
    cdef readonly Parameter _param

cdef class IndexParameterRecorder(Recorder):
    cdef readonly IndexParameter _param

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

cdef class FlowDurationCurveRecorder(NumpyArrayNodeRecorder):
    cdef object fdc_agg_func
    cdef int _fdc_agg_func
    cdef double[:] _percentiles
    cdef double[:, :] _fdc

cdef class FlowDurationCurveDeviationRecorder(FlowDurationCurveRecorder):
    cdef double[:, :] _lower_target_fdc
    cdef double[:, :] _upper_target_fdc
    cdef double[:, :] _fdc_deviations
    cdef double[:, :] _base_fdc_tile
    cdef public Scenario scenario

cdef class RollingWindowParameterRecorder(ParameterRecorder):
    cdef public int window
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

cdef class MinimumThresholdVolumeStorageRecorder(BaseConstantStorageRecorder):
    cdef public double threshold

cdef class AnnualCountIndexParameterRecorder(IndexParameterRecorder):
    cdef public int threshold
    cdef int[:] _count
    cdef int _current_year
    cdef int[:] _current_max

cdef class AnnualReturnPeriodRecorder(Recorder):
    cdef public recorder
    cdef double _nyears