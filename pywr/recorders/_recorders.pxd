from pywr._component cimport Component
from pywr._core cimport Timestep, AbstractNode, Node, AbstractStorage, Storage, ScenarioIndex, Scenario
from pywr.parameters._parameters cimport Parameter, IndexParameter


cdef class Aggregator:
    cdef object _user_func
    cdef public list func_args
    cdef public dict func_kwargs
    cdef int _func
    cpdef double aggregate_1d(self, double[:] data, ignore_nan=*) except *
    cpdef double[:] aggregate_2d(self, double[:, :] data, axis=*, ignore_nan=*) except *

cdef class Recorder(Component):
    cdef int _is_objective
    cdef double _constraint_lower_bounds
    cdef double _constraint_upper_bounds
    cdef public double epsilon
    cdef public bint ignore_nan
    cdef public Aggregator _scenario_aggregator
    cpdef double aggregated_value(self) except *
    cpdef double[:] values(self) except *

cdef class AggregatedRecorder(Recorder):
    cdef object recorder_agg_func
    cdef int _recorder_agg_func
    cdef public list recorders

cdef class NodeRecorder(Recorder):
    cdef AbstractNode _node

cdef class StorageRecorder(Recorder):
    cdef AbstractStorage _node

cdef class ParameterRecorder(Recorder):
    cdef readonly Parameter _param

cdef class IndexParameterRecorder(Recorder):
    cdef readonly IndexParameter _param

cdef class NumpyArrayNodeRecorder(NodeRecorder):
    cdef Aggregator _temporal_aggregator
    cdef double[:, :] _data
    cdef public float factor

cdef class NumpyArrayNodeDeficitRecorder(NumpyArrayNodeRecorder):
    pass

cdef class NumpyArrayNodeSuppliedRatioRecorder(NumpyArrayNodeRecorder):
    pass

cdef class NumpyArrayNodeCurtailmentRatioRecorder(NumpyArrayNodeRecorder):
    pass

cdef class NumpyArrayNodeCostRecorder(NumpyArrayNodeRecorder):
    pass

cdef class NumpyArrayAbstractStorageRecorder(StorageRecorder):
    cdef public Aggregator _temporal_aggregator
    cdef double[:, :] _data

cdef class NumpyArrayStorageRecorder(NumpyArrayAbstractStorageRecorder):
    cdef public bint proportional

cdef class NumpyArrayNormalisedStorageRecorder(NumpyArrayAbstractStorageRecorder):
    cdef readonly Parameter parameter

cdef class NumpyArrayLevelRecorder(NumpyArrayAbstractStorageRecorder):
    pass

cdef class NumpyArrayAreaRecorder(NumpyArrayAbstractStorageRecorder):
    pass

cdef class NumpyArrayParameterRecorder(ParameterRecorder):
    cdef public Aggregator _temporal_aggregator
    cdef double[:, :] _data

cdef class NumpyArrayDailyProfileParameterRecorder(ParameterRecorder):
    cdef public Aggregator _temporal_aggregator
    cdef double[:, :] _data

cdef class NumpyArrayIndexParameterRecorder(IndexParameterRecorder):
    cdef public Aggregator _temporal_aggregator
    cdef int[:, :] _data

cdef class FlowDurationCurveRecorder(NumpyArrayNodeRecorder):
    cdef double[:] _percentiles
    cdef double[:, :] _fdc

cdef class StorageDurationCurveRecorder(NumpyArrayStorageRecorder):
    cdef double[:] _percentiles
    cdef double[:, :] _sdc

cdef class FlowDurationCurveDeviationRecorder(FlowDurationCurveRecorder):
    cdef double[:, :] _lower_target_fdc
    cdef double[:, :] _upper_target_fdc
    cdef double[:, :] _fdc_deviations
    cdef double[:, :] _base_fdc_tile
    cdef public Scenario scenario

cdef class RollingWindowParameterRecorder(ParameterRecorder):
    cdef public Aggregator _temporal_aggregator
    cdef public int window
    cdef int position
    cdef double[:, :] _memory
    cdef double[:, :] _data

cdef class RollingMeanFlowNodeRecorder(NodeRecorder):
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

cdef class MeanFlowNodeRecorder(BaseConstantNodeRecorder):
    cdef public double factor

cdef class DeficitFrequencyNodeRecorder(BaseConstantNodeRecorder):
    pass

cdef class BaseConstantStorageRecorder(StorageRecorder):
    cdef double[:] _values

cdef class MinimumVolumeStorageRecorder(BaseConstantStorageRecorder):
    pass

cdef class MinimumThresholdVolumeStorageRecorder(BaseConstantStorageRecorder):
    cdef public double threshold

cdef class TimestepCountIndexParameterRecorder(IndexParameterRecorder):
    cdef public int threshold
    cdef int[:] _count

cdef class AnnualCountIndexThresholdRecorder(Recorder):
    cdef public list parameters
    cdef public int threshold
    cdef public list exclude_months
    cdef int _num_years
    cdef int _ncomb
    cdef double[:, :] _data
    cdef double[:, :] _data_this_year
    cdef int _current_year
    cdef int _start_year
    cdef Aggregator _temporal_aggregator

cdef class AnnualTotalFlowRecorder(Recorder):
    cdef public list nodes
    cdef int _num_years
    cdef int _ncomb
    cdef double[:, :] _data
    cdef int _current_year
    cdef int _start_year
    cdef Aggregator _temporal_aggregator
    cdef double [:] _factors

cdef class AnnualCountIndexParameterRecorder(IndexParameterRecorder):
    cdef public int threshold
    cdef int[:] _count
    cdef int _current_year
    cdef int[:] _current_max

cdef class SeasonalFlowDurationCurveRecorder(FlowDurationCurveRecorder):
    cdef set _months


cdef class BaseConstantParameterRecorder(ParameterRecorder):
    cdef double[:] _values

cdef class TotalParameterRecorder(BaseConstantParameterRecorder):
    cdef public double factor
    cdef public bint integrate

cdef class MeanParameterRecorder(BaseConstantParameterRecorder):
    cdef public double factor
