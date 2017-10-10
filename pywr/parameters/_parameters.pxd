from pywr.recorders._recorders cimport Recorder
from pywr._component cimport Component
from .._core cimport Timestep, Scenario, ScenarioIndex, ScenarioCollection, AbstractNode, Node

cdef class Parameter(Component):
    cdef int _size
    cdef public bint is_variable
    cdef AbstractNode _node
    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1
    cdef double[:] __values
    cdef calc_values(self, Timestep ts)
    cpdef double get_value(self, ScenarioIndex scenario_index)
    cpdef double[:] get_all_values(self)
    cpdef update(self, double[:] values)
    cpdef double[:] lower_bounds(self)
    cpdef double[:] upper_bounds(self)


cdef class ConstantParameter(Parameter):
    cdef double _value
    cdef double[:] _lower_bounds
    cdef double[:] _upper_bounds

cdef class DataFrameParameter(Parameter):
    cdef double[:,:] _values
    cdef public Scenario scenario
    cdef int _scenario_index
    cdef public object dataframe

cdef class ArrayIndexedParameter(Parameter):
    cdef double[:] values

cdef class ArrayIndexedScenarioParameter(Parameter):
    cdef Scenario _scenario
    cdef double[:, :] values
    cdef int _scenario_index

cdef class ConstantScenarioParameter(Parameter):
    cdef Scenario _scenario
    cdef double[:] _values
    cdef int _scenario_index

cdef class ArrayIndexedScenarioMonthlyFactorsParameter(Parameter):
    cdef double[:] _values
    cdef double[:, :] _factors
    cdef Scenario _scenario
    cdef int _scenario_index



cdef class DailyProfileParameter(Parameter):
    cdef double[:] _values

cdef class WeeklyProfileParameter(Parameter):
    cdef double[:] _values

cdef class MonthlyProfileParameter(Parameter):
    cdef double[:] _values
    cdef double[:] _lower_bounds
    cdef double[:] _upper_bounds


cdef class ScenarioMonthlyProfileParameter(Parameter):
    cdef double[:, :] _values
    cdef Scenario _scenario
    cdef int _scenario_index

cdef class IndexParameter(Parameter):
    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1
    cdef int[:] __indices
    cpdef int get_index(self, ScenarioIndex scenario_index)
    cpdef int[:] get_all_indices(self)

cdef class TablesArrayParameter(IndexParameter):
    cdef double[:, :] _values_dbl
    cdef int[:, :] _values_int
    cdef public Scenario scenario
    cdef public object h5file
    cdef public object h5store
    cdef public object node
    cdef public object where

    cdef int _scenario_index
    cdef int[:] _scenario_ids

cdef class IndexedArrayParameter(Parameter):
    cdef public IndexParameter index_parameter
    cdef public list params

cdef class AnnualHarmonicSeriesParameter(Parameter):
    cdef public double mean
    cdef double[:] _amplitudes, _phases
    cdef double _mean_lower_bounds, _mean_upper_bounds
    cdef double[:] _amplitude_lower_bounds, _amplitude_upper_bounds
    cdef double[:] _phase_lower_bounds, _phase_upper_bounds
    cdef double _value_cache
    cdef int _ts_index_cache

cdef class CachedParameter(IndexParameter):
    cdef public Parameter parameter
    cdef Timestep timestep
    cdef ScenarioIndex scenario_index
    cdef double cached_value
    cdef int cached_index
    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1
    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1

cdef class AggregatedParameter(Parameter):
    # This is a list rather than a set due to floating point arithmetic.
    # The order is important for maintaining determinism.
    cdef public list parameters
    cdef object _agg_user_func
    cdef int _agg_func
    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1
    cpdef add(self, Parameter parameter)
    cpdef remove(self, Parameter parameter)

cdef class AggregatedIndexParameter(IndexParameter):
    # This is a list; see above AggregatedParameter.
    cdef public list parameters
    cdef object _agg_user_func
    cdef int _agg_func
    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1
    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1
    cpdef add(self, Parameter parameter)
    cpdef remove(self, Parameter parameter)

cdef class NegativeParameter(Parameter):
    cdef public Parameter parameter

cdef class MaxParameter(Parameter):
    cdef public Parameter parameter
    cdef public double threshold

cdef class NegativeMaxParameter(MaxParameter):
    pass

cdef class MinParameter(Parameter):
    cdef public Parameter parameter
    cdef public double threshold

cdef class NegativeMinParameter(MinParameter):
    pass

cdef class DeficitParameter(Parameter):
    cdef public Node node
