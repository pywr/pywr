
# Forward declations
cdef class Parameter
cdef class ArrayIndexedParameter
cdef class ConstantScenarioParameter
cdef class ArrayIndexedScenarioMonthlyFactorsParameter

from .._core cimport Timestep, Scenario, ScenarioIndex, AbstractNode

cdef class Parameter:
    cdef int _size
    cdef bint _is_variable
    cdef AbstractNode _node
    cdef readonly object parents
    cdef readonly object children
    cpdef setup(self, model)
    cpdef reset(self)
    cpdef before(self, Timestep ts)
    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1
    cpdef after(self, Timestep ts)
    cpdef update(self, double[:] values)
    cpdef double[:] lower_bounds(self)
    cpdef double[:] upper_bounds(self)
    cdef public str name

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

cdef class IndexParameter(Parameter):
    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1

cdef class IndexedArrayParameter(Parameter):
    cdef public Parameter index_parameter
    cdef public list params

cdef class CachedParameter(IndexParameter):
    cdef public Parameter parameter
    cdef Timestep timestep
    cdef ScenarioIndex scenario_index
    cdef double cached_value
    cdef int cached_index
    cpdef reset(self)
    cpdef double value(self, Timestep timestep, ScenarioIndex scenario_index) except? -1
    cpdef int index(self, Timestep timestep, ScenarioIndex scenario_index) except? -1

cdef class CachedTimeParameter(CachedParameter):
    pass
