
# Forward declations
cdef class Parameter
cdef class ParameterArrayIndexed
cdef class ParameterConstantScenario
cdef class ParameterArrayIndexedScenarioMonthlyFactors

from _core cimport Timestep, Scenario, AbstractNode

cdef class Parameter:
    cdef AbstractNode _node
    cdef Parameter _parent
    cpdef setup(self, model)
    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1
    cpdef after(self, Timestep ts)

cdef class ParameterArrayIndexed(Parameter):
    cdef double[:] values

cdef class ParameterConstantScenario(Parameter):
    cdef Scenario _scenario
    cdef double[:] _values
    cdef int _scenario_index

cdef class ParameterArrayIndexedScenarioMonthlyFactors(Parameter):
    cdef double[:] _values
    cdef double[:, :] _factors
    cdef Scenario _scenario
    cdef int _scenario_index
