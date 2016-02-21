
# Forward declations
cdef class Parameter
cdef class ParameterArrayIndexed
cdef class ParameterConstantScenario
cdef class ParameterArrayIndexedScenarioMonthlyFactors

from _core cimport Timestep, Scenario, AbstractNode

cdef class Parameter:
    cdef int _size
    cdef bint _is_variable
    cdef AbstractNode _node
    cdef Parameter _parent
    cpdef setup(self, model)
    cpdef reset(self)
    cpdef before(self, Timestep ts)
    cpdef double value(self, Timestep ts, int[:] scenario_indices) except? -1
    cpdef after(self, Timestep ts)
    cpdef update(self, double[:] values)
    cpdef double[:] lower_bounds(self)
    cpdef double[:] upper_bounds(self)

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

cdef class ParameterDailyProfile(Parameter):
    cdef double[:] _values