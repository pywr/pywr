from .._core cimport Timestep, Scenario, ScenarioIndex, AbstractNode, Storage, AbstractStorage
from ._parameters cimport Parameter, IndexParameter



cdef class HydropowerTargetParameter(Parameter):
    cdef Parameter _water_elevation_parameter
    cdef Parameter _target
    cdef Parameter _max_flow
    cdef Parameter _min_flow
    cdef public double min_head
    cdef public double turbine_elevation
    cdef public double flow_unit_conversion
    cdef public double energy_unit_conversion
    cdef public double density
    cdef public double efficiency
