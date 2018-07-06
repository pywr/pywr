from ._recorders cimport NumpyArrayNodeRecorder, BaseConstantNodeRecorder
from pywr.parameters._parameters cimport Parameter
from .._core cimport Timestep, Scenario, ScenarioIndex


cdef class HydropowerRecorder(NumpyArrayNodeRecorder):
    cdef Parameter _water_elevation_parameter
    cdef public double turbine_elevation
    cdef public double flow_unit_conversion
    cdef public double energy_unit_conversion
    cdef public double density
    cdef public double efficiency


cdef class TotalHydroEnergyRecorder(BaseConstantNodeRecorder):
    cdef Parameter _water_elevation_parameter
    cdef public double turbine_elevation
    cdef public double flow_unit_conversion
    cdef public double energy_unit_conversion
    cdef public double density
    cdef public double efficiency