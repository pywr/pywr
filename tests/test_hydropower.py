"""
A collection of tests for pywr.domains.river

Specific additional functionality of the 'special' classes in the river domain
are tested here.
"""
import pywr.core
from pywr.core import Model, Input, Output, Catchment
from pywr.domains import river
import pytest

from helpers import assert_model, load_model


def test_reservoir_weather():
    """
        Use a simple model of a Reservoir to test that the bathymetry, weather,
        volume, area, evaporation, rainfall behave as expected

        (flow = 8.0)          (max_flow = 10.0)
        Catchment -> River -> DemandCentre
                         |        ^
        (max_flow = 2.0) v        | (max_flow = 2.0)
                        Reservoir
                        |
                        v
                        Turbine -----> TODO ???


    """
    in_flow = 8

    model = pywr.core.Model()
    catchment = river.Catchment(model, name="Catchment", flow=in_flow)
    lnk = river.River(model, name="River")
    catchment.connect(lnk)
    demand = pywr.core.Output(model, name="Demand", cost=-10.0, max_flow=10)
    lnk.connect(demand)
    from pywr.parameters import ConstantParameter

    control_curve = ConstantParameter(model, 0.8)
    #TODO: Can we make this reservoir really simple?
    reservoir = river.Reservoir(
        model,
        name="Reservoir",
    )
    reservoir.inputs[0].max_flow = 2.0
    reservoir.outputs[0].max_flow = 2.0
    lnk.connect(reservoir)
    reservoir.connect(demand)

    turbine = hydropower.Turbine(
        #TODO: ???
    )

    reservoir.connect(turbine)
    #TODO: What does the turbine connect to?

    model.setup()

    model.step()
    #TODO assert that something has changed and explain why
    model.step()
    #TODO assert that something has changedd and explain why
    model.step()
    #TODO assert that something has changedd and explain why
