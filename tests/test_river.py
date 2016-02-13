"""
A collection of tests for pywr.domains.river

Specific additional functionality of the 'special' classes in the river domain
are tested here.
"""
from __future__ import print_function
import pywr.core
from pywr.domains import river
import datetime
import numpy as np
import pytest

from helpers import assert_model

@pytest.fixture(params=[(10.0, 10.0, 10.0), (5.0, 5.0, 1.0)])
def simple_gauge_model(request, solver):
    """
    Make a simple model with a single Input and Output and RiverGauge

    Input -> PiecewiseLink -> Output

    """
    in_flow, out_flow, benefit = request.param
    min_flow_req = 5.0

    model = pywr.core.Model(solver=solver)
    inpt = river.Catchment(model, name="Catchment", flow=in_flow)
    lnk = river.RiverGauge(model, name="Gauge", mrf=min_flow_req, mrf_cost=-1.0)
    inpt.connect(lnk)
    otpt = river.DemandCentre(model, name="Demand", max_flow=out_flow, cost=-benefit)
    lnk.connect(otpt)

    default = inpt.domain

    expected_sent = in_flow if benefit > 1.0 else out_flow

    expected_node_results = {
        "Catchment": expected_sent,
        "Gauge": expected_sent,
        "Gauge Sublink 0": min(min_flow_req, expected_sent),
        "Gauge Sublink 1": expected_sent - min(min_flow_req, expected_sent),
        "Demand": expected_sent,
    }
    return model, expected_node_results


def test_piecewise_model(simple_gauge_model):
    assert_model(*simple_gauge_model)


def test_control_curve(solver):
    """
    Use a simple model of a Reservoir to test that a control curve
    behaves as expected.

    The control curve should alter the cost of the Reservoir when it
    is above or below a particular threshold.

    (flow = 8.0)          (max_flow = 10.0)
    Catchment -> River -> DemandCentre
                     |        ^
    (max_flow = 2.0) v        | (max_flow = 2.0)
                    Reservoir

    """
    in_flow = 8

    model = pywr.core.Model(solver=solver)
    catchment = river.Catchment(model, name="Catchment", flow=in_flow)
    lnk = river.River(model, name="River")
    catchment.connect(lnk)
    demand = river.DemandCentre(model, name="Demand", cost=-10.0, max_flow=10)
    lnk.connect(demand)
    from pywr.parameters import ParameterConstant
    control_curve = ParameterConstant(0.8)
    reservoir = river.Reservoir(model, name="Reservoir", max_volume=10, cost=-20, above_curve_cost=0.0,
                                control_curve=control_curve, volume=10)
    reservoir.inputs[0].max_flow = 2.0
    reservoir.outputs[0].max_flow = 2.0
    lnk.connect(reservoir)
    reservoir.connect(demand)

    model.step()
    # Reservoir is currently above control curve. 2 should be taken from the
    # reservoir
    assert(reservoir.volume == 8)
    assert(demand.flow == 10)
    # Reservoir is still at (therefore above) control curve. So 2 is still taken
    model.step()
    assert(reservoir.volume == 6)
    assert(demand.flow == 10)
    # Reservoir now below curve. Better to retain volume and divert some of the
    # inflow
    model.step()
    assert(reservoir.volume == 8)
    assert(demand.flow == 6)
    # Set the above_curve_cost function to keep filling
    from pywr.control_curves import ParameterControlCurvePiecewise
    reservoir.cost = ParameterControlCurvePiecewise(control_curve, -20.0, -20.0)
    model.step()
    assert(reservoir.volume == 10)
    assert(demand.flow == 6)
