"""
A collection of tests for pywr.domains.river

Specific additional functionality of the 'special' classes in the river domain
are tested here.
"""
import pywr.core
from pywr.core import Model, Input, Output, Catchment
from pywr.parameters import MonthlyProfileParameter
from pywr.domains import river
from numpy.testing import assert_allclose
import pytest

from helpers import assert_model, load_model


@pytest.fixture(params=[(10.0, 10.0, 10.0), (5.0, 5.0, 1.0)])
def simple_gauge_model(request):
    """
    Make a simple model with a single Input and Output and RiverGauge

    Input -> PiecewiseLink -> Output

    """
    in_flow, out_flow, benefit = request.param
    min_flow_req = 5.0

    model = pywr.core.Model()
    inpt = river.Catchment(model, name="Catchment", flow=in_flow)
    lnk = river.RiverGauge(model, name="Gauge", mrf=min_flow_req, mrf_cost=-1.0)
    inpt.connect(lnk)
    otpt = pywr.core.Output(model, name="Demand", max_flow=out_flow, cost=-benefit)
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


@pytest.fixture
def simple_river_split_gauge_model():
    """
    Make a simple model with a single Input and Output and RiverGauge

    ::

        Input -> RiverSplit -> Output 1
                     \\ --->--> Output 2

    """
    in_flow = 100.0
    min_flow_req = 40.0
    out_flow = 50.0
    model = pywr.core.Model()

    inpt = river.Catchment(model, name="Catchment", flow=in_flow)
    lnk = river.RiverSplitWithGauge(
        model,
        name="Gauge",
        mrf=min_flow_req,
        mrf_cost=-100,
        slot_names=("river", "abstraction"),
        factors=[3, 1],
    )
    inpt.connect(lnk)
    estuary = pywr.core.Output(model, name="Estuary")
    lnk.connect(estuary, from_slot="river")
    otpt = pywr.core.Output(model, name="Demand", max_flow=out_flow, cost=-10)
    lnk.connect(otpt, from_slot="abstraction")

    net_flow_after_mrf = in_flow - min_flow_req
    expected_node_results = {
        "Catchment": in_flow,
        "Gauge": in_flow,
        "Gauge Sublink 0": min_flow_req,
        "Gauge Sublink 1": net_flow_after_mrf * 0.75,
        "Gauge Sublink 2": net_flow_after_mrf * 0.25,
        "Demand": min(out_flow, net_flow_after_mrf * 0.25),
    }
    return model, expected_node_results


def test_river_gauge():
    """
    Test loading a model with a RiverGauge from JSON, modifying it, then running it
    """
    model = load_model("river_mrf1.json")

    node = model.nodes["mrf"]
    demand = model.nodes["demand"]

    # test getting properties
    assert isinstance(node.mrf, MonthlyProfileParameter)
    assert_allclose(node.mrf_cost, -1000)
    assert_allclose(node.cost, 0.0)

    # test setting properties
    node.mrf = 40
    node.mrf_cost = -999
    assert_allclose(node.mrf, 40)
    assert_allclose(node.mrf_cost, -999)

    # run the model and see if it works
    model.run()
    assert_allclose(node.flow, 40)
    assert_allclose(demand.flow, 60)


def test_piecewise_model(simple_gauge_model):
    assert_model(*simple_gauge_model)


def test_river_split_gauge(simple_river_split_gauge_model):
    assert_model(*simple_river_split_gauge_model)


def test_river_split_gauge_json():
    """As test_river_split_gauge, but model is defined in JSON"""

    model = load_model("river_split_with_gauge1.json")
    model.check()
    model.run()

    catchment_flow = 100.0
    mrf = 40.0
    expected_demand_flow = (catchment_flow - mrf) * 0.25

    catchment_node = model.nodes["Catchment"]
    demand_node = model.nodes["Demand"]
    assert_allclose(catchment_node.flow, catchment_flow)
    assert_allclose(demand_node.flow, expected_demand_flow)


def test_control_curve():
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

    model = pywr.core.Model()
    catchment = river.Catchment(model, name="Catchment", flow=in_flow)
    lnk = river.River(model, name="River")
    catchment.connect(lnk)
    demand = pywr.core.Output(model, name="Demand", cost=-10.0, max_flow=10)
    lnk.connect(demand)
    from pywr.parameters import ConstantParameter

    control_curve = ConstantParameter(model, 0.8)
    reservoir = river.Reservoir(
        model,
        name="Reservoir",
        max_volume=10,
        cost=-20,
        above_curve_cost=0.0,
        control_curve=control_curve,
        initial_volume=10,
    )
    reservoir.inputs[0].max_flow = 2.0
    reservoir.outputs[0].max_flow = 2.0
    lnk.connect(reservoir)
    reservoir.connect(demand)

    model.setup()

    model.step()
    # Reservoir is currently above control curve. 2 should be taken from the
    # reservoir
    assert reservoir.volume == 8
    assert demand.flow == 10
    # Reservoir is still at (therefore above) control curve. So 2 is still taken
    model.step()
    assert reservoir.volume == 6
    assert demand.flow == 10
    # Reservoir now below curve. Better to retain volume and divert some of the
    # inflow
    model.step()
    assert reservoir.volume == 8
    assert demand.flow == 6
    # Set the above_curve_cost function to keep filling
    from pywr.parameters.control_curves import ControlCurveParameter

    # We know what we're doing with the control_curve Parameter so unset its parent before overriding
    # the cost parameter.
    # We need to call setup() again because we're adding a parameter.
    reservoir.cost = ControlCurveParameter(
        model, reservoir, control_curve, [-20.0, -20.0]
    )
    reservoir.initial_volume = 8
    model.setup()
    model.step()
    assert reservoir.volume == 10
    assert demand.flow == 6


def test_catchment_many_successors():
    """Test if node with fixed flow can have multiple successors. See #225"""
    model = Model()
    catchment = Catchment(model, "catchment", flow=100)
    out1 = Output(model, "out1", max_flow=10, cost=-100)
    out2 = Output(model, "out2", max_flow=15, cost=-50)
    out3 = Output(model, "out3")
    catchment.connect(out1)
    catchment.connect(out2)
    catchment.connect(out3)
    model.check()
    model.run()
    assert_allclose(out1.flow, 10)
    assert_allclose(out2.flow, 15)
    assert_allclose(out3.flow, 75)
