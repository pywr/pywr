# -*- coding: utf-8 -*-
"""
A series of test based on an analytical solution to simple
network problem.


"""
import pywr.core
import datetime
import numpy as np
import pytest
from helpers import assert_model
from fixtures import simple_linear_model

import pywr.parameters


@pytest.mark.parametrize(
    "in_flow, out_flow, benefit", [(10.0, 10.0, 10.0), (10.0, 0.0, 0.0)]
)
def test_linear_model(simple_linear_model, in_flow, out_flow, benefit):
    """
    Test the simple_linear_model with different basic input and output values
    """

    simple_linear_model.nodes["Input"].max_flow = in_flow
    simple_linear_model.nodes["Output"].min_flow = out_flow
    simple_linear_model.nodes["Output"].cost = -benefit

    expected_sent = in_flow if benefit > 1.0 else out_flow

    expected_node_results = {
        "Input": expected_sent,
        "Link": expected_sent,
        "Output": expected_sent,
    }
    assert_model(simple_linear_model, expected_node_results)


@pytest.fixture(
    params=[
        (10.0, 5.0, 5.0, 0.0, 0.0, 0.0),
        (10.0, 5.0, 5.0, 0.0, 10.0, 0.0),
        (10.0, 5.0, 5.0, 0.0, 10.0, 2.0),
        (10.0, 5.0, 0.0, 5.0, 10.0, 2.0),
    ]
)
def linear_model_with_storage(request):
    """
    Make a simple model with a single Input and Output and an offline Storage Node

    Input -> Link -> Output
               |     ^
               v     |
               Storage
    """
    (
        in_flow,
        out_flow,
        out_benefit,
        strg_benefit,
        current_volume,
        min_volume,
    ) = request.param
    max_strg_out = 10.0
    max_volume = 10.0

    model = pywr.core.Model()
    inpt = pywr.core.Input(model, name="Input", min_flow=in_flow, max_flow=in_flow)
    lnk = pywr.core.Link(model, name="Link", cost=0.1)
    inpt.connect(lnk)
    otpt = pywr.core.Output(model, name="Output", min_flow=out_flow, cost=-out_benefit)
    lnk.connect(otpt)

    strg = pywr.core.Storage(
        model,
        name="Storage",
        max_volume=max_volume,
        min_volume=min_volume,
        initial_volume=current_volume,
        cost=-strg_benefit,
    )

    strg.connect(otpt)
    lnk.connect(strg)
    avail_volume = max(current_volume - min_volume, 0.0)
    avail_refill = max_volume - current_volume
    expected_sent = (
        in_flow + min(max_strg_out, avail_volume)
        if out_benefit > strg_benefit
        else max(out_flow, in_flow - avail_refill)
    )

    expected_node_results = {
        "Input": in_flow,
        "Link": in_flow,
        "Output": expected_sent,
        "Storage Output": 0.0,
        "Storage Input": min(max_strg_out, avail_volume) if out_benefit > 1.0 else 0.0,
        "Storage": min_volume if out_benefit > strg_benefit else max_volume,
    }
    return model, expected_node_results


def test_linear_model_with_storage(linear_model_with_storage):
    assert_model(*linear_model_with_storage)


@pytest.fixture
def two_domain_linear_model(request):
    """
    Make a simple model with two domains, each with a single Input and Output

    Input -> Link -> Output  : river
                        | across the domain
    Output <- Link <- Input  : grid

    """
    river_flow = 864.0  # Ml/d
    power_plant_cap = 24  # GWh/d
    power_plant_flow_req = 18.0  # Ml/GWh
    power_demand = 12  # GWh/d
    power_benefit = 10.0  # £/GWh

    river_domain = pywr.core.Domain("river")
    grid_domain = pywr.core.Domain("grid")

    model = pywr.core.Model()
    # Create river network
    river_inpt = pywr.core.Input(
        model, name="Catchment", max_flow=river_flow, domain=river_domain
    )
    river_lnk = pywr.core.Link(model, name="Reach", domain=river_domain)
    river_inpt.connect(river_lnk)
    river_otpt = pywr.core.Output(
        model, name="Abstraction", domain=river_domain, cost=0.0
    )
    river_lnk.connect(river_otpt)
    # Create grid network
    grid_inpt = pywr.core.Input(
        model,
        name="Power Plant",
        max_flow=power_plant_cap,
        domain=grid_domain,
        conversion_factor=1 / power_plant_flow_req,
    )
    grid_lnk = pywr.core.Link(model, name="Transmission", cost=1.0, domain=grid_domain)
    grid_inpt.connect(grid_lnk)
    grid_otpt = pywr.core.Output(
        model,
        name="Substation",
        max_flow=power_demand,
        cost=-power_benefit,
        domain=grid_domain,
    )
    grid_lnk.connect(grid_otpt)
    # Connect grid to river
    river_otpt.connect(grid_inpt)

    expected_requested = {"river": 0.0, "grid": 0.0}
    expected_sent = {"river": power_demand * power_plant_flow_req, "grid": power_demand}

    expected_node_results = {
        "Catchment": power_demand * power_plant_flow_req,
        "Reach": power_demand * power_plant_flow_req,
        "Abstraction": power_demand * power_plant_flow_req,
        "Power Plant": power_demand,
        "Transmission": power_demand,
        "Substation": power_demand,
    }

    return model, expected_node_results


def test_two_domain_linear_model(two_domain_linear_model):
    assert_model(*two_domain_linear_model)


@pytest.fixture
def two_cross_domain_output_single_input(request):
    """
    Make a simple model with two domains. Thre are two Output nodes
    both connect to an Input node in a different domain.

    In this example the rivers should be able to provide flow to the grid
    with a total flow equal to the sum of their respective parts.

    Input -> Link -> Output  : river
                        | across the domain
                        Input -> Link -> Output : grid
                        | across the domain
    Input -> Link -> Output  : river

    """
    river_flow = 10.0
    expected_node_results = {}

    model = pywr.core.Model()
    # Create grid network
    grid_inpt = pywr.core.Input(
        model,
        name="Input",
        domain="grid",
    )
    grid_lnk = pywr.core.Link(model, name="Link", cost=1.0, domain="grid")
    grid_inpt.connect(grid_lnk)
    grid_otpt = pywr.core.Output(
        model, name="Output", max_flow=50.0, cost=-10.0, domain="grid"
    )
    grid_lnk.connect(grid_otpt)
    # Create river network
    for i in range(2):
        river_inpt = pywr.core.Input(
            model, name="Catchment {}".format(i), max_flow=river_flow, domain="river"
        )
        river_lnk = pywr.core.Link(model, name="Reach {}".format(i), domain="river")
        river_inpt.connect(river_lnk)
        river_otpt = pywr.core.Output(
            model, name="Abstraction {}".format(i), domain="river", cost=0.0
        )
        river_lnk.connect(river_otpt)
        # Connect grid to river
        river_otpt.connect(grid_inpt)

        expected_node_results.update(
            {
                "Catchment {}".format(i): river_flow,
                "Reach {}".format(i): river_flow,
                "Abstraction {}".format(i): river_flow,
            }
        )

    expected_node_results.update(
        {
            "Input": river_flow * 2,
            "Link": river_flow * 2,
            "Output": river_flow * 2,
        }
    )

    return model, expected_node_results


@pytest.mark.xfail
def test_two_cross_domain_output_single_input(two_cross_domain_output_single_input):
    # TODO This test currently fails because of the simple way in which the cross
    # domain paths work. It can not cope with two Outputs connected to one
    # input.
    assert_model(*two_cross_domain_output_single_input)


@pytest.fixture()
def simple_linear_inline_model(request):
    """
    Make a simple model with a single Input and Output nodes inline of a route.

    Input 0 -> Input 1 -> Link -> Output 0 -> Output 1

    """
    model = pywr.core.Model()
    inpt0 = pywr.core.Input(model, name="Input 0")
    inpt1 = pywr.core.Input(model, name="Input 1")
    inpt0.connect(inpt1)
    lnk = pywr.core.Link(model, name="Link", cost=1.0)
    inpt1.connect(lnk)
    otpt0 = pywr.core.Output(model, name="Output 0")
    lnk.connect(otpt0)
    otpt1 = pywr.core.Output(model, name="Output 1")
    otpt0.connect(otpt1)

    return model


@pytest.mark.skipif(
    pywr.core.Model().solver.name == "glpk-edge",
    reason="Not valid for GLPK Edge based solver.",
)
@pytest.mark.parametrize(
    "in_flow_1, out_flow_0, link_flow", [(10.0, 10.0, 15.0), (0.0, 0.0, 10.0)]
)
def test_simple_linear_inline_model(
    simple_linear_inline_model, in_flow_1, out_flow_0, link_flow
):
    """
    Test the test_simple_linear_inline_model with different flow constraints
    """
    model = simple_linear_inline_model
    model.nodes["Input 0"].max_flow = 10.0
    model.nodes["Input 1"].max_flow = in_flow_1
    model.nodes["Link"].max_flow = link_flow
    model.nodes["Output 0"].max_flow = out_flow_0
    model.nodes["Input 1"].cost = 1.0
    model.nodes["Output 0"].cost = -10.0
    model.nodes["Output 1"].cost = -5.0

    expected_sent = min(link_flow, 10 + in_flow_1)

    expected_node_results = {
        "Input 0": 10.0,
        "Input 1": max(expected_sent - 10.0, 0.0),
        "Link": expected_sent,
        "Output 0": min(expected_sent, out_flow_0),
        "Output 1": max(expected_sent - out_flow_0, 0.0),
    }
    assert_model(model, expected_node_results)


@pytest.fixture()
def bidirectional_model(request):
    """
    Make a simple model with a single Input and Output.

    Input 0 -> Link 0 -> Output 0
               |   ^
               v   |
    Input 1 -> Link 1 -> Output 1

    """
    model = pywr.core.Model()
    for i in range(2):
        inpt = pywr.core.Input(model, name="Input {}".format(i))
        lnk = pywr.core.Link(model, name="Link {}".format(i))
        inpt.connect(lnk)
        otpt = pywr.core.Output(model, name="Output {}".format(i))
        lnk.connect(otpt)

    # Create bidirectional link (i.e. a cycle)
    model.nodes["Link 0"].connect(model.nodes["Link 1"])
    model.nodes["Link 1"].connect(model.nodes["Link 0"])

    return model


def test_bidirectional_model(bidirectional_model):
    """
    Test the simple_linear_model with different basic input and output values
    """
    model = bidirectional_model
    model.nodes["Input 0"].max_flow = 10.0
    model.nodes["Input 1"].max_flow = 10.0
    model.nodes["Output 0"].max_flow = 10.0
    model.nodes["Output 1"].max_flow = 15.0
    model.nodes["Output 0"].cost = -5.0
    model.nodes["Output 1"].cost = -10.0
    model.nodes["Link 0"].cost = 1.0
    model.nodes["Link 1"].cost = 1.0

    expected_node_results = {
        "Input 0": 10.0,
        "Input 1": 10.0,
        "Link 0": 10.0,
        "Link 1": 15.0,
        "Output 0": 5.0,
        "Output 1": 15.0,
    }
    assert_model(model, expected_node_results)


def make_simple_model(supply_amplitude, demand, frequency, initial_volume):
    """
    Make a simple model,
        supply -> reservoir -> demand.

    supply is a annual cosine function with amplitude supply_amplitude and
    frequency

    """

    model = pywr.core.Model()

    S = supply_amplitude
    w = frequency

    class SupplyFunc(pywr.parameters.Parameter):
        def value(self, ts, si):
            # Take the mean flow of the day (i.e. offset by half a day)
            t = ts.dayofyear - 0.5
            v = S * np.cos(t * w) + S
            return v

    max_flow = SupplyFunc(model)
    supply = pywr.core.Input(model, name="supply", max_flow=max_flow, min_flow=max_flow)
    demand = pywr.core.Output(model, name="demand", max_flow=demand, cost=-10)
    res = pywr.core.Storage(
        model, name="reservoir", max_volume=1e6, initial_volume=initial_volume
    )

    supply_res_link = pywr.core.Link(model, name="link1")
    res_demand_link = pywr.core.Link(model, name="link2")

    supply.connect(supply_res_link)
    supply_res_link.connect(res)
    res.connect(res_demand_link)
    res_demand_link.connect(demand)

    return model


def test_analytical():
    """
    Run the test model though a year with analytical solution values to
    ensure reservoir just contains sufficient volume.
    """

    S = 100.0  # supply amplitude
    D = S  # demand
    w = 2 * np.pi / 365  # frequency (annual)
    V0 = S / w  # initial reservoir level

    model = make_simple_model(S, D, w, V0)

    T = np.arange(1, 365)
    V_anal = S * (np.sin(w * T) / w + T) - D * T + V0
    V_model = np.empty(T.shape)

    for i, t in enumerate(T):
        model.step()
        V_model[i] = model.nodes["reservoir"].volume[0]

    # Relative error from initial volume
    error = np.abs(V_model - V_anal) / V0
    assert np.all(error < 1e-4)
