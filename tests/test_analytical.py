# -*- coding: utf-8 -*-
"""
A series of test based on an analytical solution to simple
network problem.


"""
from __future__ import print_function
import pywr.core
import datetime
import numpy as np
import pytest
from helpers import assert_model


@pytest.fixture(params=[(10.0, 10.0, 10.0), (10.0, 0.0, 0.0)])
def simple_linear_model(request, solver):
    """
    Make a simple model with a single Input and Output.

    Input -> Link -> Output

    """
    in_flow, out_flow, benefit = request.param

    model = pywr.core.Model(solver=solver)
    inpt = pywr.core.Input(model, name="Input", max_flow=in_flow)
    lnk = pywr.core.Link(model, name="Link", cost=1.0)
    inpt.connect(lnk)
    otpt = pywr.core.Output(model, name="Output", min_flow=out_flow, cost=-benefit)
    lnk.connect(otpt)
    default = inpt.domain
    expected_requested = {default: out_flow}
    expected_sent = {default: in_flow if benefit > 1.0 else out_flow}

    expected_node_results = {
        "Input": expected_sent[default],
        "Link": expected_sent[default],
        "Output": expected_sent[default],
    }
    return model, expected_node_results


def test_linear_model(simple_linear_model):
    assert_model(*simple_linear_model)


@pytest.fixture(params=[
    (10.0, 5.0, 5.0, 0.0, 0.0, 0.0),
    (10.0, 5.0, 5.0, 0.0, 10.0, 0.0),
    (10.0, 5.0, 5.0, 0.0, 10.0, 2.0),
    ])
def linear_model_with_storage(request, solver):
    """
    Make a simple model with a single Input and Output and an offline Storage Node

    Input -> Link -> Output
               |     ^
               v     |
               Storage
    """
    in_flow, out_flow, out_benefit, strg_benefit, current_volume, min_volume = request.param
    max_strg_out = 10.0

    model = pywr.core.Model(solver=solver)
    inpt = pywr.core.Input(model, name="Input", max_flow=in_flow)
    lnk = pywr.core.Link(model, name="Link", cost=0.1)
    inpt.connect(lnk)
    otpt = pywr.core.Output(model, name="Output", min_flow=out_flow, cost=-out_benefit)
    lnk.connect(otpt)

    strg = pywr.core.Storage(model, name="Storage", max_volume=10.0, min_volume=min_volume,
                             volume=current_volume, cost=-strg_benefit)

    strg.connect(otpt)
    lnk.connect(strg)
    avail_volume = max(current_volume - min_volume, 0.0)
    expected_sent = in_flow+min(max_strg_out, avail_volume) if out_benefit > 1.0 else out_flow

    expected_node_results = {
        "Input": in_flow,
        "Link": in_flow,
        "Output": expected_sent,
        "Storage Output": 0.0,
        "Storage Input": min(max_strg_out, avail_volume) if out_benefit > 1.0 else 0.0,
        "Storage": min_volume,
    }
    return model, expected_node_results

def test_linear_model_with_storage(linear_model_with_storage):
    assert_model(*linear_model_with_storage)

@pytest.fixture
def two_domain_linear_model(request, solver):
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

    model = pywr.core.Model(solver=solver)
    # Create river network
    river_inpt = pywr.core.Input(model, name="Catchment", max_flow=river_flow, domain='river')
    river_lnk = pywr.core.Link(model, name="Reach", domain='river')
    river_inpt.connect(river_lnk)
    river_otpt = pywr.core.Output(model, name="Abstraction", domain='river', cost=0.0)
    river_lnk.connect(river_otpt)
    # Create grid network
    grid_inpt = pywr.core.Input(model, name="Power Plant", max_flow=power_plant_cap, domain='grid',
                                               conversion_factor=1/power_plant_flow_req)
    grid_lnk = pywr.core.Link(model, name="Transmission", cost=1.0, domain='grid')
    grid_inpt.connect(grid_lnk)
    grid_otpt = pywr.core.Output(model, name="Substation", max_flow=power_demand,
                                 cost=-power_benefit, domain='grid')
    grid_lnk.connect(grid_otpt)
    # Connect grid to river
    river_otpt.connect(grid_inpt)

    expected_requested = {'river': 0.0, 'grid': 0.0}
    expected_sent = {'river': power_demand*power_plant_flow_req, 'grid': power_demand}

    expected_node_results = {
        "Catchment": power_demand*power_plant_flow_req,
        "Reach": power_demand*power_plant_flow_req,
        "Abstraction": power_demand*power_plant_flow_req,
        "Power Plant": power_demand,
        "Transmission": power_demand,
        "Substation": power_demand,
    }

    return model, expected_node_results


def test_two_domain_linear_model(two_domain_linear_model):
    assert_model(*two_domain_linear_model)


def make_simple_model(supply_amplitude, demand, frequency,
                      initial_volume, solver):
    """
    Make a simlpe model,
        supply -> reservoir -> demand.

    supply is a annual cosine function with amplitude supply_amplitude and
    frequency

    """

    model = pywr.core.Model(solver=solver)

    S = supply_amplitude
    w = frequency

    def supply_func(parent, index):
        t = parent.model.timestamp.timetuple().tm_yday
        return S*np.cos(t*w)+S

    supply = pywr.core.Supply(model, name='supply', max_flow=supply_func)
    demand = pywr.core.Demand(model, name='demand', demand=demand)
    res = pywr.core.Reservoir(model, name='reservoir')
    res.properties['max_volume'] = pywr.core.ParameterConstant(1e6)
    res.properties['current_volume'] = pywr.core.Variable(initial_volume)

    supply_res_link = pywr.core.Link(model, name='link1')
    res_demand_link = pywr.core.Link(model, name='link2')

    supply.connect(supply_res_link)
    supply_res_link.connect(res)
    res.connect(res_demand_link)
    res_demand_link.connect(demand)

    return model

def pytest_run_analytical(solver):
    """
    Run the test model though a year with analytical solution values to
    ensure reservoir just contains sufficient volume.
    """

    S = 100.0 # supply amplitude
    D = S # demand
    w = 2*np.pi/365 # frequency (annual)
    V0 = S/w  # initial reservoir level

    model = make_simple_model(S, D, w, V0, solver)

    model.timestamp = datetime.datetime(2015, 1, 1)

    # TODO include first timestep
    T = np.arange(1,365)
    V_anal = S*(np.sin(w*T)/w+T) - D*T + V0
    V_model = np.empty(T.shape)

    for i,t in enumerate(T):
        model.step()
        for node in model.nodes():
            if 'current_volume' in node.properties:
                V_model[i] = node.properties['current_volume'].value()


    return T, V_model, V_anal
