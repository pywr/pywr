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

from test_analytical import assert_model

@pytest.fixture(params=[(10.0, 10.0, 10.0), (5.0, 5.0, 1.0)])
def simple_gauge_model(request):
    """
    Make a simple model with a single Input and Output and RiverGauge

    Input -> PiecewiseLink -> Output

    """
    in_flow, out_flow, benefit = request.param
    min_flow_req = 5.0

    model = pywr.core.Model(solver=request.config.getoption("--solver"))
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
