
from __future__ import print_function
import pywr.core
import datetime
import numpy as np
import pytest

from helpers import assert_model

@pytest.fixture(params=[(10.0, 10.0, 10.0), (5.0, 5.0, 1.0)])
def simple_piecewise_model(request, solver):
    """
    Make a simple model with a single Input and Output and PiecewiseLink

    Input -> PiecewiseLink -> Output

    """
    in_flow, out_flow, benefit = request.param
    min_flow_req = 5.0

    model = pywr.core.Model(solver=solver)
    inpt = pywr.core.Input(model=model, name="Input", max_flow=in_flow)
    lnk = pywr.core.PiecewiseLink(model=model, name="Link", cost=[-1.0, 0.0], max_flow=[min_flow_req, None])

    inpt.connect(lnk)
    otpt = pywr.core.Output(model=model, name="Output", min_flow=out_flow, cost=-benefit)
    lnk.connect(otpt)

    default = inpt.domain


    expected_sent = in_flow if benefit > 1.0 else out_flow

    expected_node_results = {
        "Input": expected_sent,
        "Link": expected_sent,
        "Link Sublink 0": min(min_flow_req, expected_sent),
        "Link Sublink 1": expected_sent - min(min_flow_req, expected_sent),
        "Output": expected_sent,
    }
    return model, expected_node_results


def test_piecewise_model(simple_piecewise_model):
    assert_model(*simple_piecewise_model)
