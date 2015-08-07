
from __future__ import print_function
import pywr.core
import datetime
import numpy as np
import pytest

from test_analytical import assert_model

@pytest.fixture(params=[(10.0, 10.0, 10.0), (5.0, 5.0, 1.0)])
def simple_piecewise_model(request):
    """
    Make a simple model with a single Input and Output and PiecewiseLink

    Input -> PiecewiseLink -> Output

    """
    in_flow, out_flow, benefit = request.param
    min_flow_req = 5.0

    model = pywr.core.Model()
    inpt = pywr.core.Input(model, name="Input", max_flow=in_flow)
    lnk = pywr.core.PiecewiseLink(model, name="Link", cost=[-1.0, 0.0], max_flow=[min_flow_req, None])
    for sublnk in lnk.sublinks:
        inpt.connect(sublnk)
    otpt = pywr.core.Output(model, name="Output", min_flow=out_flow, benefit=benefit)
    lnk.connect(otpt)

    expected_requested = {'default': out_flow}
    expected_sent = {'default': in_flow if benefit > 1.0 else out_flow}

    expected_node_results = {
        "Input": expected_sent['default'],
        "Link": 0.0,
        "Link Sublink 0": min(min_flow_req, expected_sent['default']),
        "Link Sublink 1": expected_sent['default'] - min(min_flow_req, expected_sent['default']),
        "Output": expected_sent['default'],
    }
    return model, expected_requested, expected_sent, expected_node_results


def test_piecewise_model(simple_piecewise_model):
    assert_model(*simple_piecewise_model)