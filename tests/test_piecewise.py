import pywr.core
from pywr.parameters import ConstantParameter
import numpy as np
from numpy.testing import assert_allclose
import pytest
from helpers import assert_model, load_model


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
    lnk = pywr.core.PiecewiseLink(
        model, name="Link", nsteps=2, costs=[-1.0, 0.0], max_flows=[min_flow_req, None]
    )

    inpt.connect(lnk)
    otpt = pywr.core.Output(model, name="Output", min_flow=out_flow, cost=-benefit)
    lnk.connect(otpt)

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


def test_piecewise_json():
    """Test loading of a piecewise link from JSON"""
    model = load_model("piecewise1.json")
    sublinks = model.nodes["link1"].sublinks
    max_flows = [sublink.max_flow for sublink in sublinks]
    costs = [sublink.cost for sublink in sublinks]
    assert_allclose(max_flows, [20, np.inf])
    assert_allclose(costs, [-10, 5])
    model.run()
    assert_allclose(model.nodes["demand1"].flow, 20)


def test_piecewise_with_parameters_json():
    """Test using parameters with piecewise link."""
    model = load_model("piecewise1_with_parameters.json")
    sublinks = model.nodes["link1"].sublinks

    assert isinstance(sublinks[0].max_flow, ConstantParameter)
    assert np.isinf(sublinks[1].max_flow)
    assert isinstance(sublinks[0].cost, ConstantParameter)
    assert isinstance(sublinks[1].cost, ConstantParameter)

    model.run()

    assert_allclose(model.nodes["demand1"].flow, 20)
