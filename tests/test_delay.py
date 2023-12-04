from pywr.model import Model
from pywr.nodes import Catchment, Output, DelayNode
from pywr.recorders import NumpyArrayNodeRecorder, AssertionRecorder
from pywr.parameters import (
    ArrayIndexedScenarioParameter,
    FlowDelayParameter,
    load_parameter,
)
from pywr.core import Scenario
from numpy.testing import assert_array_almost_equal
import numpy as np
import pytest


@pytest.mark.parametrize(
    "key, delay, initial_flow",
    [
        ("days", 1, None),
        ("timesteps", 1, None),
        ("days", 1, 5.0),
        ("days", 3, 5.0),
        ("timesteps", 10, 5.0),
    ],
)
def test_delay_node(key, delay, initial_flow):
    """Test that the `DelayNode` and the `FlowDelayParameter` internal to it correctly delay node for a range of inputs and
    across scenarios"""
    model = Model()

    model.timestepper.start = "2015/01/01"
    model.timestepper.end = "2015/01/31"

    scen = Scenario(model, name="scenario", size=2)
    flow_vals = np.arange(1, 63).reshape((31, 2), order="F")
    flow = ArrayIndexedScenarioParameter(model, scen, flow_vals)

    catchment = Catchment(model, name="input", flow=flow)
    kwargs = {key: delay}
    if initial_flow:
        kwargs["initial_flow"] = initial_flow
    delaynode = DelayNode(model, name="delaynode", **kwargs)
    output = Output(model, name="output")

    catchment.connect(delaynode)
    delaynode.connect(output)

    rec = NumpyArrayNodeRecorder(model, output)

    model.run()
    if initial_flow:
        expected = np.concatenate(
            [np.full((delay, 2), initial_flow), flow_vals[:-delay, :]]
        )
    else:
        expected = np.concatenate([np.zeros((delay, 2)), flow_vals[:-delay, :]])

    assert_array_almost_equal(rec.data, expected)


def test_delay_param_load():
    """Test that the `.load` method of `FlowDelayParameter` works correctly"""

    model = Model()
    model.timestepper.start = "2015/01/01"
    model.timestepper.end = "2015/01/31"
    catchment = Catchment(model, name="input", flow=1)
    output = Output(model, name="output")
    catchment.connect(output)

    data = {"name": "delay", "node": "input", "days": 2}

    param = FlowDelayParameter.load(model, data)

    assert param.days == 2

    data = {"name": "delay2", "node": "input", "timesteps": 2}

    param2 = FlowDelayParameter.load(model, data)

    assert param2.timesteps == 2

    expected = np.concatenate([np.zeros(2), np.ones(29)]).reshape(31, 1)

    AssertionRecorder(model, param, name="rec1", expected_data=expected)
    AssertionRecorder(model, param2, name="rec2", expected_data=expected)

    model.setup()
    model.run()


@pytest.mark.parametrize("key, delay", [("days", 5), ("timesteps", 0.5)])
def test_delay_failure(key, delay):
    """Test the FlowDelayParameter returns a ValueError when the input value of the `days` attribute is not
    divisible exactly by the model timestep delta and when the `timesteps` attribute is less than 1
    """

    model = Model()
    model.timestepper.start = "2015/01/01"
    model.timestepper.end = "2015/01/31"
    model.timestepper.delta = 3

    catchment = Catchment(model, name="input", flow=1)
    output = Output(model, name="output")
    catchment.connect(output)

    FlowDelayParameter(model, catchment, **{key: delay})

    with pytest.raises(ValueError):
        model.setup()
