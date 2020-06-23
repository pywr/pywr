from pywr.model import Model
from pywr.nodes import Catchment, Output, DelayNode
from pywr.recorders import NumpyArrayNodeRecorder, AssertionRecorder
from pywr.parameters import ArrayIndexedParameter, FlowDelayParameter, load_parameter
from pywr.core import Scenario
from numpy.testing import assert_array_almost_equal
import numpy as np
import pytest


@pytest.mark.parametrize("key, delay", [("days", 1),
                                        ("timesteps", 1),
                                        ("days", 3),
                                        ("timesteps", 10)])
def test_delay_node(key, delay):
    model = Model()

    model.timestepper.start = "2015/01/01"
    model.timestepper.end = "2015/01/31"
    flow = ArrayIndexedParameter(model, np.arange(1, 32))
    Scenario(model, name="scenario", size=2)

    catchment = Catchment(model, name="input", flow=flow)
    delaynode = DelayNode(model, name="delaynode", **{key: delay})
    output = Output(model, name="output")

    catchment.connect(delaynode)
    delaynode.connect(output)

    rec = NumpyArrayNodeRecorder(model, output)

    model.run()
    expected = np.concatenate([np.zeros(delay), np.arange(1, 32 - delay)])
    expected = np.tile(expected.reshape(31, 1), 2)
    assert_array_almost_equal(rec.data, expected)


def test_delay_param_load():

    model = Model()
    model.timestepper.start = "2015/01/01"
    model.timestepper.end = "2015/01/31"
    catchment = Catchment(model, name="input", flow=1)
    output = Output(model, name="output")
    catchment.connect(output)

    data = {"name": "delay",
            "node": "input",
            "days": 2}

    param = FlowDelayParameter.load(model, data)

    assert param.days == 2

    data = {"name": "delay2",
            "node": "input",
            "timesteps": 2}

    param2 = FlowDelayParameter.load(model, data)

    assert param2.timesteps == 2

    expected = np.concatenate([np.zeros(2), np.ones(29)]).reshape(31, 1)

    AssertionRecorder(model, param, name="rec1", expected_data=expected)
    AssertionRecorder(model, param2, name="rec2", expected_data=expected)

    model.setup()
    model.run()

@pytest.mark.parametrize("key, delay", [("days", 5),
                                        ("timesteps", 0.5)])
def test_delay_failure(key, delay):

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
