
from pywr.model import Model
from pywr.nodes import Catchment, Output, DelayNode
from pywr.recorders import NumpyArrayNodeRecorder
from pywr.parameters import ArrayIndexedParameter
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
