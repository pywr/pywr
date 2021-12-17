import os
from numpy.testing import assert_allclose
import pywr.core
from pywr.core import Model, Timestep
import pandas


def load_model(filename=None, data=None, solver=None, model_klass=None, check=True):
    """Load a test model and check it"""
    if data is None:
        path = os.path.join(os.path.dirname(__file__), "models")
        with open(os.path.join(path, filename), "r") as f:
            data = f.read()
    else:
        path = None

    if model_klass is None:
        model_klass = Model

    model = model_klass.loads(data, path=path, solver=solver)
    if check:
        model.check()
    return model


def assert_model(model, expected_node_results):
    __tracebackhide__ = True
    model.step()

    for node in model.nodes:
        if node.name in expected_node_results:
            if isinstance(node, pywr.core.BaseNode):
                assert_allclose(expected_node_results[node.name], node.flow, atol=1e-7)
            elif isinstance(node, pywr.core.Storage):
                assert_allclose(
                    expected_node_results[node.name], node.volume, atol=1e-7
                )


def build_timestep(model, date):
    dt = pandas.to_datetime(date)
    timestep = Timestep(dt, (dt - model.timestepper.start).days, 1)
    return timestep
