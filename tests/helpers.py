import os
from numpy.testing import assert_allclose
import pywr.core
from pywr.core import Model


def load_model(filename=None, data=None, solver=None):
    '''Load a test model and check it'''
    if data is None:
        path = os.path.join(os.path.dirname(__file__), 'models')
        with open(os.path.join(path, filename), 'r') as f:
            data = f.read()
    else:
        path = None

    model = Model.loads(data, path=path)
    model.check()
    return model


def assert_model(model, expected_node_results):
    __tracebackhide__ = True
    model.step()

    for node in model.nodes():
        if node.name in expected_node_results:
            if isinstance(node, pywr.core.BaseNode):
                assert_allclose(expected_node_results[node.name], node.flow, atol=1e-7)
            elif isinstance(node, pywr.core.Storage):
                assert_allclose(expected_node_results[node.name], node.volume, atol=1e-7)
