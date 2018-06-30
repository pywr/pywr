from pywr.nodes import Node
from pywr.parameters import ConstantParameter
from fixtures import *


def test_node_schema(model):
    """ Basic test of loading a node directly from a schema. """

    schema = Node.Schema(context={'model': model, 'klass': Node})

    cost_param = ConstantParameter(model, 5.0, name="cost_param")

    node = schema.load({
        "name": "node1",
        "max_flow": 10.0,
        "min_flow": {
            "type": "constant",
            "value": 0.0
        },
        "cost": "cost_param"
    })

    assert node.name == "node1"
    assert node.max_flow == 10.0
    assert isinstance(node.min_flow, ConstantParameter)
    assert node.cost == cost_param
