from pywr.nodes import Node

from fixtures import simple_linear_model, simple_storage_model

model = simple_linear_model


def test_keys_and_values(model):
    expected_names = {"Input", "Link", "Output"}
    assert set(model.nodes.keys()) == expected_names
    nodes = {node for node in model.nodes}
    assert len(nodes) == 3
    assert set(model.nodes.values()) == nodes


def test_contains(model):
    assert "Input" in model.nodes
    assert "Output" in model.nodes
    output = model.nodes["Output"]
    assert isinstance(output, Node)
    assert output in model.nodes


def test_delete(model):
    del model.nodes["Output"]
    assert len(model.nodes) == 2
    assert {node.name for node in model.nodes} == {"Input", "Link"}


def test_delete_compound_node(simple_storage_model):
    """Removal of a compound node removes child nodes also"""
    assert len(list(simple_storage_model.nodes._nodes(hide_children=False))) == 5
    del simple_storage_model.nodes["Storage"]
    assert len(list(simple_storage_model.nodes._nodes(hide_children=False))) == 2
    assert {node.name for node in simple_storage_model.nodes} == {"Input", "Output"}
