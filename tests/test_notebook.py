import pytest
from numpy.testing import assert_array_equal
from pywr.notebook import pywr_model_to_d3_json, pywr_json_to_d3_json
from pywr.model import Model
import os

from helpers import load_model


def get_node(nodes, name):
    for node in nodes:
        if node["name"] == name:
            return node


def get_node_attribute(node, attr_name):
    for attr in node["attributes"]:
        if attr["attribute"] == attr_name:
            return attr


@pytest.mark.parametrize("from_json", [True, False])
def test_from_json(from_json):
    json_path = os.path.join(os.path.dirname(__file__), "models", "demand_saving2_with_variables.json")

    if from_json:
        json_dict = pywr_json_to_d3_json(json_path, attributes=True)
    else:
        model = load_model("demand_saving2_with_variables.json")
        json_dict = pywr_model_to_d3_json(model, attributes=True)

    assert "nodes" in json_dict.keys()
    assert "links" in json_dict.keys()

    node_names = ["Inflow", "Reservoir", "Demand", "Spill"]
    for node in json_dict["nodes"]:
        assert node["name"] in node_names

        if node["name"] == "Reservoir":
            assert_array_equal(node["position"], [1, 1])

    demand = get_node(json_dict["nodes"], "Demand")
    demand_max_flow = get_node_attribute(demand, "max_flow")

    assert demand_max_flow["value"] == "demand_max_flow - AggregatedParameter"


def test_d3_data():
    """ Check that the graph data returned by the pywr_json_to_d3_json and pywr_model_to_d3_json functions contains the
    same nodes and number on links. The data won't match exactly due to differences in node ordering.
    """
    json_path = os.path.join(os.path.dirname(__file__), "models", "demand_saving2_with_variables.json")
    model = load_model("demand_saving2_with_variables.json")

    d3_data_from_json = pywr_json_to_d3_json(json_path)
    d3_data_from_model = pywr_model_to_d3_json(model)

    json_nodes = {n["name"]: n for n in d3_data_from_json["nodes"]}
    model_nodes = {n["name"]: n for n in d3_data_from_model["nodes"]}

    assert json_nodes == model_nodes
    assert len(d3_data_from_json["links"]) == len(d3_data_from_model["links"])
