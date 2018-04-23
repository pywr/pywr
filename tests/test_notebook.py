import os
from pywr.notebook import pywr_model_to_d3_json, pywr_json_to_d3_json
from pywr.model import Model

def get_node(nodes, name):
    for node in nodes:
        if node["name"] == name:
            return node

def get_node_attribute(node, attr_name):
    for attr in node["attributes"]:
        if attr["attribute"] == attr_name:
            return attr 

def test_from_model():

    json_path = os.path.join(os.path.dirname(__file__), "models", "river1.json")
    model = Model.load(json_path)
    json_dict = pywr_model_to_d3_json(model, attributes=True)

    assert "nodes" in json_dict.keys()
    assert "links" in json_dict.keys()

    node_names = ["catchment1", "river1", "abs1", "link1", "term1", "demand1"]
    for node in json_dict["nodes"]:
        assert node["name"] in node_names

    catchment = get_node(json_dict["nodes"], "catchment1")
    catchment_max_flow = get_node_attribute(catchment, "max_flow")
    assert catchment_max_flow["value"] == "5.0"
    
def test_from_json():

    json_path = os.path.join(os.path.dirname(__file__), "models", "demand_saving2_with_variables.json")
    json_dict = pywr_json_to_d3_json(json_path, attributes=True)
    
    assert "nodes" in json_dict.keys()
    assert "links" in json_dict.keys()
        
    node_names = ["Inflow", "Reservoir", "Demand", "Spill"]
    for node in json_dict["nodes"]:
        assert node["name"] in node_names

    demand = get_node(json_dict["nodes"], "Demand")
    demand_max_flow = get_node_attribute(demand, "max_flow")
    assert demand_max_flow["value"] == "demand_max_flow - aggregated parameter"