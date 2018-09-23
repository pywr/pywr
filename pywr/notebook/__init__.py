import os
import json
import inspect
import numpy
from IPython.core.display import HTML, Javascript, display
from jinja2 import Template
from pywr.core import Node
from pywr.core import Model, Input, Output, Link, Storage, StorageInput, StorageOutput
from pywr.nodes import NodeMeta
from pywr._component import Component
import pywr.domains
from pywr.parameters._parameters import get_parameter_from_registry

from .figures import *

# load javascript template for d3 graph
folder = os.path.dirname(__file__)
with open(os.path.join(folder, "draw_graph.js"), "r") as f:
    draw_graph_template = Template(f.read())
with open(os.path.join(folder, "graph.css"), "r") as f:
    draw_graph_css = f.read()

def pywr_model_to_d3_json(model, attributes=False):
    """
    Convert a Pywr graph to a structure d3 can display

    Parameters
    ----------
    model : `pywr.core.Model`
    attributes: bool (default=False)
        If True, attribute data for each node is extract
    """
    nodes = []
    node_names = []
    for node in model.graph.nodes():
        if node.parent is None and node.virtual is False:
            nodes.append(node)
            node_names.append(node.name)

    edges = []
    for edge in model.graph.edges():
        node_source, node_target = edge

        # where a link is to/from a subnode, display a link to the parent instead
        if node_source.parent is not None:
            node_source = node_source.parent
        if node_target.parent is not None:
            node_target = node_target.parent

        if node_source is node_target:
            # link is between two subnodes
            continue

        index_source = node_names.index(node_source.name)
        index_target = node_names.index(node_target.name)
        edges.append({'source': index_source, 'target': index_target})

    json_nodes = []
    for n, node in enumerate(nodes):
        node_dict = {"name": node.name}
        classes = []
        cls = node.__class__
        classes.append(cls)
        while True:
            for base in cls.__bases__:
                if issubclass(base, Node) and base is not Node:
                    classes.append(base)
            if classes[-1] is cls:
                break
            else:
                cls = classes[-1]
        classes = classes[::-1]
        node_dict["clss"] = [cls.__name__.lower() for cls in classes]
        try:
            node["position"] = node.position["schematic"]
        except KeyError:
            pass

        if attributes:
            node_dict["attributes"] = get_node_attr(node)

        json_nodes.append(node_dict)

    graph = {
        "nodes": json_nodes,
        "links": edges}


    return graph

def get_node_attr(node):
    """
    Returns a dictionary that contains node attributes as strings

    Parameters
    ----------
    node : a pywr node object
    """
    attrs = inspect.getmembers(node, lambda a:not(inspect.isroutine(a)))
    attribute_data = []
    for att in attrs:

        attr_name, attr_val = att
        if attr_name.startswith("_"):
            continue
        attr_type = type(attr_val).__name__

        attrs_to_skip = ["component_attrs", "components", "color", "model",  "input", "output",
                         "inputs", "outputs", "sub_domain", "sub_output", "sublinks", "visible",
                         "fully_qualified_name", "allow_isolated"]
        if not attr_val or attr_name.lower() in attrs_to_skip:
            continue

        if isinstance(attr_val, Component):
            attr_val = attr_val.name
            if not attr_val:
                attr_val = attr_type
            else:
                attr_val = attr_val + " - " + attr_type

        if isinstance(attr_val, list):
            new_vals = []
            for val in attr_val:
                val_name = str(val)
                val_name = val_name.replace("[", "").replace("]", "")
                new_vals.append(val_name)
            attr_val = "".join(new_vals)
        else:
            attr_val = str(attr_val)

        attribute_data.append({"attribute": attr_name,  "value": attr_val})

    return attribute_data


def pywr_json_to_d3_json(model, attributes=False):
    """
    Converts a JSON file or a JSON-derived dict into structure that d3js can use.

    Parameters
    ----------
    model : dict or str
        str inputs should be a path to a json file containing the model.
    """

    if isinstance(model, str):
        with open(model) as d:
            model = json.load(d)

    nodes = [node["name"] for node in model["nodes"]]

    edges = []
    for edge in model["edges"]:
        sourceindex = nodes.index(edge[0])
        targetindex = nodes.index(edge[1])
        edges.append({'source': sourceindex, 'target': targetindex})

    nodes = []
    node_classes = create_node_class_trees()

    for node in model["nodes"]:
        json_node = {'name': node.pop("name"), 'clss': node_classes[node["type"].lower()]}
        try:
            json_node['position'] = node['position']['schematic']
        except KeyError:
            pass

        if attributes:
            json_node["attributes"] = []
            for name, val in node.items():

                if name == "type":
                    continue

                attr_val = val

                if isinstance(val, dict):
                    try:
                        attr_val = get_parameter_from_registry(val["type"]).__name__
                    except KeyError:
                        pass
                elif val in model["parameters"].keys():
                    param = model["parameters"][val]
                    attr_type = get_parameter_from_registry(param["type"]).__name__
                    attr_val = attr_val + " - " + attr_type

                attr_dict = {"attribute": name, "value": attr_val}
                json_node["attributes"].append(attr_dict)

        nodes.append(json_node)

    graph = {
        "nodes": nodes,
        "links": edges}

    return graph


def create_node_class_trees():
    # create class tree for each node type
    node_class_trees = {}
    for name, cls in NodeMeta.node_registry.items():
        classes = [cls]
        while True:
            for base in cls.__bases__:
                if issubclass(base, Node) and base is not Node:
                    classes.append(base)
            if classes[-1] is cls:
                break
            else:
                cls = classes[-1]
        clss = [cls.__name__.lower() for cls in classes[::-1]]
        node_class_trees[name] = clss
    return node_class_trees


def draw_graph(model, width=500, height=400, labels=False, attributes=False, css=None):
    """Display a Pywr model using D3 in Jupyter

    Parameters
    ----------
    model : pywr.core.Model or json-dict that describes a model
        The model to display
    width : int
        The width of the svg canvas to draw the graph on
    height : int
        The height of the svg canvas to draw the graph on
    labels : bool
        If True, each graph node is labelled with its name. If false, the node names are displayed
        during mouseover events
    attributes : bool
        If True, a table of node attributes is displayed during mouseover events
    css : string
        Stylesheet data to use instead of default
    """
    js = _draw_graph(model, width, height, labels, attributes, css)
    display(js)


def _draw_graph(model, width=500, height=400, labels=False, attributes=False, css=None):
    """Creates Javascript/D3 code for graph"""
    if isinstance(model, Model):
        graph = pywr_model_to_d3_json(model, attributes)
    else:
        graph = pywr_json_to_d3_json(model, attributes)

    if css is None:
        css = draw_graph_css

    js = Javascript(
        data=draw_graph_template.render(
            graph=graph,
            width=width,
            height=height,
            labels=labels,
            attributes=attributes,
            css=css.replace("\n","")
        ),
        lib="http://d3js.org/d3.v3.min.js",
    )
    return js
