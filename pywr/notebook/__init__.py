import os
import json
from IPython.core.display import HTML, Javascript, display
from jinja2 import Template
from pywr.core import Node
from pywr.core import Model, Input, Output, Link, Storage, StorageInput, StorageOutput
from pywr.nodes import NodeMeta
import pywr.domains

from .figures import *

# load javascript template for d3 graph
folder = os.path.dirname(__file__)
with open(os.path.join(folder, "draw_graph.js"), "r") as f:
    draw_graph_template = Template(f.read())
with open(os.path.join(folder, "graph.css"), "r") as f:
    draw_graph_css = f.read()

def pywr_model_to_d3_json(model):
    """Convert a Pywr graph to a structure d3 can display"""
    nodes = []
    for node in model.graph.nodes():
        if node.parent is None and node.virtual is False:
            nodes.append(node.name)

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

        index_source = nodes.index(node_source.name)
        index_target = nodes.index(node_target.name)
        edges.append({'source': index_source, 'target': index_target})

    json_nodes = []
    for n, name in enumerate(nodes):
        node = {"name": name}
        classes = []
        node_obj = model.nodes[name]
        cls = node_obj.__class__
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
        node["clss"] = [cls.__name__.lower() for cls in classes]
        try:
            node["position"] = node_obj.position["schematic"]
        except KeyError:
            pass
        json_nodes.append(node)

    graph = {
        "nodes": json_nodes,
        "links": edges}

    return graph


def pywr_json_to_d3_json(model):
    """
    converts a json file or a json-derived dict into structure that D3 can use
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
        json_node = {'name': node["name"], 'clss': node_classes[node["type"].lower()]}
        try:
            json_node['position'] = node['position']['schematic']
        except KeyError:
            pass

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


def draw_graph(model, width=500, height=400, css=None):
    """Display a Pywr model using D3 in Jupyter

    Parameters
    ----------
    model : pywr.core.Model or json-dict that describes a model
        The model to display
    css : string
        Stylesheet data to use instead of default
    """
    js = _draw_graph(model, width, height, css)
    display(js)


def _draw_graph(model, width=500, height=400, css=None):
    """Creates Javascript/D3 code for graph"""
    if isinstance(model, Model):
        graph = pywr_model_to_d3_json(model)
    else:
        graph = pywr_json_to_d3_json(model)

    if css is None:
        css = draw_graph_css

    js = Javascript(
        data=draw_graph_template.render(
            graph=graph,
            width=width,
            height=height,
            css=css.replace("\n","")
        ),
        lib="http://d3js.org/d3.v3.min.js",
    )
    return js 
