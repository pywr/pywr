import os
from IPython.core.display import HTML, Javascript, display
from jinja2 import Template
from pywr.core import Node

from pywr.core import Model, Input, Output, Link, Storage, StorageInput, StorageOutput

from .figures import *

# load javascript template for d3 graph
folder = os.path.dirname(__file__)
with open(os.path.join(folder, "draw_graph.js"), "r") as f:
    draw_graph_template = Template(f.read())
with open(os.path.join(folder, "graph.css"), "r") as f:
    draw_graph_css = f.read()

def nx_to_json(model):
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

def draw_graph(model, width=500, height=400, css=None):
    """Display a Pywr model using D3 in Jupyter

    Parameters
    ----------
    model : pywr.core.Model
        The model to display
    css : string
        Stylesheet data to use instead of default
    """
    graph = nx_to_json(model)

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
    display(js)
