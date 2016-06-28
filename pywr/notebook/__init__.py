import os
from IPython.core.display import Javascript, display
from jinja2 import Template

from pywr.core import Model, Input, Output, Link, Storage, StorageInput, StorageOutput

from .figures import *

# load javascript template for d3 graph
folder = os.path.dirname(__file__)
with open(os.path.join(folder, "draw_graph.js"), "r") as f:
    draw_graph_template = Template(f.read())

def nx_to_json(model):
    """Convert a Pywr graph to a structure d3 can display"""
    nodes = []
    for node in model.graph.nodes():
        if node.parent is None:
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

    graph = {
        "nodes": [
            {
                "name": name,
                "color": model.node[name].color,
            } for n, name in enumerate(nodes)
        ],
        "links": edges}

    return graph

def draw_graph(model):
    """Display a Pywr model using D3 in Jupyter

    Parameters
    ----------
    model : pywr.core.Model
        The model to display
    """
    graph = nx_to_json(model)
    data = draw_graph_template.render(graph=graph)
    js = Javascript(data=data, lib="http://d3js.org/d3.v3.min.js")
    display(js)
