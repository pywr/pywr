import os
import json
import inspect
import warnings
from IPython.core.display import Javascript, display
from jinja2 import Template
from pywr.core import Node
from pywr.core import Model
from pywr.nodes import NodeMeta
from pywr._component import Component
from pywr.parameters._parameters import get_parameter_from_registry

from .figures import *

# load javascript template for d3 graph
folder = os.path.dirname(__file__)
with open(os.path.join(folder, "draw_graph.js"), "r") as f:
    draw_graph_template = Template(f.read())
with open(os.path.join(folder, "save_graph.js"), "r") as f:
    save_graph_template = Template(f.read())
with open(os.path.join(folder, "graph.css"), "r") as f:
    draw_graph_css = f.read()
with open(os.path.join(folder, "template.html"), "r") as f:
    html_template = Template(f.read())


class PywrSchematic:
    def __init__(
        self, model, width=500, height=400, labels=False, attributes=False, css=None
    ):
        """This object contains methods that allow the graph of a pywr model network to be displayed in a jupyter
        notebook or saved to an html file.

        It also contains a method to save the node positions of a notebook graph back to a pywr model json file. Note
        that this method currently does not work if the object has be instantiated using a model object.

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
        if isinstance(model, Model):
            self.graph = pywr_model_to_d3_json(model, attributes)
            # TODO update when schema branch is merged
            self.json = None
        else:
            self.graph = pywr_json_to_d3_json(model, attributes)
            if isinstance(model, str):
                with open(model) as d:
                    self.json = json.load(d)
            else:
                self.json = model

        self.height = height
        self.width = width
        self.labels = labels
        self.attributes = attributes

        if css is None:
            self.css = draw_graph_css
        else:
            self.css = css

    def draw_graph(self):
        """Draw pywr schematic graph in a jupyter notebook"""
        js = draw_graph_template.render(
            graph=self.graph,
            width=self.width,
            height=self.height,
            element="element",
            labels=self.labels,
            attributes=self.attributes,
            css=self.css.replace("\n", ""),
        )
        display(Javascript(data=js))

    def save_graph(self, filename, save_unfixed=False, filetype="json"):
        """Save a copy of the model JSON with update schematic positions.

        When run in a jupyter notebook this will trigger a download.

        Parameters
        ----------
        filename: str
            The name of the file to save the output data to.
        save_unfixed: bool
            If True, then all node position are saved to output file. If False, only nodes who have had their position
            fixed in the d3 graph have their positions saved.
        filetype: str
            Should be either 'json' to save the model data with updated node positions to a JSON file or 'csv' to save
            node positions to a csv file.
        """

        if filetype not in ["json", "csv"]:
            warnings.warn(
                f"Output filetype '{filetype}' not recognised. Please use either 'json' or 'csv'</p>",
                stacklevel=2,
            )

        if self.json is None and filetype == "json":
            warnings.warn(
                "Node positions cannot be saved to JSON if PywrSchematic object has been instantiated using "
                "a pywr model object. Please use a JSON file path or model dict instead.",
                stacklevel=2,
            )
        else:
            display(
                Javascript(
                    save_graph_template.render(
                        model_data=json.dumps(self.json),
                        height=self.height,
                        width=self.width,
                        save_unfixed=json.dumps(save_unfixed),
                        filename=json.dumps(filename),
                        filetype=json.dumps(filetype),
                    )
                )
            )

    def to_html(self, filename="model.html", title="Model Schematic"):
        """Save an HTML file of schematic

        Parameters
        ----------
        filename: str
            The name of the html file
        title: str
            The schematic title
        """

        # TODO add option to get node position from graph that has already been drawn in a notebook

        js = draw_graph_template.render(
            graph=self.graph,
            width=self.width,
            height=self.height,
            element=json.dumps(".schematic"),
            labels=self.labels,
            attributes=self.attributes,
            css="",
        )

        html = html_template.render(title=title, css=self.css, d3_script=js)

        with open(filename, "w") as f:
            f.write(html)


def draw_graph(model, width=500, height=400, labels=False, attributes=False, css=None):
    """Display a Pywr model using D3 in Jupyter

    Functionality for creating the d3 graph is now in the PywrSchematic object
    """
    schematic = PywrSchematic(
        model, width=width, height=height, labels=labels, attributes=attributes, css=css
    )
    schematic.draw_graph()


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
        edges.append({"source": index_source, "target": index_target})

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
            node_dict["position"] = node.position["schematic"]
        except KeyError:
            pass

        if attributes:
            node_dict["attributes"] = get_node_attr(node)

        json_nodes.append(node_dict)

    graph = {"nodes": json_nodes, "links": edges}

    return graph


def get_node_attr(node):
    """
    Returns a dictionary that contains node attributes as strings

    Parameters
    ----------
    node : a pywr node object
    """
    attrs = inspect.getmembers(node, lambda a: not (inspect.isroutine(a)))
    attribute_data = []
    for att in attrs:

        attr_name, attr_val = att
        if attr_name.startswith("_"):
            continue
        attr_type = type(attr_val).__name__

        attrs_to_skip = [
            "component_attrs",
            "components",
            "color",
            "model",
            "input",
            "output",
            "inputs",
            "outputs",
            "sub_domain",
            "sub_output",
            "sublinks",
            "visible",
            "fully_qualified_name",
            "allow_isolated",
        ]
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

        attribute_data.append({"attribute": attr_name, "value": attr_val})

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

    nodes = []
    node_classes = create_node_class_trees()
    for node in model["nodes"]:

        if node["type"].lower() in [
            "annualvirtualstorage",
            "virtualstorage",
            "aggregatednode",
            "aggregatedstorage",
            "seasonalvirtualstorage",
        ]:
            # Do not add virtual nodes to the graph
            continue

        json_node = {"name": node["name"], "clss": node_classes[node["type"].lower()]}
        try:
            json_node["position"] = node["position"]["schematic"]
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
                else:
                    attr_val = str(attr_val)

                attr_dict = {"attribute": name, "value": attr_val}
                json_node["attributes"].append(attr_dict)

        nodes.append(json_node)

    nodes_names = [node["name"] for node in nodes]

    edges = []
    for edge in model["edges"]:
        sourceindex = nodes_names.index(edge[0])
        targetindex = nodes_names.index(edge[1])
        edges.append({"source": sourceindex, "target": targetindex})

    graph = {"nodes": nodes, "links": edges}

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
