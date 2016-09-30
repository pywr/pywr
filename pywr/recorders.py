import sys
from pywr._recorders import *


class CSVRecorder(Recorder):
    """
    A Recorder that saves Node values to a CSV file.

    This class uses the csv package from the Python standard library
    """
    def __init__(self, model, csvfile, scenario_index=0, nodes=None, **kwargs):
        """

        :param model: The model to record nodes from.
        :param csvfile: The path to the CSV file.
        :param scenario_index: The scenario index of the model to save.
        :param nodes: An iterable of nodes to save data. It defaults
        to None which is all nodes in the model
        :param kwargs: Additional keyword arguments to pass to the csv.writer
        object

        """
        super(CSVRecorder, self).__init__(model, **kwargs)
        self.csvfile = csvfile
        self.scenario_index = scenario_index
        self.nodes = nodes
        self.csv_kwargs = kwargs.pop('csv_kwargs', {})
        self.node_names = None

        self._fh = None
        self._writer = None

    def setup(self):
        """
        Setup the CSV file recorder.
        """

        if self.nodes is None:
            self.node_names = sorted(self.model.node.keys())
        else:
            self.node_names = sorted(n.name for n in self.nodes)

        import csv
        if sys.version_info.major >= 3:
            self._fh = open(self.csvfile, 'w', newline='')
        else:
            self._fh = open(self.csvfile, 'w')
        self._writer = csv.writer(self._fh, **self.csv_kwargs)
        # Write header data
        self._writer.writerow(['Datetime']+self.node_names)

    def save(self):
        """
        Write the node values to the CSV file
        """
        from pywr._core import Node, Storage

        values = [self.model.timestepper.current.datetime.isoformat()]
        for node_name in self.node_names:
            node = self.model.node[node_name]
            if isinstance(node, Node):
                values.append(node.flow[self.scenario_index])
            elif isinstance(node, Storage):
                values.append(node.volume[self.scenario_index])
            else:
                raise ValueError("Unrecognised Node type '{}' for CSV writer".format(type(node)))

        self._writer.writerow(values)

    def finish(self):
        self._fh.close()
recorder_registry.add(CSVRecorder)


class TablesRecorder(Recorder):
    """
    A recorder that saves to PyTables CArray

    This Recorder creates a CArray for every node passed to the constructor.
    Each CArray stores the data for all scenarios on the specific node. This
    is useful for analysis of Node statistics across multiple scenarios.
    """
    def __init__(self, model, parent, nodes=None, parameters=None, **kwargs):
        """

        :param model: The model to record nodes from.
        :param parent: The tables parent node to attach the CArray objects to.
        :param nodes: An iterable of nodes to save data. It defaults
        to None which is all nodes in the model
        """
        super(TablesRecorder, self).__init__(model, **kwargs)

        self.parent = parent
        self.nodes = nodes
        self.parameters = parameters

        self._arrays = None

    def __del__(self):
        if self._arrays is not None:
            for arr in self._arrays.values():
                arr.close()
        del(self._arrays)


    def setup(self):
        """
        Setup the tables
        """
        from pywr.parameters import IndexParameter
        import tables
        shape = len(self.model.timestepper), len(self.model.scenarios.combinations)
        # Create a CArray for each node
        self._arrays = {}

        # Default to all nodes if None given.
        if self.nodes is None:
            nodes = self.model.node.values()
        else:
            nodes = self.nodes

        nodes = list(nodes)

        if self.parameters is not None:
            for p in self.parameters:
                if p.name is None:
                    raise ValueError('Can only record named Parameter objects.')
                nodes.append(p)

        for node in nodes:
            if isinstance(node, IndexParameter):
                atom = tables.Int32Atom()
            else:
                atom = tables.Float64Atom()
            self._arrays[node] = tables.CArray(self.parent, node.name, atom, shape)

    def save(self):
        """
        Save data to the tables
        """
        from pywr._core import AbstractNode, AbstractStorage
        from pywr.parameters import BaseParameter, IndexParameter
        ts = self.model.timestepper.current
        idx = ts.index

        for node, ca in self._arrays.items():
            if isinstance(node, AbstractStorage):
                ca[idx, :] = node.volume
            elif isinstance(node, AbstractNode):
                ca[idx, :] = node.flow
            elif isinstance(node, IndexParameter):
                for si in self.model.scenarios.combinations:
                    ca[idx, si.global_id] = node.index(ts, si)
            elif isinstance(node, BaseParameter):
                for si in self.model.scenarios.combinations:
                    ca[idx, si.global_id] = node.value(ts, si)
            else:
                raise ValueError("Unrecognised Node type '{}' for TablesRecorder".format(type(node)))
recorder_registry.add(TablesRecorder)

