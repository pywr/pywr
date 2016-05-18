import sys
import numpy as np
from pywr._recorders import (Recorder, NodeRecorder, StorageRecorder,
    NumpyArrayNodeRecorder, NumpyArrayStorageRecorder, NumpyArrayLevelRecorder)


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
        super(CSVRecorder, self).__init__(model)
        self.csvfile = csvfile
        self.scenario_index = 0
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


class TablesRecorder(Recorder):
    """
    A recorder that saves to PyTables CArray

    This Recorder creates a CArray for every node passed to the constructor.
    Each CArray stores the data for all scenarios on the specific node. This
    is useful for analysis of Node statistics across multiple scenarios.
    """
    def __init__(self, model, parent, nodes=None):
        """

        :param model: The model to record nodes from.
        :param parent: The tables parent node to attach the CArray objects to.
        :param nodes: An iterable of nodes to save data. It defaults
        to None which is all nodes in the model
        """
        super(TablesRecorder, self).__init__(model)

        self.parent = parent
        self.nodes = nodes

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
        import tables
        shape = len(self.model.timestepper), len(self.model.scenarios.combinations)
        atom = tables.Float64Atom()
        # Create a CArray for each node
        self._arrays = {}

        # Default to all nodes if None given.
        if self.nodes is None:
            nodes = self.model.node.values()
        else:
            nodes = self.nodes

        for node in nodes:
            self._arrays[node] = tables.CArray(self.parent, node.name, atom, shape)

    def save(self):
        """
        Save data to the tables
        """
        from pywr._core import Node, Storage
        idx = self.model.timestepper.current.index

        for node, ca in self._arrays.items():
            if isinstance(node, Node):
                ca[idx, :] = node.flow
            elif isinstance(node, Storage):
                ca[idx, :] = node.volume
            else:
                raise ValueError("Unrecognised Node type '{}' for TablesRecorder".format(type(node)))


class BaseConstantNodeRecorder(NodeRecorder):
    """
    Base class for NodeRecorder classes with a single value for each scenario combination
    """
    def __init__(self, *args, **kwargs):
        self.agg_func = kwargs.pop('agg_func', np.mean)
        super(BaseConstantNodeRecorder, self).__init__(*args, **kwargs)
        self._values = None

    def setup(self):
        self._values = np.zeros(len(self.model.scenarios.combinations))

    def reset(self):
        self._values[...] = 0.0

    def save(self):
        raise NotImplementedError()

    def value(self):
        return self.agg_func(self._values)


class TotalDeficitNodeRecorder(BaseConstantNodeRecorder):
    """
    Recorder to total the difference between modelled flow and max_flow for a Node
    """
    def save(self):
        ts = self.model.timestepper.current
        node = self.node
        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(ts, scenario_index)
            self._values[scenario_index.global_id] += max_flow - node.flow[scenario_index.global_id]


class TotalFlowRecorder(BaseConstantNodeRecorder):
    """
    Recorder to total the flow for a Node.

    A factor can be provided to scale the total flow (e.g. for calculating operational costs).
    """
    def __init__(self, *args, **kwargs):
        self.factor = kwargs.pop('factor', 1.0)
        super(TotalFlowRecorder, self).__init__(*args, **kwargs)

    def save(self):
        self._values += self.node.flow*self.factor


class AggregatedRecorder(Recorder):
    def __init__(self, model, recorders, **kwargs):
        self.agg_func = kwargs.pop('agg_func', np.mean)
        super(AggregatedRecorder, self).__init__(model, **kwargs)
        self.recorders = recorders

    def value(self):
        return self.agg_func([r.value() for r in self.recorders])
