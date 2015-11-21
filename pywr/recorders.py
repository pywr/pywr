from pywr._recorders import Recorder, NodeRecorder, StorageRecorder, NumpyArrayNodeRecorder


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