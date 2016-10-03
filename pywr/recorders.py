import sys
from pywr._recorders import *
from past.builtins import basestring
from .h5tools import H5Store


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
            self.node_names = sorted(self.model.nodes.keys())
        else:
            self.node_names = sorted(n.name for n in self.nodes)

    def reset(self):
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
            node = self.model.nodes[node_name]
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
    def __init__(self, model, h5file, nodes=None, parameters=None, where='/', **kwargs):
        """

        Parameters
        ----------
        model : pywr.core.Model
            The model to record nodes from.
        h5file : tables.File or filename
            The tables file handle or filename to attach the CArray objects to. If a
            filename is given the object will open and close the file handles.
        nodes : iterable or None
            Nodes to save in the tables database. Can be an iterable of Node objects or
            node names. It can also be a iterable of tuples with a node specific where
            keyword as the first item and a Node object or name as the second item. If
            an iterable of tuples is provided then the node specific where keyword is
            used in preference to the where keyword (see below).
        parameters : iterable or None
            Parameters to save. Similar to the nodes keyword, except refers to Parameter
            objects or names thereof.
        where : string
            Default path to create the CArrays inside the database.
        filter_kwds : dict
            Filter keywords to pass to tables.open_file when opening a file.
        mode : string
            Model argument to pass to tables.open_file. Defaults to 'w'
        """
        self.filter_kwds = kwargs.pop('filter_kwds', {})
        self.mode = kwargs.pop('mode', 'w')

        title = kwargs.pop('title', None)
        if title is None:
            try:
                title = model.metadata['title']
            except KeyError:
                title = ''
        self.title = title
        super(TablesRecorder, self).__init__(model, **kwargs)

        self.h5file = h5file
        self.h5store = None
        self._arrays = {}
        self.nodes = nodes
        self.parameters = parameters
        self.where = where

        self._arrays = None

    @classmethod
    def load(cls, model, data):
        import os
        url = data.pop("url")
        if not os.path.isabs(url) and model.path is not None:
            url = os.path.join(model.path, url)
        return cls(model, url, **data)

    def setup(self):
        """
        Setup the tables
        """
        from pywr.parameters import IndexParameter
        import tables
        shape = len(self.model.timestepper), len(self.model.scenarios.combinations)

        self.h5store = H5Store(self.h5file, self.filter_kwds, self.mode, title=self.title)

        # Create a CArray for each node
        self._arrays = {}

        # Default to all nodes if None given.
        if self.nodes is None:
            nodes = [(self.where, n) for n in self.model.nodes.values()]
        else:
            nodes = []
            for n in self.nodes:

                try:
                    where, node = n
                except (TypeError, ValueError):
                    node = n
                    where = self.where

                # Accept a str, and lookup node by name instead.
                if isinstance(node, basestring):
                    node = self.model.nodes[node]
                # Otherwise assume it is a node object anyway

                nodes.append((where, node))

        if self.parameters is not None:
            for p in self.parameters:

                try:
                    where, param = p
                except (TypeError, ValueError):
                    param = p
                    where = self.where

                if isinstance(param, basestring):
                    param = self.model.parameters[param]
                else:
                    param = p

                if param.name is None:
                    raise ValueError('Can only record named Parameter objects.')
                nodes.append((where, param))

        self._nodes = nodes

        for where, node in self._nodes:
            if isinstance(node, IndexParameter):
                atom = tables.Int32Atom()
            else:
                atom = tables.Float64Atom()
            self.h5store.file.create_carray(where, node.name, atom, shape, createparents=True)

        self.h5store = None

    def reset(self):
        mode = "r+"  # always need to append, as file already created in setup
        self.h5store = H5Store(self.h5file, self.filter_kwds, mode)
        self._arrays = {}
        for where, node in self._nodes:
            self._arrays[node] = self.h5store.file.get_node(where, node.name)

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

    def finish(self):
        self.h5store = None
        self._arrays = {}

recorder_registry.add(TablesRecorder)
