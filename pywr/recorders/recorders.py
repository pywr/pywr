import sys
import numpy as np
from functools import wraps
from ._recorders import *
from ._thresholds import *
from .events import EventRecorder, Event
from .calibration import *
from past.builtins import basestring
from pywr.h5tools import H5Store

def assert_rec(model, parameter, name=None):
    """Decorator for creating AssertionRecorder objects

    Example
    -------
    @assert_rec(model, parameter)
    def expected_func(timestep, scenario_index):
        return timestep.dayofyear * 2.0
    """
    def assert_rec_(f):
        rec = AssertionRecorder(model, parameter, expected_func=f, name=name)
        return f
    return assert_rec_

class AssertionRecorder(Recorder):
    """A recorder that asserts the value of a parameter for testing purposes"""
    def __init__(self, model, parameter, expected_data=None, expected_func=None, **kwargs):
        """
        Parameters
        ----------
        model : pywr.model.Model
        parameter : pywr.parameters.Parameter
        expected_data : np.ndarray[timestep, scenario] (optional)
        expected_func : function

        See also
        --------
        pywr.recorders.assert_rec
        """
        super(AssertionRecorder, self).__init__(model, **kwargs)
        self.parameter = parameter
        self.expected_data = expected_data
        self.expected_func = expected_func

    def setup(self):
        super(AssertionRecorder, self).setup()
        self.count = 0

    def after(self):
        timestep = self.model.timestep
        self.count += 1
        for scenario_index in self.model.scenarios.combinations:
            if self.expected_func:
                expected_value = self.expected_func(timestep, scenario_index)
            elif self.expected_data is not None:
                expected_value = self.expected_data[timestep.index, scenario_index.global_id]
            value = self.parameter.get_value(scenario_index)
            np.testing.assert_allclose(value, expected_value)

    def finish(self):
        super(AssertionRecorder, self).finish()
        if sys.exc_info():
            # exception was raised before we had a chance! (e.g. ModelStructureError)
            pass
        elif self.count == 0:
            # this still requires model.run() to have been called...
            raise RuntimeError("AssertionRecorder was never called!")

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

    def after(self):
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
CSVRecorder.register()


class TablesRecorder(Recorder):
    """
    A recorder that saves to PyTables CArray

    This Recorder creates a CArray for every node passed to the constructor.
    Each CArray stores the data for all scenarios on the specific node. This
    is useful for analysis of Node statistics across multiple scenarios.
    """
    def __init__(self, model, h5file, nodes=None, parameters=None, where='/', time='/time', scenarios='/scenarios', **kwargs):
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
        time : string
            Default full node path to save a time tables.Table. If None no table is created.
        scenarios : string
            Default full node path to save a scenarios tables.Table. If None no table is created.
        filter_kwds : dict
            Filter keywords to pass to tables.open_file when opening a file.
        mode : string
            Model argument to pass to tables.open_file. Defaults to 'w'
        metadata : dict
            Dict of user defined attributes to save on the root node (`root._v_attrs`)
        create_directories : bool
            If a file path is given and create_directories is True then attempt to make the intermediate
            directories. This uses os.makedirs() underneath.
        """
        self.filter_kwds = kwargs.pop('filter_kwds', {})
        self.mode = kwargs.pop('mode', 'w')
        self.metadata = kwargs.pop('metadata', {})
        self.create_directories = kwargs.pop('create_directories', False)

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
        self.time = time
        self.scenarios = scenarios

        self._arrays = None
        self._time_table = None

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

        # The first dimension is the number of timesteps.
        # The following dimensions are sized per scenario
        scenario_shape = list(self.model.scenarios.shape)
        shape = [len(self.model.timestepper)] + scenario_shape

        self.h5store = H5Store(self.h5file, self.filter_kwds, self.mode, title=self.title, metadata=self.metadata,
                               create_directories=self.create_directories)

        # Create a CArray for each node
        self._arrays = {}

        # Default to all nodes if None given.
        if self.nodes is None:
            nodes = [((self.where + "/" + n.name).replace("//", "/"), n) for n in self.model.nodes.values()]
        else:
            nodes = []
            for n in self.nodes:

                try:
                    where, node = n
                except (TypeError, ValueError):
                    node = n
                    where = self.where + "/" + node

                # Accept a str, and lookup node by name instead.
                if isinstance(node, basestring):
                    node = self.model.nodes[node]
                # Otherwise assume it is a node object anyway

                where = where.replace("//", "/")
                nodes.append((where, node))

        if self.parameters is not None:
            for p in self.parameters:

                try:
                    where, param = p
                except (TypeError, ValueError):
                    param = p
                    where = None

                if isinstance(param, basestring):
                    param = self.model.parameters[param]

                if param.name is None:
                    raise ValueError('Can only record named Parameter objects.')

                if where is None:
                    where = self.where + "/" + param.name

                where = where.replace("//", "/")
                nodes.append((where, param))

        self._nodes = nodes

        for where, node in self._nodes:
            if isinstance(node, IndexParameter):
                atom = tables.Int32Atom()
            else:
                atom = tables.Float64Atom()
            group_name, node_name = where.rsplit("/", 1)
            if "group_name" == "/":
                group_name = self.h5store.file.root
            self.h5store.file.create_carray(group_name, node_name, atom, shape, createparents=True)

        # Create time table
        if self.time is not None:
            group_name, node_name = self.time.rsplit('/', 1)
            if "group_name" == "/":
                group_name = self.h5store.file.root
            description = {c: tables.Int64Col() for c in ('year', 'month', 'day', 'index')}
            self.h5store.file.create_table(group_name, node_name, description=description, createparents=True)

        # Create scenario tables
        if self.scenarios is not None:
            group_name, node_name = self.scenarios.rsplit('/', 1)
            if "group_name" == "/":
                group_name = self.h5store.file.root
            description = {
                # TODO make string length configurable
                'name': tables.StringCol(1024),
                'size': tables.Int64Col()
            }
            tbl = self.h5store.file.create_table(group_name, node_name, description=description, createparents=True)
            # Now add the scenarios
            entry = tbl.row
            for scenario in self.model.scenarios.scenarios:
                entry['name'] = scenario.name
                entry['size'] = scenario.size
                entry.append()

        self.h5store = None

    def reset(self):
        mode = "r+"  # always need to append, as file already created in setup
        self.h5store = H5Store(self.h5file, self.filter_kwds, mode)
        self._arrays = {}
        for where, node in self._nodes:
            self._arrays[node] = self.h5store.file.get_node(where)

        self._time_table = None
        if self.time is not None:
            self._time_table = self.h5store.file.get_node(self.time)

    def after(self):
        """
        Save data to the tables
        """
        from pywr._core import AbstractNode, AbstractStorage
        from pywr.parameters import Parameter, IndexParameter
        scenario_shape = list(self.model.scenarios.shape)
        ts = self.model.timestepper.current
        idx = ts.index
        dt = ts.datetime

        if self._time_table is not None:
            entry = self._time_table.row
            entry['year'] = dt.year
            entry['month'] = dt.month
            entry['day'] = dt.day
            entry['index'] = idx
            entry.append()

        for node, ca in self._arrays.items():
            if isinstance(node, AbstractStorage):
                ca[idx, :] = np.reshape(node.volume, scenario_shape)
            elif isinstance(node, AbstractNode):
                ca[idx, :] = np.reshape(node.flow, scenario_shape)
            elif isinstance(node, IndexParameter):
                a = node.get_all_indices()
                ca[idx, :] = np.reshape(a, scenario_shape)
            elif isinstance(node, Parameter):
                a = node.get_all_values()
                ca[idx, :] = np.reshape(a, scenario_shape)
            else:
                raise ValueError("Unrecognised Node type '{}' for TablesRecorder".format(type(node)))

    def finish(self):
        if self._time_table is not None:
            self._time_table.flush()
        self.h5store = None
        self._arrays = {}

TablesRecorder.register()
