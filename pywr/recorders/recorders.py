import sys
import pandas
import numpy as np
from functools import wraps
from pywr._core import AbstractNode, AbstractStorage
from ._recorders import *
from ._thresholds import *
from ._hydropower import *
from .events import *
from .calibration import *
from .kde import *
from pywr.h5tools import H5Store
from ..parameter_property import parameter_property
import warnings


class ParameterNameWarning(UserWarning):
    pass


def assert_rec(model, parameter, name=None, get_index=False):
    """Decorator for creating AssertionRecorder objects

    Example
    -------
    @assert_rec(model, parameter)
    def expected_func(timestep, scenario_index):
        return timestep.dayofyear * 2.0
    """

    def assert_rec_(f):
        rec = AssertionRecorder(
            model, parameter, expected_func=f, name=name, get_index=get_index
        )
        return f

    return assert_rec_


class AssertionRecorder(Recorder):
    """A recorder that asserts the value of a parameter for testing purposes"""

    def __init__(
        self,
        model,
        parameter,
        expected_data=None,
        expected_func=None,
        get_index=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : pywr.model.Model
        parameter : pywr.parameters.Parameter
        expected_data : np.ndarray[timestep, scenario] (optional)
        expected_func : function
        get_index : bool

        See also
        --------
        pywr.recorders.assert_rec
        """
        super(AssertionRecorder, self).__init__(model, **kwargs)
        self._parameter = None
        self.parameter = parameter
        self.expected_data = expected_data
        self.expected_func = expected_func
        self.get_index = get_index

    parameter = parameter_property("_parameter")

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
                expected_value = self.expected_data[
                    timestep.index, scenario_index.global_id
                ]
            if self.get_index:
                value = self._parameter.get_index(scenario_index)
            else:
                value = self._parameter.get_value(scenario_index)
            try:
                np.testing.assert_allclose(value, expected_value)
            except AssertionError:
                raise AssertionError(
                    'Expected {}, got {} from "{}" [timestep={}, scenario={}]'.format(
                        expected_value,
                        value,
                        self._parameter.name,
                        timestep.index,
                        scenario_index.global_id,
                    )
                )

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

    Parameters
    ----------

    model : `pywr.model.Model`
        The model to record nodes from.
    csvfile : str
        The path to the CSV file.
    scenario_index : int
        The scenario index of the model to save.
    nodes : iterable (default=None)
        An iterable of nodes to save data. It defaults to None which is all nodes in the model
    kwargs : Additional keyword arguments to pass to the `csv.writer` object

    """

    def __init__(
        self,
        model,
        csvfile,
        scenario_index=0,
        nodes=None,
        complib=None,
        complevel=9,
        **kwargs,
    ):
        super(CSVRecorder, self).__init__(model, **kwargs)
        self.csvfile = csvfile
        self.scenario_index = scenario_index
        self.nodes = nodes
        self.csv_kwargs = kwargs.pop("csv_kwargs", {})
        self._node_names = None
        self._fh = None
        self._writer = None
        self.complib = complib
        self.complevel = complevel

    @classmethod
    def load(cls, model, data):
        import os

        url = data.pop("url")
        if not os.path.isabs(url) and model.path is not None:
            url = os.path.join(model.path, url)
        return cls(model, url, **data)

    def setup(self):
        """
        Setup the CSV file recorder.
        """

        if self.nodes is None:
            self._node_names = sorted(self.model.nodes.keys())
        else:
            node_names = []
            for node_ in self.nodes:
                # test if the node name is provided
                if isinstance(node_, str):
                    # lookup node by name
                    node_names.append(node_)
                else:
                    node_names.append((node_.name))
            self._node_names = node_names

    def reset(self):
        import csv

        kwargs = {"newline": "", "encoding": "utf-8"}
        mode = "wt"

        if self.complib == "gzip":
            import gzip

            self._fh = gzip.open(self.csvfile, mode, self.complevel, **kwargs)
        elif self.complib in ("bz2", "bzip2"):
            import bz2

            self._fh = bz2.open(self.csvfile, mode, self.complevel, **kwargs)
        elif self.complib is None:
            self._fh = open(self.csvfile, mode, **kwargs)
        else:
            raise KeyError("Unexpected compression library: {}".format(self.complib))
        self._writer = csv.writer(self._fh, **self.csv_kwargs)
        # Write header data
        row = ["Datetime"] + [name for name in self._node_names]
        self._writer.writerow(row)

    def after(self):
        """
        Write the node values to the CSV file
        """
        values = [self.model.timestepper.current.datetime.isoformat()]
        for node_name in self._node_names:
            node = self.model.nodes[node_name]
            if isinstance(node, AbstractStorage):
                values.append(node.volume[self.scenario_index])
            elif isinstance(node, AbstractNode):
                values.append(node.flow[self.scenario_index])
            else:
                raise ValueError(
                    "Unrecognised Node type '{}' for CSV writer".format(type(node))
                )

        self._writer.writerow(values)

    def finish(self):
        if self._fh:
            self._fh.close()


CSVRecorder.register()


class TablesRecorder(Recorder):
    """
    A recorder that saves to PyTables CArray

    This Recorder creates a CArray for every node passed to the constructor.
    Each CArray stores the data for all scenarios on the specific node. This
    is useful for analysis of Node statistics across multiple scenarios.
    """

    def __init__(
        self,
        model,
        h5file,
        nodes=None,
        parameters=None,
        where="/",
        time="/time",
        routes_flows=None,
        routes="/routes",
        scenarios="/scenarios",
        **kwargs,
    ):
        """

        Parameters
        ----------
        model : `pywr.model.Model`
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
        routes_flows : string
            Relative (to `where`) node path to save the routes flow CArray. If None (default) no array is created.
        routes : string
            Full node path to save the routes tables.Table. If None not table is created.
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
        self.filter_kwds = kwargs.pop("filter_kwds", {})
        self.mode = kwargs.pop("mode", "w")
        self.metadata = kwargs.pop("metadata", {})
        self.create_directories = kwargs.pop("create_directories", False)

        title = kwargs.pop("title", None)
        if title is None:
            try:
                title = model.metadata["title"]
            except KeyError:
                title = ""
        self.title = title
        super(TablesRecorder, self).__init__(model, **kwargs)

        self.h5file = h5file
        self.h5store = None
        self._arrays = {}
        self.nodes = nodes
        self.where = where
        self.time = time
        self.scenarios = scenarios
        self.routes = routes
        self.routes_flows = routes_flows

        # Enable saving routes in the solver.
        if routes_flows:
            self.model.solver.save_routes_flows = True

        self.parameters = []
        if parameters:
            for parameter in parameters:
                self._add_parameter(parameter)

        self._arrays = None
        self._time_table = None
        self._routes_flow_array = None

    def _add_parameter(self, parameter):
        try:
            where, param = parameter
        except (TypeError, ValueError):
            where = None
            param = parameter
        if isinstance(param, str):
            from ..parameters import load_parameter

            param = load_parameter(self.model, param)
        if not param.name:
            raise ValueError("Can only record named Parameter objects")
        if where is None:
            name = param.name.replace("/", "_")
            if name != param.name:
                warnings.warn(
                    'Recorded parameter has "/" in name, replaced with "_" to avoid creation of subgroup: {}'.format(
                        param.name
                    ),
                    ParameterNameWarning,
                )
            where = self.where + "/" + name
        where = where.replace("//", "/")
        self.children.add(param)
        self.parameters.append((where, param))

    def _remove_parameter(self, parameter):
        if isinstance(parameter, str):
            parameter = self.model.parameters[parameter]
        index = None
        for n, (where, param) in enumerate(self.parameters):
            if param is parameter:
                index = n
        if index is None:
            raise KeyError("Parameter is not in TablesRecorder: {}".format(parameter))
        self.parameters.pop(index)
        self.children.remove(parameter)

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

        self.h5store = H5Store(
            self.h5file,
            self.filter_kwds,
            self.mode,
            title=self.title,
            metadata=self.metadata,
            create_directories=self.create_directories,
        )

        # Create a CArray for each node
        self._arrays = {}

        # Default to all nodes if None given.
        if self.nodes is None:
            nodes = [
                ((self.where + "/" + n.name).replace("//", "/"), n)
                for n in self.model.nodes.values()
            ]
        else:
            nodes = []
            for n in self.nodes:

                try:
                    where, node = n
                except (TypeError, ValueError):
                    node = n
                    where = self.where + "/" + node

                # Accept a str, and lookup node by name instead.
                if isinstance(node, str):
                    node = self.model.nodes[node]
                # Otherwise assume it is a node object anyway

                where = where.replace("//", "/")
                nodes.append((where, node))

        if self.parameters is not None:
            nodes.extend(self.parameters)

        self._nodes = nodes

        for where, node in self._nodes:
            if isinstance(node, IndexParameter):
                atom = tables.Int32Atom()
            else:
                atom = tables.Float64Atom()
            group_name, node_name = where.rsplit("/", 1)
            if group_name == "":
                group_name = "/"
            self.h5store.file.create_carray(
                group_name, node_name, atom, shape, createparents=True
            )

        # Create scenario tables
        if self.scenarios is not None:
            group_name, node_name = self.scenarios.rsplit("/", 1)
            if group_name == "":
                group_name = "/"
            description = {
                # TODO make string length configurable
                "name": tables.StringCol(1024),
                "size": tables.Int64Col(),
            }
            tbl = self.h5store.file.create_table(
                group_name, node_name, description=description, createparents=True
            )
            # Now add the scenarios
            entry = tbl.row
            for scenario in self.model.scenarios.scenarios:
                entry["name"] = scenario.name.encode("utf-8")
                entry["size"] = scenario.size
                entry.append()
            tbl.flush()

            if self.model.scenarios.user_combinations is not None:
                description = {
                    s.name: tables.Int64Col() for s in self.model.scenarios.scenarios
                }
                tbl = self.h5store.file.create_table(
                    group_name, "scenario_combinations", description=description
                )
                entry = tbl.row
                for comb in self.model.scenarios.user_combinations:
                    for s, i in zip(self.model.scenarios.scenarios, comb):
                        entry[s.name] = i
                    entry.append()
                tbl.flush()

        self.h5store = None

    def reset(self):
        import tables

        mode = "r+"  # always need to append, as file already created in setup
        self.h5store = H5Store(self.h5file, self.filter_kwds, mode)
        self._arrays = {}
        for where, node in self._nodes:
            self._arrays[node] = self.h5store.file.get_node(where)

        self._time_table = None
        # Create time table
        # This is created in reset so that the table is always recreated
        if self.time is not None:
            group_name, node_name = self.time.rsplit("/", 1)
            if group_name == "":
                group_name = "/"
            description = {
                c: tables.Int64Col() for c in ("year", "month", "day", "index")
            }

            try:
                self.h5store.file.remove_node(group_name, node_name)
            except tables.NoSuchNodeError:
                pass
            finally:
                self._time_table = self.h5store.file.create_table(
                    group_name, node_name, description=description, createparents=True
                )

        self._routes_flow_array = None
        if self.routes_flows is not None:
            # Create a CArray for the flows
            # The first dimension is the number of timesteps.
            # The second dimension is the number of routes
            # The following dimensions are sized per scenario
            scenario_shape = list(self.model.scenarios.shape)
            shape = [
                len(self.model.timestepper),
                len(self.model.solver.routes),
            ] + scenario_shape
            atom = tables.Float64Atom()

            try:
                self.h5store.file.remove_node(self.where, self.routes_flows)
            except tables.NoSuchNodeError:
                pass
            finally:
                self._routes_flow_array = self.h5store.file.create_carray(
                    self.where, self.routes_flows, atom, shape, createparents=True
                )

            # Create routes table. This must be done in reset
            if self.routes is not None:
                group_name, node_name = self.routes.rsplit("/", 1)
                if group_name == "":
                    group_name = "/"

                description = {
                    # TODO make string length configurable
                    "start": tables.StringCol(1024),
                    "end": tables.StringCol(1024),
                }
                try:
                    self.h5store.file.remove_node(group_name, node_name)
                except tables.NoSuchNodeError:
                    pass
                finally:
                    tbl = self.h5store.file.create_table(
                        group_name,
                        node_name,
                        description=description,
                        createparents=True,
                    )

                entry = tbl.row
                for route in self.model.solver.routes:
                    node_first = route[0]
                    node_last = route[-1]

                    if node_first.parent is not None:
                        node_first = node_first.parent
                    if node_last.parent is not None:
                        node_last = node_last.parent

                    entry["start"] = node_first.name.encode("utf-8")
                    entry["end"] = node_last.name.encode("utf-8")
                    entry.append()

                tbl.flush()

    def after(self):
        """
        Save data to the tables
        """
        from pywr._core import AbstractNode, AbstractStorage
        from pywr.parameters import Parameter, IndexParameter

        scenario_shape = list(self.model.scenarios.shape)
        ts = self.model.timestepper.current
        idx = ts.index

        if self._time_table is not None:
            entry = self._time_table.row
            entry["year"] = ts.year
            entry["month"] = ts.month
            entry["day"] = ts.day
            entry["index"] = idx
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
                raise ValueError(
                    "Unrecognised Node type '{}' for TablesRecorder".format(type(node))
                )

        if self._routes_flow_array is not None:
            routes_shape = [
                len(self.model.solver.routes),
            ] + scenario_shape
            self._routes_flow_array[idx, ...] = np.reshape(
                self.model.solver.routes_flows_array.T, routes_shape
            )

    def finish(self):
        if self._time_table is not None:
            self._time_table.flush()
        self.h5store = None
        self._arrays = {}
        self._routes_flow_array = None

    @staticmethod
    def generate_dataframes(h5file, time="/time", scenarios="/scenarios"):
        """Helper function to generate pandas dataframes from `TablesRecorder` data.

        Parameters
        h5file : str
            A path to a H5 file created by `TablesRecorder`.
        time : str
            The internal table that contains the time information (default "/time")
        scenarios : str
            The internal table that contains the scenario information (default "/scenarios")
        """
        store = H5Store(h5file, mode="r")

        # Get the time information
        if time:
            time_table = store.file.get_node(time)
            index = pandas.to_datetime(
                {k: time_table.col(k) for k in ("year", "month", "day")}
            )
        else:
            index = None

        # Get the scenario information
        if scenarios:
            scenarios_table = store.file.get_node(scenarios)
            scenarios = pandas.DataFrame(
                {k: scenarios_table.col(k) for k in ("name", "size")}
            )
            columns = pandas.MultiIndex.from_product(
                [range(row["size"]) for _, row in scenarios.iterrows()],
                names=[row["name"].decode() for _, row in scenarios.iterrows()],
            )
        else:
            columns = None

        for node in store.file.walk_nodes("/", "CArray"):
            data = node.read()
            data = data.reshape((data.shape[0], -1))
            df = pandas.DataFrame(data, index=index, columns=columns)
            yield node._v_name, df


TablesRecorder.register()
