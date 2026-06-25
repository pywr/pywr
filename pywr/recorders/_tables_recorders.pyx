from typing import Tuple
import pandas
cimport numpy as np
import numpy as np
from pywr.h5tools import H5Store
import warnings
from pywr.parameters._parameters cimport IndexParameter, Parameter
from pywr.recorders._recorders cimport Recorder
from .._core cimport Timestep, Scenario, ScenarioIndex, AbstractNode, AbstractStorage, Storage

class ParameterNameWarning(UserWarning):
    pass


class TablesRecorder2(Recorder):
    """
    A recorder that saves model outputs to an HDF file.

    This Recorder is an extension of the original 'TablesRecorder'. It creates
    multiple CArrays in the HDF file for every node passed to the
    constructor. Each node has a CArray for flow/volume, max flow/volume, min flow/volume.
    This enables downstream processing to calculate statistics such as deficit and
    utilisation. Each CArray stores the data for all scenarios on the specific node.
    This is useful for analysis of Node statistics across multiple scenarios.
    Parameter values can also be optionally stored in CArrays within the file.

    By default, the recorder also stores time and scenario metadata tables. The
    time table stores a row containing index, year, month and day values for every
    timestep. A scenario table is created containing the name and size of each
    scenario defined for the Model. If scenario combinations are defined for the
    model then a separate table is created that saves the scenario indices of each
    combination. If there are no combinations but some of the scenarios have slices
    defined then scenario slice information, including the slice start, end, and
    step, is stored in a table.
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
            Full node path to save the routes tables.Table. If None no table is created.
        filter_kwds : dict
            Filter keywords to pass to tables.open_file when opening a file.
        mode : string
            Model argument to pass to tables.open_file. Defaults to 'w'
        metadata : dict
            Dict of user defined attributes to save on the root node (`root._v_attrs`)
        create_directories : bool
            If a file path is given and create_directories is True then attempt to make the intermediate
            directories. This uses os.makedirs() underneath.
        buffer_size : int
            The size in megabytes of buffer to keep in memory before appending to the PyTables data. This
            buffer is a total buffer for the whole recorder and is divided amongst the items to be recorded
            equally. Default is 100.
        buffer_timesteps: int
            The number of timesteps to buffer in memory before appending to the PyTables data. This is an alternative
            to buffer_size, and if given will be used in preference to buffer_size.
        """
        self.filter_kwds = kwargs.pop("filter_kwds", {})
        self.mode = kwargs.pop("mode", "w")
        self.metadata = kwargs.pop("metadata", {})
        self.create_directories = kwargs.pop("create_directories", False)
        self.buffer_size = kwargs.pop("buffer_size", 100)
        self.buffer_timesteps = kwargs.pop("buffer_timesteps", None)
        self._buffer_num_timesteps = 1

        title = kwargs.pop("title", None)
        if title is None:
            try:
                title = model.metadata["title"]
            except KeyError:
                title = ""
        self.title = title
        super(TablesRecorder2, self).__init__(model, **kwargs)

        self.h5file = h5file
        self.h5store = None
        # Separate internal storage
        self._node_arrays = None
        self._nodes = None
        self._storage_node_arrays = None
        self._storage_nodes = None
        self._parameter_arrays = None
        self._parameters = None
        self._index_parameter_arrays = None
        self._index_parameters = None
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

    @staticmethod
    def _node_attribute(node) -> Tuple[str, str, str]:
        """Return the type of attribute recorded from a particular "node" type."""

        if isinstance(node, AbstractStorage):
            return ("max_volume", "volume", "min_volume")
        elif isinstance(node, AbstractNode):
            return ("max_flow", "flow", "min_flow")
        elif isinstance(node, IndexParameter):
            return (None, "parameter_index", None)
        elif isinstance(node, Parameter):
            return (None, "parameter", None)
        else:
            raise ValueError(
                "Unrecognised Node type '{}' for TablesRecorder".format(type(node))
            )

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
            metadata={"PYWR_FORMAT": 2, "PYWR_VERSION": 1, **self.metadata},
            create_directories=self.create_directories,
        )

        # Create a CArray for each node
        self._node_arrays = {}
        self._storage_node_arrays = {}
        self._parameter_arrays = {}
        self._index_parameter_arrays = {}

        # Collect all items
        if self.nodes is None:
            # Default to all nodes if None given.
            items = [
                ((self.where + "/" + n.name).replace("//", "/"), n)
                for n in self.model.nodes.values()
            ]
        else:
            items = []
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
                items.append((where, node))

        if self.parameters is not None:
            items.extend(self.parameters)

        # Work out the number of timesteps to buffer
        if self.buffer_timesteps is not None:
            self._buffer_num_timesteps = min(self.buffer_timesteps, len(self.model.timestepper))
        elif self.buffer_size is not None:
            # Divide by 8 as we are storing 64-bit floats
            # Also assume most data is nodes, and we store 3 values for each
            total_buffer_values = self.buffer_size * 1024 * 1024 / 8 / 3
            buffer_per_item = total_buffer_values // len(items)
            self._buffer_num_timesteps = min(max(1, int(buffer_per_item // np.prod(scenario_shape))), len(self.model.timestepper))
        else:
            self._buffer_num_timesteps = 1

        self._nodes = []
        self._storage_nodes = []
        self._parameters = []
        self._index_parameters = []

        for where, item in items:
            if isinstance(item, IndexParameter):
                atom = tables.Int32Atom()
            else:
                atom = tables.Float64Atom()
            group_name = where
            if group_name == "":
                group_name = "/"

            node_attributes = self._node_attribute(item)
            for attr in node_attributes:
                if attr is not None:
                    carray = self.h5store.file.create_carray(
                        group_name, attr, atom, shape, createparents=True
                    )
                    # Save some metadata about the type of data this is
                    carray._v_attrs["PYWR_ATTRIBUTE"] = attr
                    carray._v_attrs["PYWR_TYPE"] = item.__class__.__name__

            # Collect the items into their own dictionaries organised by type
            if isinstance(item, AbstractStorage):
                self._storage_nodes.append((where, item))
            elif isinstance(item, AbstractNode):
                self._nodes.append((where, item))
            elif isinstance(item, IndexParameter):
                self._index_parameters.append((where, item))
            elif isinstance(item, Parameter):
                self._parameters.append((where, item))
            else:
                raise ValueError(
                    "Unrecognised type '{}' for TablesRecorder".format(type(item))
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

            ensemble_names = [
                scenario.ensemble_names for scenario in self.model.scenarios.scenarios
            ]
            if len(ensemble_names) > 0:
                indices = [
                    [
                        str(ensemble_names[n][i])
                        for n, i in enumerate(scenario_index.indices)
                    ]
                    for scenario_index in self.model.scenarios.get_combinations()
                ]
                _ = self.h5store.file.create_carray(
                    "/", "scenario_ensemble_names", obj=indices
                )

            if self.model.scenarios.user_combinations is not None:
                combs = np.array(self.model.scenarios.user_combinations, dtype=np.int64)
                ds = self.h5store.file.create_carray(
                    "/", "scenario_combinations", obj=combs
                )
                ds.attrs.columns = [s.name for s in self.model.scenarios.scenarios]
            elif any([s.slice for s in self.model.scenarios.scenarios]):
                # Slices are only applied in a model run if there are no user combinations
                description = {
                    "name": tables.StringCol(1024),
                    "start": tables.Int64Col(),
                    "stop": tables.Int64Col(),
                    "step": tables.Int64Col(),
                }
                tbl = self.h5store.file.create_table(
                    group_name,
                    "scenario_slices",
                    description=description,
                    createparents=True,
                )

                entry = tbl.row
                for scenario in self.model.scenarios.scenarios:
                    if scenario.slice is not None:
                        entry["name"] = scenario.name.encode("utf-8")
                        entry["start"] = scenario.slice.start
                        entry["stop"] = scenario.slice.stop
                        if scenario.slice.step:
                            entry["step"] = scenario.slice.step
                        else:
                            entry["step"] = 1
                        entry.append()
                tbl.flush()

        self.h5store = None

    def reset(self):
        import tables

        mode = "r+"  # always need to append, as file already created in setup
        self.h5store = H5Store(self.h5file, self.filter_kwds, mode)

        if self._buffer_num_timesteps > 1:
            scenario_shape = list(self.model.scenarios.shape)
            buffer_shape = [self._buffer_num_timesteps, ] + scenario_shape
        else:
            buffer_shape = None

        self._buffer_write_position = 0

        self._node_arrays = {}
        for where, node in self._nodes:
            if buffer_shape is not None:
                buffer = np.empty([3, ] + buffer_shape, dtype=np.float64)
            else:
                buffer = None

            self._node_arrays[node] = (self.h5store.file.get_node(where), buffer)

        self._storage_node_arrays = {}
        for where, node in self._storage_nodes:
            if buffer_shape is not None:
                buffer = np.empty([3, ] + buffer_shape, dtype=np.float64)
            else:
                buffer = None
            self._storage_node_arrays[node] = (self.h5store.file.get_node(where), buffer)

        self._index_parameter_arrays = {}
        for where, node in self._index_parameters:
            if buffer_shape is not None:
                buffer = np.empty([1, ] + buffer_shape, dtype=np.int32)
            else:
                buffer = None
            self._index_parameter_arrays[node] = (self.h5store.file.get_node(where), buffer)

        self._parameter_arrays = {}
        for where, node in self._parameters:
            if buffer_shape is not None:
                buffer = np.empty([1, ] + buffer_shape, dtype=np.float64)
            else:
                buffer = None
            self._parameter_arrays[node] = (self.h5store.file.get_node(where), buffer)

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
        cdef AbstractNode node
        cdef AbstractStorage storage_node
        cdef Parameter parameter
        cdef IndexParameter index_parameter
        cdef double[:] v
        cdef int[:] i
        cdef double[:] out = np.empty(len(self.model.scenarios.combinations))

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

        buffer_idx = idx % self._buffer_num_timesteps

        for node, (arrays, buffer) in self._node_arrays.items():
            if buffer is None:
                node.get_all_max_flow(out)
                arrays.max_flow[idx, :] = np.reshape(out, scenario_shape)

                node.get_all_min_flow(out)
                arrays.min_flow[idx, :] = np.reshape(out, scenario_shape)

                arrays.flow[idx, :] = np.reshape(node._flow, scenario_shape)
            else:
                node.get_all_max_flow(out)
                buffer[0, buffer_idx, ...] = np.reshape(out, scenario_shape)
                node.get_all_min_flow(out)
                buffer[1, buffer_idx, ...] = np.reshape(out, scenario_shape)
                buffer[2, buffer_idx, ...] = np.reshape(node._flow, scenario_shape)

        for storage_node, (arrays, buffer) in self._storage_node_arrays.items():
            if buffer is None:
                storage_node.get_all_max_volume(out)
                arrays.max_volume[idx, :] = np.reshape(out, scenario_shape)

                storage_node.get_all_min_volume(out)
                arrays.min_volume[idx, :] = np.reshape(out, scenario_shape)

                arrays.volume[idx, :] = np.reshape(storage_node._volume, scenario_shape)
            else:
                storage_node.get_all_max_volume(out)
                buffer[0, buffer_idx, ...] = np.reshape(out, scenario_shape)
                storage_node.get_all_min_volume(out)
                buffer[1, buffer_idx, ...] = np.reshape(out, scenario_shape)
                buffer[2, buffer_idx, ...] = np.reshape(storage_node._volume, scenario_shape)

        for parameter, (arrays, buffer) in self._parameter_arrays.items():
            if buffer is None:
                v = parameter.get_all_values()
                arrays.parameter[idx, :] = np.reshape(v, scenario_shape)
            else:
                buffer[0, buffer_idx, ...] = np.reshape(parameter.get_all_values(), scenario_shape)

        for index_parameter, (arrays, buffer) in self._index_parameter_arrays.items():
            if buffer is None:
                i = index_parameter.get_all_indices()
                arrays.parameter_index[idx, :] = np.reshape(i, scenario_shape)
            else:
                buffer[0, buffer_idx, ...] = np.reshape(index_parameter.get_all_indices(), scenario_shape)

        # Flush before we overwrite the data if this is the last entry in the buffer
        if self._buffer_num_timesteps > 1 and (idx + 1) % self._buffer_num_timesteps == 0:
            self.flush_buffer(idx, self._buffer_num_timesteps)

        if self._routes_flow_array is not None:
            routes_shape = [
                len(self.model.solver.routes),
            ] + scenario_shape
            self._routes_flow_array[idx, ...] = np.reshape(
                self.model.solver.routes_flows_array.T, routes_shape
            )

    def flush_buffer(self, int idx, int num_timesteps):
        """Flush the buffer arrays to PyTables"""
        cdef int idx_start = idx + 1 - num_timesteps

        for _, (arrays, buffer) in self._node_arrays.items():
            arrays.max_flow[idx_start:idx + 1, ...] = buffer[0, :num_timesteps, ...]
            arrays.min_flow[idx_start:idx + 1, ...] = buffer[1, :num_timesteps, ...]
            arrays.flow[idx_start:idx + 1, ...] = buffer[2, :num_timesteps, ...]

        for _, (arrays, buffer) in self._storage_node_arrays.items():
            arrays.max_volume[idx_start:idx + 1, ...] = buffer[0, :num_timesteps, ...]
            arrays.min_volume[idx_start:idx + 1, ...] = buffer[1, :num_timesteps, ...]
            arrays.volume[idx_start:idx + 1, ...] = buffer[2, :num_timesteps, ...]

        for _, (arrays, buffer) in self._parameter_arrays.items():
            arrays.parameter[idx_start:idx + 1, ...] = buffer[0, :num_timesteps, ...]

        for _, (arrays, buffer) in self._index_parameter_arrays.items():
            arrays.parameter_index[idx_start:idx + 1, ...] = buffer[0, :num_timesteps, ...]

        if self._time_table is not None:
            self._time_table.flush()

    def finish(self):

        if self._buffer_num_timesteps > 1:
            ts = self.model.timestepper.current
            # The final time-step is one past the end of the data
            idx = ts.index
            residual_timesteps = idx % self._buffer_num_timesteps

            # Flush the buffer if there are residual time-steps, or our buffer covered the entire simulation
            if residual_timesteps > 0:
                self.flush_buffer(idx - 1, residual_timesteps)
            elif self._buffer_num_timesteps == len(self.model.timestepper):
                self.flush_buffer(idx - 1, self._buffer_num_timesteps)

        if self._time_table is not None:
            self._time_table.flush()
        self.h5store = None
        self._node_arrays = {}
        self._storage_node_arrays = {}
        self._parameter_arrays = {}
        self._index_parameter_arrays = {}
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
            if node.name in ("scenario_combinations", "scenario_ensemble_names"):
                continue
            data = node.read()
            data = data.reshape((data.shape[0], -1))
            df = pandas.DataFrame(data, index=index, columns=columns)
            yield f"{node._v_parent._v_name}.{node._v_name}", df


TablesRecorder2.register()
