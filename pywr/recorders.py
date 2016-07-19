import sys
import numpy as np
from pywr._recorders import (Recorder, NodeRecorder, StorageRecorder, ParameterRecorder, IndexParameterRecorder,
    NumpyArrayNodeRecorder, NumpyArrayStorageRecorder, NumpyArrayLevelRecorder, NumpyArrayParameterRecorder,
    NumpyArrayIndexParameterRecorder, MeanParameterRecorder)
from pywr._recorders import recorder_registry
from pywr._core import Storage


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
    def __init__(self, model, parent, nodes=None, **kwargs):
        """

        :param model: The model to record nodes from.
        :param parent: The tables parent node to attach the CArray objects to.
        :param nodes: An iterable of nodes to save data. It defaults
        to None which is all nodes in the model
        """
        super(TablesRecorder, self).__init__(model, **kwargs)

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
recorder_registry.add(TablesRecorder)


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

    def value(self, aggregate=True):
        if aggregate:
            return self.agg_func(self._values)
        else:
            return self._values


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
recorder_registry.add(TotalDeficitNodeRecorder)


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
recorder_registry.add(TotalFlowRecorder)


# TODO: safe evaluation of arbitrary aggregation functions from strings (e.g. "min(a, b) + c")
agg_funcs = {
    "mean": np.mean,
    "sum": np.sum,
    "max": np.max,
    "min": np.min,
    "product": np.product,
}
class AggregatedRecorder(Recorder):
    """
    This Recorder is used to aggregate across multiple other Recorder objects.

    The class provides a method to produce a complex aggregated recorder by taking
     the results of other records. The value() method first collects unaggregated values
     from the provided recorders. These are then aggregated on a per scenario basis before
     aggregation across the scenarios to a single value (assuming aggregate=True).

    By default the same `agg_func` function is used for both steps, but an optional
     `recorder_agg_func` can undertake a different aggregation across scenarios. For
      example summing recorders per scenario, and then taking a mean of the sum totals.

    This method allows `AggregatedRecorder` to be used as a recorder for in other
     `AggregatedRecorder` instances.
    """
    def __init__(self, model, recorders, **kwargs):
        """

        :param model: pywr.core.Model instance
        :param recorders: iterable of `Recorder` objects to aggregate
        :keyword agg_func: function used for aggregating across the recorders.
            Numpy style functions that support an axis argument are supported.
        :keyword recorder_agg_func: optional different function for aggregating
            across scenarios.
        """
        self.agg_func = kwargs.pop('agg_func', np.mean)
        if isinstance(self.agg_func, str):
            self.agg_func = agg_funcs[self.agg_func]

        # Opitional different method for aggregating across self.recorders scenarios
        self.recorder_agg_func = kwargs.pop('recorder_agg_func', None)
        if isinstance(self.recorder_agg_func, str):
            self.recorder_agg_func = agg_funcs[self.recorder_agg_func]

        super(AggregatedRecorder, self).__init__(model, **kwargs)
        self.recorders = recorders

    def value(self, aggregate=True):
        # First aggregate across the recorders
        agg_func = self.recorder_agg_func
        # If no specific recorder aggregation function is given, we use the scenario one
        if agg_func is None:
            agg_func = self.agg_func

        values = agg_func(np.r_[[r.value(aggregate=False) for r in self.recorders]], axis=0)

        # Finally perform the aggregation across scenarios if requested.
        if aggregate:
            return self.agg_func(values)
        else:
            return values

    @classmethod
    def load(cls, model, data):
        recorder_names = data["recorders"]
        recorders = [model.recorders[name] for name in recorder_names]
        print(recorders)
        del(data["recorders"])
        rec = cls(model, recorders, **data)
        print(rec.name)

recorder_registry.add(AggregatedRecorder)


def load_recorder(model, data):
    recorder_type = data['type']
    
    # lookup the recorder class in the registry
    cls = None
    name2 = recorder_type.lower().replace('recorder', '')
    for recorder_class in recorder_registry:
        name1 = recorder_class.__name__.lower().replace('recorder', '')
        if name1 == name2:
            cls = recorder_class

    if cls is None:
        raise NotImplementedError('Unrecognised recorder type "{}"'.format(recorder_type))

    del(data["type"])
    rec = cls.load(model, data)
    
    return rec  # not strictly needed
