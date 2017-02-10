import numpy as np
cimport numpy as np
from pywr._core cimport Timestep
import pandas as pd
from past.builtins import basestring

recorder_registry = {}

cdef enum AggFuncs:
    SUM = 0
    MIN = 1
    MAX = 2
    MEAN = 3
    MEDIAN = 4
    PRODUCT = 5
    CUSTOM = 6
    ANY = 7
    ALL = 8
_agg_func_lookup = {
    "sum": AggFuncs.SUM,
    "min": AggFuncs.MIN,
    "max": AggFuncs.MAX,
    "mean": AggFuncs.MEAN,
    "median": AggFuncs.MEDIAN,
    "product": AggFuncs.PRODUCT,
    "custom": AggFuncs.CUSTOM,
    "any": AggFuncs.ANY,
    "all": AggFuncs.ALL,
}

cdef class Recorder:
    def __init__(self, model, name=None, agg_func="mean", comment=None):
        self._model = model
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = name
        self.comment = comment
        self.agg_func = agg_func
        model.recorders.append(self)

    property agg_func:
        def __set__(self, agg_func):
            self._agg_user_func = None
            if isinstance(agg_func, basestring):
                agg_func = _agg_func_lookup[agg_func.lower()]
            elif callable(agg_func):
                self._agg_user_func = agg_func
                agg_func = AggFuncs.CUSTOM
            else:
                raise ValueError("Unrecognised aggregation function: \"{}\".".format(agg_func))
            self._agg_func = agg_func

    property name:
        def __get__(self):
            return self._name

        def __set__(self, name):
            # check for name collision
            if name in self.model.recorders.keys():
                raise ValueError('A recorder with the name "{}" already exists.'.format(name))
            # apply new name
            self._name = name

    def __repr__(self):
        return '<{} "{}">'.format(self.__class__.__name__, self.name)

    cpdef setup(self):
        pass

    cpdef reset(self):
        pass

    cpdef int save(self) except -1:
        return 0

    cpdef finish(self):
        pass

    property model:
        def __get__(self, ):
            return self._model

    property is_objective:
        def __get__(self):
            return self._is_objective

        def __set__(self, value):
            self._is_objective = value

    cpdef double aggregated_value(self) except? -1:
        cdef double[:] values = self.values()

        if self._agg_func == AggFuncs.PRODUCT:
            return np.product(values)
        elif self._agg_func == AggFuncs.SUM:
            return np.sum(values)
        elif self._agg_func == AggFuncs.MAX:
            return np.max(values)
        elif self._agg_func == AggFuncs.MIN:
            return np.min(values)
        elif self._agg_func == AggFuncs.MEAN:
            return np.mean(values)
        elif self._agg_func == AggFuncs.MEDIAN:
            return np.median(values)
        else:
            return self._agg_user_func(np.array(values))

    cpdef double[:] values(self):
        raise NotImplementedError()

    @classmethod
    def load(cls, model, data):
        try:
            node_name = data["node"]
        except KeyError:
            pass
        else:
            data["node"] = model._get_node_from_ref(model, node_name)
        return cls(model, **data)

    @classmethod
    def register(cls):
        recorder_registry[cls.__name__.lower()] = cls

    @classmethod
    def unregister(cls):
        del(recorder_registry[cls.__name__.lower()])

cdef class AggregatedRecorder(Recorder):
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
        # Opitional different method for aggregating across self.recorders scenarios
        agg_func = kwargs.pop('recorder_agg_func', kwargs.get('agg_func'))

        if isinstance(agg_func, basestring):
            agg_func = _agg_func_lookup[agg_func.lower()]
        elif callable(agg_func):
            self.recorder_agg_func = agg_func
            agg_func = AggFuncs.CUSTOM
        else:
            raise ValueError("Unrecognised recorder aggregation function: \"{}\".".format(agg_func))
        self._recorder_agg_func = agg_func

        super(AggregatedRecorder, self).__init__(model, **kwargs)
        self.recorders = list(recorders)

    cpdef double[:] values(self):
        cdef Recorder recorder
        cdef double[:] value, value2
        assert(len(self.recorders))
        cdef int n = len(self.model.scenarios.combinations)
        cdef int i

        if self._recorder_agg_func == AggFuncs.PRODUCT:
            value = np.ones(n, np.float64)
            for recorder in self.recorders:
                value2 = recorder.values()
                for i in range(n):
                    value[i] *= value2[i]
        elif self._recorder_agg_func == AggFuncs.SUM:
            value = np.zeros(n, np.float64)
            for recorder in self.recorders:
                value2 = recorder.values()
                for i in range(n):
                    value[i] += value2[i]
        elif self._recorder_agg_func == AggFuncs.MAX:
            value = np.empty(n)
            value[:] = np.NINF
            for recorder in self.recorders:
                value2 = recorder.values()
                for i in range(n):
                    if value2[i] > value[i]:
                        value[i] = value2[i]
        elif self._recorder_agg_func == AggFuncs.MIN:
            value = np.empty(n)
            value[:] = np.INF
            for recorder in self.recorders:
                value2 = recorder.values()
                for i in range(n):
                    if value2[i] < value[i]:
                        value[i] = value2[i]
        elif self._recorder_agg_func == AggFuncs.MEAN:
            value = np.zeros(n, np.float64)
            for recorder in self.recorders:
                value2 = recorder.values()
                for i in range(n):
                    value[i] += value2[i]
            for i in range(n):
                value[i] /= len(self.recorders)
        else:
            value = self.recorder_agg_func([recorder.values() for recorder in self.recorders], axis=0)
        return value

    @classmethod
    def load(cls, model, data):
        recorder_names = data["recorders"]
        recorders = [model.recorders[name] for name in recorder_names]
        print(recorders)
        del(data["recorders"])
        rec = cls(model, recorders, **data)
        print(rec.name)

AggregatedRecorder.register()


cdef class NodeRecorder(Recorder):
    def __init__(self, model, AbstractNode node, name=None, **kwargs):
        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), node.name)
        super(NodeRecorder, self).__init__(model, name=name, **kwargs)
        self._node = node
        node._recorders.append(self)

    property node:
        def __get__(self):
            return self._node

    def __repr__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.node, self.name)

NodeRecorder.register()


cdef class StorageRecorder(Recorder):
    def __init__(self, model, Storage node, name=None, **kwargs):
        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), node.name)
        super(StorageRecorder, self).__init__(model, name=name, **kwargs)
        self._node = node
        node._recorders.append(self)

    property node:
        def __get__(self):
            return self._node

    def __repr__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.node, self.name)

StorageRecorder.register()


cdef class ParameterRecorder(Recorder):
    def __init__(self, model, Parameter param, name=None, **kwargs):
        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), param.name)
        super(ParameterRecorder, self).__init__(model, name=name, **kwargs)
        self._param = param
        param._recorders.append(self)

    property parameter:
        def __get__(self):
            return self._param

    def __repr__(self):
        return '<{} on {} "{}" ({})>'.format(self.__class__.__name__, repr(self.parameter), self.name, hex(id(self)))

    def __str__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.parameter, self.name)

    @classmethod
    def load(cls, model, data):
        # when the parameter being recorder is defined inline (i.e. not in the
        # parameters section, but within the node) we need to make sure the
        # node has been loaded first
        try:
            node_name = data["node"]
        except KeyError:
            node = None
        else:
            del(data["node"])
            node = model._get_node_from_ref(model, node_name)
        from .parameters import load_parameter
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)

ParameterRecorder.register()


cdef class IndexParameterRecorder(Recorder):
    def __init__(self, model, IndexParameter param, name=None, **kwargs):
        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), param.name)
        super(IndexParameterRecorder, self).__init__(model, name=name, **kwargs)
        self._param = param
        param._recorders.append(self)

    property parameter:
        def __get__(self):
            return self._param

    def __repr__(self):
        return '<{} on {} "{}" ({})>'.format(self.__class__.__name__, repr(self.parameter), self.name, hex(id(self)))

    def __str__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.parameter, self.name)

    @classmethod
    def load(cls, model, data):
        from .parameters import load_parameter
        parameter = load_parameter(model, data.pop("parameter"))
        return cls(model, parameter, **data)

IndexParameterRecorder.register()


cdef class NumpyArrayNodeRecorder(NodeRecorder):
    cpdef setup(self):
        cdef int ncomb = len(self._model.scenarios.combinations)
        cdef int nts = len(self._model.timestepper)
        self._data = np.zeros((nts, ncomb))

    cpdef reset(self):
        self._data[:, :] = 0.0

    cpdef int save(self) except -1:
        cdef int i
        cdef Timestep ts = self._model.timestepper.current
        for i in range(self._data.shape[1]):
            self._data[ts._index,i] = self._node._flow[i]
        return 0

    property data:
        def __get__(self, ):
            return np.array(self._data)

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)

NumpyArrayNodeRecorder.register()



cdef class FlowDurationCurveRecorder(NumpyArrayNodeRecorder):
    """
    This recorder calculates a flow duration curve for each scenario.
    ----------
    model : `pywr.core.Model`
    node : `pywr.core.Node`
        The node to record
    percentiles : array
        The percentiles to use in the calculation of the flow duration curve.
        Values must be in the range 0-100.
    agg_func: str, optional
        function used for aggregating the FDC across percentiles.
        Numpy style functions that support an axis argument are supported.
    fdc_agg_func: str, optional
        optional different function for aggregating across scenarios.
    """

    def __init__(self, model, AbstractNode node, percentiles, **kwargs):

        # Optional different method for aggregating across percentiles
        agg_func = kwargs.pop('fdc_agg_func', kwargs.get('agg_func', 'mean'))
        super(FlowDurationCurveRecorder, self).__init__(model, node, **kwargs)

        if isinstance(agg_func, basestring):
            agg_func = _agg_func_lookup[agg_func.lower()]
        elif callable(agg_func):
            self.agg_user_func = agg_func
            agg_func = AggFuncs.CUSTOM
        else:
            raise ValueError("Unrecognised recorder aggregation function: \"{}\".".format(agg_func))
        self._fdc_agg_func = agg_func

        self._percentiles = np.asarray(percentiles, dtype=np.float64)


    cpdef finish(self):
        self._fdc = np.percentile(np.asarray(self._data), np.asarray(self._percentiles), axis=0)

    property fdc:
        def __get__(self, ):
            return np.array(self._fdc)

    cpdef double[:] values(self):

        if self._fdc_agg_func == AggFuncs.PRODUCT:
            return np.product(self._fdc, axis=0)
        elif self._fdc_agg_func == AggFuncs.SUM:
            return np.sum(self._fdc, axis=0)
        elif self._fdc_agg_func == AggFuncs.MAX:
            return np.max(self._fdc, axis=0)
        elif self._fdc_agg_func == AggFuncs.MIN:
            return np.min(self._fdc, axis=0)
        elif self._fdc_agg_func == AggFuncs.MEAN:
            return np.mean(self._fdc, axis=0)
        elif self._fdc_agg_func == AggFuncs.MEDIAN:
            return np.median(self._fdc, axis=0)
        else:
            return self._agg_user_func(np.array(self._fdc), axis=0)

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self._percentiles
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self.fdc), index=index, columns=sc_index)

FlowDurationCurveRecorder.register()

cdef class FlowDurationCurveDeviationRecorder(FlowDurationCurveRecorder):
    """
    This recorder calculates a Flow Duration Curve (FDC) for each scenario and then
    calculates their deviation from an input FDC. The 2nd dimension of the input flow
    duration curve and percentiles list must be of the same length and have the same
    order (high to low values or low to high values)
    ----------
    model : `pywr.core.Model`
    node : `pywr.core.Node`
        The node to record
    percentiles : array
        The percentiles to use in the calculation of the flow duration curve.
        Values must be in the range 0-100.
    target_fdc : array
        The FDC against which the scenario FDCs are compared
    agg_func: str, optional
        Function used for aggregating the FDC deviations across percentiles.
        Numpy style functions that support an axis argument are supported.
    fdc_agg_func: str, optionalgit push 
        Optional different function for aggregating across scenarios.
    """
    def __init__(self, model, AbstractNode node, percentiles, target_fdc, scenario=None, name=None, **kwargs):
        super(FlowDurationCurveDeviationRecorder, self).__init__(model, node, percentiles, name=None, **kwargs)
        self._target_fdc = np.asarray(target_fdc, dtype=np.float64)

        if len(self._percentiles) != self._target_fdc.shape[0]:
            raise ValueError("The lengths of the input FDC and the percentiles list do not match")

        if np.any(np.argmin(self._target_fdc, axis=0) != np.argmin(self._percentiles)):
            raise ValueError("The orders of input FDC and the percentiles list do not match")

    cpdef setup(self):
        super(FlowDurationCurveDeviationRecorder, self).setup()
        # Check target FDC is the correct size; this is done in setup rather than __init__
        # because the scenarios might change after the Recorder is created.
        if self.scenario is not None:
            if self._target_fdc.shape[1] != self.scenario.size:
                raise ValueError("The number of target FDCs does not match the number of scenarios")
        else:
            if self._target_fdc.shape[1] != len(self.model.scenarios.combinations):
                raise ValueError("The number of target FDCs does not match the number of scenarios")

    cpdef finish(self):
        super(FlowDurationCurveDeviationRecorder, self).finish()

        cdef int i, j, sc_index
        cdef ScenarioIndex scenario_index
        cdef double[:] trgt_fdc

        if self.scenario is not None:
            # We have to do this the slow way by iterating through all scenario combinations
            sc_index = self.model.scenarios.get_scenario_index(self.scenario)
            self._fdc_deviations = np.empty((self._target_fdc.shape[0], len(self.model.scenarios.combinations)), dtype=np.float64)
            for i, scenario_index in enumerate(self.model.scenarios.combinations):
                # Get the scenario specific ensemble id for this combination
                j = scenario_index._indices[sc_index]
                # Cache the target FDC to use in this combination
                trgt_fdc = self._target_fdc[:, j]
                # Finally calculate deviation
                self._fdc_deviations[:, i] = np.divide(np.subtract(self._fdc[:, i], trgt_fdc), trgt_fdc)
        else:
            self._fdc_deviations = np.divide(np.subtract(self._fdc, self._target_fdc), self._target_fdc)

    property fdc_deviations:
        def __get__(self, ):
            return np.array(self._fdc_deviations)

    cpdef double[:] values(self):

        if self._fdc_agg_func == AggFuncs.PRODUCT:
            return np.product(self._fdc_deviations, axis=0)
        elif self._fdc_agg_func == AggFuncs.SUM:
            return np.sum(self._fdc_deviations, axis=0)
        elif self._fdc_agg_func == AggFuncs.MAX:
            return np.max(self._fdc_deviations, axis=0)
        elif self._fdc_agg_func == AggFuncs.MIN:
            return np.min(self._fdc_deviations, axis=0)
        elif self._fdc_agg_func == AggFuncs.MEAN:
            return np.mean(self._fdc_deviations, axis=0)
        elif self._fdc_agg_func == AggFuncs.MEDIAN:
            return np.median(self._fdc_deviations, axis=0)
        else:
            return self._agg_user_func(np.array(self._fdc_deviations), axis=0)

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self._percentiles
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._fdc_deviations), index=index, columns=sc_index)

FlowDurationCurveDeviationRecorder.register()


cdef class NumpyArrayStorageRecorder(StorageRecorder):
    cpdef setup(self):
        cdef int ncomb = len(self._model.scenarios.combinations)
        cdef int nts = len(self._model.timestepper)
        self._data = np.zeros((nts, ncomb))

    cpdef reset(self):
        self._data[:, :] = 0.0

    cpdef int save(self) except -1:
        cdef int i
        cdef Timestep ts = self._model.timestepper.current
        for i in range(self._data.shape[1]):
            self._data[ts._index,i] = self._node._volume[i]
        return 0

    property data:
        def __get__(self, ):
            return np.array(self._data)

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)

NumpyArrayStorageRecorder.register()

cdef class NumpyArrayLevelRecorder(StorageRecorder):
    cpdef setup(self):
        cdef int ncomb = len(self._model.scenarios.combinations)
        cdef int nts = len(self._model.timestepper)
        self._data = np.zeros((nts, ncomb))

    cpdef reset(self):
        self._data[:, :] = 0.0

    cpdef int save(self) except -1:
        cdef int i
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self._model.timestepper.current
        for i, scenario_index in enumerate(self._model.scenarios.combinations):
            self._data[ts._index,i] = self._node.get_level(ts, scenario_index)
        return 0

    property data:
        def __get__(self, ):
            return np.array(self._data)

NumpyArrayLevelRecorder.register()

cdef class NumpyArrayParameterRecorder(ParameterRecorder):
    cpdef setup(self):
        cdef int ncomb = len(self._model.scenarios.combinations)
        cdef int nts = len(self._model.timestepper)
        self._data = np.zeros((nts, ncomb))

    cpdef reset(self):
        self._data[:, :] = 0.0

    cpdef int save(self) except -1:
        cdef int i
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self._model.timestepper.current
        for i, scenario_index in enumerate(self._model.scenarios.combinations):
            self._data[ts._index, i] = self._param.value(ts, scenario_index)
        return 0

    property data:
        def __get__(self, ):
            return np.array(self._data)

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data
        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)
NumpyArrayParameterRecorder.register()

cdef class NumpyArrayIndexParameterRecorder(IndexParameterRecorder):
    cpdef setup(self):
        cdef int ncomb = len(self._model.scenarios.combinations)
        cdef int nts = len(self._model.timestepper)
        self._data = np.zeros((nts, ncomb), dtype=np.int32)

    cpdef reset(self):
        self._data[:, :] = 0

    cpdef int save(self) except -1:
        cdef int i
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self._model.timestepper.current
        for i, scenario_index in enumerate(self._model.scenarios.combinations):
            self._data[ts._index, i] = self._param.index(ts, scenario_index)
        return 0

    property data:
        def __get__(self, ):
            return np.array(self._data)

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data
        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)
NumpyArrayIndexParameterRecorder.register()

cdef class MeanParameterRecorder(ParameterRecorder):
    """Records the mean value of a Parameter for the last N timesteps"""
    def __init__(self, model, Parameter param, int timesteps, *args, **kwargs):
        super(MeanParameterRecorder, self).__init__(model, param, *args, **kwargs)
        self.timesteps = timesteps

    cpdef setup(self):
        cdef int ncomb = len(self._model.scenarios.combinations)
        cdef int nts = len(self._model.timestepper)
        self._data = np.zeros((nts, ncomb,), np.float64)
        self._memory = np.empty((nts, ncomb,), np.float64)
        self.position = 0

    cpdef reset(self):
        self._data[...] = 0
        self.position = 0

    cpdef int save(self) except -1:
        cdef int i, n
        cdef double[:] mean_value
        cdef ScenarioIndex scenario_index
        cdef Timestep timestep = self._model.timestepper.current

        for i, scenario_index in enumerate(self._model.scenarios.combinations):
            self._memory[self.position, i] = self._param.value(timestep, scenario_index)

        if timestep.index < self.timesteps:
            n = timestep.index + 1
        else:
            n = self.timesteps

        mean_value = np.mean(self._memory[0:n, :], axis=0)
        self._data[<int>(timestep.index), :] = mean_value

        self.position += 1
        if self.position >= self.timesteps:
            self.position = 0

    property data:
        def __get__(self):
            return np.array(self._data, dtype=np.float64)

    def to_dataframe(self):
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        return pd.DataFrame(data=self.data, index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        from .parameters import load_parameter
        parameter = load_parameter(model, data.pop("parameter"))
        timesteps = int(data.pop("timesteps"))
        return cls(model, parameter, timesteps, **data)

MeanParameterRecorder.register()

cdef class MeanFlowRecorder(NodeRecorder):
    """Records the mean flow of a Node for the previous N timesteps
    Parameters
    ----------
    model : `pywr.core.Model`
    node : `pywr.core.Node`
        The node to record
    timesteps : int
        The number of timesteps to calculate the mean flow for
    name : str (optional)
        The name of the recorder
    """
    def __init__(self, model, node, timesteps=None, days=None, name=None, **kwargs):
        super(MeanFlowRecorder, self).__init__(model, node, name=name, **kwargs)
        self._model = model
        if not timesteps and not days:
            raise ValueError("Either `timesteps` or `days` must be specified.")
        if timesteps:
            self.timesteps = int(timesteps)
        else:
            self.timesteps = 0
        if days:
            self.days = int(days)
        else:
            self.days = 0
        self._data = None

    cpdef setup(self):
        super(MeanFlowRecorder, self).setup()
        self.position = 0
        self._data = np.empty([len(self._model.timestepper), len(self._model.scenarios.combinations)])
        if self.days:
            self.timesteps = self.days // self._model.timestepper.delta.days
        if self.timesteps == 0:
            raise ValueError("Timesteps property of MeanFlowRecorder is less than 1.")
        self._memory = np.zeros([len(self._model.scenarios.combinations), self.timesteps])

    cpdef int save(self) except -1:
        cdef Timestep timestep
        cdef int i, n
        cdef double[:] mean_flow
        # save today's flow
        for i in range(0, self._memory.shape[0]):
            self._memory[i, self.position] = self._node._flow[i]
        # calculate the mean flow
        timestep = self._model.timestepper.current
        if timestep.index < self.timesteps:
            n = timestep.index + 1
        else:
            n = self.timesteps
        # save the mean flow
        mean_flow = np.mean(self._memory[:, 0:n], axis=1)
        self._data[<int>(timestep.index), :] = mean_flow
        # prepare for the next timestep
        self.position += 1
        if self.position >= self.timesteps:
            self.position = 0

    property data:
        def __get__(self):
            return np.array(self._data, dtype=np.float64)

    @classmethod
    def load(cls, model, data):
        name = data.get("name")
        node = model._get_node_from_ref(model, data["node"])
        if "timesteps" in data:
            timesteps = int(data["timesteps"])
        else:
            timesteps = None
        if "days" in data:
            days = int(data["days"])
        else:
            days = None
        return cls(model, node, timesteps=timesteps, days=days, name=name)

MeanFlowRecorder.register()

cdef class BaseConstantNodeRecorder(NodeRecorder):
    """
    Base class for NodeRecorder classes with a single value for each scenario combination
    """

    cpdef setup(self):
        self._values = np.zeros(len(self.model.scenarios.combinations))

    cpdef reset(self):
        self._values[...] = 0.0

    cpdef int save(self) except -1:
        raise NotImplementedError()

    cpdef double[:] values(self):
        return self._values


cdef class TotalDeficitNodeRecorder(BaseConstantNodeRecorder):
    """
    Recorder to total the difference between modelled flow and max_flow for a Node
    """
    cpdef int save(self) except -1:
        cdef double max_flow
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef AbstractNode node = self._node
        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(ts, scenario_index)
            self._values[scenario_index._global_id] += max_flow - node._flow[scenario_index._global_id]

        return 0
TotalDeficitNodeRecorder.register()


cdef class TotalFlowNodeRecorder(BaseConstantNodeRecorder):
    """
    Recorder to total the flow for a Node.

    A factor can be provided to scale the total flow (e.g. for calculating operational costs).
    """
    def __init__(self, *args, **kwargs):
        self.factor = kwargs.pop('factor', 1.0)
        super(TotalFlowNodeRecorder, self).__init__(*args, **kwargs)

    cpdef int save(self) except -1:
        cdef ScenarioIndex scenario_index
        cdef int i
        cdef int days = self.model.timestepper.delta.days
        for scenario_index in self.model.scenarios.combinations:
            i = scenario_index._global_id
            self._values[i] += self._node._flow[i]*self.factor*days
        return 0
TotalFlowNodeRecorder.register()


cdef class DeficitFrequencyNodeRecorder(BaseConstantNodeRecorder):
    """
    Recorder to total the difference between modelled flow and max_flow for a Node
    """
    cpdef int save(self) except -1:
        cdef double max_flow
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current
        cdef AbstractNode node = self._node
        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(ts, scenario_index)
            if abs(node._flow[scenario_index._global_id] - max_flow) > 1e-6:
                self._values[scenario_index._global_id] += 1.0

    cpdef finish(self):
        cdef int i
        cdef int nt = self.model.timestepper.current.index
        for i in range(self._values.shape[0]):
            self._values[i] /= nt
DeficitFrequencyNodeRecorder.register()

cdef class BaseConstantStorageRecorder(StorageRecorder):
    """
    Base class for StorageRecorder classes with a single value for each scenario combination
    """

    cpdef setup(self):
        self._values = np.zeros(len(self.model.scenarios.combinations))

    cpdef reset(self):
        self._values[...] = 0.0

    cpdef int save(self) except -1:
        raise NotImplementedError()

    cpdef double[:] values(self):
        return self._values
BaseConstantStorageRecorder.register()

cdef class MinimumVolumeStorageRecorder(BaseConstantStorageRecorder):

    cpdef reset(self):
        self._values[...] = np.inf

    cpdef int save(self) except -1:
        cdef int i
        for i in range(self._values.shape[0]):
            self._values[i] = np.min([self._node._volume[i], self._values[i]])
        return 0
MinimumVolumeStorageRecorder.register()

cdef class MinimumThresholdVolumeStorageRecorder(BaseConstantStorageRecorder):

    def __init__(self, model, node, threshold, *args, **kwargs):
        self.threshold = threshold
        super(MinimumThresholdVolumeStorageRecorder, self).__init__(model, node, *args, **kwargs)
    
    cpdef reset(self):
        self._values[...] = 0.0

    cpdef int save(self) except -1:
        cdef int i
        for i in range(self._values.shape[0]):
            if self._node._volume[i] <= self.threshold:
                self._values[i] = 1.0 
        return 0
MinimumThresholdVolumeStorageRecorder.register()


cdef class AnnualCountIndexParameterRecorder(IndexParameterRecorder):
    """ Record the number of years where an IndexParameter is greater than or equal to a threshold """
    def __init__(self, model, IndexParameter param, int threshold, *args, **kwargs):
        super(AnnualCountIndexParameterRecorder, self).__init__(model, param, *args, **kwargs)
        self.threshold = threshold

    cpdef setup(self):
        self._count = np.zeros(len(self.model.scenarios.combinations), np.int32)
        self._current_max = np.zeros_like(self._count)

    cpdef reset(self):
        self._count[...] = 0
        self._current_max[...] = 0
        self._current_year = -1

    cpdef int save(self) except -1:
        cdef int i, ncomb, value
        cdef ScenarioIndex scenario_index
        cdef Timestep ts = self.model.timestepper.current

        ncomb = len(self.model.scenarios.combinations)

        if ts.year != self._current_year:
            # A new year
            if self._current_year != -1:
                # As long as at least one year has been run
                # then update the count if threshold equal to or exceeded
                for i in range(ncomb):
                    if self._current_max[i] >= self.threshold:
                        self._count[i] += 1

            # Finally reset current maximum and update current year
            self._current_max[...] = 0
            self._current_year = ts.year

        for scenario_index in self.model.scenarios.combinations:
            # Get current parameter value
            value = self._param.index(ts, scenario_index)

            # Update annual max if a new maximum is found
            if value > self._current_max[scenario_index._global_id]:
                self._current_max[scenario_index._global_id] = value

        return 0

    cpdef finish(self):
        cdef int i
        cdef int ncomb = len(self.model.scenarios.combinations)
        # Complete the current year by updating the count if threshold equal to or exceeded
        for i in range(ncomb):
            if self._current_max[i] >= self.threshold:
                self._count[i] += 1

    cpdef double[:] values(self):
        return np.asarray(self._count).astype(np.float64)
AnnualCountIndexParameterRecorder.register()


def load_recorder(model, data):
    recorder = None

    if isinstance(data, basestring):
        recorder_name = data
    else:
        recorder_name = None

    # check if recorder has already been loaded
    for rec in model.recorders:
        if rec.name == recorder_name:
            recorder = rec
            break

    if recorder is None and isinstance(data, basestring):
        # recorder was requested by name, but hasn't been loaded yet
        if hasattr(model, "_recorders_to_load"):
            # we're still in the process of loading data from JSON and
            # the parameter requested hasn't been loaded yet - do it now
            try:
                data = model._recorders_to_load[recorder_name]
            except KeyError:
                raise KeyError("Unknown recorder: '{}'".format(data))
            recorder = load_recorder(model, data)
        else:
            raise KeyError("Unknown recorder: '{}'".format(data))

    if recorder is None:
        recorder_type = data['type']

        name = recorder_type.lower()
        try:
            cls = recorder_registry[name]
        except KeyError:
            if name.endswith("recorder"):
                name = name.replace("recorder", "")
            else:
                name += "recorder"
            try:
                cls = recorder_registry[name]
            except KeyError:
                raise NotImplementedError('Unrecognised recorder type "{}"'.format(recorder_type))

        del(data["type"])
        recorder = cls.load(model, data)

    return recorder
