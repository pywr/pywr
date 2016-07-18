import numpy as np
cimport numpy as np

from pywr._core cimport Timestep

import pandas as pd

recorder_registry = set()

cdef class Recorder:
    def __init__(self, model, name=None):
        self._model = model
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = name
        model.recorders._recorders.append(self)

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

    cpdef value(self):
        raise NotImplementedError("Implement value() in subclasses to return an aggregated values.")

    @classmethod
    def load(cls, model, data):
        try:
            node_name = data["node"]
        except KeyError:
            pass
        else:
            data["node"] = model._get_node_from_ref(model, node_name)
        return cls(model, **data)

cdef class NodeRecorder(Recorder):
    def __init__(self, model, AbstractNode node, name=None):
        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), node.name)
        Recorder.__init__(self, model, name=name)
        self._node = node
        node._recorders.append(self)

    property node:
        def __get__(self):
            return self._node

    def __repr__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.node, self.name)

recorder_registry.add(NodeRecorder)


cdef class StorageRecorder(Recorder):
    def __init__(self, model, Storage node, name=None):
        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), node.name)
        Recorder.__init__(self, model, name=name)
        self._node = node
        node._recorders.append(self)

    property node:
        def __get__(self):
            return self._node

    def __repr__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.node, self.name)

recorder_registry.add(StorageRecorder)


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

recorder_registry.add(NumpyArrayNodeRecorder)


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

recorder_registry.add(NumpyArrayStorageRecorder)

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

recorder_registry.add(NumpyArrayLevelRecorder)

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
    def __init__(self, model, node, timesteps, name=None):
        super(MeanFlowRecorder, self).__init__(model, node, name=name)
        self._model = model
        self.timesteps = timesteps
        self._data = None

    cpdef setup(self):
        super(MeanFlowRecorder, self).setup()
        self._memory = np.zeros([len(self._model.scenarios.combinations), self.timesteps])
        self.position = 0
        self._data = np.empty([len(self._model.timestepper), len(self._model.scenarios.combinations)])

    cpdef int save(self) except -1:
        cdef double mean_flow
        cdef Timestep timestep
        # save today's flow
        self._memory[:, self.position] = self.node.flow
        # calculate the mean flow
        timestep = self._model.timestepper.current
        if timestep.index < self.timesteps:
            n = timestep.index + 1
        else:
            n = self.timesteps
        mean_flow = np.mean(self._memory[:, 0:n], axis=1)
        # save the mean flow
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
        timesteps = int(data["timesteps"])
        return cls(model, node, timesteps, name=name)

recorder_registry.add(MeanFlowRecorder)

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
        recorder = cls.load(model, data)

    return recorder
