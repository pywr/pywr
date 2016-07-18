import numpy as np
cimport numpy as np

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
        return '<{} "{}" ({})>'.format(self.__class__.__name__, self.name, hex(id(self)))

    def __str__(self):
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
            node = model.nodes[node_name]
            data["node"] = node
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
        return '<{} on {} "{}" ({})>'.format(self.__class__.__name__, repr(self.node), self.name, hex(id(self)))

    def __str__(self):
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
        return '<{} on {} "{}" ({})>'.format(self.__class__.__name__, repr(self.node), self.name, hex(id(self)))

    def __str__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.node, self.name)
recorder_registry.add(StorageRecorder)


cdef class ParameterRecorder(Recorder):
    def __init__(self, model, Parameter param, name=None):
        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), param.name)
        Recorder.__init__(self, model, name=name)
        self._param = param
        param._recorders.append(self)

    property parameter:
        def __get__(self):
            return self._param

    def __repr__(self):
        return '<{} on {} "{}" ({})>'.format(self.__class__.__name__, repr(self.parameter), self.name, hex(id(self)))

    def __str__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.parameter, self.name)
recorder_registry.add(ParameterRecorder)


cdef class IndexParameterRecorder(Recorder):
    def __init__(self, model, IndexParameter param, name=None):
        if name is None:
            name = "{}.{}".format(self.__class__.__name__.lower(), param.name)
        Recorder.__init__(self, model, name=name)
        self._param = param
        param._recorders.append(self)

    property parameter:
        def __get__(self):
            return self._param

    def __repr__(self):
        return '<{} on {} "{}" ({})>'.format(self.__class__.__name__, repr(self.parameter), self.name, hex(id(self)))

    def __str__(self):
        return '<{} on {} "{}">'.format(self.__class__.__name__, self.parameter, self.name)
recorder_registry.add(IndexParameterRecorder)


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
recorder_registry.add(NumpyArrayParameterRecorder)

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
recorder_registry.add(NumpyArrayIndexParameterRecorder)
