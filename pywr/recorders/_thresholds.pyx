cimport numpy as np
import numpy as np


cdef enum Predicates:
    LT = 0
    GT = 1
    EQ = 2
    LE = 3
    GE = 4
_predicate_lookup = {
    "LT": Predicates.LT, "<": Predicates.LT,
    "GT": Predicates.GT, ">": Predicates.GT,
    "EQ": Predicates.EQ, "=": Predicates.EQ,
    "LE": Predicates.LE, "<=": Predicates.LE,
    "GE": Predicates.GE, ">=": Predicates.GE,
}

cdef int compare(double x, double threshold, int predicate) except? -1:
    """Returns 1 if the predicate evalutes True, else 0"""

    if predicate == Predicates.LT:
        ind = x < threshold
    elif predicate == Predicates.GT:
        ind = x > threshold
    elif predicate == Predicates.LE:
        ind = x <= threshold
    elif predicate == Predicates.GE:
        ind = x >= threshold
    else:
        ind = x == threshold
    return ind


cdef class StorageThresholdRecorder(StorageRecorder):
    """ Recorder for tracking state of Storage volume against a threshold.

    Parameters
    ----------
    node : Storage
        Storage instance to compare with the threshold.
    threshold : double
        Threshold to compare the value of the recorder to
    values : iterable of doubles
        If the predicate evaluates False the zeroth value is returned,
        otherwise the first value is returned.
    predicate : string
        One of {"LT", "GT", "EQ", "LE", "GE"}.


    """
    def __init__(self, model, Storage node, threshold, *args,  predicate=None, **kwargs):
        super(StorageThresholdRecorder, self).__init__(model, node, *args, **kwargs)
        self.threshold = threshold

        if predicate is None:
            predicate = Predicates.LT
        elif isinstance(predicate, str):
            predicate = _predicate_lookup[predicate.upper()]
        self.predicate = predicate

    cpdef setup(self):
        self._state = np.zeros(len(self.model.scenarios.combinations), dtype=np.int32)

    cpdef reset(self):
        self._state[...] = 0

    cpdef double[:] values(self) except *:
        return np.array(self._state, dtype=np.float64)

    cpdef after(self):
        cdef double volume
        cdef ScenarioIndex scenario_index
        for scenario_index in self.model.scenarios.combinations:
            volume = self._node._volume[scenario_index.global_id]
            self._state[scenario_index.global_id] = compare(volume, self.threshold, self.predicate)
        return 0


cdef class NodeThresholdRecorder(NodeRecorder):
    """ Recorder for tracking state of Node flow against a threshold.

    Parameters
    ----------
    node : Node
        Node instance to compare with the threshold.
    threshold : double
        Threshold to compare the value of the recorder to
    values : iterable of doubles
        If the predicate evaluates False the zeroth value is returned,
        otherwise the first value is returned.
    predicate : string
        One of {"LT", "GT", "EQ", "LE", "GE"}.


    """
    def __init__(self, model, AbstractNode node, threshold, *args,  predicate=None, **kwargs):
        super(NodeThresholdRecorder, self).__init__(model, node, *args, **kwargs)
        self.threshold = threshold

        if predicate is None:
            predicate = Predicates.LT
        elif isinstance(predicate, str):
            predicate = _predicate_lookup[predicate.upper()]
        self.predicate = predicate

    cpdef setup(self):
        self._state = np.zeros(len(self.model.scenarios.combinations), dtype=np.int32)

    cpdef reset(self):
        self._state[...] = 0

    cpdef double[:] values(self) except *:
        return np.array(self._state, dtype=np.float64)

    cpdef after(self):
        cdef double flow
        cdef ScenarioIndex scenario_index
        for scenario_index in self.model.scenarios.combinations:
            flow = self._node._flow[scenario_index.global_id]
            self._state[scenario_index.global_id] = compare(flow, self.threshold, self.predicate)
        return 0



