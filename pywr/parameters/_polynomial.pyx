""" Module contains Polynomial Parameters """
from ._parameters import load_parameter
import numpy as np
cimport numpy as np


cdef class Polynomial1DParameter(Parameter):
    """ Parameter that returns the result of 1D polynomial evaluation

    The input to the polynomial can be either:
     - The previous flow of the attached node (default)
     - The previous flow of another `AbstractNode`
     - The current storage of an `AbstractStorage` node
     - The current value of another `Parameter`

    Parameters
    ----------
    node : `AbstractNode`
        An optional `AbstractNode` the flow of which is used to evaluate the polynomial.
    storage_node : `Storage`
        An optional `Storage` node the volume of which is used to evaluate the polynomial.
    parameter : iterable of Parameter objects or single Parameter
        An optional `Parameter` the value of which is used to evaluate the polynomial.
    use_proportional_volume : bool
        An optional boolean only used with a `Storage` node to switch between using absolute
         or proportional volume when evaluating the polynomial.
    """
    def __init__(self, coefficients, *args, **kwargs):
        self.coefficients = np.array(coefficients, dtype=np.float64)
        self._other_node = kwargs.pop('node', None)
        self._storage_node = kwargs.pop('storage_node', None)
        self._parameter = kwargs.pop('parameter', None)
        self.use_proportional_volume = kwargs.pop('use_proportional_volume', False)
        # Check only one of the above is given
        arg_check = [
            self._other_node is not None,
            self._storage_node is not None,
            self._parameter is not None,
        ]
        # Check we haven't been given an ambiguous number of objects
        if arg_check.count(True) > 1:
            raise ValueError('Only one of "node", "storage_node" or "parameter" keywords should be given.')
        super(Polynomial1DParameter, self).__init__(*args, **kwargs)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        cdef int i
        cdef double x, y

        # Get the 'x' value to put in the polynomial
        if self._parameter is not None:
            x = self._parameter.value(ts, scenario_index)
        elif self._storage_node is not None:
            if self.use_proportional_volume:
                x = self._storage_node.current_pc[scenario_index._global_id]
            else:
                x = self._storage_node.volume[scenario_index._global_id]
        elif self._other_node is not None:
            x = self._other_node.flow[scenario_index._global_id]
        else:
            x = self._node.flow[scenario_index._global_id]

        y = 0.0
        for i in range(self.coefficients.shape[0]):
            y += self.coefficients[i]*x**i
        return y

    @classmethod
    def load(cls, model, data):
        node = None
        if 'node' in data:
            node = model._get_node_from_ref(model, data["node"])
        storage_node = None
        if 'storage_node' in data:
            storage_node = model._get_node_from_ref(model, data["storage_node"])
        parameter = None
        if 'parameter' in data:
            parameter = load_parameter(model, data["parameter"])

        coefficients = data.pop("coefficients")
        use_proportional_volume = data.pop("use_proportional_volume", False)
        parameter = cls(coefficients, node=node, storage_node=storage_node, parameter=parameter,
                        use_proportional_volume=use_proportional_volume)
        return parameter
Polynomial1DParameter.register()