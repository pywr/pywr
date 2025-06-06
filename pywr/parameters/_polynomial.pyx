""" Module contains Polynomial Parameters """
from ._parameters import load_parameter
import numpy as np
cimport numpy as np


cdef class Polynomial1DParameter(Parameter):
    """This parameter returns the result of a 1D polynomial evaluation. The input to the polynomial
    can be either:
    
    - The previous flow of a node.
    - The current storage of a storage node.
    - The current value of parameter.

    The degree of the polynomial is determined based on the number of given coefficients minus 1.

    Examples
    -------
    The following example use the parameter to find the surface area of a reservoir based 
    on the current storage. The area-volume relationship is represented using the following
    2-degree polynomial:

    area = 19.1 + 123 * x + 12 * x<sup>1</sup> + 0.192 * x<sup>2</sup>

    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes imort Storage
    from pywr.parameters import Polynomial1DParameter

    model = Model()
    storage_node = Storage(
        model=model,
        name="reservoir",
        max_volume=100, 
        initial_volume=100
    )
    Polynomial1DParameter(
        model=model, 
        storage=storage,
        name="My parameter", 
        coefficients=[19.1, 123, 12, 0.192], 
        use_proportional_volume=True
    )
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "Polynomial1DParameter",
            "storage" "reservoir"
            "coefficients": [19.1, 123, 12, 0.192],
            "use_proportional_volume": true
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    coefficients : Iterable[float]
        The 1 dimensional array of polynomial coefficients. The first item is the polynomial
        constant, the second item is the coefficient associated to x<sup>1</sup>, the third 
        item to x<sup>2</sup>, and so on.
    node : Optional[Node]
        An optional [pywr.core.Node][] the flow of which is used to evaluate the polynomial.
    storage_node : Optional[Storage]
        An optional [pywr.core.Storage][] node the volume of which is used to evaluate the polynomial.
    parameter : Optional[Parameter]
        An optional [pywr.parameters.Parameter][] the value of which is used to evaluate the polynomial.
    use_proportional_volume : bool
        An optional boolean only used with a storage node to switch between using absolute
        or proportional volume when evaluating the polynomial.
    scale : float
        An optional scaling factor to apply to the polynomial input before calculation. This is
        applied before any offset.
    offset : float
        An optional offset to apply to the polynomial input before calculation. This is applied after
        and scaling.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """
    def __init__(self, model, coefficients, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        coefficients : Iterable[float]
            The 1 dimensional array of polynomial coefficients. The first item is the polynomial
            constant, the second item is the coefficient associated to x<sup>1</sup>, the third 
            item to x<sup>2</sup>, and so on.
        
        Other parameters
        ----------
        node : Optional[Node], default=None
            An optional [pywr.core.Node][] the flow of which is used to evaluate the polynomial.
        storage_node : Optional[Storage], default=None
            An optional [pywr.core.Storage][] node the volume of which is used to evaluate the polynomial.
        parameter : Optional[Parameter], default=None
            An optional [pywr.parameters.Parameter][] the value of which is used to evaluate the polynomial.
        use_proportional_volume : bool, default=False
            An optional boolean only used with a storage node to switch between using absolute
            or proportional volume when evaluating the polynomial.
        scale : float, default=1
            An optional scaling factor to apply to the polynomial input before calculation. This is
            applied before any offset.
        offset : float, default=0
            An optional offset to apply to the polynomial input before calculation. This is applied after
            scaling.
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        
        Raises
        ------
        ValueError
            If the `node` or `storage_node` or `parameter` is not provided.
        """
        self.coefficients = np.array(coefficients, dtype=np.float64)
        self._other_node = kwargs.pop('node', None)
        self._storage_node = kwargs.pop('storage_node', None)
        self._parameter = kwargs.pop('parameter', None)
        self.use_proportional_volume = kwargs.pop('use_proportional_volume', False)
        self.offset = kwargs.pop('offset', 0.0)
        self.scale = kwargs.pop('scale', 1.0)
        # Check only one of the above is given
        arg_check = [
            self._other_node is not None,
            self._storage_node is not None,
            self._parameter is not None,
        ]
        # Check we haven't been given an ambiguous number of objects
        if arg_check.count(True) > 1:
            raise ValueError('Only one of "node", "storage_node" or "parameter" keywords should be given.')

        super(Polynomial1DParameter, self).__init__(model, *args, **kwargs)

        # Finally register parent relationship if parameter is given
        if self._parameter is not None:
            self.children.add(self._parameter)


    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        cdef int i
        cdef double x, y

        # Get the 'x' value to put in the polynomial
        if self._parameter is not None:
            x = self._parameter._Parameter__values[scenario_index.global_id]
        elif self._storage_node is not None:
            if self.use_proportional_volume:
                x = self._storage_node.get_current_pc(scenario_index)
            else:
                x = self._storage_node._volume[scenario_index.global_id]
        elif self._other_node is not None:
            x = self._other_node._flow[scenario_index.global_id]
        else:
            x = self._node._flow[scenario_index.global_id]

        # Apply scaling and offset
        x = x*self.scale + self.offset
        # No calculate polynomial
        y = 0.0
        for i in range(self.coefficients.shape[0]):
            y += self.coefficients[i]*x**i
        return y

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        Polynomial1DParameter
            The loaded class.
        """
        node = None
        if 'node' in data:
            node = model.nodes[data.pop("node")]
        storage_node = None
        if 'storage_node' in data:
            storage_node = model.nodes[data.pop("storage_node")]
        parameter = None
        if 'parameter' in data:
            parameter = load_parameter(model, data.pop("parameter"))

        coefficients = data.pop("coefficients")
        parameter = cls(model, coefficients, node=node, storage_node=storage_node, parameter=parameter, **data)
        return parameter
Polynomial1DParameter.register()


cdef class Polynomial2DStorageParameter(Parameter):
    """This parameter returns the result of 2D polynomial evaluation where the two independent 
    variables are the volume of a storage node and the current value of a parameter respectively.
    
    Attributes
    ----------
    model : Model
        The model instance.
    coefficients : numpy.typing.NDArray[numpy.number]
        The 2 dimensional array of polynomial coefficients. Index (0, 0) is the polynomial
        constant, index (1, 0) the coefficient for x<sub>1</sub><sup>1</sup> and x<sub>2</sub><sup>0</sup>,
        index (1, 1) the coefficient for x<sub>1</sub><sup>1</sup> and x<sub>2</sub><sup>1</sup>,
        and so on.
    use_proportional_volume : bool
        An optional boolean only used with a storage` node to switch between using absolute
        or proportional volume when evaluating the polynomial.
    storage_scale : float
        A scaling factor to apply to the storage value before calculation. This is
        applied before any offset.
    storage_offset : float
        An offset to apply to the storage value before calculation. This is applied after
        and scaling
    parameter_scale : float
        An scaling factor to apply to the parameter value before calculation. This is
        applied before any offset.
    parameter_offset : float
        An offset to apply to the parameter value before calculation. This is applied after
        and scaling.
    """
    def __init__(self, model, coefficients, storage_node, parameter, *args, **kwargs):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        coefficients : numpy.typing.NDArray[numpy.number]
            The 2 dimensional array of polynomial coefficients. Index (0, 0) is the polynomial
            constant, index (1, 0) the coefficient for x<sub>1</sub><sup>1</sup> and x<sub>2</sub><sup>0</sup>,
            index (1, 1) the coefficient for x<sub>1</sub><sup>1</sup> and x<sub>2</sub><sup>1</sup>,
            and so on.
        storage_node : Storage
            A storage node the volume of which is used to evaluate the polynomial (as first independent variable).
        parameter : Parameter
            A parameter the value of which is used to evaluate the polynomial (as second independent variable).
        

        Other parameters
        ----------
        use_proportional_volume : Optional[bool], default=False
            An optional boolean only used with a storage` node to switch between using absolute
            or proportional volume when evaluating the polynomial.
        storage_scale : Optional[float], default=1
            A scaling factor to apply to the storage value before calculation. This is
            applied before any offset.
        storage_offset : Optional[float], default=0
            An offset to apply to the storage value before calculation. This is applied after
            and scaling
        parameter_scale : Optional[float], default=1
            An scaling factor to apply to the parameter value before calculation. This is
            applied before any offset.
        parameter_offset : Optional[float], default=0
            An offset to apply to the parameter value before calculation. This is applied after
            and scaling.
        name : Optional[str]
            The name of the parameter.
        comment : Optional[str]
            An optional comment for the parameter.
        tags : Optional[dict]
            An optional container of key-value pairs that the user can set to help group and identify parameters.
        
        """
        self.coefficients = np.array(coefficients, dtype=np.float64)
        self._storage_node = storage_node
        self._parameter = parameter
        self.use_proportional_volume = kwargs.pop('use_proportional_volume', False)
        self.storage_offset = kwargs.pop('storage_offset', 0.0)
        self.storage_scale = kwargs.pop('storage_scale', 1.0)
        self.parameter_offset = kwargs.pop('parameter_offset', 0.0)
        self.parameter_scale = kwargs.pop('parameter_scale', 1.0)
        super(Polynomial2DStorageParameter, self).__init__(model, *args, **kwargs)

        # Register parameter relationships
        self.children.add(parameter)

    cpdef double value(self, Timestep ts, ScenarioIndex scenario_index) except? -1:
        """Get the parameter value for the given timestep and scenario.

        Parameters
        ----------
        ts : Timestep
            The timestep instance.
        scenario_index : ScenarioIndex
            The scenario index instance.
        
        Returns
        -------
        float
            The parameter value.
        """
        cdef int i, j
        cdef double x, y, z

        # Storage volume is 1st dimension
        if self.use_proportional_volume:
            x = self._storage_node.get_current_pc(scenario_index)
        else:
            x = self._storage_node._volume[scenario_index.global_id]
        # Parameter value is 2nd dimension
        y = self._parameter._Parameter__values[scenario_index.global_id]

        # Apply scaling and offset
        x = self.storage_scale*x + self.storage_offset
        y = self.parameter_scale*y + self.parameter_offset
        z = 0.0
        for i in range(self.coefficients.shape[0]):
            for j in range(self.coefficients.shape[1]):
                z += self.coefficients[i, j]*x**i*y**j
        return z

    @classmethod
    def load(cls, model, data):
        """Load the parameter from the data dictionary (i.e. when the parameter is defined in JSON format).

        Parameters
        ---------
        model : Model
            The model instance.
        data : dict
            The dictionary with the parameter configuration.

        Returns
        -------
        Polynomial2DStorageParameter
            The loaded class.
        """
        storage_node = model.nodes[data.pop("storage_node")]
        parameter = load_parameter(model, data.pop("parameter"))
        coefficients = data.pop("coefficients")
        parameter = cls(model, coefficients, storage_node, parameter, **data)
        return parameter
Polynomial2DStorageParameter.register()
