from typing_extensions import TYPE_CHECKING
from typing import Iterable, Optional

if TYPE_CHECKING:
    from ..core import ScenarioIndex, Timestep, Model, Node

from ..parameter_property import parameter_property
from ._parameters import (
    Parameter,
    parameter_registry,
    UnutilisedDataWarning,
    ConstantParameter,
    ConstantScenarioParameter,
    ConstantScenarioIndexParameter,
    AnnualHarmonicSeriesParameter,
    ArrayIndexedParameter,
    ConstantScenarioParameter,
    IndexedArrayParameter,
    ArrayIndexedScenarioMonthlyFactorsParameter,
    TablesArrayParameter,
    DailyProfileParameter,
    MonthlyProfileParameter,
    WeeklyProfileParameter,
    ArrayIndexedScenarioParameter,
    ScenarioMonthlyProfileParameter,
    ScenarioDailyProfileParameter,
    ScenarioWeeklyProfileParameter,
    align_and_resample_dataframe,
    DataFrameParameter,
    IndexParameter,
    AggregatedParameter,
    AggregatedIndexParameter,
    PiecewiseIntegralParameter,
    DiscountFactorParameter,
    NegativeParameter,
    MaxParameter,
    NegativeMaxParameter,
    MinParameter,
    NegativeMinParameter,
    DeficitParameter,
    DivisionParameter,
    FlowParameter,
    FlowDelayParameter,
    StorageParameter,
    OffsetParameter,
    RbfProfileParameter,
    UniformDrawdownProfileParameter,
    RollingMeanFlowNodeParameter,
    load_parameter,
    load_parameter_values,
    load_dataframe,
)

from . import licenses
from ._polynomial import Polynomial1DParameter, Polynomial2DStorageParameter
from ._thresholds import (
    AbstractThresholdParameter,
    StorageThresholdParameter,
    NodeThresholdParameter,
    ParameterThresholdParameter,
    RecorderThresholdParameter,
    CurrentYearThresholdParameter,
    CurrentOrdinalDayThresholdParameter,
    MultipleThresholdIndexParameter,
    MultipleThresholdParameterIndexParameter,
)
from ._hydropower import HydropowerTargetParameter
from ._activation_functions import (
    BinaryStepParameter,
    RectifierParameter,
    LogisticParameter,
)  # noqa
from .control_curves import (
    ControlCurveParameter,
    BaseControlCurveParameter,
    ControlCurveInterpolatedParameter,
    ControlCurveIndexParameter,
    ControlCurvePiecewiseInterpolatedParameter,
    WeightedAverageProfileParameter,
)
from .multi_model_parameters import (
    OtherModelParameterValueParameter,
    OtherModelNodeFlowParameter,
    OtherModelNodeStorageParameter,
    OtherModelIndexParameterValueIndexParameter,
)
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import pandas


class FunctionParameter(Parameter):
    def __init__(self, model, parent, func, *args, **kwargs):
        super(FunctionParameter, self).__init__(model, *args, **kwargs)
        self._parent = parent
        self._func = func

    def value(self, ts, scenario_index):
        return self._func(self._parent, ts, scenario_index)


FunctionParameter.register()


class ScaledProfileParameter(Parameter):
    def __init__(self, model, scale, profile, *args, **kwargs):
        super(ScaledProfileParameter, self).__init__(model, *args, **kwargs)
        self.scale = scale

        profile.parents.add(self)
        self.profile = profile

    @classmethod
    def load(cls, model, data):
        scale = float(data.pop("scale"))
        profile = load_parameter(model, data.pop("profile"))
        return cls(model, scale, profile, **data)

    def value(self, ts, si):
        p = self.profile.get_value(si)
        return self.scale * p


ScaledProfileParameter.register()


class AbstractInterpolatedParameter(Parameter):
    def __init__(self, model, x, y, interp_kwargs=None, **kwargs):
        super(AbstractInterpolatedParameter, self).__init__(model, **kwargs)
        self.x = x
        self.y = y
        self.interp = None
        default_interp_kwargs = dict(kind="linear", bounds_error=True)
        if interp_kwargs is not None:
            # Overwrite or add to defaults with given values
            default_interp_kwargs.update(interp_kwargs)
        self.interp_kwargs = default_interp_kwargs

    def _value_to_interpolate(self, ts, scenario_index):
        raise NotImplementedError()

    @property
    def interp_kwargs(self):
        return self._interp_kwargs

    @interp_kwargs.setter
    def interp_kwargs(self, data):
        if "fill_value" in data and isinstance(data["fill_value"], list):
            # SciPy's interp1d expects a tuple when defining fill values for the upper and lower bounds
            data["fill_value"] = tuple(data["fill_value"])
        self._interp_kwargs = data

    def setup(self):
        super(AbstractInterpolatedParameter, self).setup()
        self.interp = interp1d(self.x, self.y, **self.interp_kwargs)

    def value(self, ts, scenario_index):
        v = self._value_to_interpolate(ts, scenario_index)
        return self.interp(v)


class InterpolatedParameter(AbstractInterpolatedParameter):
    """Given a function with discrete data points x and y coordinates,
    this parameter interpolates over the function using the value from another
    parameter.

    Notes
    -----
    The interpolation relies on the `interp1d` function in the Scipy package.
    By default, the interpolation method defaults to linear and the `bounds_error`
    option is set to True. This means that a `ValueError` is raised any time the
    interpolation is attempted on a value outside the range of x.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.parameters import ConstantParameter, InterpolatedParameter
    x = [0, 5, 10, 20]
    y = [0, 10, 30, -5]
    model = Model()
    p1 = ConstantParameter(model=model, value=9.3, name="p1")
    p2 = InterpolatedParameter(
        model=model,
        parameter=p1,
        x=x,
        y=y,
        interp_kwargs={"kind": "linear"}
    )
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "InterpolatedParameter",
            "parameter": "p1",
            "x": [0, 5, 10, 20],
            "y": [0, 10, 30, -5],
            "interp_kwargs": {"kind": "linear"}
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    parameter : Parameter
        The parameter instance whose value is used in the interpolation.
    x : Iterable[float]
        The x coordinates of the data points for interpolation.
    y : Iterable[float]
        The y coordinates of the data points for interpolation.
    interp_kwargs : dict
        The scipy.interp1d keyword arguments.
    interp : scipy.interpolate.interp1d
        The interpolation instance.
    """

    def __init__(
        self,
        model: "Model",
        parameter: "Parameter",
        x: Iterable[float],
        y: Iterable[float],
        interp_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Initialise the parameter.

        Parameters
        ----------
        model : Model
            The model instance.
        parameter : Parameter
            The parameter instance whose value is used in the interpolation.
        x : Iterable[float]
            The x coordinates of the data points for interpolation.
        y : Iterable[float]
            The y coordinates of the data points for interpolation.
        interp_kwargs : Optional[dict]
            The option to pass to the scipy.interp1d keyword arguments.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#interp1d
            The interpolation method defaults to linear with exception raised when the parameter value
            is outside the range of x.
        """
        super(InterpolatedParameter, self).__init__(
            model, x, y, interp_kwargs, **kwargs
        )
        self._parameter = None
        self.parameter = parameter

    parameter = parameter_property("_parameter")

    def _value_to_interpolate(self, ts, scenario_index):
        return self._parameter.get_value(scenario_index)

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
        InterpolatedParameter
            The loaded class.
        """
        parameter = load_parameter(model, data.pop("parameter"))
        x = np.array(data.pop("x"))
        y = np.array(data.pop("y"))
        interp_kwargs = data.pop("interp_kwargs", None)
        return cls(model, parameter, x, y, interp_kwargs=interp_kwargs, **data)


InterpolatedParameter.register()


class InterpolatedVolumeParameter(AbstractInterpolatedParameter):
    """
    Given a function between volume data points (as x coordinates) as some
    values (as y coordinates), this parameter interpolates over the given
    relationship using the current storage from a node.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import InterpolatedParameter

    model = Model()
    storage = Storage(model=model, max_volume=500, name="Storage")
    p2 = InterpolatedVolumeParameter(
        model=model,
        node=storage,
        volumes=[100, 200, 340, 450, 510],
        values=[0.1, 0.32, 0.98, 1.2, 2.5],
        interp_kwargs={"kind": "quadratic"}
    )
    ```

    JSON
    ======

    ```json
    {
        "My parameter": {
            "type": "InterpolatedVolumeParameter",
            "node": "Storage",
            "volumes": [100, 200, 340, 450, 510],
            "values": [0.1, 0.32, 0.98, 1.2, 2.5],
            "interp_kwargs": {"kind": "quadratic"}
        }
    }
    ```

    The x and y coordinates can also be loaded from a file in the JSON document:
    ```json
    {
        "My parameter": {
            "type": "InterpolatedVolumeParameter",
            "node": "Storage",
            "volumes": {
                "table": "Storage table",
                "column": "Volume"
            },
            "values": {
                "table": "Storage table",
                "column": "Area"
            },
            "interp_kwargs": {"kind": "quadratic"}
        }
    }
    ```

    Attributes
    ----------
    x : Iterator[float]
        The x coordinates of the data points for interpolation.
    y : Iterator[float]
        The y coordinates of the data points for interpolation.
    interp_kwargs : dict
        The scipy.interp1d keyword arguments.
    interp : scipy.interpolate.interp1d
        The interpolation instance.
    """

    def __init__(
        self,
        model: "Model",
        node: "Node",
        volumes: Iterable[float],
        values: Iterable[float],
        interp_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Initialise the parameter.

        Parameters
        ----------
        model : Model
            The model instance.
        node: Node
            Storage node to provide input volume values to interpolation calculation.
        volumes : Iterable[float]
            The x coordinates of the data points for interpolation.
        values : Iterable[float]
            The y coordinates of the data points for interpolation.
        interp_kwargs : Optional[dict]
            The option to pass to the scipy.interp1d keyword arguments.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#interp1d
            The interpolation method defaults to linear with exception when the storage
            is outside the range of x.
        """
        super(InterpolatedVolumeParameter, self).__init__(
            model, volumes, values, interp_kwargs, **kwargs
        )
        self._node = node

    def _value_to_interpolate(self, ts, scenario_index):
        return self._node.volume[scenario_index.global_id]

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
        InterpolatedVolumeParameter
            The loaded class.
        """
        node = model.nodes[data.pop("node")]
        volumes = data.pop("volumes")
        if isinstance(volumes, list):
            volumes = np.asarray(volumes, np.float64)
        elif isinstance(volumes, dict):
            volumes = load_parameter_values(model, volumes)
        else:
            raise TypeError('Unexpected type for "volumes" in {}'.format(cls.__name__))
        values = data.pop("values")
        if isinstance(values, list):
            values = np.asarray(values, np.float64)
        elif isinstance(values, dict):
            values = load_parameter_values(model, values)
        else:
            raise TypeError('Unexpected type for "values" in {}'.format(cls.__name__))
        interp_kwargs = data.pop("interp_kwargs", None)
        return cls(model, node, volumes, values, interp_kwargs=interp_kwargs, **data)


InterpolatedVolumeParameter.register()


class InterpolatedFlowParameter(AbstractInterpolatedParameter):
    """
    Given a function between flow data points (as x coordinates) as some
    values (as y coordinates), this parameter interpolates over the given
    relationship using a node's flow at the previous time-step.

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.nodes import Link
    from pywr.parameters import InterpolatedFlowParameter

    model = Model()
    link = Link(model=model, max_flow=20, name="Link")
    parameter = InterpolatedFlowParameter(
        model=model,
        node=link,
        flows=[0, 5, 10, 20],
        values=[0, 10, 30, -5],
        interp_kwargs={"kind": "linear"}
    )
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "InterpolatedFlowParameter",
            "node": "Link",
            "flows": [0, 5, 10, 20],
            "values": [0, 10, 30, -5],
            "interp_kwargs": {"kind": "linear"}
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    x : Iterable[float]
        The x coordinates of the data points for interpolation.
    y : Iterable[float]
        The y coordinates of the data points for interpolation.
    interp_kwargs : dict
        The scipy.interp1d keyword arguments.
    interp : scipy.interpolate.interp1d
        The interpolation instance.
    """

    def __init__(
        self,
        model: "Model",
        node: "Node",
        flows: Iterable[float],
        values: Iterable[float],
        interp_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Initialise the parameter.

        Parameters
        ----------
        model : Model
            The model instance.
        node : Node
            The node instance whose flow is used in the interpolation.
        flows : Iterable[float]
            The x coordinates of the data points for interpolation.
        values : Iterable[float]
            The y coordinates of the data points for interpolation.
        interp_kwargs : Optional[dict]
            The option to pass to the scipy.interp1d keyword arguments.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html#interp1d
            The interpolation method defaults to linear with exception when the parameter value
            is outside the range of x.
        """
        super().__init__(model, flows, values, interp_kwargs, **kwargs)
        self._node = node

    def _value_to_interpolate(self, ts, scenario_index):
        return self._node.prev_flow[scenario_index.global_id]

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
        InterpolatedFlowParameter
            The loaded class.
        """
        node = model.nodes[data.pop("node")]
        flows = np.array(data.pop("flows"))
        values = np.array(data.pop("values"))
        interp_kwargs = data.pop("interp_kwargs", None)
        return cls(model, node, flows, values, interp_kwargs=interp_kwargs, **data)


InterpolatedFlowParameter.register()


class InterpolatedQuadratureParameter(AbstractInterpolatedParameter):
    """This parameter integrates a function over an interval. The function is defined
    as a set of x and y coordinate, whereas the interval is defined using the values from
    two parameters, one that returns that lower limit and the other that returns
    the upper limit. When the x value needed for the integration is not available
    in the given x values, the value for y is interpolated.

    Notes
    -----
    The integration is performed using [scipy.integrate.quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#quad)

    Examples
    -------
    Python
    ======
    ```python
    from pywr.core import Model
    from pywr.parameters import (
        ConstantParameter,
        InterpolatedQuadratureParameter
    )

    x =  [0, 5, 10, 20]
    y = [0, 10, 30, -5]
    model = Model()
    p1 = ConstantParameter(model=model, value=9.3, name="p1")
    p2 = InterpolatedQuadratureParameter(
        model=model,
        upper_parameter=p1,
        x=x,
        y=y,
        interp_kwargs={"kind": "linear"}
    )
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "InterpolatedQuadratureParameter",
            "upper_parameter": "p1",
            "x": [0, 5, 10, 20],
            "y": [0, 10, 30, -5],
            "interp_kwargs": {"kind": "linear"}
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    x : Iterable[float]
        The x coordinates of the data points for interpolation.
    y : Iterable[float]
        The y coordinates of the data points for interpolation.
    lower_parameter : Optional[Parameter]
        Lower value of the interpolation interval to integrate
        over. It can be `None` in which case the lower value of the interval is zero.
    upper_parameter : Parameter
        Upper value of the interpolated interval to integrate over.
    interp_kwargs : dict
        The scipy.interp1d keyword arguments.
    interp : scipy.interpolate.interp1d
        The interpolation instance.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.
    """

    def __init__(
        self,
        model: "Model",
        upper_parameter: Parameter,
        x: Iterable[float],
        y: Iterable[float],
        lower_parameter: Optional[Parameter] = None,
        interp_kwargs: dict = None,
        **kwargs,
    ):
        """Initialise the parameter.

        Parameters
        ----------
        model : Model
            The model instance.
        upper_parameter : Parameter
            Upper value of the interpolated interval to integrate over.
        x : Iterable[float]
            The x coordinates of the data points for interpolation.
        y : Iterable[float]
            The y coordinates of the data points for interpolation.
        lower_parameter : Optional[Parameter]
            Lower value of the interpolation interval to integrate
            over. Can be `None` in which case the lower value of the interval is zero.
        interp_kwargs : dict
            The scipy.interp1d keyword arguments.
        """
        super().__init__(model, x, y, interp_kwargs, **kwargs)
        self._upper_parameter = None
        self.upper_parameter = upper_parameter
        self._lower_parameter = None
        self.lower_parameter = lower_parameter

    upper_parameter = parameter_property("_upper_parameter")
    lower_parameter = parameter_property("_lower_parameter")

    def _value_to_interpolate(self, ts, scenario_index):
        return self._upper_parameter.get_value(scenario_index)

    def value(self, ts: "Timestep", scenario_index: "ScenarioIndex") -> float:
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
        a = 0
        if self._lower_parameter is not None:
            a = self._lower_parameter.get_value(scenario_index)
        b = self._value_to_interpolate(ts, scenario_index)

        cost, err = quad(self.interp, a, b)
        return cost

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
        InterpolatedQuadratureParameter
            The loaded class.
        """
        upper_parameter = load_parameter(model, data.pop("upper_parameter"))
        lower_parameter = load_parameter(model, data.pop("lower_parameter", None))
        x = np.array(data.pop("x"))
        y = np.array(data.pop("y"))
        interp_kwargs = data.pop("interp_kwargs", None)
        return cls(
            model,
            upper_parameter,
            x,
            y,
            lower_parameter=lower_parameter,
            interp_kwargs=interp_kwargs,
            **data,
        )


InterpolatedQuadratureParameter.register()


class ScenarioWrapperParameter(Parameter):
    """This parameter uses a different parameter depending on the scenario ensemble being modelled
    in a given `Scenario`. It can be used to vary data in a non-scenario-aware
    parameter type across multiple scenario ensembles. For example, many of the control curves or
    interpolation parameters do not explicitly support scenarios. This parameter can be used
    to test multiple control curve definitions as part of a single simulation.

    Examples
    -------
    In the example below, when the model runs the first scenario, the first control curve
    parameter `p1` is used, otherwise `p2` is used to assign the cost to the reservoir node.

    Python
    ======
    ```python
    from pywr.core import Model, Scenario
    from pywr.nodes import Storage
    from pywr.parameters import ScenarioWrapperParameter, ControlCurveInterpolatedParameter, ConstantParameter

    model = Model()
    scenario = Scenario(
        model=model,
        name="Demand",
        size=2,
        ensemble_names=["Low demand", "High demand"]
    )
    storage_node = Storage(
        model=model,
        name="reservoir",
        max_volume=100,
        initial_volume=100
    )
    p1 = ControlCurveInterpolatedParameter(
        name="CC1",
        storage_node=storage_node,
        control_curves=[ConstantParameter(model, 0.5), ConstantParameter(model, 0.3)],
        values=[0.0, -5.0, -10.0, -20.0]
    )
    p2 = ControlCurveInterpolatedParameter(
        name="CC2",
        storage_node=storage_node,
        control_curves=[ConstantParameter(model, 0.3), ConstantParameter(model, 0.1)],
        values=[0.0, -5.0, -10.0, -20.0]
    )
    storage.cost = ScenarioWrapperParameter(model=model, scenario=scenario, parameters[p1, p2])
    ```

    JSON
    ======
    ```json
    {
        "My parameter": {
            "type": "ScenarioWrapperParameter",
            "scenario": "Demand",
            "parameters": ["CC1,", "CC2"]
        }
    }
    ```

    Attributes
    ----------
    model : Model
        The model instance.
    scenario : Scenario
        The scenario instance which is used to select the parameters.
    parameters : Iterable[Parameter]
        The child parameters that are used in each of `scenario`'s ensembles.
    name : Optional[str]
        The name of the parameter.
    comment : Optional[str]
        An optional comment for the parameter.
    tags : Optional[dict]
        An optional container of key-value pairs that the user can set to help group and identify parameters.

    """

    def __init__(
        self,
        model: "Model",
        scenario: "Scenario",
        parameters: list["Parameter"],
        **kwargs,
    ):
        """Initialise the class.

        Parameters
        ----------
        model : Model
            The model instance.
        scenario : Scenario
            The scenario instance which is used to select the parameters.
        parameters : Iterable[Parameter]
            The child parameters that are used in each of `scenario`'s ensembles. The number
            of parameters must equal the size of the given scenario.

        Other Parameters
        ----------------
        name : Optional[str], default=None
            The name of the parameter.
        comment : Optional[str], default=None
            An optional comment for the parameter.
        tags : Optional[dict], default=None
            An optional container of key-value pairs.

        Raises
        ------
        ValueError
            If the number of parameters is different from the scenario size.
        """
        super().__init__(model, **kwargs)
        if scenario.size != len(parameters):
            raise ValueError(
                "The number of parameters must equal the size of the scenario."
            )
        self.scenario = scenario
        self.parameters = []
        for p in parameters:
            self.children.add(p)
            self.parameters.append(p)
        # Initialise internal attributes
        self._scenario_index = None

    def setup(self):
        """Setup the internal variables."""
        super().setup()
        # This setup must find out the index of self._scenario in the model
        # so that it can return the correct value in value()
        self._scenario_index = self.model.scenarios.get_scenario_index(self.scenario)

    def value(self, ts: "Timestep", scenario_index: "ScenarioIndex") -> float:
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
        # This is a bit confusing.
        # scenario_indices contains the current scenario number for all
        # the Scenario objects in the model run. We have cached the
        # position of self._scenario in self._scenario_index to lookup the
        # correct number to use in this instance.
        parameter = self.parameters[scenario_index.indices[self._scenario_index]]
        return parameter.get_value(scenario_index)

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
        ScenarioWrapperParameter
            The loaded class.
        """
        scenario = model.scenarios[data.pop("scenario")]

        parameters = [load_parameter(model, p) for p in data.pop("parameters")]
        return cls(model, scenario, parameters, **data)


ScenarioWrapperParameter.register()


def pop_kwarg_parameter(kwargs, key, default):
    """Pop a parameter from the keyword arguments dictionary

    Parameters
    ----------
    kwargs : dict
        A keyword arguments dictionary
    key : string
        The argument name, e.g. 'flow'
    default : object
        The default value to use if the dictionary does not have that key

    Returns a Parameter
    """
    value = kwargs.pop(key, default)
    if isinstance(value, Parameter):
        return value
    elif callable(value):
        # TODO this is broken?
        return FunctionParameter(self, value)
    else:
        return value


class PropertiesDict(dict):
    def __setitem__(self, key, value):
        if not isinstance(value, Property):
            value = ConstantParameter(value)
        dict.__setitem__(self, key, value)
