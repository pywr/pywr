""" This module contains `Parameter` subclasses for performing radial basis function interpolation.

"""
import numpy as np
from scipy.interpolate import Rbf
from .parameters import Parameter, load_parameter
from ..nodes import Storage


class RbfData:
    """Container for Rbf interpolation data.

    This object is intended to be used with `RbfParameter` where one set of data
    is required for each item to be used as an exogenous variable. This object
    contains the interpolation values and data specifying whether this particular
    item is to be considered a variable.

    """
    def __init__(self, values, is_variable=False, upper_bounds=None, lower_bounds=None):
        self.values = values
        self.is_variable = is_variable
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds

    def __len__(self):
        return len(self.values)

    def get_upper_bounds(self):
        if self.upper_bounds is None:
            return None
        return [self.upper_bounds]*len(self.values)

    def get_lower_bounds(self):
        if self.lower_bounds is None:
            return None
        return [self.lower_bounds]*len(self.values)


class RbfParameter(Parameter):
    """ A general Rbf parameter.

    This parameter is designed to perform general multi-dimensional interpolation using
    radial basis functions. It utilises the `scipy.interpolate.Rbf` functionality for evaluation
    of the radial basis function, and is mostly a wrapper around that class.

    Parameters
    ==========

    """
    def __init__(self, model, y, nodes=None, parameters=None, days_of_year=None, rbf_kwargs=None, **kwargs):
        super(RbfParameter, self).__init__(model, **kwargs)

        # Initialise defaults of no data
        if nodes is None:
            nodes = {}

        if parameters is None:
            parameters = {}

        for parameter in parameters.keys():
            # Make these parameter's children.
            self.children.add(parameter)

        # Attributes
        self.nodes = nodes
        self.parameters = parameters
        self.days_of_year = days_of_year
        self.y = y
        self.rbf_kwargs = rbf_kwargs if rbf_kwargs is not None else {}
        self._rbf_func = None
        self._node_order = None
        self._parameter_order = None

    def setup(self):
        super().setup()
        double_size = 0

        for node, x in self.nodes.items():
            if x.is_variable:
                double_size += len(x)

        for parameter, x in self.parameters.items():
            if x.is_variable:
                double_size += len(x)

        if self.days_of_year is not None:
            if self.days_of_year.is_variable:
                double_size += len(self.days_of_year)

        if self.y.is_variable:
            double_size += len(self.y)

        self.double_size = double_size
        if self.double_size > 0:
            self.is_variable = True
        else:
            self.is_variable = False

    def reset(self):
        # Create the Rbf object here.
        # This is done in `reset` rather than `setup` because
        # we wish to have support for optimising some of the Rbf parameters.
        # Therefore it needs recreating each time.
        nodes = self.nodes
        parameters = self.parameters
        days_of_year = self.days_of_year
        y = self.y

        if len(nodes) == 0 and len(parameters) == 0 and days_of_year is None:
            raise ValueError('There must be at least one exogenous variable defined.')

        # Create the arguments for the Rbf function.
        # also cache the order of the nodes, so that when the are evaluated later we know
        # the correct order
        args = []
        node_order = []
        for node, x in nodes.items():
            if len(x) != len(y):
                raise ValueError('The length of the exogenous variables for node "{}"'
                                 ' must be the same as length of "y".'.format(node.name))
            args.append(x.values)
            node_order.append(node)

        parameter_order = []
        for parameter, x in parameters.items():
            if len(x) != len(y):
                raise ValueError('The length of the exogenous variables for parameter "{}"'
                                 ' must be the same as the length of "y".'.format(parameter.name))
            args.append(x.values)
            parameter_order.append(parameter)

        if days_of_year is not None:
            # Normalise DoY using cosine & sine harmonics
            x = [2 * np.pi * (doy - 1) / 365 for doy in self.days_of_year.values]
            args.append(np.sin(x))
            args.append(np.cos(x))

        # Finally append the known y values
        args.append(y.values)

        # Convention here is that DoY is the first independent variable.
        self._rbf_func = Rbf(*args, **self.rbf_kwargs)

        # Save the node and parameter order caches
        self._node_order = node_order
        self._parameter_order = parameter_order

    def set_double_variables(self, values):
        """Assign an array of variables to the interpolation data."""
        N = len(self.y)

        values = np.reshape(values, (-1, N))
        item = 0
        for node, x in self.nodes.items():
            if x.is_variable:
                x.values = values[item, :]
                item += 1

        for parameter, x in self.parameters.items():
            if x.is_variable:
                x.values = values[item, :]
                item += 1

        if self.days_of_year is not None:
            if self.days_of_year.is_variable:
                self.days_of_year.values = values[item, :]
                item += 1

        if self.y.is_variable:
            self.y.values = values[item, :]
            item += 1

        # Make sure all variables have been used.
        assert item == values.shape[0]

    def get_double_variables(self):
        """Get the current values of variable interpolation data."""
        values = []

        for node, x in self.nodes.items():
            if x.is_variable:
                values.extend(x.values)

        for parameter, x in self.parameters.items():
            if x.is_variable:
                values.extend(x.values)

        if self.days_of_year is not None:
            if self.days_of_year.is_variable:
                values.extend(self.days_of_year.values)

        if self.y.is_variable:
            values.extend(self.y.values)

        return np.array(values)

    def get_double_upper_bounds(self):
        """Returns an array of the upper bounds of the variables."""
        values = []

        for node, x in self.nodes.items():
            if x.is_variable:
                values.extend(x.get_upper_bounds())

        for parameter, x in self.parameters.items():
            if x.is_variable:
                values.extend(x.get_upper_bounds())

        if self.days_of_year is not None:
            if self.days_of_year.is_variable:
                values.extend(self.days_of_year.get_upper_bounds())

        if self.y.is_variable:
            values.extend(self.y.get_upper_bounds())

        return np.array(values)
    
    def get_double_lower_bounds(self):
        """Returns an array of the lower bounds of the variables."""
        values = []

        for node, x in self.nodes.items():
            if x.is_variable:
                values.extend(x.get_lower_bounds())

        for parameter, x in self.parameters.items():
            if x.is_variable:
                values.extend(x.get_lower_bounds())

        if self.days_of_year is not None:
            if self.days_of_year.is_variable:
                values.extend(self.days_of_year.get_lower_bounds())

        if self.y.is_variable:
            values.extend(self.y.get_lower_bounds())

        return np.array(values)

    def value(self, ts, scenario_index):
        """Calculate the interpolate Rbf value from the current state."""
        # Use the cached node and parameter orders so that the exogenous inputs
        # are in the correct order.
        nodes = self._node_order
        parameters = self._parameter_order
        days_of_year = self.days_of_year

        # Create the arguments for the Rbf function.
        args = []
        for node in nodes:
            if isinstance(node, Storage):
                # Storage nodes use the current volume
                x = node.current_pc[scenario_index.global_id]
            else:
                # Other nodes are based on the flow
                x = node.flow[scenario_index.global_id]
            args.append(x)

        for parameter in parameters:
            x = parameter.get_value(scenario_index)
            args.append(x)

        if days_of_year is not None:
            # Normalise DoY to be between 0 and 1.
            x = 2 * np.pi * (ts.dayofyear - 1) / 365
            args.append(np.sin(x))
            args.append(np.cos(x))

        # Perform interpolation.
        return self._rbf_func(*args)

    @classmethod
    def load(cls, model, data):
        y = RbfData(**data.pop('y'))
        days_of_year = data.pop('days_of_year', None)
        if days_of_year is not None:
            days_of_year = RbfData(**days_of_year)

        nodes = {}
        for node_name, node_data in data.pop('nodes', {}).items():
            node = model._get_node_from_ref(model, node_name)
            nodes[node] = RbfData(**node_data)

        parameters = {}
        for param_name, param_data in data.pop('parameters', {}).items():
            parameter = load_parameter(model, param_name)
            parameters[parameter] = RbfData(**param_data)

        if 'is_variable' in data:
            raise ValueError('The RbfParameter does not support specifying the `is_variable` key '
                             'at the root level of its definition. Instead specify individual items '
                             '(e.g. nodes or parameters) to be variables instead.')

        return cls(model, y, nodes=nodes, parameters=parameters, days_of_year=days_of_year, **data)

RbfParameter.register()


# TODO write a test for this. Perhaps abstract common elements from this with above class
class RbfVolumeParameter(Parameter):
    """ A simple Rbf parameter that uses day of year and volume for interpolation.


    """
    def __init__(self, model, node, days_of_year, volume_proportions, y, rbf_kwargs=None, **kwargs):
        super(RbfVolumeParameter, self).__init__(model, **kwargs)

        self.node = node
        self.days_of_year = days_of_year
        self.volume_proportions = volume_proportions
        self.y = y
        self.rbf_kwargs = rbf_kwargs
        self._rbf_func = None
        # TODO expose variables (e.g. epsilon, the y vector).

    def reset(self):
        # Create the Rbf object here.
        # This is done in `reset` rather than `setup` because
        # we wish to have support for optimising some of the Rbf parameters.
        # Therefore it needs recreating each time.

        # Normalise DoY to be between 0 and 1.
        norm_doy = self.days_of_year / 366
        # Convention here is that DoY is the first independent variable.
        self._rbf_func = Rbf(norm_doy, self.volume_proportions, self.y)

    def value(self, ts, scenario_index):

        norm_day = ts.dayofyear / 366
        volume_pc = self.node.current_pc
        # Perform interpolation.
        return self._rbf_func(norm_day, volume_pc)
