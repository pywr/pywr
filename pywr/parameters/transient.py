""" This module contains `Parameter` subclasses for modelling transient changes.

Examples include the modelling of a decision at a fixed point during a simulation.

"""
from .._component import Component
from ._parameters import Parameter, ConstantParameter, BinaryVariableParameter
import numpy as np
import pandas


class TransientDecisionParameter(Parameter):
    """ Return one of two values depending on the current time-step

    This `Parameter` can be used to model a discrete decision event
     that happens at a given date. Prior to this date the `before`
     value is returned, and post this date the `after` value is returned.

    Parameters
    ----------
    decision_date : string or pandas.Timestamp
        The trigger date for the decision.
    before_parameter : Parameter
        The value to use before the decision date.
    after_parameter : Parameter
        The value to use after the decision date.
    earliest_date : string or pandas.Timestamp or None
        Earliest date that the variable can be set to. Defaults to `model.timestepper.start`
    latest_date : string or pandas.Timestamp or None
        Latest date that the variable can be set to. Defaults to `model.timestepper.end`
    decision_freq : pandas frequency string (default 'AS')
        The resolution of feasible dates. For example 'AS' would create feasible dates every
        year between `earliest_date` and `latest_date`. The `pandas` functions are used
        internally for delta date calculations.

    """
    def __init__(self, model, decision_date, before_parameter, after_parameter, 
                 earliest_date=None, latest_date=None, decision_freq='AS', **kwargs):
        super(TransientDecisionParameter, self).__init__(model, **kwargs)
        self._decision_date = None
        self.decision_date = decision_date

        if not isinstance(before_parameter, Parameter):
            raise ValueError('The `before` value should be a Parameter instance.')
        before_parameter.parents.add(self)
        self.before_parameter = before_parameter

        if not isinstance(after_parameter, Parameter):
            raise ValueError('The `after` value should be a Parameter instance.')
        after_parameter.parents.add(self)
        self.after_parameter = after_parameter
        
        # These parameters are mostly used if this class is used as variable.
        self._earliest_date = None
        self.earliest_date = earliest_date
        
        self._latest_date = None
        self.latest_date = latest_date

        self.decision_freq = decision_freq
        self._feasible_dates = None
        self.integer_size = 1  # This parameter has a single integer variable

    def decision_date():
        def fget(self):
            return self._decision_date
        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._decision_date = value
            else:
                self._decision_date = pandas.to_datetime(value)
        return locals()
    decision_date = property(**decision_date())
    
    def earliest_date():
        def fget(self):
            if self._earliest_date is not None:
                return self._earliest_date
            else:
                return self.model.timestepper.start
        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._earliest_date = value
            else:
                self._earliest_date = pandas.to_datetime(value)
        return locals()
    earliest_date = property(**earliest_date())
    
    def latest_date():
        def fget(self):
            if self._latest_date is not None:
                return self._latest_date
            else:
                return self.model.timestepper.end
        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._latest_date = value
            else:
                self._latest_date = pandas.to_datetime(value)
        return locals()
    latest_date = property(**latest_date())

    def setup(self):
        super(TransientDecisionParameter, self).setup()

        # Now setup the feasible dates for when this object is used as a variable.
        self._feasible_dates = pandas.date_range(self.earliest_date, self.latest_date,
                                                 freq=self.decision_freq)

    def value(self, ts, scenario_index):

        if ts is None:
            v = self.before_parameter.get_value(scenario_index)
        elif ts.datetime >= self.decision_date:
            v = self.after_parameter.get_value(scenario_index)
        else:
            v = self.before_parameter.get_value(scenario_index)
        return v

    def get_integer_lower_bounds(self):
        return np.array([0, ], dtype=np.int)

    def get_integer_upper_bounds(self):
        return np.array([len(self._feasible_dates)-1, ], dtype=np.int)

    def set_integer_variables(self, values):
        # Update the decision date with the corresponding feasible date
        self.decision_date = self._feasible_dates[values[0]]

    def dump(self):

        data = {
            'earliest_date': self.earliest_date.isoformat(),
            'latest_date': self.latest_date.isoformat(),
            'decision_date': self.decision_date.isoformat(),
            'decision_frequency': self.decision_freq
        }

        return data


class ScenarioTreeDecisionItem(Component):
    def __init__(self, model, name, end_date, **kwargs):
        super(ScenarioTreeDecisionItem, self).__init__(model, name, **kwargs)
        self.end_date = end_date
        self.scenarios = []

    @property
    def start_date(self):
        # Find if there is a parent stage in the tree
        for parent in self.parents:
            if isinstance(parent, ScenarioTreeDecisionItem):
                # The start of this stage is the end of parent stage
                return parent.end_date
        # Otherwise the start is the start of the model
        return self.model.timestepper.start

    @property
    def paths(self):
        if len(self.children) == 0:
            yield (self, )
        else:
            for child in self.children:
                for path in child.paths:
                    yield tuple([self, ] + [c for c in path])

    def end_date():
        def fget(self):
            if self._latest_date is not None:
                return self._latest_date
            else:
                return self.model.timestepper.end
        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._latest_date = value
            else:
                self._latest_date = pandas.to_datetime(value)
        return locals()
    end_date = property(**end_date())


class ScenarioTreeDecisionParameter(Parameter):
    def __init__(self, model, root, parameter_factory, **kwargs):
        super(ScenarioTreeDecisionParameter, self).__init__(model, **kwargs)
        self.root = root
        self.parameter_factory = parameter_factory

        self.path_index = None
        self._cached_paths = None
        self.parameters = None
        # Setup the parameters associated with the tree
        self._create_scenario_parameters()

    def _create_scenario_parameters(self):
        parameters = {}
        def make_parameter(scenario):
            p = self.parameter_factory(self.model, scenario)
            parameters[scenario] = p
            self.children.add(p)  # Ensure that these parameters are children of this
            # Recursively call to make parameters for children
            for child in scenario.children:
                make_parameter(child)

        make_parameter(self.root)
        self.parameters = parameters

    def setup(self):
        super(ScenarioTreeDecisionParameter, self).setup()
        # During setup we take the tree to scenario mapping to make
        # a more efficiency index based lookup array

        # Cache the scenario tree paths
        self._cached_paths = paths = tuple(p for p in self.root.paths)

        nscenarios = len(self.model.scenarios.combinations)
        path_index = np.empty(nscenarios, dtype=np.int)
        path_index[:] = np.nan

        for i, path in enumerate(paths):
            final_stage = path[-1]
            for scenario_index in final_stage.scenarios:
                path_index[scenario_index.global_id] = i

        if len(np.where(path_index == np.nan)[0]) > 0:
            raise ValueError('One or more scenarios are not assigned to final stages in the scenario tree.')

        self.path_index = path_index

    def value(self, ts, scenario_index):

        i = self.path_index[scenario_index.global_id]
        path = self._cached_paths[i]

        for stage in path:
            if ts.datetime < stage.end_date:
                parameter = self.parameters[stage]
                return parameter.get_value(scenario_index)

        raise ValueError('No parameter found from stages for current time-step.')


class TransientScenarioTreeDecisionParameter(ScenarioTreeDecisionParameter):
    def __init__(self, model, root, enabled_parameter_factory, earliest_date=None, latest_date=None,
                 decision_freq='AS', **kwargs):

        self.enabled_parameter_factory = enabled_parameter_factory

        # These parameters are mostly used if this class is used as variable.
        self._earliest_date = None
        self.earliest_date = earliest_date
        self._latest_date = None
        self.latest_date = latest_date
        self.decision_freq = decision_freq

        super(TransientScenarioTreeDecisionParameter, self).__init__(model, root, self._transient_parameter_factory, **kwargs)

    def _transient_parameter_factory(self, model, stage):
        """ Private factory function for creating the transient parameters """

        name = '{}.{}.{}'.format(self.name, stage.name, '{}')

        # When the parameter is not active (either off or before decision) data
        # default to a zero value.
        # TODO make this disabled value configurable.
        disabled_parameter = ConstantParameter(model, 0, name=name.format('disabled'))

        # Use the given factory function to create the enabled parameter
        enabled_parameter = self.enabled_parameter_factory(model, stage)
        enabled_parameter.name = name.format('enabled')

        # Make the transient parameter
        earliest_date = stage.start_date
        latest_date = stage.end_date
        current_date = latest_date

        p = TransientDecisionParameter(model, current_date, disabled_parameter, enabled_parameter,
                                       earliest_date=earliest_date, latest_date=latest_date,
                                       name=name.format('transient'))

        # Finally wrap the transient parameter in a binary variable
        # This defaults to `is_variable=True` because that is the intended use of this overall parameter.
        return BinaryVariableParameter(model, p, disabled_parameter, is_variable=True, name=name.format('binary'))

    def earliest_date():
        def fget(self):
            if self._earliest_date is not None:
                return self._earliest_date
            else:
                return self.model.timestepper.start

        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._earliest_date = value
            else:
                self._earliest_date = pandas.to_datetime(value)

        return locals()

    earliest_date = property(**earliest_date())

    def latest_date():
        def fget(self):
            if self._latest_date is not None:
                return self._latest_date
            else:
                return self.model.timestepper.end

        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._latest_date = value
            else:
                self._latest_date = pandas.to_datetime(value)

        return locals()

    latest_date = property(**latest_date())

    def value(self, ts, scenario_index):

        i = self.path_index[scenario_index.global_id]
        path = self._cached_paths[i]

        for stage in path:
            parameter = self.parameters[stage]
            # Fetch the state from the binary variable parameter
            # The index returns 0 or 1 depending on the internal state
            active = parameter.get_index(scenario_index)

            # The stages are iterated through in time order (first to last)
            # Therefore if this stage has an active binary variable we
            # ignore the current time-step and use the value of this stage's parameter
            print(self.name, ts.datetime, stage.name, stage.end_date, active, parameter.get_value(scenario_index))
            if ts.datetime < stage.end_date or active:
                return parameter.get_value(scenario_index)

        raise ValueError('No parameter found from stages for current time-step.')