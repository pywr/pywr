""" This module contains `Parameter` subclasses for modelling transient changes.

Examples include the modelling of a decision at a fixed point during a simulation.

"""
from ._parameters import Parameter
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
        self.size = 1  # This parameter is always size 1

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

        if ts.datetime >= self.decision_date:
            v = self.after_parameter.get_value(scenario_index)
        else:
            v = self.before_parameter.get_value(scenario_index)
        return v

    def lower_bounds(self):
        return np.array([0.0, ])

    def upper_bounds(self):
        return np.array([len(self._feasible_dates)-1, ], dtype=np.float64)

    def update(self, values):
        # Update the decision date with the corresponding feasible date
        self.decision_date = self._feasible_dates[int(round(values[0]))]


class ScenarioTreeDecisionItem:
    def __init__(self, name, end_data):
        self.name = name
        self.end_data = end_data
        self.children = []

    @property
    def paths(self):
        if len(self.children) == 0:
            yield (self, )
        else:
            for child in self.children:
                for path in child.paths:
                    yield tuple([self, ] + [c for c in path])


class ScenarioTreeDecisionParameter(Parameter):
    def __init__(self, model, root, tree_scenario_mapping, parameter_factory, **kwargs):
        super(ScenarioTreeDecisionParameter, self).__init__(model, **kwargs)
        self.root = root
        self.tree_scenario_mapping = tree_scenario_mapping
        self.parameter_factory = parameter_factory

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

    def setup(self):

        # During setup we take the tree to scenario mapping to make
        # a more efficiency index based lookup array

        # Cache the scenario tree paths
        self._cached_paths = paths = tuple(p for p in self.root.paths)

        nscenarios = len(self.model.scenarios.combinations)
        path_index = np.array()



    def value(self, ts, scenario_index):
        if ts.datetime >= self.decision_date:
            v = self.after_parameter.get_value(scenario_index)
        else:
            v = self.before_parameter.get_value(scenario_index)
        return v