""" This module contains `Parameter` subclasses for modelling transient changes.

Examples include the modelling of a decision at a fixed point during a simulation.

"""
from ._parameters import Parameter
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
    before : Parameter
        The value to use before the decision date.
    after: Parameter
        The value to use after the decision date.

    """
    def __init__(self, model, decision_date, before, after, **kwargs):
        super(TransientDecisionParameter, self).__init__(model, **kwargs)
        self._decision_date = None
        self.decision_date = decision_date

        if not isinstance(before, Parameter):
            raise ValueError('The `before` value should be a Parameter instance.')
        before.parents.add(self)
        self.before = before

        if not isinstance(after, Parameter):
            raise ValueError('The `after` value should be a Parameter instance.')
        after.parents.add(self)
        self.after = after

    def decision_date():
        def fget(self):
            return self._start
        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._decision_date = value
            else:
                self._decision_date = pandas.to_datetime(value)
        return locals()
    decision_date = property(**decision_date())

    def value(self, ts, scenario_index):

        if ts.datetime > self.decision_date:
            self.after.get_value(scenario_index)
        else:
            self.before.get_value(scenario_index)


            