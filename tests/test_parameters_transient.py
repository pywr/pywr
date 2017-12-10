from pywr.parameters.transient import TransientDecisionParameter
from pywr.parameters import ConstantParameter
from pywr.core import ScenarioIndex
import pytest
import numpy as np
import pandas
from pywr.recorders import AssertionRecorder, assert_rec
from fixtures import simple_linear_model, simple_storage_model
from test_parameters import model


class TestTransientDecisionParameter:
    """ Tests for the `TransientDecisionParameter` """

    @pytest.mark.parametrize('date', ['2015-02-01', pandas.to_datetime('2015-02-01')])
    def test_simple_date_string(self, date, simple_linear_model):
        model = simple_linear_model

        before = ConstantParameter(model, 0)
        after = ConstantParameter(model, 1)

        p = TransientDecisionParameter(model, date, before, after)

        @assert_rec(model, p)
        def expected_func(timestep, scenario_index):
            if model.timestepper.current.datetime < pandas.to_datetime(date):
                return before.get_value(scenario_index)
            else:
                return after.get_value(scenario_index)

        model.run()

    def test_start_end_dates(self, model):
        """ Test the assignment of earliest_date and latest_date keywords """
        date = '2015-02-01'
        before = ConstantParameter(model, 0)
        after = ConstantParameter(model, 1)

        # Default to model start and end dates
        p = TransientDecisionParameter(model, date, before, after)

        assert p.earliest_date == model.timestepper.start
        assert p.latest_date == model.timestepper.end

        # Test with user defined start and end dates
        p = TransientDecisionParameter(model, date, before, after, earliest_date='2015-02-01',
                                       latest_date='2020-02-03')

        assert p.earliest_date == pandas.to_datetime('2015-02-01')
        assert p.latest_date == pandas.to_datetime('2020-02-03')

    def test_variable_setup(self, simple_linear_model):
        """ Test the setup of the feasible dates for when using as a variable.

         This example should create 6 feasible dates at annual year start frequency.
         """
        model = simple_linear_model

        # This model run is 5 year; it should create 6 possible dates
        model.timestepper.start = '2015-01-01'
        model.timestepper.end = '2020-01-01'

        date = '2017-01-01'
        before = ConstantParameter(model, 0)
        after = ConstantParameter(model, 1)

        # The decision resolution is annual
        # With this setup the model should have 5 possible values
        p = TransientDecisionParameter(model, date, before, after)

        model.setup()

        np.testing.assert_allclose(p.lower_bounds(), [0.0, ])
        np.testing.assert_allclose(p.upper_bounds(), [5.0, ])  # 6th index
        for i, feasible_date in enumerate(p._feasible_dates):
            assert feasible_date == model.timestepper.start + pandas.DateOffset(years=i)

        # Finally check updating the variable works as expected
        p.update(np.array([0.8, ]))
        assert p.decision_date == pandas.to_datetime('2016-01-01')


