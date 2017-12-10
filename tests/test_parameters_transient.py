from pywr.parameters.transient import TransientDecisionParameter
from pywr.parameters import ConstantParameter
from pywr.core import ScenarioIndex
import pytest
import numpy as np
import pandas
from pywr.recorders import AssertionRecorder, assert_rec
from fixtures import simple_linear_model, simple_storage_model


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