from pywr.parameters.transient import TransientDecisionParameter, ScenarioTreeDecisionItem, ScenarioTreeDecisionParameter, TransientScenarioTreeDecisionParameter
from pywr.parameters import ConstantParameter
from pywr.core import ScenarioIndex, Scenario
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


@pytest.fixture()
def simple_model_with_scenario_tree(simple_linear_model):
    model = simple_linear_model

    stage1 = ScenarioTreeDecisionItem(model, 'stage 1', '2030-01-01')
    stage2a = ScenarioTreeDecisionItem(model, 'stage 2a', '2050-01-01')
    stage2b = ScenarioTreeDecisionItem(model, 'stage 2b', '2050-01-01')

    stage1.children.add(stage2a)
    stage1.children.add(stage2b)

    # Add a single scenario to the model
    scenario = Scenario(model, 'A', size=10)

    # Now add the scenario indices to the tree
    # The first 5 are on stage1a and the last five on 2b
    for scenario_index in model.scenarios.get_combinations():
        if scenario_index.global_id < 5:
            stage2a.scenarios.append(scenario_index)
        else:
            stage2b.scenarios.append(scenario_index)

    return model, (stage1, stage2a, stage2b)



class TestScenarioTreeDecisionParameter:

    def test_creating_simple_scenario_tree(self, simple_model_with_scenario_tree):
        """ Test the basic API for creating a scenario tree

        The test creates a simple tree as follows:

        |-- stage 1 --|--- stage 2a ---|
                      |--- stage 2b ---|

        """
        model, (stage1, stage2a, stage2b) = simple_model_with_scenario_tree

        paths = (
            (stage1, stage2a),
            (stage1, stage2b)
        )

        for p1, p2 in zip(stage1.paths, paths):
            assert p1 == p2

        def factory(model, stage):
            if '1' in stage.name:
                p = ConstantParameter(model, 1)
            elif '2a' in stage.name:
                p = ConstantParameter(model, 2)
            elif '2b' in stage.name:
                p = ConstantParameter(model, 3)
            else:
                raise ValueError('Unrecognised stage name.')
            return p

        # Now create the parameters
        parameter = ScenarioTreeDecisionParameter(model, stage1, factory)

        # There should be one parameter for each of the stages in the tree
        assert len(parameter.children) == 3

        # This model run is 5 year; it should create 6 possible dates
        model.timestepper.start = '2025-01-01'
        model.timestepper.end = '2040-01-01'

        @assert_rec(model, parameter)
        def expected_func(timestep, scenario_index):
            if timestep.year < 2030:
                return 1
            else:
                if scenario_index.global_id < 5:
                    return 2
                else:
                    return 3

        model.run()

    def test_creating_transient_scenario_tree(self, simple_model_with_scenario_tree):
        """ Test creating the specialised transient scenario tree.

        """
        model, (stage1, stage2a, stage2b) = simple_model_with_scenario_tree

        def factory(model, stage):
            if '1' in stage.name:
                p = ConstantParameter(model, 1)
            elif '2a' in stage.name:
                p = ConstantParameter(model, 2)
            elif '2b' in stage.name:
                p = ConstantParameter(model, 3)
            else:
                raise ValueError('Unrecognised stage name.')
            return p

        # Now create the parameters
        parameter_all_off = TransientScenarioTreeDecisionParameter(model, stage1, factory, name='all_off')
        parameter_stage1_on = TransientScenarioTreeDecisionParameter(model, stage1, factory, name='stage1_on')
        parameter_stage2_on = TransientScenarioTreeDecisionParameter(model, stage1, factory, name='stage2_on')

        # There should be one parameter for each of the stages in the tree
        assert len(parameter_all_off.children) == 3

        # Default setup is all disabled.
        @assert_rec(model, parameter_all_off, name='all_off_rec')
        def expected_func(timestep, scenario_index):
            return 0

        # Turn on stage 1
        model.parameters['stage1_on.stage 1.binary'].update(np.array([1.0, ]))
        model.parameters['stage1_on.stage 1.transient'].decision_date = '2027-01-01'

        @assert_rec(model, parameter_stage1_on, name='stage1_on_rec')
        def expected_func(timestep, scenario_index):
            if timestep.year < 2027:
                return 0
            else:
                return 1

        # Turn on stage 2
        # This tests that stage 1 is zero, but there are different decisions dates
        # in the the two different stage 2 branches.
        model.parameters['stage2_on.stage 2a.binary'].update(np.array([1.0, ]))
        model.parameters['stage2_on.stage 2a.transient'].decision_date = '2035-01-01'
        model.parameters['stage2_on.stage 2b.binary'].update(np.array([1.0, ]))
        model.parameters['stage2_on.stage 2b.transient'].decision_date = '2037-01-01'

        @assert_rec(model, parameter_stage2_on, name='stage2_on_rec')
        def expected_func(timestep, scenario_index):
            if timestep.year < 2030:
                val = 0
            else:
                if scenario_index.global_id < 5:
                    val = 0 if timestep.year < 2035 else 2
                else:
                    val = 0 if timestep.year < 2037 else 3
            return val

        # This model run is 5 year; it should create 6 possible dates
        model.timestepper.start = '2025-01-01'
        model.timestepper.end = '2040-01-01'
        model.timestepper.delta = 10

        model.run()

