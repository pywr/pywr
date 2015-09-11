# -*- coding: utf-8 -*-
"""
A series of tests of the Scenario objects and associated infrastructure


"""
import pywr.core
from helpers import assert_model
# To gete simple_linear_model fixture
from test_analytical import simple_linear_model
from numpy.testing import assert_equal


def test_scenario_collection():
    """ Basic test of Scenario and ScenarioCollection API """

    model = pywr.core.Model()
    # There is 1 combination when there are no Scenarios
    assert(model.scenarios.get_number_of_combinations() == 1)
    assert(model.number_of_scenario_combinations == 1)
    assert(model.number_of_scenarios == 0)
    scA = pywr.core.Scenario(model, 'Scenario A', size=3)
    assert(model.scenarios.get_number_of_combinations() == 3)
    assert(model.number_of_scenarios == 1)
    assert(model.number_of_scenario_combinations == 3)
    scA = pywr.core.Scenario(model, 'Scenario B', size=2)
    assert(model.scenarios.get_number_of_combinations() == 6)
    assert(model.number_of_scenarios == 2)
    assert(model.number_of_scenario_combinations == 6)

    assert_equal(model.scenarios.get_combinations(),
                 [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])


def test_scenario(simple_linear_model, ):
    """Basic test of Scenario functionality"""
    model = simple_linear_model  # Convenience renaming

    scenario = pywr.core.Scenario(model, 'Inflow', size=2)
    model.node["Input"].max_flow = pywr.core.ParameterConstantScenario(scenario, [5.0, 10.0])

    model.node["Output"].max_flow = 5.0
    model.node["Output"].cost = -2.0

    expected_node_results = {
        "Input": [5.0, 5.0],
        "Link": [5.0, 5.0],
        "Output": [5.0, 5.0],
    }

    assert_model(model, expected_node_results)


def test_two_scenarios(simple_linear_model, ):
    """Basic test of Scenario functionality"""
    model = simple_linear_model  # Convenience renaming

    scenario_input = pywr.core.Scenario(model, 'Inflow', size=2)
    model.node["Input"].max_flow = pywr.core.ParameterConstantScenario(scenario_input, [5.0, 10.0])

    scenario_outflow = pywr.core.Scenario(model, 'Outflow', size=2)
    model.node["Output"].max_flow = pywr.core.ParameterConstantScenario(scenario_outflow, [3.0, 8.0])
    model.node["Output"].cost = -2.0

    expected_node_results = {
        "Input": [3.0, 5.0, 3.0, 8.0],
        "Link": [3.0, 5.0, 3.0, 8.0],
        "Output": [3.0, 5.0, 3.0, 8.0],
    }

    assert_model(model, expected_node_results)


def test_scenario_two_parameter(simple_linear_model, ):
    """Basic test of Scenario functionality"""
    model = simple_linear_model  # Convenience renaming

    scenario_input = pywr.core.Scenario(model, 'Inflow', size=2)
    model.node["Input"].max_flow = pywr.core.ParameterConstantScenario(scenario_input, [5.0, 10.0])

    model.node["Output"].max_flow = pywr.core.ParameterConstantScenario(scenario_input, [8.0, 3.0])
    model.node["Output"].cost = -2.0

    expected_node_results = {
        "Input": [5.0, 3.0],
        "Link": [5.0, 3.0],
        "Output": [5.0, 3.0],
    }

    assert_model(model, expected_node_results)
