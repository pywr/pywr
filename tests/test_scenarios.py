# -*- coding: utf-8 -*-
"""
A series of tests of the Scenario objects and associated infrastructure


"""
import pywr.core
from pywr.core import Model, Input, Output, Link, Storage
from pywr.recorders import NumpyArrayStorageRecorder
from helpers import assert_model
# To get simple_linear_model fixture
from fixtures import simple_linear_model
from numpy.testing import assert_equal, assert_allclose


def test_scenario_collection(solver):
    """ Basic test of Scenario and ScenarioCollection API """

    model = pywr.core.Model(solver=solver)
    # There is 1 combination when there are no Scenarios
    assert(len(model.scenarios.combinations) == 1)
    assert(len(model.scenarios) == 0)
    scA = pywr.core.Scenario(model, 'Scenario A', size=3)
    assert(len(model.scenarios.combinations) == 3)
    assert(len(model.scenarios) == 1)
    scA = pywr.core.Scenario(model, 'Scenario B', size=2)
    assert(len(model.scenarios.combinations) == 6)
    assert(len(model.scenarios) == 2)

    assert_equal([comb for comb in model.scenarios.combinations],
                 [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])


def test_scenario(simple_linear_model, ):
    """Basic test of Scenario functionality"""
    model = simple_linear_model  # Convenience renaming

    scenario = pywr.core.Scenario(model, 'Inflow', size=2)
    model.node["Input"].max_flow = pywr.parameters.ConstantScenarioParameter(scenario, [5.0, 10.0])

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
    model.node["Input"].max_flow = pywr.parameters.ConstantScenarioParameter(scenario_input, [5.0, 10.0])

    scenario_outflow = pywr.core.Scenario(model, 'Outflow', size=2)
    model.node["Output"].max_flow = pywr.parameters.ConstantScenarioParameter(scenario_outflow, [3.0, 8.0])
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
    model.node["Input"].max_flow = pywr.parameters.ConstantScenarioParameter(scenario_input, [5.0, 10.0])

    model.node["Output"].max_flow = pywr.parameters.ConstantScenarioParameter(scenario_input, [8.0, 3.0])
    model.node["Output"].cost = -2.0

    expected_node_results = {
        "Input": [5.0, 3.0],
        "Link": [5.0, 3.0],
        "Output": [5.0, 3.0],
    }

    assert_model(model, expected_node_results)


def test_scenario_storage(solver):
    """Test the behaviour of Storage nodes with multiple scenarios

    The model defined has two inflow scenarios: 5 and 10. It is expected that
    the volume in the storage node should increase at different rates in the
    two scenarios.
    """
    model = Model(solver=solver)

    i = Input(model, 'input', max_flow=999)
    s = Storage(model, 'storage', num_inputs=1, num_outputs=1, max_volume=1000, volume=500)
    o = Output(model, 'output', max_flow=999)

    scenario_input = pywr.core.Scenario(model, 'Inflow', size=2)
    i.min_flow = pywr.parameters.ConstantScenarioParameter(scenario_input, [5.0, 10.0])

    i.connect(s)
    s.connect(o)

    s_rec = NumpyArrayStorageRecorder(model, s)

    model.run()

    assert_allclose(i.flow, [5, 10])
    assert_allclose(s_rec.data[0], [505, 510])
    assert_allclose(s_rec.data[1], [510, 520])
