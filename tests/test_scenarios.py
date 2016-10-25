# -*- coding: utf-8 -*-
"""
A series of tests of the Scenario objects and associated infrastructure


"""
from pywr.core import Model, Input, Output, Link, Storage, Scenario
from pywr.parameters import ConstantScenarioParameter
from pywr.recorders import NumpyArrayStorageRecorder, NumpyArrayNodeRecorder
from helpers import assert_model, load_model
from fixtures import simple_linear_model
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest


def test_scenario_collection(solver):
    """ Basic test of Scenario and ScenarioCollection API """

    model = Model(solver=solver)

    # There is 1 combination when there are no Scenarios
    model.scenarios.setup()
    assert(len(model.scenarios.combinations) == 1)
    assert(len(model.scenarios) == 0)
    scA = Scenario(model, 'Scenario A', size=3)
    model.scenarios.setup()
    assert(len(model.scenarios.combinations) == 3)
    assert(len(model.scenarios) == 1)
    scA = Scenario(model, 'Scenario B', size=2)
    model.scenarios.setup()
    assert(len(model.scenarios.combinations) == 6)
    assert(len(model.scenarios) == 2)

    assert_equal([comb.indices for comb in model.scenarios.combinations],
                 [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])

    names = model.scenarios.combination_names
    for n, (ia, ib) in zip(names, [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]):
        assert n == 'Scenario A.{:03d}-Scenario B.{:03d}'.format(ia, ib)

    index = model.scenarios.multiindex
    assert_equal(index.tolist(),
                 [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
    assert_equal(index.names, ['Scenario A', 'Scenario B'])


def test_scenario(simple_linear_model, ):
    """Basic test of Scenario functionality"""
    model = simple_linear_model  # Convenience renaming

    scenario = Scenario(model, 'Inflow', size=2)
    model.nodes["Input"].max_flow = ConstantScenarioParameter(scenario, [5.0, 10.0])

    model.nodes["Output"].max_flow = 5.0
    model.nodes["Output"].cost = -2.0

    expected_node_results = {
        "Input": [5.0, 5.0],
        "Link": [5.0, 5.0],
        "Output": [5.0, 5.0],
    }

    assert_model(model, expected_node_results)


def test_two_scenarios(simple_linear_model, ):
    """Basic test of Scenario functionality"""
    model = simple_linear_model  # Convenience renaming

    scenario_input = Scenario(model, 'Inflow', size=2)
    model.nodes["Input"].max_flow = ConstantScenarioParameter(scenario_input, [5.0, 10.0])

    scenario_outflow = Scenario(model, 'Outflow', size=2)
    model.nodes["Output"].max_flow = ConstantScenarioParameter(scenario_outflow, [3.0, 8.0])
    model.nodes["Output"].cost = -2.0

    # add numpy recorders to input and output nodes
    NumpyArrayNodeRecorder(model, model.nodes["Input"], "input")
    NumpyArrayNodeRecorder(model, model.nodes["Output"], "output")

    expected_node_results = {
        "Input": [3.0, 5.0, 3.0, 8.0],
        "Link": [3.0, 5.0, 3.0, 8.0],
        "Output": [3.0, 5.0, 3.0, 8.0],
    }

    assert_model(model, expected_node_results)

    model.run()

    # combine recorder outputs to a single dataframe
    df = model.to_dataframe()
    assert(df.shape == (365, 2 * 2 * 2))
    assert_allclose(df["input", 0, 0].iloc[0], 3.0)
    assert_allclose(df["input", 0, 1].iloc[0], 5.0)
    assert_allclose(df["input", 1, 0].iloc[0], 3.0)
    assert_allclose(df["input", 1, 1].iloc[0], 8.0)


def test_scenario_two_parameter(simple_linear_model, ):
    """Basic test of Scenario functionality"""
    model = simple_linear_model  # Convenience renaming

    scenario_input = Scenario(model, 'Inflow', size=2)
    model.nodes["Input"].max_flow = ConstantScenarioParameter(scenario_input, [5.0, 10.0])

    model.nodes["Output"].max_flow = ConstantScenarioParameter(scenario_input, [8.0, 3.0])
    model.nodes["Output"].cost = -2.0

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
    s = Storage(model, 'storage', num_inputs=1, num_outputs=1, max_volume=1000, initial_volume=500)
    o = Output(model, 'output', max_flow=999)

    scenario_input = Scenario(model, 'Inflow', size=2)
    i.min_flow = ConstantScenarioParameter(scenario_input, [5.0, 10.0])

    i.connect(s)
    s.connect(o)

    s_rec = NumpyArrayStorageRecorder(model, s)

    model.run()

    assert_allclose(i.flow, [5, 10])
    assert_allclose(s_rec.data[0], [505, 510])
    assert_allclose(s_rec.data[1], [510, 520])


def test_scenarios_from_json(solver):

    model = load_model('simple_with_scenario.json', solver=solver)
    assert len(model.scenarios) == 2

    model.setup()
    assert len(model.scenarios.combinations) == 20
    model.run()


def test_timeseries_with_scenarios(solver):

    model = load_model('timeseries2.json', solver=solver)

    model.setup()

    assert len(model.scenarios) == 1

    model.step()
    catchment1 = model.nodes['catchment1']

    step1 = np.array([21.64, 21.72, 23.97, 23.35, 21.79, 21.52, 21.21, 22.58, 26.19, 25.71])
    assert_allclose(catchment1.flow, step1)

    model.step()
    step2 = np.array([20.03, 20.10, 22.18, 21.62, 20.17, 19.92, 19.63, 20.90, 24.24, 23.80])
    # Low tolerance because test values were truncated to 2 decimal places.
    assert_allclose(catchment1.flow, step2)

    model.finish()


def test_timeseries_with_scenarios_hdf(solver):
    # this test uses TablesArrayParameter
    model = load_model('timeseries2_hdf.json', solver=solver)

    model.setup()

    assert len(model.scenarios) == 1

    catchment1 = model.nodes['catchment1']

    model.step()
    step1 = np.array([21.64, 21.72, 23.97, 23.35, 21.79, 21.52, 21.21, 22.58, 26.19, 25.71])
    # Low tolerance because test values were truncated to 2 decimal places.
    assert_allclose(catchment1.flow, step1, atol=1e-1)

    model.step()
    step2 = np.array([20.03, 20.10, 22.18, 21.62, 20.17, 19.92, 19.63, 20.90, 24.24, 23.80])
    # Low tolerance because test values were truncated to 2 decimal places.
    assert_allclose(catchment1.flow, step2, atol=1e-1)

    model.finish()

def test_tables_array_index_error(solver):
    # check an exception is raised (before the model starts) if the length
    # of the data passed to a TablesArrayParameter is not long enough
    model = load_model('timeseries2_hdf.json', solver=solver)
    model.timestepper.start = "1920-01-01"
    model.timestepper.end = "1980-01-01"
    with pytest.raises(IndexError):
        model.run()
    assert(model.timestepper.current is None)

    # check the HDF5 file was closed, despite an exception being raised
    catchment1 = model.nodes['catchment1']
    print(type(catchment1))
    param = catchment1.max_flow
    assert(param.h5store is None)

def test_dirty_scenario(simple_linear_model):
    """Adding a scenario to a model makes it dirty"""
    model = simple_linear_model
    model.setup()
    assert(not model.dirty)
    scenario = Scenario(model, "test", size=42)
    assert(model.dirty)
