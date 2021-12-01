# -*- coding: utf-8 -*-
"""
A series of tests of the Scenario objects and associated infrastructure


"""
from pywr.core import Model, Input, Output, Link, Storage, Scenario
from pywr.parameters import ConstantScenarioParameter
from pywr.recorders import NumpyArrayStorageRecorder, NumpyArrayNodeRecorder
from pywr.hashes import HashMismatchError
from helpers import assert_model, load_model
from fixtures import simple_linear_model
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest


def test_scenario_collection():
    """Basic test of Scenario and ScenarioCollection API"""

    model = Model()

    # There is 1 combination when there are no Scenarios
    model.scenarios.setup()
    assert len(model.scenarios.combinations) == 1
    assert len(model.scenarios) == 0
    scA = Scenario(model, "Scenario A", size=3)
    model.scenarios.setup()
    assert len(model.scenarios.combinations) == 3
    assert len(model.scenarios) == 1
    scA = Scenario(model, "Scenario B", size=2)
    model.scenarios.setup()
    assert len(model.scenarios.combinations) == 6
    assert len(model.scenarios) == 2

    assert_equal(
        [comb.indices for comb in model.scenarios.combinations],
        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]],
    )

    names = model.scenarios.combination_names
    for n, (ia, ib) in zip(names, [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]):
        assert n == "Scenario A.{:03d}-Scenario B.{:03d}".format(ia, ib)

    index = model.scenarios.multiindex
    assert_equal(index.tolist(), [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
    assert_equal(index.names, ["Scenario A", "Scenario B"])


def test_scenario(
    simple_linear_model,
):
    """Basic test of Scenario functionality"""
    model = simple_linear_model  # Convenience renaming

    scenario = Scenario(model, "Inflow", size=2)
    model.nodes["Input"].max_flow = ConstantScenarioParameter(
        model, scenario, [5.0, 10.0]
    )

    model.nodes["Output"].max_flow = 5.0
    model.nodes["Output"].cost = -2.0

    expected_node_results = {
        "Input": [5.0, 5.0],
        "Link": [5.0, 5.0],
        "Output": [5.0, 5.0],
    }

    assert_model(model, expected_node_results)


def test_two_scenarios(
    simple_linear_model,
):
    """Basic test of Scenario functionality"""
    model = simple_linear_model  # Convenience renaming

    scenario_input = Scenario(model, "Inflow", size=2)
    model.nodes["Input"].max_flow = ConstantScenarioParameter(
        model, scenario_input, [5.0, 10.0]
    )

    scenario_outflow = Scenario(
        model, "Outflow", size=2, ensemble_names=["High", "Low"]
    )
    model.nodes["Output"].max_flow = ConstantScenarioParameter(
        model, scenario_outflow, [3.0, 8.0]
    )
    model.nodes["Output"].cost = -2.0

    # Check ensemble names are provided in the multi-index
    index = model.scenarios.multiindex
    assert index.levels[0].name == "Inflow"
    assert index.levels[1].name == "Outflow"
    assert np.all(index.levels[1] == ["High", "Low"])

    # add numpy recorders to input and output nodes
    NumpyArrayNodeRecorder(model, model.nodes["Input"], name="input")
    NumpyArrayNodeRecorder(model, model.nodes["Output"], name="output")

    expected_node_results = {
        "Input": [3.0, 5.0, 3.0, 8.0],
        "Link": [3.0, 5.0, 3.0, 8.0],
        "Output": [3.0, 5.0, 3.0, 8.0],
    }

    assert_model(model, expected_node_results)

    model.run()

    # combine recorder outputs to a single dataframe
    df = model.to_dataframe()
    assert df.shape == (365, 2 * 2 * 2)
    assert_allclose(df["input", 0, "High"].iloc[0], 3.0)
    assert_allclose(df["input", 0, "Low"].iloc[0], 5.0)
    assert_allclose(df["input", 1, "High"].iloc[0], 3.0)
    assert_allclose(df["input", 1, "Low"].iloc[0], 8.0)


def test_scenario_two_parameter(
    simple_linear_model,
):
    """Basic test of Scenario functionality"""
    model = simple_linear_model  # Convenience renaming

    scenario_input = Scenario(model, "Inflow", size=2)
    model.nodes["Input"].max_flow = ConstantScenarioParameter(
        model, scenario_input, [5.0, 10.0]
    )

    model.nodes["Output"].max_flow = ConstantScenarioParameter(
        model, scenario_input, [8.0, 3.0]
    )
    model.nodes["Output"].cost = -2.0

    expected_node_results = {
        "Input": [5.0, 3.0],
        "Link": [5.0, 3.0],
        "Output": [5.0, 3.0],
    }

    assert_model(model, expected_node_results)


def test_scenario_storage():
    """Test the behaviour of Storage nodes with multiple scenarios

    The model defined has two inflow scenarios: 5 and 10. It is expected that
    the volume in the storage node should increase at different rates in the
    two scenarios.
    """
    model = Model()

    i = Input(model, "input", max_flow=999)
    s = Storage(
        model, "storage", inputs=1, outputs=1, max_volume=1000, initial_volume=500
    )
    o = Output(model, "output", max_flow=999)

    scenario_input = Scenario(model, "Inflow", size=2)
    i.min_flow = ConstantScenarioParameter(model, scenario_input, [5.0, 10.0])

    i.connect(s)
    s.connect(o)

    s_rec = NumpyArrayStorageRecorder(model, s)

    model.run()

    assert_allclose(i.flow, [5, 10])
    assert_allclose(s_rec.data[0], [505, 510])
    assert_allclose(s_rec.data[1], [510, 520])


@pytest.mark.parametrize(
    "json_file", ["simple_with_scenario.json", "simple_with_scenario_wrapper.json"]
)
def test_scenarios_from_json(json_file):
    """
    Test a simple model with two scenarios.

    The model varies in the inflow by "scenario A" and the demand
    by "scenario B". The test ensures the correct size of model is
    created, and uses a `NumpyArrayNodeRecorder` to check the output
    in multiple dimensions is correct. The latter is done using
    the `MultiIndex` on the `DataFrame` from the recorder.
    """

    model = load_model(json_file)
    assert len(model.scenarios) == 2

    model.setup()
    assert len(model.scenarios.combinations) == 20
    model.run()

    # Test the recorder data is correct
    df = model.recorders["demand1"].to_dataframe()

    assert df.shape[1] == 20
    assert df.columns.names[0] == "scenario A"
    assert_equal(df.columns.levels[0], np.arange(10))
    assert df.columns.names[1] == "scenario B"
    assert_equal(df.columns.levels[1], np.array(["First", "Second"]))
    # Data for first demand (B) ensemble
    d1 = df.xs("First", level="scenario B", axis=1).iloc[0, :].values
    assert_allclose(d1, [10] * 10)
    # Data for second demand (B) ensemble
    d2 = df.xs("Second", level="scenario B", axis=1).iloc[0, :]
    assert_allclose(d2, [10, 11, 12, 13, 14] + [15] * 5)


def test_timeseries_with_scenarios():

    model = load_model("timeseries2.json")

    model.setup()

    assert len(model.scenarios) == 1

    model.step()
    catchment1 = model.nodes["catchment1"]

    step1 = np.array(
        [21.64, 21.72, 23.97, 23.35, 21.79, 21.52, 21.21, 22.58, 26.19, 25.71],
        dtype=np.float64,
    )
    assert_allclose(catchment1.flow, step1)

    model.step()
    step2 = np.array(
        [20.03, 20.10, 22.18, 21.62, 20.17, 19.92, 19.63, 20.90, 24.24, 23.80],
        dtype=np.float64,
    )
    # Low tolerance because test values were truncated to 2 decimal places.
    assert_allclose(catchment1.flow, step2)

    model.finish()


def test_timeseries_with_scenarios_hdf():
    # this test uses TablesArrayParameter
    model = load_model("timeseries2_hdf.json")

    model.setup()

    assert len(model.scenarios) == 1

    catchment1 = model.nodes["catchment1"]

    model.step()
    step1 = np.array(
        [21.64, 21.72, 23.97, 23.35, 21.79, 21.52, 21.21, 22.58, 26.19, 25.71],
        dtype=np.float64,
    )
    # Low tolerance because test values were truncated to 2 decimal places.
    assert_allclose(catchment1.flow, step1, atol=1e-1)

    model.step()
    step2 = np.array(
        [20.03, 20.10, 22.18, 21.62, 20.17, 19.92, 19.63, 20.90, 24.24, 23.80],
        dtype=np.float64,
    )
    # Low tolerance because test values were truncated to 2 decimal places.
    assert_allclose(catchment1.flow, step2, atol=1e-1)

    model.finish()


def test_timeseries_with_wrong_hash():
    with pytest.raises(HashMismatchError):
        load_model("timeseries2_hdf_wrong_hash.json")


def test_tablesarrayparameter_scenario_slice():
    model = load_model("timeseries2_hdf.json")
    catchment1 = model.nodes["catchment1"]
    scenario = model.scenarios["scenario A"]
    scenario.slice = slice(0, 10, 2)
    model.setup()
    model.reset()
    model.step()
    step1 = np.array(
        [21.64, 21.72, 23.97, 23.35, 21.79, 21.52, 21.21, 22.58, 26.19, 25.71],
        dtype=np.float64,
    )
    assert_allclose(catchment1.flow, step1[::2], atol=1e-1)
    model.step()
    step2 = np.array(
        [20.03, 20.10, 22.18, 21.62, 20.17, 19.92, 19.63, 20.90, 24.24, 23.80],
        dtype=np.float64,
    )
    assert_allclose(catchment1.flow, step2[::2], atol=1e-1)
    model.finish()


def test_tablesarrayparameter_scenario_user_combinations():
    """Test TablesArrayParameter with user defined combination of scenarios"""
    model = load_model("timeseries2_hdf.json")
    catchment1 = model.nodes["catchment1"]
    scenario = model.scenarios["scenario A"]
    scenario2 = Scenario(model, "scenario B", size=2)
    # combinations are intentially out of order and with duplicates
    model.scenarios.user_combinations = [(0, 0), (2, 0), (6, 1), (2, 1)]
    model.setup()
    model.reset()
    assert len(model.scenarios.combinations) == 4
    model.step()
    step1 = np.array(
        [21.64, 21.72, 23.97, 23.35, 21.79, 21.52, 21.21, 22.58, 26.19, 25.71],
        dtype=np.float64,
    )
    assert_allclose(catchment1.flow, step1[[0, 2, 6, 2]], atol=1e-1)
    model.step()
    step2 = np.array(
        [20.03, 20.10, 22.18, 21.62, 20.17, 19.92, 19.63, 20.90, 24.24, 23.80],
        dtype=np.float64,
    )
    assert_allclose(catchment1.flow, step2[[0, 2, 6, 2]], atol=1e-1)
    model.finish()


def test_tables_array_index_error():
    # check an exception is raised (before the model starts) if the length
    # of the data passed to a TablesArrayParameter is not long enough
    model = load_model("timeseries2_hdf.json")
    model.timestepper.start = "1920-01-01"
    model.timestepper.end = "1980-01-01"
    with pytest.raises(IndexError):
        model.run()
    assert model.timestepper.current is None

    # check the HDF5 file was closed, despite an exception being raised
    catchment1 = model.nodes["catchment1"]
    print(type(catchment1))
    param = catchment1.max_flow
    assert param.h5store is None


def test_dirty_scenario(simple_linear_model):
    """Adding a scenario to a model makes it dirty"""
    model = simple_linear_model
    model.setup()
    assert not model.dirty
    scenario = Scenario(model, "test", size=42)
    assert model.dirty


def test_scenario_slices(simple_linear_model):
    """Test slicing of scenarios"""
    model = simple_linear_model

    # create two scenarios
    s1 = Scenario(model=model, name="A", size=20)
    s2 = Scenario(model=model, name="B", size=3)

    combinations = model.scenarios.get_combinations()
    assert len(combinations) == 20 * 3

    s1.slice = slice(0, None, 2)
    combinations = model.scenarios.get_combinations()
    assert len(combinations) == 10 * 3

    # check multiindex respects scenario slices
    index = model.scenarios.multiindex
    assert len(index.levels) == 2
    assert len(index.levels[0]) == 10
    assert len(index.levels[1]) == 3
    assert len(index.codes) == 2
    assert len(index.codes[0]) == 10 * 3
    assert len(index.codes[1]) == 10 * 3
    assert index.names == ["A", "B"]

    s2.slice = slice(1, 3, 1)
    combinations = model.scenarios.get_combinations()
    assert len(combinations) == 10 * 2

    assert combinations[0].global_id == 0
    assert tuple(combinations[0].indices) == (0, 1)

    assert combinations[-1].global_id == 19
    assert tuple(combinations[-1].indices) == (18, 2)

    model.run()

    node = model.nodes["Input"]
    assert (len(combinations),) == node.flow.shape

    s1.slice = None
    s2.slice = None
    combinations = model.scenarios.get_combinations()
    assert len(combinations) == 20 * 3


def test_scenario_user_combinations(simple_linear_model):
    model = simple_linear_model

    # create two scenarios
    s1 = Scenario(model=model, name="A", size=20)
    s2 = Scenario(model=model, name="B", size=3)

    model.scenarios.user_combinations = [[0, 1], [1, 1]]
    combinations = model.scenarios.get_combinations()
    assert len(combinations) == 2
    # ScenarioCollection.shape is simply the number of combinations
    assert model.scenarios.shape == (2,)

    # Test wrong number of dimensions
    with pytest.raises(ValueError):
        model.scenarios.user_combinations = [0, 1, 1, 1]

    # Test out of range index
    with pytest.raises(ValueError):
        model.scenarios.user_combinations = [[19, 3], [2, 2]]
    with pytest.raises(ValueError):
        model.scenarios.user_combinations = [[-1, 2], [2, 2]]


def test_scenario_slices_json():
    model = load_model("scenario_with_slices.json")
    scenarios = model.scenarios
    assert len(scenarios) == 2
    assert scenarios["scenario A"].slice == slice(0, None, 2)
    assert scenarios["scenario B"].slice == slice(0, 1, 1)
    combinations = model.scenarios.get_combinations()
    assert len(combinations) == 5


def test_scenario_slices_json():
    model = load_model("scenario_with_user_combinations.json")
    scenarios = model.scenarios
    assert len(scenarios) == 2
    combinations = model.scenarios.get_combinations()
    assert len(combinations) == 3
