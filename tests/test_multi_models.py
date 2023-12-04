import os
from pathlib import Path

import numpy as np
import pandas
import pytest

from pywr.parameters import DataFrameParameter
from pywr.model import MultiModel, Model
from pywr.core import Scenario

# def load_multi_model(filename: Path):
from pywr.nodes import Input, Link, Output
from pywr.parameters.multi_model_parameters import (
    OtherModelParameterValueParameter,
    OtherModelNodeFlowParameter,
    OtherModelNodeStorageParameter,
)
from pywr.recorders import (
    NumpyArrayNodeRecorder,
    NumpyArrayParameterRecorder,
    NumpyArrayStorageRecorder,
)


def make_simple_model(num: int) -> Model:
    m = Model()
    inpt = Input(m, name=f"input-{num}", max_flow=5 + num)
    lnk = Link(m, name=f"link-{num}")
    otpt = Output(m, name=f"output-{num}", max_flow=5 + num, cost=-10)

    inpt.connect(lnk)
    lnk.connect(otpt)
    return m


def test_run_two_independent_models():
    multi_model = MultiModel()

    for i in range(2):
        m = make_simple_model(i)
        multi_model.add_model(f"model-{i}", m)

    multi_model.run()

    np.testing.assert_allclose(
        multi_model.models["model-0"].nodes["output-0"].flow[0], 5.0
    )
    np.testing.assert_allclose(
        multi_model.models["model-1"].nodes["output-1"].flow[0], 6.0
    )


def test_setup_profile_two_independent_models(tmp_path):
    multi_model = MultiModel()

    for i in range(2):
        m = make_simple_model(i)
        multi_model.add_model(f"model-{i}", m)

    profile_out = tmp_path / "stats.csv"
    multi_model.setup(profile=True, profile_dump_filename=profile_out)

    assert profile_out.exists()
    df = pandas.read_csv(profile_out)
    assert len(df) == 12


def test_run_two_independent_models_from_json():
    """Test two independent models running together"""
    path = Path(os.path.dirname(__file__)) / "models" / "two-independent-sub-models"
    multi_model = MultiModel.load(path / "integrated-model.json")

    multi_model.run()

    np.testing.assert_allclose(
        multi_model.models["model1"].nodes["demand1"].flow[0], 10.0
    )
    np.testing.assert_allclose(
        multi_model.models["model2"].nodes["demand1"].flow[0], 10.0
    )


def test_run_two_dependent_models_from_json():
    """Test two simple but dependent models."""

    path = Path(os.path.dirname(__file__)) / "models" / "two-dependent-sub-models"
    multi_model = MultiModel.load(path / "integrated-model.json")

    sub_model2 = multi_model.models["model2"]
    supply2 = sub_model2.nodes["supply2"]
    demand2 = sub_model2.nodes["demand2"]
    assert isinstance(supply2.max_flow, OtherModelParameterValueParameter)

    # Add recorder for flow
    demand2_rec = NumpyArrayNodeRecorder(sub_model2, demand2)
    # Demand should equal the inflow the model1
    expected_flow = pandas.read_csv(path / "timeseries1.csv", index_col=0)

    multi_model.run()

    np.testing.assert_allclose(
        multi_model.models["model1"].nodes["demand1"].flow[0], 10.0
    )
    np.testing.assert_allclose(demand2_rec.data, expected_flow)


def test_run_two_dependent_models_with_flow_transfer_from_json():
    """Test two simple but dependent models."""

    path = (
        Path(os.path.dirname(__file__))
        / "models"
        / "two-dependent-sub-models-flow-transfer"
    )
    multi_model = MultiModel.load(path / "integrated-model.json")

    sub_model2 = multi_model.models["model2"]
    supply2 = sub_model2.nodes["supply2"]
    demand2 = sub_model2.nodes["demand2"]
    assert isinstance(supply2.max_flow, OtherModelNodeFlowParameter)

    # Add recorder for flow
    demand2_rec = NumpyArrayNodeRecorder(sub_model2, demand2)
    # Demand should equal the flow supplied in model1
    expected_flow = pandas.read_csv(path / "timeseries1.csv", index_col=0)

    multi_model.run()

    np.testing.assert_allclose(
        multi_model.models["model1"].nodes["demand1"].flow[0], 21.92
    )
    np.testing.assert_allclose(demand2_rec.data, expected_flow)


def test_run_three_dependent_storage_sub_models():
    """Test three dependent models."""

    path = (
        Path(os.path.dirname(__file__))
        / "models"
        / "three-dependent-storage-sub-models"
    )
    multi_model = MultiModel.load(path / "integrated-model.json")

    sub_model0 = multi_model.models["model0"]
    sub_model1 = multi_model.models["model1"]
    sub_model2 = multi_model.models["model2"]

    # Create some recorders
    sm0_sv1 = NumpyArrayParameterRecorder(
        sub_model0, sub_model0.parameters["storage1-volume"]
    )
    sm1_sv1 = NumpyArrayStorageRecorder(sub_model1, sub_model1.nodes["storage1"])
    sm0_sr1 = NumpyArrayParameterRecorder(
        sub_model0, sub_model0.parameters["storage1-release"]
    )

    multi_model.setup()

    assert isinstance(
        sub_model0.parameters["storage1-volume"], OtherModelNodeStorageParameter
    )

    multi_model.run()

    # Reservoir releases on the first time-step because combined volume > 0
    # From second time-step onwards release is turned off from model0 because combined volume < 0
    # The storage parameter in model0 has the volume at the end of the previous day ...
    np.testing.assert_allclose(
        sm0_sv1.data[:, 0],
        [
            510.0,
            490.0,
            480.0,
            470.0,
            460.0,
        ],
    )
    # The volume in storage is recorded on the node at the end of the day
    np.testing.assert_allclose(sm1_sv1.data[:, 0], [490.0, 480.0, 470.0, 460.0, 450.0])
    # The release is calculated using previous day's volume
    np.testing.assert_allclose(sm0_sr1.data[:, 0], [10.0, 0.0, 0.0, 0.0, 0.0])


def test_error_with_different_timesteps():
    """Check a RuntimeError is raised if the models have different timesteps."""

    path = Path(os.path.dirname(__file__)) / "models" / "two-independent-sub-models"
    multi_model = MultiModel.load(path / "integrated-model.json")

    multi_model.models["model1"].timestepper.start = "1900-01-01"

    with pytest.raises(RuntimeError):
        multi_model.run()


@pytest.mark.parametrize(
    "sizes1,names1,sizes2,names2",
    [
        [[10], ["A"], [11], ["B"]],
        [[11], ["A"], [10], ["B"]],
        [[10, 20], ["A", "B"], [10], ["A"]],
        [[10], ["A"], [10, 20], ["A", "B"]],
        [[10], ["A"], [5, 2], ["A", "B"]],
        [[1], ["A"], [10], ["A"]],
    ],
)
def test_error_with_different_scenarios(sizes1, names1, sizes2, names2):
    """Check a ValueError is raised if the models have different scenarios."""

    path = Path(os.path.dirname(__file__)) / "models" / "two-dependent-sub-models"
    multi_model = MultiModel.load(path / "integrated-model.json")

    for s, n in zip(sizes1, names1):
        Scenario(multi_model.models["model1"], size=s, name=n)
    for s, n in zip(sizes2, names2):
        Scenario(multi_model.models["model2"], size=s, name=n)

    with pytest.raises(ValueError):
        multi_model.run()


def test_error_with_different_scenario_combinations():
    """Check a ValueError is raised if the models have different scenarios."""

    path = Path(os.path.dirname(__file__)) / "models" / "two-dependent-sub-models"
    multi_model = MultiModel.load(path / "integrated-model.json")

    # Define the same scenarios in each sub-model
    Scenario(multi_model.models["model1"], size=10, name="A")
    Scenario(multi_model.models["model1"], size=2, name="B")

    Scenario(multi_model.models["model2"], size=10, name="A")
    Scenario(multi_model.models["model2"], size=2, name="B")

    # Only run the first two scenarios in model1
    multi_model.models["model1"].scenarios.user_combinations = [[0, 0], [1, 0]]

    with pytest.raises(ValueError):
        multi_model.run()
