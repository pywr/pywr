import os
from pathlib import Path

import numpy as np
import pandas

from pywr.parameters import DataFrameParameter
from pywr.model import MultiModel, Model

# def load_multi_model(filename: Path):
from pywr.nodes import Input, Link, Output
from pywr.parameters.multi_model_parameters import OtherModelParameterValue
from pywr.recorders import NumpyArrayNodeRecorder


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
    assert isinstance(supply2.max_flow, OtherModelParameterValue)

    # Add recorder for flow
    demand2_rec = NumpyArrayNodeRecorder(sub_model2, supply2)
    # Demand should equal the inflow the model1
    expected_flow = pandas.read_csv(path / "timeseries1.csv", index_col=0)

    multi_model.run()

    np.testing.assert_allclose(
        multi_model.models["model1"].nodes["demand1"].flow[0], 10.0
    )
    np.testing.assert_allclose(demand2_rec.data, expected_flow)
