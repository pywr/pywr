from pywr.recorders import TotalFlowNodeRecorder
from pywr.parameters import ConstantParameter
import pytest
import os
from fixtures import simple_storage_model


TEST_FOLDER = os.path.dirname(__file__)


@pytest.fixture()
def optimisation_model(simple_storage_model):
    model = simple_storage_model

    # Modify the model to have two variables.
    inpt = model.nodes["Input"]
    inpt.max_flow = ConstantParameter(
        model,
        inpt.max_flow,
        lower_bounds=0.0,
        upper_bounds=10.0,
        name="Input flow",
        is_variable=True,
    )

    otpt = model.nodes["Output"]
    otpt.max_flow = ConstantParameter(
        model,
        otpt.max_flow,
        lower_bounds=0.0,
        upper_bounds=10.0,
        name="Output flow",
        is_variable=True,
    )

    # And two objectives.
    TotalFlowNodeRecorder(model, inpt, name="Total inflow", is_objective="maximise")
    TotalFlowNodeRecorder(model, otpt, name="Total outflow", is_objective="maximise")

    return model


def test_variables(optimisation_model):
    """Test that the number of variables, objectives and constraints are correct."""

    model = optimisation_model

    assert len(model.variables) == 2
    assert len(model.objectives) == 2
    assert len(model.constraints) == 0
