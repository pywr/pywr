"""Tests for activation function parameters."""
from pywr.parameters import load_parameter
from pywr.recorders import assert_rec
from fixtures import simple_linear_model
import numpy as np
import pytest


class TestBinaryStepParameter:
    @pytest.mark.parametrize(
        [
            "internal_value",
            "output",
        ],
        [
            (
                -1.0,
                1.0,
            ),
            (
                0.0,
                1.0,
            ),
            (
                0.5,
                1.0,
            ),
        ],
    )
    def test_binary_step(self, simple_linear_model, internal_value, output):
        """Test the binary step profile parameter."""

        m = simple_linear_model
        m.timestepper.start = "2015-01-01"
        m.timestepper.end = "2015-12-31"

        if internal_value <= 0.0:
            expected_values = np.zeros(365)
        else:
            expected_values = np.ones(365) * output

        data = {
            "type": "binarystep",
            "value": internal_value,
            "output": output,
        }

        p = load_parameter(m, data)

        @assert_rec(m, p)
        def expected_func(timestep, _scenario_index):
            return expected_values[timestep.index]

        m.run()

    @pytest.mark.parametrize(
        [
            "lb",
            "ub",
        ],
        [
            (
                -1.0,
                1.0,
            ),
            (
                -10.0,
                5.0,
            ),
        ],
    )
    def test_bounds(self, simple_linear_model, lb, ub):

        m = simple_linear_model

        data = {
            "type": "binarystep",
            "value": 1.0,
            "output": 1.0,
            "lower_bounds": lb,
            "upper_bounds": ub,
        }

        p = load_parameter(m, data)
        np.testing.assert_allclose(p.get_double_lower_bounds(), [lb])
        np.testing.assert_allclose(p.get_double_upper_bounds(), [ub])


class TestRectifierParameter:
    @pytest.mark.parametrize(
        ["internal_value", "max_output", "upper_bounds", "min_output"],
        [
            (-1.0, 1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0, 0.0),
            (0.5, 1.0, 1.0, 0.0),
            (0.5, 10.0, 1.0, 0.0),
            (0.5, 10.0, 5.0, 0.0),
            (1.0, 10.0, 1.0, 0.0),
            (0.0, 10.0, 1.0, 5.0),
            (0.5, 10.0, 1.0, 5.0),
            (1.0, 10.0, 1.0, 5.0),
        ],
    )
    def test_binary_step(
        self, simple_linear_model, internal_value, max_output, upper_bounds, min_output
    ):
        """Test the binary step profile parameter."""

        m = simple_linear_model
        m.timestepper.start = "2015-01-01"
        m.timestepper.end = "2015-12-31"

        if internal_value <= 0.0:
            expected_values = np.zeros(365)
        else:
            range = max_output - min_output
            expected_values = np.ones(365) * (
                min_output + range * internal_value / upper_bounds
            )

        data = {
            "type": "rectifier",
            "value": internal_value,
            "max_output": max_output,
            "min_output": min_output,
            "upper_bounds": upper_bounds,
        }

        p = load_parameter(m, data)

        @assert_rec(m, p)
        def expected_func(timestep, _scenario_index):
            return expected_values[timestep.index]

        m.run()

    @pytest.mark.parametrize(
        [
            "lb",
            "ub",
        ],
        [
            (
                -1.0,
                1.0,
            ),
            (
                -10.0,
                5.0,
            ),
        ],
    )
    def test_bounds(self, simple_linear_model, lb, ub):

        m = simple_linear_model

        data = {
            "type": "rectifier",
            "value": 1.0,
            "max_output": 1.0,
            "lower_bounds": lb,
            "upper_bounds": ub,
        }

        p = load_parameter(m, data)
        np.testing.assert_allclose(p.get_double_lower_bounds(), [lb])
        np.testing.assert_allclose(p.get_double_upper_bounds(), [ub])


class TestLogisticParameter:
    @pytest.mark.parametrize(
        [
            "internal_value",
            "max_output",
            "growth_rate",
        ],
        [
            (-1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.5, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ],
    )
    def test_logistic(
        self, simple_linear_model, internal_value, max_output, growth_rate
    ):
        """Test the binary step profile parameter."""

        m = simple_linear_model
        m.timestepper.start = "2015-01-01"
        m.timestepper.end = "2015-12-31"

        expected_values = (
            np.ones(365) * max_output / (1.0 + np.exp(-growth_rate * internal_value))
        )

        data = {
            "type": "logistic",
            "value": internal_value,
            "max_output": max_output,
        }

        p = load_parameter(m, data)

        @assert_rec(m, p)
        def expected_func(timestep, _scenario_index):
            return expected_values[timestep.index]

        m.run()

    @pytest.mark.parametrize(
        [
            "lb",
            "ub",
        ],
        [
            (
                -1.0,
                1.0,
            ),
            (
                -10.0,
                5.0,
            ),
        ],
    )
    def test_bounds(self, simple_linear_model, lb, ub):

        m = simple_linear_model

        data = {
            "type": "logistic",
            "value": 1.0,
            "max_output": 1.0,
            "lower_bounds": lb,
            "upper_bounds": ub,
        }

        p = load_parameter(m, data)
        np.testing.assert_allclose(p.get_double_lower_bounds(), [lb])
        np.testing.assert_allclose(p.get_double_upper_bounds(), [ub])
