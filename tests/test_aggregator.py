from functools import partial
from pywr.recorders import Aggregator
from pywr.recorders._recorders import _agg_func_lookup
from scipy.stats import percentileofscore
import numpy as np
import pytest

npy_funcs = {
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "sum": np.sum,
    "median": np.median,
    "product": np.product,
    "all": np.all,
    "any": np.any,
    "count_nonzero": np.count_nonzero,
}


def custom_test_func(array, axis=None):
    return np.sum(array ** 2, axis=axis)


def percentileofscore_with_axis(values, *args, axis=0, **kwargs):
    if values.ndim == 1:
        # For 1D data we just calculate the percentile
        out = percentileofscore(values, *args, **kwargs)
    elif axis == 0:
        # 2D data by axis 0
        out = [
            percentileofscore(values[:, i], *args, **kwargs)
            for i in range(values.shape[1])
        ]
    elif axis == 1:
        # 2D data by axis 1
        out = [
            percentileofscore(values[i, :], *args, **kwargs)
            for i in range(values.shape[0])
        ]
    else:
        raise ValueError('Axis "{}" not supported'.format(axis))
    return out


@pytest.fixture(params=_agg_func_lookup.keys())
def agg_func(request):
    agg_func_name = request.param

    if agg_func_name == "custom":
        # When using custom you assign the function rather than a string.
        agg_func_name = npy_func = custom_test_func
    elif agg_func_name == "percentile":
        agg_func_name = {"func": "percentile", "args": [95], "kwargs": {}}
        npy_func = partial(np.percentile, q=95)
    elif agg_func_name == "percentileofscore":
        agg_func_name = {
            "func": "percentileofscore",
            "kwargs": {"score": 0.5, "kind": "rank"},
        }
        npy_func = partial(percentileofscore_with_axis, score=0.5, kind="rank")
    else:
        npy_func = npy_funcs[agg_func_name]
    return agg_func_name, npy_func


def test_get_set_aggregator(agg_func):
    """Test getter and setter for Aggregator.func"""
    agg_func_name, _ = agg_func
    agg = Aggregator(agg_func_name)
    if isinstance(agg_func_name, dict):
        assert agg.func == agg_func_name["func"]
    else:
        assert agg.func == agg_func_name
    agg.func = "sum"
    assert agg.func == "sum"


def test_aggregator_1d(agg_func):
    """Test Aggregator.aggregate_1d function."""
    agg_func_name, npy_func = agg_func

    agg = Aggregator(agg_func_name)

    data = np.random.rand(10)

    np.testing.assert_allclose(agg.aggregate_1d(data), npy_func(data))


@pytest.mark.parametrize("axis", [0, 1])
def test_aggregator_2d(agg_func, axis):
    """Test Aggregator.aggregate_2d function."""
    agg_func_name, npy_func = agg_func

    agg = Aggregator(agg_func_name)

    data = np.random.rand(100, 10)

    result = agg.aggregate_2d(data, axis=axis)
    assert len(result) == data.shape[1 - axis]
    np.testing.assert_allclose(result, npy_func(data, axis=axis))
