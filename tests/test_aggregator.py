
from pywr.recorders import Aggregator
from pywr.recorders._recorders import _agg_func_lookup
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
    "any": np.any
}


def custom_test_func(array, axis=None):
    return np.sum(array**2, axis=axis)


@pytest.fixture(params=_agg_func_lookup.keys())
def agg_func(request):
    agg_func_name = request.param

    if agg_func_name == "custom":
        # When using custom you assign the function rather than a string.
        agg_func_name = npy_func = custom_test_func
    else:
        npy_func = npy_funcs[agg_func_name]
    return agg_func_name, npy_func


def test_get_set_aggregator(agg_func):
    """Test getter and setter for Aggregator.func"""
    agg_func_name, _ = agg_func
    agg = Aggregator(agg_func_name)
    assert agg.func == agg_func_name
    agg.func = "sum"
    assert agg.func == "sum"


def test_aggregator_1d(agg_func):
    """ Test Aggregator.aggregate_1d function. """
    agg_func_name, npy_func = agg_func

    agg = Aggregator(agg_func_name)

    data = np.random.rand(10)

    np.testing.assert_allclose(agg.aggregate_1d(data), npy_func(data))


def test_aggregator_2d(agg_func):
    """ Test Aggregator.aggregate_2d function. """
    agg_func_name, npy_func = agg_func

    agg = Aggregator(agg_func_name)

    data = np.random.rand(100, 10)

    np.testing.assert_allclose(agg.aggregate_2d(data), npy_func(data, axis=0))
