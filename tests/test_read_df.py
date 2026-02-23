import pandas as pd
import numpy as np
import pytest

from pywr.dataframe_tools import read_dataframe


class DummyModel:
    def __init__(self, path=None):
        self.path = path

    def check_hash(self, url, hash, algorithm=None):
        pass


def dataframe_for_test():
    return pd.DataFrame(
        {"a": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="D")
    )


def test_read_dataframe_csv(tmp_path):
    df = dataframe_for_test()
    file = tmp_path / "test.csv"
    df.to_csv(file)
    model = DummyModel(path=str(tmp_path))
    data = {"url": "test.csv", "index_col": 0, "parse_dates": [0]}
    result = read_dataframe(model, data.copy())
    assert np.allclose(result["a"], df["a"])
    assert isinstance(result.index, pd.DatetimeIndex)


def test_read_dataframe_excel(tmp_path):
    df = dataframe_for_test()
    file = tmp_path / "test.xlsx"
    df.to_excel(file)
    model = DummyModel(path=str(tmp_path))
    data = {"url": "test.xlsx", "index_col": 0}
    result = read_dataframe(model, data.copy())
    assert np.allclose(result["a"], df["a"])


def test_read_dataframe_hdf(tmp_path):
    df = dataframe_for_test()
    file = tmp_path / "test.h5"
    df.to_hdf(file, key="mykey")
    model = DummyModel(path=str(tmp_path))
    data = {"url": "test.h5", "key": "mykey"}
    result = read_dataframe(model, data.copy())
    assert np.allclose(result["a"], df["a"])


def test_read_dataframe_dict():
    df = dataframe_for_test()
    model = DummyModel()
    data = {"data": df.to_dict(orient="list")}
    result = read_dataframe(model, data.copy())
    assert np.allclose(result["a"], df["a"])


def test_read_dataframe_rds(tmp_path):
    pytest.importorskip("pyreadr")
    import pyreadr

    df = pd.DataFrame(
        {"a": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3, freq="D")
    )
    file = tmp_path / "test.rds"
    pyreadr.write_rds(str(file), df)
    model = DummyModel(path=str(tmp_path))
    data = {"url": "test.rds", "key": None}
    result = read_dataframe(model, data.copy())
    assert np.allclose(result["a"], df["a"])
    assert isinstance(result.index, pd.DatetimeIndex) or isinstance(
        result.index, pd.Index
    )
