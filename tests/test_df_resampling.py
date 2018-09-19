from pywr.parameters import align_and_resample_dataframe
import numpy as np
import pandas as pd
import pytest


def make_df(freq, start='2015-01-01', end='2015-12-31'):
    # Daily time-step
    index = pd.date_range(start, end, freq=freq)
    series = pd.Series(np.arange(len(index), dtype=np.float64), index=index)
    return series


def test_upsampling_D_7D():
    """Test aligning and resampling daily data to 7-day data.
    """
    # Example daily data
    df1D = make_df('D')

    # Date range we want to end up with
    index = pd.period_range('2015-01-01', '2015-12-31', freq='7D')

    df7D = align_and_resample_dataframe(df1D, index)

    assert df7D.index[0] == index[0].start_time
    assert df7D.index[-1] == index[-1].start_time
    assert df7D.index.freq == index.freq
    assert df7D[0] == 3.0
    assert df7D[1] == 10.0
    assert df7D[2] == 17.0

    # Required range now one day after start of our data
    index = pd.period_range('2015-01-02', '2015-12-31', freq='7D')

    df7D = align_and_resample_dataframe(df1D, index)

    assert df7D.index[0] == index[0].start_time
    assert df7D.index[-1] == index[-1].start_time
    assert df7D.index.freq == index.freq
    assert df7D[0] == 4.0
    assert df7D[1] == 11.0
    assert df7D[2] == 18.0

    # Required range now one day before start of our data
    index = pd.period_range('2014-12-31', '2015-12-31', freq='7D')

    with pytest.raises(ValueError):
        df7D = align_and_resample_dataframe(df1D, index)

    # Required range now after end of our data
    index = pd.period_range('2015-01-01', '2016-01-31', freq='7D')

    with pytest.raises(ValueError):
        df7D = align_and_resample_dataframe(df1D, index)


def test_upsampling_2D_7D():
    """Test aligning and resampling bi-daily data to 7-day data.
    """

    # Example bi-daily data
    df2D = make_df('2D')

    # Date range we want to end up with
    index = pd.period_range('2015-01-01', '2015-12-31', freq='7D')

    df7D = align_and_resample_dataframe(df2D, index)

    assert df7D.index.freq == index.freq
    assert df7D[0] == 1.5
    assert df7D[1] == 5.0
    assert df7D[2] == 8.5

    assert df7D.index[0] == index[0].start_time
    assert df7D.index[-1] == index[-1].start_time

    # Required range now one day after start of our data
    index = pd.period_range('2015-01-02', '2015-12-31', freq='7D')

    # This should raise an error because the resulting index
    # won't align with the desired on.
    with pytest.raises(ValueError):
        df7D = align_and_resample_dataframe(df2D, index)

    # Required range now two days after start of our data
    index = pd.period_range('2015-01-03', '2015-12-31', freq='7D')

    df7D = align_and_resample_dataframe(df2D, index)

    assert df7D.index[0] == index[0].start_time
    # The end does not necessarily align
    # assert df7D.index[-1] == index[-1]

    assert df7D.index.freq == index.freq
    assert df7D[0] == 2.5
    assert df7D[1] == 6.0
    assert df7D[2] == 9.5

    # Required range now one day before start of our data
    index = pd.period_range('2014-12-31', '2015-12-31', freq='7D')

    with pytest.raises(ValueError):
        df7D = align_and_resample_dataframe(df2D, index)

    # Required range now after end of our data
    index = pd.period_range('2015-01-01', '2016-01-31', freq='7D')

    with pytest.raises(ValueError):
        df7D = align_and_resample_dataframe(df2D, index)