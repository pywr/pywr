from pywr.dataframe_tools import align_and_resample_dataframe, ResamplingError
import numpy as np
import pandas as pd
import pytest


def make_df(freq, start="2015-01-01", end="2015-12-31"):
    # Daily time-step
    index = pd.period_range(start, end, freq=freq)
    series = pd.DataFrame(np.arange(len(index), dtype=np.float64), index=index)
    return series


def make_model_index(freq, start="2015-01-01", end="2015-12-31"):
    return pd.period_range(start, end, freq=freq)


class TestDownSampling:
    """Test for down-sampling a dataframe to lower frequency model time-step."""

    @pytest.mark.parametrize("resample_func", ["mean", "max"])
    def test_daily_to_monthly(self, resample_func):
        """Test daily data to monthly model time-step."""
        input_df = make_df("D")
        model_index = make_model_index("M")

        input_resampled = align_and_resample_dataframe(
            input_df, model_index, resample_func=resample_func
        )

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(
            input_resampled, input_resampled.resample("M").agg(resample_func)
        )

    @pytest.mark.parametrize("resample_func", ["mean", "max"])
    def test_daily_to_weekly(self, resample_func):
        """Test daily to weekly model time-step."""
        input_df = make_df("D")
        model_index = make_model_index("W")

        input_resampled = align_and_resample_dataframe(
            input_df, model_index, resample_func=resample_func
        )

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(
            input_resampled, input_resampled.resample("W").agg(resample_func)
        )

    @pytest.mark.parametrize("resample_func", ["mean", "max"])
    def test_daily_to_7daily(self, resample_func):
        """Test daily to 7-day model time-step."""
        input_df = make_df("D")
        model_index = make_model_index("7D")

        input_resampled = align_and_resample_dataframe(
            input_df, model_index, resample_func=resample_func
        )

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(
            input_resampled, input_resampled.resample("7D").agg(resample_func)
        )

    @pytest.mark.parametrize("resample_func", ["mean", "max"])
    def test_misaligned_daily_to_7daily(self, resample_func):
        """Test daily to 7-day model time-step."""
        input_df = make_df("D", start="2014-12-20")
        model_index = make_model_index("7D")

        input_resampled = align_and_resample_dataframe(
            input_df, model_index, resample_func=resample_func
        )

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(
            input_resampled, input_resampled.resample("7D").agg(resample_func)
        )

    @pytest.mark.parametrize("resample_func", ["mean", "max"])
    def test_weekly_to_monthly(self, resample_func):
        """Test weekly to monthly model time-step."""
        input_df = make_df("W")
        model_index = make_model_index("M")

        with pytest.raises(ResamplingError):
            input_resampled = align_and_resample_dataframe(
                input_df, model_index, resample_func=resample_func
            )

    @pytest.mark.parametrize("resample_func", ["mean", "max"])
    def test_weekly_to_two_weekly(self, resample_func):
        """Test weekly to two-weekly model time-step."""
        input_df = make_df("W")
        model_index = make_model_index("2W")

        input_resampled = align_and_resample_dataframe(
            input_df, model_index, resample_func=resample_func
        )

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(
            input_resampled, input_resampled.resample("2W").agg(resample_func)
        )


class TestUpSampling:
    """Test for up-sampling a dataframe to higher frequency model time-step."""

    def test_monthly_to_daily(self):
        """Test monthly data to daily model time-step."""
        input_df = make_df("M")
        model_index = make_model_index("D")

        input_resampled = align_and_resample_dataframe(input_df, model_index)

        expected_df = input_df.resample("D").ffill()
        expected_df = expected_df["2015-01-01":"2015-12-31"]

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(input_resampled, expected_df)

    def test_weekly_to_daily(self):
        """Test weekly aligned to daily model time-step."""
        # Note that we have to choose a week starting on Wednesday because 2015-01-01 is a Wednesday.
        input_df = make_df("W-WED")
        model_index = make_model_index("D")

        input_resampled = align_and_resample_dataframe(input_df, model_index)

        expected_df = input_df.resample("D").ffill()
        expected_df = expected_df["2015-01-01":"2015-12-31"]

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(input_resampled, expected_df)

    def test_7daily_to_daily(self):
        """Test weekly aligned to daily model time-step."""
        # Note that we have to choose a week starting on Wednesday because 2015-01-01 is a Wednesday.
        input_df = make_df("7D")
        model_index = make_model_index("D")

        input_resampled = align_and_resample_dataframe(input_df, model_index)

        expected_df = input_df.resample("D").ffill()
        expected_df = expected_df["2015-01-01":"2015-12-31"]

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(input_resampled, expected_df)

    def test_nonaligned_weekly_to_daily(self):
        """Test weekly non-aligned data to daily model time-step."""
        input_df = make_df("W")
        model_index = make_model_index("D")

        input_resampled = align_and_resample_dataframe(input_df, model_index)

        expected_df = input_df.resample("D").ffill()
        expected_df = expected_df["2015-01-01":"2015-12-31"]

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(input_resampled, expected_df)

    @pytest.mark.parametrize("resample_func", ["mean", "max"])
    def test_weekly_to_7daily(self, resample_func):
        """Test weekly to 7-day model time-step."""
        # Note that we have to choose a week starting on Wednesday because 2015-01-01 is a Wednesday.
        input_df = make_df("W-WED")
        model_index = make_model_index("7D")

        input_resampled = align_and_resample_dataframe(
            input_df, model_index, resample_func=resample_func
        )

        expected_df = input_df.resample("D").ffill()
        expected_df = expected_df["2015-01-01":"2015-12-31"]
        expected_df = expected_df.resample("7D").agg(resample_func)

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(input_resampled, expected_df)

    @pytest.mark.parametrize("resample_func", ["mean", "max"])
    def test_nonaligned_weekly_to_7daily(self, resample_func):
        """Test invalid weekly data to 7-day model time-step."""
        # Because the weeks do not align with the 7-day time-step it is not possible to align this data.
        input_df = make_df("W")
        model_index = make_model_index("7D")

        input_resampled = align_and_resample_dataframe(
            input_df, model_index, resample_func=resample_func
        )

        pd.testing.assert_index_equal(input_resampled.index, model_index)

        expected_df = input_df.resample("D").ffill()
        expected_df = expected_df["2015-01-01":"2015-12-31"]
        expected_df = expected_df.resample("7D").agg(resample_func)

        pd.testing.assert_frame_equal(input_resampled, expected_df)

    def test_annual_to_daily(self):

        input_df = make_df("A", "2010-01-01", "2020-01-01")
        model_index = make_model_index("D", "2010-01-01", "2020-01-01")

        input_resampled = align_and_resample_dataframe(input_df, model_index)

        expected_df = input_df.resample("D").ffill()
        expected_df = expected_df["2010-01-01":"2020-01-01"]

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(input_resampled, expected_df)

    def test_annual_to_monthly(self):

        input_df = make_df("A", "2010-01-01", "2020-01-01")
        model_index = make_model_index("M", "2010-01-01", "2020-01-01")

        input_resampled = align_and_resample_dataframe(
            input_df, model_index, resample_func="ffill"
        )

        expected_df = input_df.resample("M").ffill()
        expected_df = expected_df["2010-01-01":"2020-01-01"]

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(input_resampled, expected_df)

    def test_two_weekly_to_weekly(self):
        """Test weekly to two-weekly model time-step."""
        input_df = make_df("2W")
        model_index = make_model_index("W")

        input_resampled = align_and_resample_dataframe(
            input_df, model_index, resample_func="ffill"
        )

        expected_df = input_df.resample("W").ffill()
        expected_df = expected_df["2015-01-01":"2015-12-31"]

        pd.testing.assert_index_equal(input_resampled.index, model_index)
        pd.testing.assert_frame_equal(input_resampled, expected_df)
