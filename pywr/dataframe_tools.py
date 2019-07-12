"""





"""
import pandas
from pandas.tseries.offsets import Tick, DateOffset
from  pandas._libs.tslibs.period import IncompatibleFrequency


class ResamplingError(Exception):
    def __init__(self, original_dataframe, target_index):
        self.original_dataframe = original_dataframe
        self.target_index = target_index
        message = f'Failed to convert input dataframe with index "{original_dataframe.index}" to model index ' \
            f'"{target_index}".'
        super().__init__(message)


def align_and_resample_dataframe(df, datetime_index, resample_func='mean'):
    # Must resample and align the DataFrame to the model.
    start = datetime_index[0]
    end = datetime_index[-1]

    if not isinstance(df.index, pandas.PeriodIndex):
        df = df.to_period(df.index.freq.freqstr)

    if not isinstance(datetime_index, pandas.PeriodIndex):
        raise ValueError('Period index expected.')

    model_freq = datetime_index.freq
    df_freq = df.index.freq

    # Determine how to do the resampling based on the frequency type
    # and whether to do up or down sampling.
    if isinstance(model_freq, Tick):
        # Model is tick based e.g. daily or hourly
        if isinstance(df_freq, Tick):
            # Dataframe is also tick based
            if model_freq >= df_freq:
                # Down sampling (i.e. from high freq to lower model freq)
                df = _down_sample_tick_to_tick(df, datetime_index)
            else:
                df = _up_sample_tick_to_tick(df, datetime_index)
        else:
            # Dataframe must be offset based
            # Generally this is going to be up sampling to a higher model freq
            df = _up_sample_date_offset_to_tick(df, datetime_index)

    elif isinstance(model_freq, DateOffset):
        # Model is based on date offsets.
        if isinstance(df_freq, Tick):
            # Dataframe is tick based
            # Generally this is going to be down sampling to a lower model freq
            df = _down_sample_tick_to_date_offset(df, datetime_index)
        else:
            # Dataframe must be offset based
            # Down sampling (i.e. from high freq to lower model freq)
            try:
                df = _resample_date_offset_to_date_offset(df, datetime_index, resample_func=resample_func)
            except IncompatibleFrequency:
                raise ResamplingError(df, datetime_index)

    df = df[start:end]

    if not df.index.equals(datetime_index):
        raise ResamplingError(df, datetime_index)

    if df.isnull().values.any():
        raise ValueError('Missing values detected after resampling dataframe.')

    return df


def _resample_date_offset_to_date_offset(df, target_index, resample_func):
    """Down sample a date offset data to a target index with date offset."""
    new_df = df.resample(target_index.freq).agg(resample_func)
    new_df = new_df[target_index[0]:target_index[-1]]
    return new_df


def _down_sample_tick_to_date_offset(df, target_index):

    new_df = df.resample(target_index.freq).mean()
    new_df = new_df[target_index[0]:target_index[-1]]
    return new_df


def _down_sample_tick_to_tick(df, target_index):
    # First we try to align the higher frequency data to lower the frequency
    start = target_index[0].asfreq(df.index.freq, how='start')
    end = target_index[-1].asfreq(df.index.freq, how='end')
    new_df = df[start:end]
    # Second we re-sample the aligned data
    new_df = new_df.resample(target_index.freq).mean()
    return new_df


def _up_sample_date_offset_to_tick(df, target_index):
    """Re-sample a date offset to tick based index.
    """

    target_tick = target_index.freq

    # Create a copy of the target tick with a period of 1
    target_single_tick = target_tick.base

    # Forward fill the data to the single period tick
    new_df = df.resample(target_single_tick).ffill()

    # Now align to the target index
    start = target_index[0].asfreq(target_single_tick, how='start')
    end = target_index[-1].asfreq(target_single_tick, how='end')
    new_df = new_df[start:end]

    # If the target is multi-frequency
    if target_tick.n > 1:
        new_df = new_df.resample(target_tick).mean()
    return new_df


def _up_sample_tick_to_tick(df, target_index):

    new_df = df.resample(target_index.freq).ffill()
    return new_df
