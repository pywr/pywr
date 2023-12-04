"""Utilities for working with pandas DataFrame objects."""
import pandas
from pandas.tseries.offsets import Tick, DateOffset
from pandas._libs.tslibs.period import IncompatibleFrequency
import os
from .hashes import check_hash


class ResamplingError(Exception):
    def __init__(self, original_dataframe, target_index):
        self.original_dataframe = original_dataframe
        self.target_index = target_index
        message = (
            f'Failed to convert input dataframe with index "{original_dataframe.index}" to model index '
            f'"{target_index}".'
        )
        super().__init__(message)


def align_and_resample_dataframe(df, target_index, resample_func="mean"):
    """Align and resample a DataFrame to the provided index.

    This function attempts to align an input dataframe to a target index. The
     index of the incoming dataframe must be time based and will be cast to a
    `PeriodIndex` (using `to_period()`). Up or down sampling is selected based
     on the relative frequencies of the two indexes. The exact method depends
     also on the type of frequencies (offsets) involved (e.g. `Tick` or
     `DataOffset`). Up-sampling uses forward filling. Whereas down-sampling
     uses the provided `resample_func`.

    Parameters
    ==========

    df : `pandas.DataFrame`
        The input data that needs to be aligned and/or resampled.
    target_index : `pandas.PeriodIndex`
        The target index that the input should be aligned and resampled to match.
    resample_func : str, func
        Function to be used when down-sampling from high frequency data to lower
        frequency.

    """
    # Must resample and align the DataFrame to the model.
    start = target_index[0]
    end = target_index[-1]

    if not isinstance(df.index, pandas.PeriodIndex):
        # Converting to period is sometimes unreliable. E.g. with freq='7D'
        # If the target frequency is passed explicitly this can help, but
        # not all Timestamp frequencies convert to Period frequencies. Therefore,
        # this can not be the default.
        try:
            df = df.to_period()
        except AttributeError:
            df = df.to_period(df.index.freq.freqstr)

    if not isinstance(target_index, pandas.PeriodIndex):
        raise ValueError("Period index expected.")

    model_freq = target_index.freq
    df_freq = df.index.freq

    # Determine how to do the resampling based on the frequency type
    # and whether to do up or down sampling.
    if isinstance(model_freq, Tick):
        # Model is tick based e.g. daily or hourly
        if isinstance(df_freq, Tick):
            # Dataframe is also tick based
            if model_freq >= df_freq:
                # Down sampling (i.e. from high freq to lower model freq)
                df = _down_sample_tick_to_tick(df, target_index, resample_func)
            else:
                df = _up_sample_tick_to_tick(df, target_index)
        else:
            # Dataframe must be offset based
            # Generally this is going to be up sampling to a higher model freq
            df = _resample_date_offset_to_tick(df, target_index, resample_func)

    elif isinstance(model_freq, DateOffset):
        # Model is based on date offsets.
        if isinstance(df_freq, Tick):
            # Dataframe is tick based
            # Generally this is going to be down sampling to a lower model freq
            df = _down_sample_tick_to_date_offset(df, target_index, resample_func)
        else:
            # Dataframe must be offset based
            # Down sampling (i.e. from high freq to lower model freq)
            try:
                df = _resample_date_offset_to_date_offset(
                    df, target_index, resample_func
                )
            except IncompatibleFrequency:
                raise ResamplingError(df, target_index)

    df = df[start:end]

    if not df.index.equals(target_index):
        raise ResamplingError(df, target_index)

    if df.isnull().values.any():
        raise ValueError("Missing values detected after resampling dataframe.")

    return df


def _resample_date_offset_to_date_offset(df, target_index, resample_func):
    """Down sample a date offset data to a target index with date offset."""
    new_df = df.resample(target_index.freq).agg(resample_func)
    new_df = new_df[target_index[0] : target_index[-1]]
    return new_df


def _down_sample_tick_to_date_offset(df, target_index, resample_func):
    """Down sampling tick data to a target index with date offset."""
    new_df = df.resample(target_index.freq).agg(resample_func)
    new_df = new_df[target_index[0] : target_index[-1]]
    return new_df


def _down_sample_tick_to_tick(df, target_index, resample_func):
    """Down sampling tick data to a tick based target index."""
    # First we try to align the higher frequency data to lower the frequency
    start = target_index[0].asfreq(df.index.freq, how="start")
    end = target_index[-1].asfreq(df.index.freq, how="end")
    new_df = df[start:end]
    # Second we re-sample the aligned data
    new_df = new_df.resample(target_index.freq).agg(resample_func)
    return new_df


def _resample_date_offset_to_tick(df, target_index, resample_func):
    """Re-sample a date offset to tick based index."""

    target_tick = target_index.freq

    # Create a copy of the target tick with a period of 1
    target_single_tick = target_tick.base

    # Forward fill the data to the single period tick
    new_df = df.resample(target_single_tick).ffill()

    # Now align to the target index
    start = target_index[0].asfreq(target_single_tick, how="start")
    end = target_index[-1].asfreq(target_single_tick, how="end")
    new_df = new_df[start:end]

    # If the target is multi-frequency
    if target_tick.n > 1:
        new_df = new_df.resample(target_tick).agg(resample_func)
    return new_df


def _up_sample_tick_to_tick(df, target_index):
    """Up sample tick data to a tick based target index."""
    new_df = df.resample(target_index.freq).ffill()
    return new_df


def load_dataframe(model, data):
    column = data.pop("column", None)
    if isinstance(column, list):
        # Cast multiindex to a tuple to ensure .loc works correctly
        column = tuple(column)

    index = data.pop("index", None)
    if isinstance(index, list):
        # Cast multiindex to a tuple to ensure .loc works correctly
        index = tuple(index)

    indexes = data.pop("indexes", None)

    table_ref = data.pop("table", None)
    if table_ref is not None:
        name = table_ref
        df = model.tables[table_ref]
    else:
        name = data.get("url", None)
        df = read_dataframe(model, data)

    # if column is not specified, use the whole dataframe
    if column is not None:
        try:
            df = df[column]
        except KeyError:
            raise KeyError('Column "{}" not found in dataset "{}"'.format(column, name))

    if index is not None:
        try:
            df = df.loc[index]
        except KeyError:
            raise KeyError('Index "{}" not found in dataset "{}"'.format(index, name))

    if indexes is not None:
        try:
            df = df.loc[indexes, :]
        except KeyError:
            raise KeyError('Indexes "{}" not found in dataset "{}"'.format(index, name))

    try:
        if isinstance(df.index, pandas.DatetimeIndex):
            # Only infer freq if one isn't already found.
            # E.g. HDF stores the saved freq, but CSV tends to have None, but infer to Weekly for example
            if df.index.freq is None:
                freq = pandas.infer_freq(df.index)
                if freq is None:
                    raise IndexError(
                        'Failed to identify frequency of dataset "{}"'.format(name)
                    )
                df = df.asfreq(freq)
    except AttributeError:
        # Probably wasn't a pandas dataframe at this point.
        pass

    return df


def read_dataframe(model, data):
    # values reference data in an external file
    url = data.pop("url", None)
    if url is not None:
        if not os.path.isabs(url) and model.path is not None:
            url = os.path.join(model.path, url)
    else:
        # Must be an embedded dataframe
        df_data = data.pop("data", None)

    if url is None and df_data is None:
        raise ValueError('No data specified. Provide a "url" or "data" key.')

    if url is not None:
        # Check hashes if given before reading the data
        checksums = data.pop("checksum", {})
        for algo, hash in checksums.items():
            check_hash(url, hash, algorithm=algo)

        try:
            filetype = data.pop("filetype")
        except KeyError:
            # guess file type based on extension
            if url.endswith((".xls", ".xlsx")):
                filetype = "excel"
            elif url.endswith((".csv", ".gz")):
                filetype = "csv"
            elif url.endswith((".hdf", ".hdf5", ".h5")):
                filetype = "hdf"
            else:
                raise NotImplementedError('Unknown file extension: "{}"'.format(url))
    else:
        if "filetype" in data:
            raise ValueError('"filetype" is only valid when loading data from a URL.')
        if "checksum" in data:
            raise ValueError('"checksum" is only valid when loading data from a URL.')

        filetype = "dict"

    data.pop("comment", None)  # remove kwargs from data before passing to Pandas

    if filetype == "csv":
        df = pandas.read_csv(url, **data)  # automatically decompressed gzipped data!
    elif filetype == "excel":
        df = pandas.read_excel(url, **data)
    elif filetype == "hdf":
        key = data.pop("key", None)
        df = pandas.read_hdf(url, key=key, **data)
    elif filetype == "dict":
        parse_dates = data.pop("parse_dates", False)
        df = pandas.DataFrame.from_dict(df_data, **data)
        if parse_dates:
            df.index = pandas.DatetimeIndex(df.index)

    if df.index.dtype.name == "object" and data.get("parse_dates", False):
        # catch dates that haven't been parsed yet
        raise TypeError(
            'Invalid DataFrame index type "{}" in "{}".'.format(
                df.index.dtype.name, url
            )
        )

    # clean up
    # Assume all keywords are consumed by pandas.read_* functions
    data.clear()

    return df
