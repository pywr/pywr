from typing import Union

import pandas
import numpy as np
import datetime


from pywr import _core

# Constants
SECONDS_IN_DAY = 3600 * 24


class Timestepper(object):
    """
    The timestepper object to handle the start, end date, and the model timestep.

    Attributes
    ----------
    start : str
        The start date of the model simulation.
    end : str
        The end date of the model simulation.
    delta: int
        The time step in days.
    """

    def __init__(self, start="2015-01-01", end="2015-12-31", delta=1):
        """Initialise the class.

        Parameters
        ----------
        start : str
            The start date of the model simulation.
        end : str
            The end date of the model simulation.
        delta: int
            The time step in days.
        """
        self.start = start
        self.end = end
        self.delta = delta
        self._last_length = None
        self._periods = None
        self._deltas = None
        self._current = None
        self._next = None
        self.setup()
        self.reset()
        self._dirty = True

    def __iter__(
        self,
    ):
        return self

    def __len__(
        self,
    ):
        return len(self._periods)

    @property
    def dirty(self):
        return self._dirty

    def setup(self):
        """
        Setup the timestep.

        Returns
        -------
        None
            This function returns `None`.
        """
        periods = self.datetime_index

        # Compute length of each period
        deltas = periods.to_timestamp(how="e") - periods.to_timestamp(how="s")
        # Round to nearest second
        deltas = np.round(deltas.total_seconds())
        # Convert to days
        deltas = deltas / SECONDS_IN_DAY
        self._periods = periods
        self._deltas = deltas
        self.reset()
        self._dirty = False

    def reset(self, start=None):
        """Reset the timestepper.

        Parameters
        ----------
        start: pd.Timestamp | None
            The start date. If None it resets to the original `self.start`,
            otherwise start is used as the new starting point.

        Returns
        -------
        None
            This function returns `None`.
        """
        self._current = None
        current_length = len(self)

        if start is None:
            current_index = 0
        else:
            # Determine which period the start time is within
            for index, period in enumerate(self._periods):
                if period.start_time <= start < period.end_time:
                    current_index = index
                    break
            else:
                raise ValueError(
                    "New starting position is outside the range of the model timesteps."
                )

        self._next = _core.Timestep(
            self._periods[current_index], current_index, self._deltas[current_index]
        )
        length_changed = self._last_length != current_length
        self._last_length = current_length
        return length_changed

    def __next__(
        self,
    ):
        return self.next()

    def next(
        self,
    ):
        """
        Advance the timestep.

        Returns
        -------
        None
            This function returns `None`.
        """
        self._current = current = self._next

        if current.index >= len(self._periods):
            raise StopIteration()

        # Increment to next timestep
        next_index = current.index + 1
        if next_index >= len(self._periods):
            # The final time-step is one offset beyond the end of the model.
            # Here we compute its delta and create the object.
            final_period = current.period + self.offset
            delta = final_period.end_time - final_period.start_time
            delta = np.round(delta.total_seconds())
            delta = delta / SECONDS_IN_DAY
            self._next = _core.Timestep(final_period, next_index, delta)
        else:
            self._next = _core.Timestep(
                self._periods[next_index], next_index, self._deltas[next_index]
            )

        # Return this timestep
        return current

    @property
    def start(self):
        """
        The timestepper start date.

        **Setter:** set the start date as `pandas.Timestamp` or any type accepted
        by `pandas.to_datetime`.

        Returns
        -------
        start : pandas.Timestamp
            The start date.
        """
        return self._start

    @start.setter
    def start(self, value):
        if isinstance(value, pandas.Timestamp):
            self._start = value
        else:
            self._start = pandas.to_datetime(value)
        self._dirty = True

    @property
    def start_period(self) -> pandas.Period:
        """
        Get the start date as pandas `Period`.

        Returns
        -------
        period: pandas.Period
            The period.
        """
        return pandas.Period(self.start, freq=self.freq)

    @property
    def end(self) -> pandas.Timestamp:
        """
        Get the end date.

        **Setter:** set the end date as `pandas.Timestamp` or any type accepted
        by `pandas.to_datetime`.

        Returns
        -------
        period: pandas.Timestamp
            The end date.
        """
        return self._end

    @end.setter
    def end(self, value: Union[pandas.Period, datetime.datetime]) -> None:
        """
        Set the timestepper's end date.

        Parameters
        ----------
        value: pd.Period
            The end date to set.

        Returns
        -------
        None
            This function returns `None`.
        """
        if isinstance(value, pandas.Timestamp):
            self._end = value
        else:
            self._end = pandas.to_datetime(value)
        self._dirty = True

    @property
    def end_period(self) -> pandas.Period:
        """
        Get the end date as pandas `Period`.

        Returns
        -------
        period: pandas.Period
            The period.
        """
        return pandas.Period(self.end, freq=self.freq)

    @property
    def delta(self) -> str:
        """
        Get the delta.

        **Setter:** Set a new delta as `Union[str, int, datetime.timedelta]`. This
        can be a string for example "1D" or an integer for the number of days or
        a `datetime.timedelta` object.

        Returns
        -------
        delta: str
            The timestepper delta as string.
        """
        return self._delta

    @delta.setter
    def delta(self, value: Union[str, int, datetime.timedelta]) -> None:
        """
        Set the timestepper's delta.

        Parameters
        ----------
        value : Union[str, int, datetime.timedelta]
        The delta to set. This can be a string for example "1D" or an integer for the
        number of days or a `datetime.timedelta` object.

        Returns
        -------
        None
            This function returns `None`.

        """
        self._delta = value
        self._dirty = True

    @property
    def freq(self) -> str:
        """
        Get the frequency as string. This represents the time-step followed by the frequency
        identifier, for example `3D` for three days.

        Returns
        -------
        frequency : str
            The frequency as string.
        """
        d = self._delta
        if isinstance(d, int):
            freq = "{}D".format(d)
        elif isinstance(d, datetime.timedelta):
            freq = "{}D".format(d.days)
        else:
            freq = d
        return freq

    @property
    def offset(self) -> pandas.DateOffset:
        """
        Get the timestepper offset as `pandas.DateOffset`.

        Returns
        -------
        offset : pandas.DateOffset
            The offset.
        """
        # noinspection PyTypeChecker
        return pandas.tseries.frequencies.to_offset(self.freq)

    @property
    def current(self) -> Union["Timestepper", None]:
        """The current timestep.

        Returns
        -------
        current : Union["Timestepper", None]
            The current timestep or `None` if the iteration has not begun.
        """
        return self._current

    @property
    def datetime_index(self) -> pandas.PeriodIndex:
        """Return a `pandas.PeriodIndex` using the start, end and delta of this object.

        This is useful for creating `pandas.DataFrame` objects from Model results.

        Returns
        -------
            object : pandas.PeriodIndex
        The `PeriodIndex` object using the start, end and delta.
        """
        return pandas.period_range(self.start, self.end, freq=self.freq)

    def __repr__(self):
        start = self.start.strftime("%Y-%m-%d")
        end = self.end.strftime("%Y-%m-%d")
        return '<Timestepper start="{}" end="{}" freq="{}">'.format(
            start, end, self.freq
        )
