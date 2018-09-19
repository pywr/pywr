import pandas
import datetime
from pywr import _core

class Timestepper(object):
    def __init__(self, start="2015-01-01", end="2015-12-31", delta=1):
        self.start = start
        self.end = end
        self.delta = delta
        self._last_length = None
        self._periods = None
        self.setup()
        self.reset()
        self._dirty = True

    def __iter__(self, ):
        return self

    def __len__(self, ):
        return len(self._periods)

    @property
    def dirty(self):
        return self._dirty

    def setup(self):
        self._periods = self.datetime_index
        self.reset()
        self._dirty = False

    def reset(self, start=None):
        """ Reset the timestepper

        If start is None it resets to the original self.start, otherwise
        start is used as the new starting point.
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
                raise ValueError('New starting position is outside the range of the model timesteps.')

        period = self.start_period
        for _ in range(current_index):
            period += self.offset
        self._next = _core.Timestep(period, current_index)

        length_changed = self._last_length != current_length
        self._last_length = current_length
        return length_changed

    def __next__(self, ):
        return self.next()

    def next(self, ):
        self._current = current = self._next

        if current.period > self.end_period:
            raise StopIteration()

        # Increment to next timestep
        self._next = _core.Timestep(current.period + self.offset, current.index + 1)

        # Return this timestep
        return current
    #
    # def next(self, ):
    #     if self._next is None:
    #         raise StopIteration()
    #
    #     self._current = current = self._next
    #     # Increment to next timestep
    #     self._current_index += 1
    #     try:
    #         period = self._periods[self._current_index]
    #     except IndexError:
    #         self._next = None
    #     else:
    #         self._next = _core.Timestep(period, self._current_index)
    #
    #     # Return this timestep
    #     return current

    def start():
        def fget(self):
            return self._start
        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._start = value
            else:
                self._start = pandas.to_datetime(value)
            self._dirty = True
        return locals()
    start = property(**start())

    @property
    def start_period(self):
        return pandas.Period(self.start, freq=self.freq)

    def end():
        def fget(self):
            return self._end
        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._end = value
            else:
                self._end = pandas.to_datetime(value)
            self._dirty = True
        return locals()
    end = property(**end())

    @property
    def end_period(self):
        return pandas.Period(self.end, freq=self.freq)

    def delta():
        def fget(self):
            return self._delta
        def fset(self, value):
            self._delta = value
            self._dirty = True
        return locals()
    delta = property(**delta())

    @property
    def freq(self):
        d = self._delta
        if isinstance(d, int):
            freq = '{}D'.format(d)
        elif isinstance(d, datetime.timedelta):
            freq = '{}D'.format(d.days)
        else:
            freq = d
        return freq

    @property
    def offset(self):
        return pandas.tseries.frequencies.to_offset(self.freq)

    @property
    def current(self):
        """The current timestep

        If iteration has not begun this will return None.
        """
        return self._current

    @property
    def datetime_index(self):
        """ Return a `pandas.DatetimeIndex` using the start, end and delta of this object

        This is useful for creating `pandas.DataFrame` objects from Model results
        """
        return pandas.period_range(self.start, self.end, freq=self.freq)

    def __repr__(self):
        start = self.start.strftime("%Y-%m-%d")
        end = self.end.strftime("%Y-%m-%d")
        return "<Timestepper start=\"{}\" end=\"{}\" freq=\"{}\">".format(
            start, end, self.freq
        )
