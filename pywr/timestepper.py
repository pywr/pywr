import pandas

from pywr import _core

class Timestepper(object):
    def __init__(self, start="2015-01-01", end="2015-12-31", delta=1):
        self.start = start
        self.end = end
        self.delta = delta
        self._last_length = None
        self.reset()

    def __iter__(self, ):
        return self

    def __len__(self, ):
        return int((self.end-self.start)/self.delta) + 1

    def reset(self, start=None):
        """ Reset the timestepper

        If start is None it resets to the original self.start, otherwise
        start is used as the new starting point.
        """
        self._current = None
        current_length = len(self)

        if start is None:
            self._next = _core.Timestep(self.start, 0, self.delta.days)
        else:
            # Calculate actual index from new position
            diff = start - self.start
            if diff.days % self.delta.days != 0:
                raise ValueError('New starting position is not compatible with the existing starting position and timestep.')
            index = diff.days / self.delta.days
            self._next = _core.Timestep(start, index, self.delta.days)

        length_changed = self._last_length != current_length
        self._last_length = current_length
        return length_changed

    def __next__(self, ):
        return self.next()

    def next(self, ):
        self._current = current = self._next
        if current.datetime > self.end:
            raise StopIteration()

        # Increment to next timestep
        self._next = _core.Timestep(current.datetime + self.delta, current.index + 1, self.delta.days)

        # Return this timestep
        return current

    def start():
        def fget(self):
            return self._start
        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._start = value
            else:
                self._start = pandas.to_datetime(value)
        return locals()
    start = property(**start())

    def end():
        def fget(self):
            return self._end
        def fset(self, value):
            if isinstance(value, pandas.Timestamp):
                self._end = value
            else:
                self._end = pandas.to_datetime(value)
        return locals()
    end = property(**end())

    def delta():
        def fget(self):
            return self._delta
        def fset(self, value):
            try:
                self._delta = pandas.Timedelta(days=value)
            except TypeError:
                self._delta = pandas.to_timedelta(value)
        return locals()
    delta = property(**delta())

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
        freq = '{}D'.format(self.delta.days)
        return pandas.date_range(self.start, self.end, freq=freq)

    def __repr__(self):
        start = self.start.strftime("%Y-%m-%d")
        end = self.end.strftime("%Y-%m-%d")
        delta = self.delta.days
        return "<Timestepper start=\"{}\" end=\"{}\" delta=\"{}\">".format(
            start, end, delta
        )
