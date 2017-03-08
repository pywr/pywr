from ._recorders import Recorder
from pywr.parameters import Parameter, IndexParameter
import numpy as np
import pandas


class Event(object):
    """ Container for event information """
    def __init__(self, start, scenario_index):
        self.start = start
        self.scenario_index = scenario_index
        self.end = None

    @property
    def duration(self):
        td = self.end.datetime - self.start.datetime
        return td.days


class EventRecorder(Recorder):
    """ Track discerete events based on a threshold parameter.

    The recorder is intended to work with `IndexParameter` objects
     that return a binary value. An event is considered active
     while the threshold returns a non-zero index value.

    The events are stored in a flat list across all scenarios. Each
     event is stored as a separate object.

    Parameters
    ----------
    threshold - IndexParameter
       The parameter that defines the start and end of an event.

     See also
     --------
     `pywr.parameters._thresholds`


     """
    def __init__(self, model, threshold, *args, **kwargs):
        super(EventRecorder, self).__init__(model, *args, **kwargs)
        self.threshold = threshold
        self.threshold.parents.add(self)

        self.events = None
        self._current_events = None

    def setup(self):
        pass

    def reset(self):
        self.events = []
        # This list stores if an event is current active in each scenario.
        self._current_events = [None for si in self.model.scenarios.combinations]

    def after(self):
        # Current timestep
        ts = self.model.timestepper.current

        if isinstance(self.threshold, Recorder):
            all_triggered = np.array(self.threshold.values(), dtype=np.int)
        elif isinstance(self.threshold, IndexParameter):
            all_triggered = self.threshold.get_all_indices()
        elif isinstance(self.threshold, Parameter):
            all_triggered = np.array(self.threshold.get_all_values(), dtype=np.int)
        else:
            raise TypeError("Threshold must be either a Recorder or Parameter instance.")

        for si in self.model.scenarios.combinations:
            # Determine if an event is active this time-step/scenario combination
            triggered = all_triggered[si.global_id]

            # Get the current event
            current_event = self._current_events[si.global_id]
            if current_event is not None:
                # A current event is active
                if triggered:
                    # Current event continues
                    pass
                else:
                    # Current event ends
                    current_event.end = ts
                    self.events.append(current_event)
                    # Event has ended; no further updates
                    current_event = None
            else:
                # No current event
                if triggered:
                    # Start of a new event
                    current_event = Event(ts, si)
                else:
                    # No event active and one hasn't started
                    # Therefore do nothing.
                    pass

            # Update list of current events
            self._current_events[si.global_id] = current_event

    def finish(self):
        ts = self.model.timestepper.current
        # Complete any unfinished events
        for si in self.model.scenarios.combinations:
            # Get the current event
            current_event = self._current_events[si.global_id]
            if current_event is not None:
                # Unfinished event
                current_event.end = ts
                self.events.append(current_event)
                self._current_events[si.global_id] = None

    def to_dataframe(self):
        """ Returns a DataFrame containing all of the events. """

        df = pandas.DataFrame(columns=['scenario_id', 'start', 'end'])

        for evt in self.events:
            df = df.append({
                'scenario_id': evt.scenario_index.global_id,
                'start': evt.start.datetime,
                'end': evt.end.datetime,
            }, ignore_index=True)

        # Coerce ID to correct type. It defaults to float
        df['scenario_id'] = df['scenario_id'].astype(int)
        return df


class EventDurationRecorder(Recorder):
    """ Recorder for the duration of events found by an EventRecorder

    This Recorder uses the results of an EventRecorder to calculate the duration
    of those events in each scenario. Aggregation by scenario is done via
    the pandas.DataFrame.groupby() method.

    Any scenario which has no events will contain a NaN value.

    Parameters
    ----------
    event_recorder : EventRecorder
        EventRecorder instance to calculate the events.

    """
    def __init__(self, model, event_recorder, **kwargs):
        # Optional different method for aggregating across self.recorders scenarios
        agg_func = kwargs.pop('recorder_agg_func', kwargs.get('agg_func'))
        self.recorder_agg_func = agg_func

        super(EventDurationRecorder, self).__init__(model, **kwargs)
        self.event_recorder = event_recorder
        self.event_recorder.parents.add(self)

    def setup(self):
        self._values = np.empty(len(self.model.scenarios.combinations))

    def reset(self):
        self._values[...] = 0.0

    def after(self):
        pass

    def values(self):
        return self._values

    def finish(self):
        df = self.event_recorder.to_dataframe()

        self._values[...] = 0.0
        # No events found
        if len(df) == 0:
            return

        # Calculate duration
        df['duration'] = df['end'] - df['start']
        # Convert to int of days
        df['duration'] = df['duration'].dt.days
        # Drop other columns
        df = df[['scenario_id', 'duration']]

        # Group by scenario ...
        grouped = df.groupby('scenario_id').agg(self.recorder_agg_func)
        # ... and update the internal values
        for index, row in grouped.iterrows():
            self._values[index] = row['duration']
