from ._recorders import Recorder


class Event(object):
    """ Container for event information """
    def __init__(self, start, scenario_index):
        self.start = start
        self.scenario_index = scenario_index
        self.end = None


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

        self.events = None
        self._current_events = None

    def setup(self):
        pass

    def reset(self):
        self.events = []
        # This list stores if an event is current active in each scenario.
        self._current_events = [None for si in self.model.scenarios.combinations]

    def save(self):
        # Current timestep
        # TODO replace this when dependency branch is merged.
        ts = self.model.timestepper.current
        for si in self.model.scenarios.combinations:
            # Determine if an event is active this time-step/scenario combination
            triggered = self.threshold.index(ts, si)

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
