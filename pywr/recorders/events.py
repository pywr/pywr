from ._recorders import Recorder, load_recorder
import numpy as np
import pandas


class Event(object):
    """Container for event information"""

    def __init__(self, start, scenario_index):
        self.start = start
        self.scenario_index = scenario_index
        self.end = None
        self.values = None  # to record any  tracked values

    @property
    def duration(self):
        td = self.end.datetime - self.start.datetime
        return td.days


class EventRecorder(Recorder):
    """Track discrete events using a Parameter or Recorder

    The recorder works with an `IndexParameter`, `Parameter` or `Recorder`. An
    event is considered active while the value of the threshold is non-zero.

    The events are stored in a flat list across all scenarios. Each
    event is stored as a separate `Event` object. Events can be accessed as a
    dataframe using the `to_dataframe` method.

    Parameters
    ----------
    threshold - IndexParameter, Parameter or Recorder
       The object that defines the start and end of an event.
    minimum_event_length - int (default=1)
        The minimum number of time-steps that an event must last for
        to be recorded. This is useful to not record events that are
        caused by model hysteresis. The default will cause all events
        to be recorded.
    agg_func - string, callable
        Function used for aggregating across the recorders. Numpy style functions that
        support an axis argument are supported.
    event_agg_func - string, callable
        Optional different function for aggregating the `tracked_parameter` across events.
        If given this aggregation will be added as a `value` column in the `to_dataframe` method.
    tracked_parameter - `Parameter`
        The parameter to track across each event. The values from this parameter are appended each
        time-step to each event. These can then be used with other event recorders for statistical
        aggregation, or with `event_agg_func`.

     See also
     --------
     `pywr.parameters._thresholds`


    """

    def __init__(
        self, model, threshold, minimum_event_length=1, tracked_parameter=None, **kwargs
    ):
        self.event_agg_func = kwargs.pop("event_agg_func", kwargs.get("agg_func"))
        super(EventRecorder, self).__init__(model, **kwargs)
        self.threshold = threshold
        self.threshold.parents.add(self)
        if minimum_event_length < 1:
            raise ValueError('Keyword "minimum_event_length" must be >= 1')
        self.minimum_event_length = minimum_event_length
        self.events = None
        self._current_events = None
        # TODO make this more generic to track components or  nodes (e.g. storage volume)
        self.tracked_parameter = tracked_parameter
        if self.tracked_parameter is not None:
            self.tracked_parameter.parents.add(self)

    def setup(self):
        pass

    def reset(self):
        self.events = []
        # This list stores if an event is current active in each scenario.
        self._current_events = [None for si in self.model.scenarios.combinations]

    def after(self):
        # Current timestep
        ts = self.model.timestepper.current

        from pywr.parameters import Parameter, IndexParameter

        if isinstance(self.threshold, Recorder):
            all_triggered = np.array(self.threshold.values(), dtype=int)
        elif isinstance(self.threshold, IndexParameter):
            all_triggered = self.threshold.get_all_indices()
        elif isinstance(self.threshold, Parameter):
            all_triggered = np.array(self.threshold.get_all_values(), dtype=int)
        else:
            raise TypeError(
                "Threshold must be either a Recorder or Parameter instance."
            )

        for si in self.model.scenarios.combinations:
            # Determine if an event is active this time-step/scenario combination
            triggered = all_triggered[si.global_id]

            # Get the current event
            current_event = self._current_events[si.global_id]
            if current_event is not None:
                # A current event is active
                if triggered:
                    # Current event continues
                    # Update the timeseries of event data
                    if self.tracked_parameter is not None:
                        value = self.tracked_parameter.get_value(si)
                        current_event.values.append(value)
                else:
                    # Update the end of the current event.
                    current_event.end = ts
                    current_event.values = np.array(
                        current_event.values
                    )  # Convert list to nparray
                    current_length = ts.index - current_event.start.index

                    if current_length >= self.minimum_event_length:
                        # Current event ends
                        self.events.append(current_event)
                        # Event has ended; no further updates
                        current_event = None
                    else:
                        # Event wasn't long enough; don't append
                        current_event = None
            else:
                # No current event
                if triggered:
                    # Start of a new event
                    current_event = Event(ts, si)
                    # Start the timeseries of event data
                    if self.tracked_parameter is not None:
                        value = self.tracked_parameter.get_value(si)
                        current_event.values = [
                            value,
                        ]
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
        """Returns a `pandas.DataFrame` containing all of the events.

        If `event_agg_func` is a valid aggregation function and `tracked_parameter`
         is given then a "value" column is added to the dataframe containing the
         result of the aggregation.

        """
        # Return empty dataframe if no events are found.
        if len(self.events) == 0:
            return pandas.DataFrame(columns=["scenario_id", "start", "end"])

        scen_id = np.empty(len(self.events), dtype=int)
        start = np.empty_like(scen_id, dtype=object)
        end = np.empty_like(scen_id, dtype=object)
        values = np.empty_like(scen_id, dtype=float)

        for i, evt in enumerate(self.events):
            scen_id[i] = evt.scenario_index.global_id
            start[i] = evt.start.datetime
            end[i] = evt.end.datetime
            if self.tracked_parameter is not None and self.event_agg_func is not None:
                values[i] = pandas.Series(evt.values).aggregate(self.event_agg_func)

        df_dict = {"scenario_id": scen_id, "start": start, "end": end}
        if self.tracked_parameter is not None and self.event_agg_func is not None:
            df_dict["value"] = values

        return pandas.DataFrame(df_dict)

    @classmethod
    def load(cls, model, data):
        from pywr.parameters import load_parameter

        threshold = data.pop("threshold")
        try:
            threshold = load_parameter(model, threshold)
        except KeyError:
            threshold = load_recorder(model, threshold)
        tracked_param = data.pop("tracked_parameter", None)
        if tracked_param:
            tracked_param = load_parameter(model, tracked_param)
        return cls(model, threshold, tracked_parameter=tracked_param, **data)


EventRecorder.register()


class EventDurationRecorder(Recorder):
    """Recorder for the duration of events found by an EventRecorder

    This Recorder uses the results of an EventRecorder to calculate the duration
    of those events in each scenario. Aggregation by scenario is done via
    the pandas.DataFrame.groupby() method.

    Any scenario which has no events will contain a NaN value.

    Parameters
    ----------
    event_recorder : EventRecorder
        EventRecorder instance to calculate the events.
    agg_func - string, callable
        Function used for aggregating across the recorders. Numpy style functions that
        support an axis argument are supported. Defulat value is 'mean'.
    recorder_agg_func - string, callable
        Optional aggregating function for all events in each scenario. The function
        must be supported by the `DataFrame.group_by` method.  If no value is provided
        then the value of 'agg_func' is used.
    """

    def __init__(self, model, event_recorder, **kwargs):
        # Optional different method for aggregating across self.recorders scenarios
        agg_func = kwargs.pop("recorder_agg_func", kwargs.get("agg_func", "mean"))
        self.recorder_agg_func = agg_func

        super(EventDurationRecorder, self).__init__(model, **kwargs)
        self.event_recorder = event_recorder
        self.event_recorder.parents.add(self)

    def setup(self):
        self._values = np.empty(len(self.model.scenarios.combinations))

    def reset(self):
        self._values[...] = 0.0

    def values(self):
        return self._values

    def finish(self):
        df = self.event_recorder.to_dataframe()

        self._values[...] = 0.0
        # No events found
        if len(df) == 0:
            return

        # Calculate duration
        df["duration"] = df["end"] - df["start"]
        # Convert to int of days
        df["duration"] = df["duration"].dt.days
        # Drop other columns
        df = df[["scenario_id", "duration"]]

        # Group by scenario ...
        grouped = df.groupby("scenario_id").agg(self.recorder_agg_func)
        # ... and update the internal values
        for index, row in grouped.iterrows():
            self._values[index] = row["duration"]

    @classmethod
    def load(cls, model, data):
        event_rec = load_recorder(model, data.pop("event_recorder"))
        return cls(model, event_rec, **data)


EventDurationRecorder.register()


class EventStatisticRecorder(Recorder):
    """Recorder for the duration of events found by an EventRecorder

    This Recorder uses the results of an EventRecorder to calculate aggregated statistics
    of those events in each scenario. This requires the EventRecorder to be given a `tracked_parameter`
    in order to save an array of values during each event. This recorder uses `event_agg_func` to aggregate
    those saved values in each event before applying `recorder_agg_func` to those values in each scenario.
    Aggregation by scenario is done via the pandas.DataFrame.groupby() method.

    Any scenario which has no events will contain a NaN value regardless of the aggregation function defined.

    Parameters
    ----------
    model : pywr.model.Model
    event_recorder : EventRecorder
        EventRecorder instance to calculate the events.
    agg_func - string, callable
        Function used for aggregating across the recorders. Numpy style functions that
        support an axis argument are supported. Default value is 'mean'.
    recorder_agg_func - string, callable
        Optional aggregating function for all events in each scenario. The function
        must be supported by the `DataFrame.group_by` method. If no value is provided
        then the value of 'agg_func' is used.
    event_agg_func - string, callable
        Optional different function for aggregating the `tracked_parameter` across events.
        If given this aggregation will be added as a `value` column in the `to_dataframe` method.
        If no value is provided then the value of 'agg_func' is used.
    """

    def __init__(self, model, event_recorder, **kwargs):
        # Optional different method for aggregating across self.recorders scenarios
        agg_func = kwargs.pop("event_agg_func", kwargs.get("agg_func", "mean"))
        self.event_agg_func = agg_func
        agg_func = kwargs.pop("recorder_agg_func", kwargs.get("agg_func", "mean"))
        self.recorder_agg_func = agg_func

        super(EventStatisticRecorder, self).__init__(model, **kwargs)
        self.event_recorder = event_recorder
        self.event_recorder.parents.add(self)

    def setup(self):
        self._values = np.empty(len(self.model.scenarios.combinations))

        if self.event_recorder.tracked_parameter is None:
            raise ValueError(
                "To calculate event statistics requires the parent `EventRecorder` to have a `tracked_parameter`."
            )

    def reset(self):
        self._values[...] = np.nan

    def values(self):
        return self._values

    def finish(self):
        """Compute the aggregated value in each scenario based on the parent `EventRecorder` events"""
        events = self.event_recorder.events
        # Return NaN if no events found
        if len(events) == 0:
            return

        scen_id = np.empty(len(events), dtype=int)
        values = np.empty_like(scen_id, dtype=np.float64)

        for i, evt in enumerate(events):
            scen_id[i] = evt.scenario_index.global_id
            values[i] = pandas.Series(evt.values).aggregate(self.event_agg_func)

        df = pandas.DataFrame({"scenario_id": scen_id, "value": values})

        # Group by scenario ...
        grouped = df.groupby("scenario_id").agg(self.recorder_agg_func)
        # ... and update the internal values
        for index, row in grouped.iterrows():
            self._values[index] = row["value"]

    @classmethod
    def load(cls, model, data):
        event_rec = load_recorder(model, data.pop("event_recorder"))
        return cls(model, event_rec, **data)


EventStatisticRecorder.register()
