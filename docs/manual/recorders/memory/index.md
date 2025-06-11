# Introduction
Memory recorders are Python classes which internally stores data depending on what they are recording. 
They have two key methods:

- `.values()`. This is implemented for a specific recorder and returns an array containing the one number
for each scenario. For example, 

    - the recorder that calculates the minimum storage at the end of a simulation ([pywr.recorders.MinimumVolumeStorageRecorder]),
    returns an array with the minimum storage for each scenario.
    - a numpy recorder, such as the [pywr.recorders.NumpyArrayNodeRecorder][] which saves the timeseries of a node's flow,
      temporally aggregates the timeseries for each scenario using a user-defined function (min, mean, max, sum, etc.). This
      can be used, for example, to calculate the total operational cost for the node.

- `.aggregated_value()`. This method takes the array from `.values()` and aggregates the numbers among all scenarios using
   using a user-defined function to return one number. This is normally used by an optimisation wrapper.

You can use these two methods to access the reordered data from Python. Other recorders
implement additional properties to read additional information.