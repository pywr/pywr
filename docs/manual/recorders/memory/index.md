# Introduction
Memory recorders are Python classes which internally store data depending on what they are recording. 
They have two key methods:

- `.values()`. This is implemented for a specific recorder and **returns an array containing one number
for each scenario**. For example, 

    - the recorder that calculates the minimum storage at the end of a simulation ([pywr.recorders.MinimumVolumeStorageRecorder]),
    returns an array with the minimum storage for each scenario.
    - a numpy recorder, such as the [pywr.recorders.NumpyArrayNodeRecorder][] which saves the timeseries of a node's flow,
      temporally aggregates the timeseries for each scenario using a user-defined function (min, mean, max, sum, etc.). This
      can be used, for example, to calculate the total delivered flow from a node for each scenario.

- `.aggregated_value()`. This method takes the array from `.values()` and aggregates the numbers among all scenarios
   using a user-defined function to **return one number**. This is normally used by an optimisation wrapper.

You can use these two methods to access the reordered data from Python. Other recorders
implement additional properties to read additional information.

## Aggregation functions
The following aggregation functions can be used in the `.aggregated_value()` method (and `.values()` when this method
supports a temporal aggregation):

- sum
- min
- max
- mean
- median
- product
- [percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
- [percentileofscore](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html)
- [count_nonzero](https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html#numpy-count-nonzero)

The aggregation function is a parameter supported by all recorders. 