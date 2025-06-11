# Recorders 
A key part of Pywr's functionality lies in its recorders, which are used to capture and store data 
generated during a simulation. The following sections explain how recorders work and provide examples on how to
use them, with a particular focus on charting recorded data using the [matplotlib library](https://matplotlib.org).

There are two types of recorders:

- **file recorders** which stores the timeseries results into a file; results can be process with the software of your choice (Python, R or Excel).
- **memory recorders** which stores the results in memory. Once the Python interpreters terminate
or the model instance is destroyed, the results are deleted.

Memory recorders can be configured to record various attributes of a model component. For example, you can record:

- the flow through a link or the demand at a demand node.
- the volume of water in a storage node.
- any parameter, such as cost, reservoir level, control curves position, etc.
- any statistics on the model results (such as min storage, mean flow or flow duration curve).

The following sections explain how to use most of the recorders available in Pywr.
For a comprehensive list of all available recorders, please refer to the
[API reference](../../api/recorders/core/recorder.md).