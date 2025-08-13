# Introduction
You can build a pywr model in two different ways:
    
1. by writing [Python code](python.md)
2. by writing a [JSON file](json.md) with a specific format (or schema)

The next two sections explain step by step how to build and run a very simple model with a [pywr.nodes.Input][] and 
[pywr.nodes.Output][] node.

!!! tip
    You should prefer the first approach for building very simple models. The JSON format allows
    you to define any mode component without writing Python code and it is easy to share
    with stakeholders who aren’t coders.