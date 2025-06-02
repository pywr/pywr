# Abstraction license with flow bands
This section explains how to implement a river intake where you have an abstraction license that limits 
the abstraction based on the river's flow. This can be achieved by using the [pywr.parameters.MultipleThresholdIndexParameter][] and 
the [pywr.parameters.IndexedArrayParameter][].

In this example the maximum abstraction is 14.184 Ml/d subjected to the following band conditions:

| River flow below (Ml/d) | Max abstraction (Ml/d) |
|-------------------------|------------------------|
| 112                     | 13.58                  |
| 81                      | 8.32                   |
| 59                      | 3.07                   |

We can define two parameters:

```json
{
  "parameters": {
    "License band index": {
      "type": "MultipleThresholdIndexParameter",
      "node": "My river node",
      "thresholds": [
        112,
        81,
        59
      ]
    },
    "License value": {
      "type": "IndexedArrayParameter",
      "index_parameter": "License band index",
      "parameters": [
        14.184,
        13.58,
        8.32,
        3.07
      ]
    }
  }
}
```

The `"License band index"` reads the flow from the `My river node` node and returns a integer (or index):

- when flow is above `13.58`, returns `0`
- when flow is below `13.58`, returns `1`
- when flow is below `8.32`, returns `2`
- when flow is below `3.07`, returns `3`

The parameter `"License value"` reads then the integer from `"License band index"`  (the parameter name is 
provided in the `index_parameter` key) and returns one of the values in the `parameters` list:

- if the index is `0`, it returns `14.184`.
- if the index is `1`, it returns `13.58`.
- if the index is `2`, it returns `8.32`.
- if the index is `3`, it returns `3.07`.

The value in the `parameters` key is in fact a list that can be indexed by its item number:

| Index | 0      | 1     | 2    | 3    |
|-------|--------|-------|------|------|
| Value | 14.184 | 13.58 | 8.32 | 3.07 |

You can then set the `"License value"` parameter as value of the `max_flow` attribute of the node controlling the
abstraction.