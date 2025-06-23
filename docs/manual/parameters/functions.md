# Functions

# 1D polynomial
The [pywr.parameters.Polynomial1DParameter][] returns the result of a 1D polynomial evaluation. The input to the polynomial
can be either:
    
- The previous flow of a node.
- The current storage of a storage node.
- The current value of parameter.

The degree of the polynomial is determined based on the number of given coefficients minus 1.

The following example uses the parameter to find the surface area of a reservoir based
on the current relative storage. The area-volume relationship is represented using the following
2-degree polynomial:

Area = 19.1 + 123 * x + 12 * x<sup>1</sup> + 0.192 * x<sup>2</sup>

where `x` is the storage.

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import Polynomial1DParameter

    model = Model()
    storage_node = Storage(
        model=model,
        name="reservoir",
        max_volume=100, 
        initial_volume=100
    )
    Polynomial1DParameter(
        model=model, 
        storage=storage,
        name="My parameter", 
        coefficients=[19.1, 123, 12, 0.192], 
        use_proportional_volume=True
    )
    ```

=== "JSON"
    ```json
    {
        "My parameter": {
            "type": "Polynomial1DParameter",
            "storage": "reservoir",
            "coefficients": [19.1, 123, 12, 0.192],
            "use_proportional_volume": true
        }
    }
    ```

The `use_proportional_volume` can only be used if the dependant variable is of type storage and always
defaults to `False`.

!!!warning "Coefficient order"
    The coefficients are sorted from the lowest degree to the maximum. The first value in the array
    corresponds to degree `0`, the second to degree `1` and so on.

If you want to use a parameter instead, you can provide the parameter instance (for Python) 
or name (for JSON) using the `parameter` option: 

```json
{
    "My parameter": {
        "type": "Polynomial1DParameter",
        "parameter": "Dependant parameter",
        "coefficients": [19.1, 123, 12, 0.192]
    }
}
```

If you want to use a node's flow, you can provide the node instance (for Python) 
or name (for JSON) using the `node` option: 

```json
{
    "My parameter": {
        "type": "Polynomial1DParameter",
        "node": "My node",
        "coefficients": [19.1, 123, 12, 0.192]
    }
}
```

You can also scale and offset the dependant variable before the polynomial calculation using the
`scale` and `offset` options, which defaults to 1 and 0 respectively. In the first storage example,
if both options are provided, the parameter will evaluate the following:

Area = 19.1 + 123 * (x*`scale` + `offset`) + 12 * (x*`scale` + `offset`) + 0.192 * (x*`scale` + `offset`)<sup>2</sup>

```json
{
    "My parameter": {
        "type": "Polynomial1DParameter",
        "storage": "reservoir",
        "coefficients": [19.1, 123, 12, 0.192],
        "use_proportional_volume": true,
        "scale": 10,
        "offset": -1
    }
}
```

# Piecewise integral
The [pywr.parameters.PiecewiseIntegralParameter][] integrates a piecewise function given as two arrays of `x` and `y`. 
`x` should be monotonically increasing and greater than zero and `y` should start from 0.
At a given timestep, this parameter then integrates `y` over `x` between 0 and the value
given by a given parameter value.

In the following example, a [pywr.parameters.StorageParameter][] tracks the storage of a reservoir.
If the storage is `45%`, the parameter will integrate between `0.01` and `0.45` the values in `y`
(between `0.2` and `0.5`).

=== "Python"
    ```python
    from pywr.core import Model
    from pywr.nodes import Storage
    from pywr.parameters import StorageParameter, PiecewiseIntegralParameter

    model = Model()
    node = Storage(model=model, name="Reservoir", max_volume=100)
    parameter = StorageParameter(
        model=model, 
        storage_node=node, 
        use_proportional_volume=True,
        name="Storage"
    )
    PiecewiseIntegralParameter(
        model=model, 
        name="My parameter", 
        x=[0.01, 0.45, 0.90, 1], 
        y=[0.2, 0.5, 0.67, 0.7],
        parameter=parameter
    )
    ```

=== "JSON"
    ```json
    {
        "My parameter": {
            "type": "PiecewiseIntegralParameter",
            "x": [1, 45, 90, 100], 
            "y": [0.2, 0.5, 0.67, 0.7],
            "parameter": "Storage"
        }
    }
    ```

## Other parameters
The following parameters are also available:

- [pywr.parameters.AnnualHarmonicSeriesParameter][]: this parameter returns the value from an annual harmonic series.
- [pywr.parameters.Polynomial2DStorageParameter][]: this parameter returns the result of 2D polynomial evaluation
where the two independent variables are the volume of a storage node and the current value of a parameter respectively.

These are only documented in the API section.