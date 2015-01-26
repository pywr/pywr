# Pywr

Pywr is a water resource model written in Python. It aims to be fast, free, and extendable.

## Overview

A water supply network is represented as a directional graph using [NetworkX](https://networkx.github.io/). Timeseries representing variations such as river flow and demand are handled by [pandas](http://pandas.pydata.org/). The supply-demand balance is solved for each timestep using linear programming provided by [CyLP](https://github.com/coin-or/CyLP); however, the solver is decoupled from the network allowing the potential for alternate solvers. A graphical user interface is being developed using [Qt](http://qt-project.org/) and [PySide](http://qt-project.org/wiki/PySide).

Pywr is released under the Simplified BSD license (see LICENSE.txt).

## Development and testing

To install pywr (and it's dependencies) in a virtual environment:

```
$ virtualenv venv
$ source venv/bin/activate
(env)$ pip install -r requirements.txt
(env)$ pip install -e .
```

To run the unit tests:

```
(env)$ py.test tests
```
