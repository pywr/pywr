# Development and testing

The source code for Pywr is managed using Git and is hosted on GitHub:
<https://github.com/pywr/pywr/> .

There are a collection of unit tests for Pywr written using `pytest`.
These can be run using:

    pytest tests

This will run all available tests using the default solver. A specific
solver can be tested by specifying the `PYWR_SOLVER` environment variable:

    PYWR_SOLVER=lpsolve pytest tests

## Continuous Integration
Pywr is automatically built and tested on Linux and Windows using
GitHub CI. Creating a pull request on GitHub will automatically trigger tests
and a build.

