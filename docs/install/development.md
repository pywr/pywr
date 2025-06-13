# Development

The source code for Pywr is managed using Git and is hosted on GitHub:
<https://github.com/pywr/pywr/> .

## Testing
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

## Build this documentation
This documentation is built using [mkdocs-material](https://github.com/squidfunk/mkdocs-material/tree/master). If you
want to run or build it, you need to install the necessary dependencies first:

    pip install ".[docs]"

MkDocs includes a live preview server, so you can preview your changes as you write your documentation. The server will
automatically rebuild the site upon saving. Start it with:

    mkdocs serve

otherwise, you can build a static site from your Markdown files with:

    mkdocs build
