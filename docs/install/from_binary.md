# Installing binary package

## Installing from PyPI
The easiest way to install pywr is to use the binary wheel distributed and hosted on [PyPi](https://pypi.org) 
for Windows and Linux.

    pip install pywr

The official page of the wheel is: [https://pypi.org/project/pywr/](https://pypi.org/project/pywr/).

## Installing with Anaconda
A binary distribution of Pywr is provided for Python 3.7+ (64-bit) on Windows,
Linux and OS X for the [Anaconda Python distribution](https://www.continuum.io/downloads). 
You will need to install and configure Anaconda before proceeding. Please read
[getting started section](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
available on the official Conda website.

Add the `conda-forge` channel first, as Pywr is hosted on that channel.

    conda config --add channels conda-forge

Install it using

    conda install pywr

The official page of the package is: [https://anaconda.org/conda-forge/pywr](https://anaconda.org/conda-forge/pywr)

!!!warning
    This release may lag behind the development version.
