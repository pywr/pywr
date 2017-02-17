====
Pywr
====

Pywr is a generalised network resource allocation model written in Python. It aims to be fast, free, and extendable.

.. image:: https://travis-ci.org/pywr/pywr.svg?branch=master
   :target: https://travis-ci.org/pywr/pywr

.. image:: https://ci.appveyor.com/api/projects/status/j1llo3j6o4ww9t1t/branch/master?svg=true
   :target: https://ci.appveyor.com/project/snorfalorpagus/pywr/branch/master

Overview
========

Pywr is a tool for solving network resource allocation problems at discrete timesteps using a linear programming approach. It's principal application is in resource allocation in water supply network, although other uses are conceivable. A network is represented as a directional graph using `NetworkX <https://networkx.github.io/>`__. Nodes in the network can be given constraints (e.g. minimum/maximum flows) and costs, and can be connected as required. Parameters in the model can vary time according to boundary conditions (e.g. an inflow timeseries) or based on states in the model (e.g. the current volume of a reservoir).

Models can be developed using the Python API, either in a script or interactively using `IPython <https://ipython.org/>`__/`Jupyter <https://jupyter.org/>`__. Alternatively, models can be defined in a rich `JSON-based document format <https://pywr.github.io/pywr-docs/json.html>`__.

.. image:: https://raw.githubusercontent.com/pywr/pywr/master/docs/source/_static/pywr_d3.png
   :width: 250px
   :height: 190px

New users are encouraged to read the `Pywr Tutorial <https://pywr.github.io/pywr-docs/tutorial.html>`__.

Installation
============

Pywr should work on Python 2.7 (or later) and 3.4 (or later) on Windows, Linux or OS X.

See the documentation for `detailed installation instructions <https://pywr.github.io/pywr-docs/install.html>`__.

Provided that you have the required `dependencies <https://pywr.github.io/pywr-docs/install.html#dependencies>`__ already installed, it's as simple as:

.. code-block:: console

    python setup.py install --with-glpk --with-lpsolve

For most users it will be easier to install the `binary packages made available for the Anaconda Python distribution <https://anaconda.org/pywr/pywr>`__. See install docs for more information. Note that these packages may lag behind the development version.

License
=======

Copyright (C) 2014-17 Joshua Arnott, James E. Tomlinson, Atkins, University of Manchester


This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 1, or (at your option)
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston MA  02110-1301 USA.
