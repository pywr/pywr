====
Pywr
====

Pywr is a generalised network resource allocation model written in Python. It aims to be fast, free, and extendable.

.. image:: https://github.com/pywr/pywr/workflows/Build/badge.svg?branch=master
   :target: https://github.com/pywr/pywr/actions?query=workflow%3ABuild

.. image:: https://img.shields.io/badge/chat-on%20gitter-blue.svg
   :target: https://gitter.im/pywr/pywr

.. image:: https://codecov.io/gh/pywr/pywr/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/pywr/pywr

Overview
========

`Documentation <https://pywr.github.io/pywr>`__

Pywr is a tool for solving network resource allocation problems at discrete timesteps using a linear programming approach. It's principal application is in resource allocation in water supply networks, although other uses are conceivable. A network is represented as a directional graph using `NetworkX <https://networkx.github.io/>`__. Nodes in the network can be given constraints (e.g. minimum/maximum flows) and costs, and can be connected as required. Parameters in the model can vary time according to boundary conditions (e.g. an inflow timeseries) or based on states in the model (e.g. the current volume of a reservoir).

Models can be developed using the Python API, either in a script or interactively using `IPython <https://ipython.org/>`__/`Jupyter <https://jupyter.org/>`__. Alternatively, models can be defined in a rich `JSON-based document format <https://pywr.github.io/pywr/json.html>`__.

.. image:: https://raw.githubusercontent.com/pywr/pywr/master/docs/source/_static/pywr_d3.png
   :width: 250px
   :height: 190px

New users are encouraged to read the `Pywr Tutorial <https://pywr.github.io/pywr/tutorial.html>`__.

Design goals
============

Pywr is a tool for solving network resource allocation problems. It has many similarities with other software packages such as WEAP, Wathnet, Aquator and MISER, but also has some significant differences. Pywr’s principle design goals are that it is:

- Fast enough to handle large stochastic datasets and large numbers of scenarios and function evaluations required by advanced decision making methodologies;
- Free to use without restriction – licensed under the GNU General Public Licence;
- Extendable – uses the Python programming language to define complex operational rules and control model runs.

Installation
============

Pywr should work on Python 3.7 (or later) on Windows, Linux or OS X.

See the documentation for `detailed installation instructions <https://pywr.github.io/pywr/install.html>`_.

For a quick start use pip:

.. code-block:: console

    pip install pywr

For most users it will be easier to install the binary packages made available from `PyPi <https://pypi.org/project/pywr/>`_ or the `Anaconda Python distribution <https://anaconda.org/conda-forge/pywr>`__. Note that these packages may lag behind the development version.

Citation
========

Please consider citing the following paper when using Pywr:


    Tomlinson, J.E., Arnott, J.H. and Harou, J.J., 2020. A water resource simulator in Python. Environmental Modelling & Software. https://doi.org/10.1016/j.envsoft.2020.104635


License
=======

Copyright (C) 2014-20 Joshua Arnott, James E. Tomlinson, Atkins, University of Manchester


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
