====
Pywr
====

Pywr is a generalised network resource allocation model written in Python. It aims to be fast, free, and extendable.

.. image:: https://travis-ci.org/pywr/pywr.svg?branch=master
   :target: https://travis-ci.org/pywr/pywr

.. image:: https://ci.appveyor.com/api/projects/status/ik9u75bxfvracimh?svg=true
   :target: https://ci.appveyor.com/project/pywr-admin/pywr

.. image:: https://img.shields.io/badge/chat-on%20gitter-blue.svg
   :target: https://gitter.im/pywr/pywr

.. image:: https://codecov.io/gh/pywr/pywr/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/pywr/pywr

Overview
========

Pywr is a tool for solving network resource allocation problems at discrete timesteps using a linear programming approach. It's principal application is in resource allocation in water supply networks, although other uses are conceivable. A network is represented as a directional graph using `NetworkX <https://networkx.github.io/>`__. Nodes in the network can be given constraints (e.g. minimum/maximum flows) and costs, and can be connected as required. Parameters in the model can vary time according to boundary conditions (e.g. an inflow timeseries) or based on states in the model (e.g. the current volume of a reservoir).

Models can be developed using the Python API, either in a script or interactively using `IPython <https://ipython.org/>`__/`Jupyter <https://jupyter.org/>`__. Alternatively, models can be defined in a rich `JSON-based document format <https://pywr.github.io/pywr-docs/master/json.html>`__.

.. image:: https://raw.githubusercontent.com/pywr/pywr/master/docs/source/_static/pywr_d3.png
   :width: 250px
   :height: 190px

New users are encouraged to read the `Pywr Tutorial <https://pywr.github.io/pywr-docs/master/tutorial.html>`__.

Design goals
============

Pywr is a tool for solving network resource allocation problems. It has many similarities with other software packages such as WEAP, Wathnet, Aquator and MISER, but also has some significant differences. Pywr’s principle design goals are that it is:

- Fast enough to handle large stochastic datasets and large numbers of scenarios and function evaluations required by advanced decision making methodologies;
- Free to use without restriction – licensed under the GNU General Public Licence;
- Extendable – uses the Python programming language to define complex operational rules and control model runs.

Installation
============

Pywr should work on Python 3.6 (or later) on Windows, Linux or OS X.

See the documentation for `detailed installation instructions <https://pywr.github.io/pywr-docs/master/install.html>`__.

Provided that you have the required `dependencies <https://pywr.github.io/pywr-docs/master/install.html#dependencies>`__ already installed, it's as simple as:

.. code-block:: console

    python setup.py install --with-glpk --with-lpsolve

For most users it will be easier to install the `binary packages made available for the Anaconda Python distribution <https://anaconda.org/pywr/pywr>`__. See install docs for more information. Note that these packages may lag behind the development version.

License
=======

Copyright (C) 2014-19 Joshua Arnott, James E. Tomlinson, Atkins, University of Manchester


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
