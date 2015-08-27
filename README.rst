====
Pywr
====

Pywr is a conjunctive use water resource model written in Python. It aims to be fast, free, and extendable.

.. image:: https://travis-ci.org/snorfalorpagus/pywr.svg?branch=master
   :target: https://travis-ci.org/snorfalorpagus/pywr

Overview
========

A water supply network is represented as a directional graph using `NetworkX <https://networkx.github.io/>`__. Timeseries representing variations such as river flow and demand are handled by `pandas <http://pandas.pydata.org/>`__. The supply-demand balance is solved for each timestep using linear programming provided by `GLPK <https://www.gnu.org/software/glpk/>`__; however, the solver is decoupled from the network allowing the potential for alternate solvers. A graphical user interface is being developed using `Qt <http://qt-project.org/>`__ and `PySide <http://qt-project.org/wiki/PySide>`__.

Development and testing
=======================

To install pywr (and it's dependencies) in a virtual environment:

.. code-block:: console

    $ virtualenv venv
    $ source venv/bin/activate
    (env)$ pip install -r requirements.txt
    (env)$ pip install -e .

To run the unit tests:

.. code-block:: console

    (env)$ py.test tests

License
=======

Copyright (C) 2015  Joshua Arnott

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


.. image:: https://badges.gitter.im/Join%20Chat.svg
   :alt: Join the chat at https://gitter.im/snorfalorpagus/pywr
   :target: https://gitter.im/snorfalorpagus/pywr?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge