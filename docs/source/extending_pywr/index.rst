.. _extending_pywr:

Extending Pywr
--------------

Pywr is a Python library and as such is designed in an object oriented fashion. These objects (or classes) form
the basic structure of Pywr, but much of the functionality in Pywr is provided via specific sub-classes designed
to do specific instances. For example, there exist many specialised ``Parameter`` classes to provide specific types
of input data. One of the benefits of being a Python library is that users can extend the base functionality in their
own projects to provide custom functionality. In this section of the documentation there are some examples and guidance
for extending Pywr in your projects.


.. toctree::
   :maxdepth: 1

   Extending nodes <extending_pywr_nodes.rst>
   Extending parameters <extending_pywr_parameters.rst>
   Extending recorders <extending_pywr_recorders.rst>

If you write a custom node/parameter/recorder that is of potential use to others consider creating a pull request to
have it included in the core library.
