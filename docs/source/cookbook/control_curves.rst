Dynamic behaviour and control curves
------------------------------------

Control curves are an important concept of many Pywr models. They are the most common way in which complex dynamic
behaviour is created. The typical application uses control curves to parameterise state dependent rules. For example,
a reservoir release that is dependent on the current volume. In this section of the documentation we discuss
different strategies for implementing dynamic behaviour and control curves in Pywr models.


Basic concept
=============

It is very common for resource allocation models to include rules and behaviours that are dependent on the
current state of the model. The most common example of state in Pywr models is the current volume in ``Storage``
node(s). During a simulation this state will be updated each timestep. Resource allocation rules and constraints
can be made dependent on this state.

In Pywr this behaviour is implemented through ``Parameters``. Some ``Parameters``, as discussed below, use information
(i.e. state) from nodes, recorders or other parameters within the model. When such parameters are used within a model
the behaviour is said to be state dependent. A key advantage of such an approach is that operational rules are
parameterised independently of the model's boundary conditions. Such rules should be capable of dynamically responding
to different boundary conditions (e.g. changes in demand, future flow scenarios, etc.).


Storage dependent control curves
================================

Pywr provides a few different ways to implement dynamic state dependent behaviour. The most common approach compares
current volume in a ``Storage`` node against one or more curves. These are typically referred to as a *curves*
because they are often defined as a profile which varies through the year. More complex systems will often contain
multiple ordered control curves that define progressively changing behaviour.

TBC

Other control curves
====================

TBC

Custom control curves
=====================

TBC

See also :ref:`extending-pywr-parameters`.
