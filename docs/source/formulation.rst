Problem formulation
===================

Pywr attempts to solve the supply demand balance each timestep by framing the problem as a `linear programme <https://en.wikipedia.org/wiki/Linear_programming>`_.

.. math::

    \begin{align}
    & \text{maximize}   && \mathbf{c}^\mathrm{T} \mathbf{x}\\
    & \text{subject to} && A \mathbf{x} \leq \mathbf{b} \\
    & \text{and} && \mathbf{x} \ge \mathbf{0}
    \end{align} 
