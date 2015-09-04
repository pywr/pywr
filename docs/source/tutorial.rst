Tutorial
========

.. code-block:: python

   from pywr.core import Model, Input, Output
   
   model = Model()
   
   supply = Input(model, name='supply')
   demand = Input(model, name='demand')
   
   supply.connect(demand)
   
   model.run()
