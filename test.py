#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pywr import *

model = Model()

source1 = Supply(model, position=(1,1), max_flow=1.0)
pipe1 = Link(model, position=(2,1))
demand1 = Demand(model, position=(3,1))
source2 = Supply(model, position=(2,2), max_flow=2.0)
source3 = Supply(model, position=(3,3), max_flow=0.5)
pipe2 = Link(model, position=(3,2))
pipe3 = Link(model, position=(4,2))

source1.connect(pipe1)
pipe1.connect(demand1)

source2.connect(pipe1)

source3.connect(pipe2)
pipe2.connect(demand1)
source3.connect(pipe3)
pipe3.connect(demand1)

demand2 = Demand(model, position=(2,0))
pipe1.connect(demand2)

catch1 = Catchment(model, position=(1, 0))
conf1 = River(model, position=(2, -1))
abs1 = RiverAbstraction(model, position=(3, -1), max_flow=2)
abs2 = RiverAbstraction(model, position=(4, -1), max_flow=2)
term1 = Terminator(model, position=(5,-1))

conf1.connect(abs1)
catch1.connect(conf1)

abs1.connect(abs2)
abs1.connect(demand1)

abs2.connect(demand1)
abs2.connect(term1)

catch2 = Catchment(model, position=(1, -2))
catch2.connect(conf1)

model.check()

#model.routes()

status, links, nodes = model.solve()

model.plot(links, nodes)
