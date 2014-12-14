#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pywr import *
import matplotlib.pyplot as pyplot

model = Model()

source1 = Supply(model, position=(1,1), max_flow=1.0, name='supply1')
pipe1 = Link(model, position=(2,1))
demand1 = Demand(model, position=(3,1), name='demand1')
source2 = Supply(model, position=(2,2), max_flow=2.0, name='supply2')
source3 = Supply(model, position=(3,3), max_flow=0.5, name='supply3')
pipe2 = Link(model, position=(3,2))
pipe3 = Link(model, position=(4,2))

source1.connect(pipe1)
pipe1.connect(demand1)

source2.connect(pipe1)

source3.connect(pipe2)
pipe2.connect(demand1)
source3.connect(pipe3)
pipe3.connect(demand1)

demand2 = Demand(model, position=(2,0), name='demand2')
pipe1.connect(demand2)

catch1 = Catchment(model, position=(1, 0), name='catch1')
conf1 = River(model, position=(2, -1), name='conf1')
abs1 = RiverAbstraction(model, position=(3, -1), max_flow=2, name='abs1')
abs2 = RiverAbstraction(model, position=(4, -1), max_flow=2, name='abs2')
term1 = Terminator(model, position=(5,-1))

conf1.connect(abs1)
catch1.connect(conf1)

abs1.connect(abs2)
abs1.connect(demand1)

abs2.connect(demand1)
abs2.connect(term1)

term2 = Terminator(model, position=(2, -3), name='term2')

split1 = RiverSplit(model, position=(1, -2), name='split1')
split1.connect(conf1)
split1.connect(term2)
split1.split = [conf1, term2]

abs3 = RiverAbstraction(model, position=(0, -2), max_flow=1.0, name='abs3')
abs3.connect(split1)
demand3 = Demand(model, position=(0, -3))
abs3.connect(demand3)

catch2 = Catchment(model, position=(-1, -2), name='catch2')
catch2.connect(abs3)

split1.connect(term2)

model.check()

#model.routes()

status, links, nodes = model.solve()

model.plot(links, nodes)
pyplot.show()
