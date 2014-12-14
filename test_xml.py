#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import xmlutils
import matplotlib.pyplot as pyplot

data = file('simple1.xml', 'r').read()
model = xmlutils.parse_xml(data)

model.plot()
pyplot.show()
