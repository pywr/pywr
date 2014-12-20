#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import datetime

import pywr.core
import pywr.xmlutils

def test_run_simple1():
    # parse the XML into a model
    data = file(os.path.join(os.path.dirname(__file__), 'simple1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)

    # run the model
    t0 = model.timestamp
    result = model.step()
    
    # check results
    assert(result == ('optimal', 10.0, 10.0))
    
    # check the timestamp incremented
    assert(model.timestamp - t0 == datetime.timedelta(1))

def test_run_reservoir1():
    data = file(os.path.join(os.path.dirname(__file__), 'reservoir1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()

    for delivered in [10.0, 10.0, 10.0, 5.0, 0.0]:
        result = model.step()
        assert(result == ('optimal', 10.0, delivered))

def test_run_river1():
    data = file(os.path.join(os.path.dirname(__file__), 'river1.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    
    result = model.step()
    assert(result == ('optimal', 10.0, 5.0))

def test_run_river2():
    data = file(os.path.join(os.path.dirname(__file__), 'river2.xml'), 'r').read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    
    result = model.step()
    assert(result == ('optimal', 12.0, 9.25))
