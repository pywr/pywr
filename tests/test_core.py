#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import pytest

import pywr.core

def test_names():
    '''Test node names'''
    model = pywr.core.Model()
    
    node1 = pywr.core.Supply(model, name='A')
    node2 = pywr.core.Demand(model, name='B')
    assert(model.node['A'] is node1)
    assert(model.node['B'] is node2)
    
    nodes = sorted(model.nodes(), key=lambda node: node.name)
    assert(nodes == [node1, node2])

    # rename node
    node1.name = 'C'
    assert(model.node['C'] is node1)
    assert('A' not in model.node)
