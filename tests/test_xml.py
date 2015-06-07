#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import pandas
import datetime
import xml.etree.ElementTree as ET

import pywr.core
import pywr.licenses

from helpers import load_model

def test_simple1():
    '''Test parsing a simple XML document'''
    # parse the XML into a model
    model = load_model('simple1.xml')

    # metadata
    assert(model.metadata['title'] == 'Simple 1')
    assert(model.metadata['description'] == 'A very simple example.')

    # node names
    nodes = model.nodes()
    assert(len(nodes) == 3)
    supply1 = model.node['supply1']
    link1 = model.node['link1']
    demand1 = model.node['demand1']

    # node types
    assert(type(supply1) is pywr.core.Supply)
    assert(type(link1) is pywr.core.Link)
    assert(type(demand1) is pywr.core.Demand)

    # node positions
    assert(supply1.position == (1,1))
    assert(link1.position == (2,1))
    assert(demand1.position == (3,1))

    # edges
    edges = model.graph.edges()
    assert(len(edges) == 2)
    assert((supply1, link1) in edges)
    assert((link1, demand1) in edges)

    model.check()

def test_timestamps():
    '''Test datetime related model parameters'''
    model = load_model('timeseries1.xml')
    
    assert(model.parameters['timestamp_start'] == pandas.to_datetime('1970-01-01'))
    assert(model.parameters['timestamp_finish'] == pandas.to_datetime('3027-08-22'))
    assert(model.parameters['timestep'] == datetime.timedelta(1))

def test_xml_parameter_constant_float():
    """Test serialisation/deserialisation of a constant parameter"""
    model = pywr.core.Model()
    parameter = pywr.core.ParameterConstant(42.0)
    
    # to xml
    parameter_xml = parameter.xml('max_flow')
    assert(parameter_xml.get('key') == 'max_flow')
    assert(float(parameter_xml.text) == 42.0)
    assert(parameter_xml.get('type') == 'const')
    
    # and back again
    key, parameter = pywr.core.ParameterConstant.from_xml(model, parameter_xml)
    assert(key == 'max_flow')
    assert(parameter._value == 42.0)

def test_xml_parameter_constant_datetime():
    """Test serialisation/deserialisation of a datetime parameter"""
    model = pywr.core.Model()
    parameter = pywr.core.ParameterConstant(pandas.to_datetime('2015-01-01'))
    
    # to xml
    parameter_xml = parameter.xml('test')
    assert(parameter_xml.text == '2015-01-01 00:00:00')
    assert(parameter_xml.get('type') == 'datetime')
    
    # and back again
    key, parameter = pywr.core.ParameterConstant.from_xml(model, parameter_xml)
    assert(parameter == pandas.to_datetime('2015-01-01'))

def test_xml_parameter_constant_timedelta():
    """Test serialisation/deserialisation of a timedelta parameter"""
    model = pywr.core.Model()
    parameter = pywr.core.ParameterConstant(datetime.timedelta(days=2))
    
    # to xml
    parameter_xml = parameter.xml('test')
    assert(parameter_xml.get('type') == 'timedelta')
    value = float(parameter_xml.text)
    units = parameter_xml.get('units')
    if units == 'days':
        seconds = value * 60 * 60 * 24
    elif units == 'hours':
        seconds = value * 60 * 60
    elif units == 'minutes':
        seconds = value * 60
    else:
        seconds = value
    assert(seconds == 172800)

    # and back again
    key, parameter = pywr.core.ParameterConstant.from_xml(model, parameter_xml)
    assert(parameter == datetime.timedelta(days=2))

def test_xml_node():
    """Test serialisation/deserialisation of a generic node"""
    model = pywr.core.Model()
    node = pywr.core.Node(model, name='node1', position=(3, 4))
    node.properties['max_flow'] = pywr.core.ParameterConstant(42.0)
    node.check()
    
    # to xml
    node_xml = node.xml()
    assert(node_xml.tag == 'node')
    assert(node_xml.get('name') == 'node1')
    properties = dict([(prop.get('key'), prop) for prop in node_xml.findall('parameter')])
    assert('max_flow' in properties)
    assert(properties['max_flow'].get('type').startswith('const'))
    assert(float(properties['max_flow'].text) == 42.0)
    
    # and back again
    del(model, node)
    model = pywr.core.Model()
    node = pywr.core.Node.from_xml(model, node_xml)
    assert(node.name == 'node1')
    assert('max_flow' in node.properties)
    assert(isinstance(node.properties['max_flow'], pywr.core.ParameterConstant))
    assert(node.properties['max_flow'].value(None) == 42.0)
    assert(node.position == (3, 4))

def test_xml_node_supply_without_license():
    """Test serialisation/deserialisation of supply without a license"""
    model = pywr.core.Model()
    node = pywr.core.Supply(model, name='supply1', position=(3, 4))
    
    # to_xml
    supply_xml = node.xml()
    
    # and back again
    del(model, node)
    model = pywr.core.Model()
    node = pywr.core.Supply.from_xml(model, supply_xml)
    assert(node.name == 'supply1')
    assert(not node.licenses)

def test_xml_node_supply_with_license():
    """Test serialisation/deserialisation of supply with a license"""
    model = pywr.core.Model()
    node = pywr.core.Supply(model, name='supply1', position=(3, 4))
    license = pywr.licenses.DailyLicense(42.0)
    licensecollection = pywr.licenses.LicenseCollection([license])
    node.licenses = licensecollection
    
    # to_xml
    supply_xml = node.xml()
    licensecollection_xml = supply_xml.find('licensecollection')
    assert(licensecollection_xml is not None)
    licenses = licensecollection_xml.getchildren()
    assert(len(licenses) == 1)
    license = licenses[0]
    assert(license.get('type') == 'daily')
    assert(float(license.text) == 42.0)
    
    # and back again
    del(model, node, license, licensecollection)
    model = pywr.core.Model()
    node = pywr.core.Supply.from_xml(model, supply_xml)
    assert(node.name == 'supply1')
    assert(isinstance(node.licenses, pywr.licenses.LicenseCollection))
    assert(len(node.licenses) == 1)
    license = list(node.licenses._licenses)[0]
    assert(isinstance(license, pywr.licenses.DailyLicense))
    assert(license._amount == 42.0)
