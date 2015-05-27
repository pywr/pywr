#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import pywr.core

import xml.etree.ElementTree as ET
import pandas
import dateutil

def parse_xml(data):
    '''Create a new Model from XML data'''
    root = ET.fromstring(data)
    
    # parse solver
    xml_solver = root.find('solver')
    if xml_solver is not None:
        solver = xml_solver.get('name')
    else:
        solver = None
    
    model = pywr.core.Model(solver=solver)
    
    # parse metadata
    xml_metadatas = root.find('metadata')
    for xml_metadata in xml_metadatas.getchildren():
        tag = xml_metadata.tag.lower()
        text = xml_metadata.text.strip()
        model.metadata[tag] = text
    
    # parse parameters
    xml_parameters = root.find('parameters')
    if xml_parameters:
        for xml_parameter in xml_parameters.getchildren():
            tag = xml_parameter.tag.lower()
            if tag == 'parameter':
                key = xml_parameter.get('key')
                value = xml_parameter.text
                model.parameters[key] = value
            else:
                raise NotImplementedError()
        try:
            model.timestamp = dateutil.parser.parse(model.parameters['timestamp'])
        except KeyError:
            pass

    # parse data
    xml_datas = root.find('data')
    if xml_datas:
        for xml_data in xml_datas.getchildren():
            tag = xml_data.tag.lower()
            name = xml_data.get('name')
            properties = {}
            for child in xml_data.getchildren():
                properties[child.tag] = child.text
            if properties['type'] == 'pandas':
                # TODO: better handling of british/american dates (currently assumes british)
                df = pandas.read_csv(properties['path'], index_col=0, parse_dates=True, dayfirst=True)
                df = df[properties['column']]
                ts = pywr.core.Timeseries(df)
                model.data[name] = ts
            else:
                raise NotImplementedError()

    # parse nodes
    nodes = {}
    xml_nodes = root.find('nodes')
    for xml_node in xml_nodes.getchildren():
        tag = xml_node.tag.lower()
        try:
            cls = pywr.core.node_registry[tag]
        except KeyError:
            raise KeyError('Unrecognised node type ({})'.format(tag))
        attrs = dict(xml_node.items())
        name = attrs['name']
        position = (float(attrs['x']), float(attrs['y']),)
        node = cls(model, position=position, **attrs)
        nodes[name] = node
        for child in xml_node.getchildren():
            if child.tag == 'parameter':
                child_type = child.get('type')
                key = child.get('key')
                if child_type == 'constant':
                    try:
                        value = float(child.text)
                    except:
                        value = child.text
                    node.properties[key] = pywr.core.Parameter(value=value)
                elif child_type == 'timeseries':
                    name = child.text
                    node.properties[key] = model.data[name]
                else:
                    raise NotImplementedError()
            elif child.tag == 'variable':
                key = child.get('key')
                value = float(child.text)
                node.properties[key] = pywr.core.Variable(initial=value)

    # parse edges
    xml_edges = root.find('edges')
    for xml_edge in xml_edges.getchildren():
        tag = xml_edge.tag.lower()
        if tag != 'edge':
            raise ValueError()
        from_name = xml_edge.get('from')
        to_name = xml_edge.get('to')
        from_node = nodes[from_name]
        to_node = nodes[to_name]
        slot = xml_edge.get('slot')
        if slot is not None:
            slot = int(slot)
        to_slot = xml_edge.get('to_slot')
        if to_slot is not None:
            to_slot = int(to_slot)
        from_node.connect(to_node, slot=slot, to_slot=to_slot)

    return model
