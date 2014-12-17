#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import pywr.core

import xml.etree.ElementTree as ET

def parse_xml(data):
    '''Create a new Model from XML data'''
    root = ET.fromstring(data)
    
    model = pywr.core.Model()
    
    # parse metadata
    xml_metadatas = root.find('metadata')
    for xml_metadata in xml_metadatas.getchildren():
        tag = xml_metadata.tag.lower()
        text = xml_metadata.text.strip()
        model.metadata[tag] = text

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
                if child.get('type') == 'constant':
                    key = child.get('key')
                    try:
                        value = float(child.text)
                    except:
                        value = child.text
                    node.properties[key] = pywr.core.Parameter(value=value)
                else:
                    raise NotImplementedError()

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
        from_node.connect(to_node)

    return model
