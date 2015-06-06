import os
import xml.etree.ElementTree as ET

import pywr.core

def load_model(filename=None, data=None):
    '''Load a test model and check it'''
    if data is None:
        with open(os.path.join(os.path.dirname(__file__), 'models', filename), 'r') as f:
            data = f.read()
    xml = ET.fromstring(data)
    model = pywr.core.Model.from_xml(xml)
    model.check()
    return model
