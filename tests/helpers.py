import os
import xml.etree.ElementTree as ET

import pywr.core

def load_model(filename=None, data=None):
    '''Load a test model and check it'''
    if data is None:
        path = os.path.join(os.path.dirname(__file__), 'models', filename)
        with open(path, 'r') as f:
            data = f.read()
    else:
        path = None
    xml = ET.fromstring(data)
    model = pywr.core.Model.from_xml(xml, path=path)
    model.check()
    return model
