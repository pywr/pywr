import os

import pywr.xmlutils

def load_model(filename):
    '''Load a test model and check it'''
    with open(os.path.join(os.path.dirname(__file__), 'models', filename), 'r') as f:
        data = f.read()
    model = pywr.xmlutils.parse_xml(data)
    model.check()
    return model
