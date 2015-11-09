# -*- coding: utf-8 -*-
"""
Test the Recorder object API

"""
from __future__ import print_function
import pywr.core
import numpy as np
import pytest
from test_analytical import simple_linear_model
from pywr.recorders import NumpyArrayNodeRecorder, CSVRecorder


def test_numpy_recorder(simple_linear_model):
    """
    Test the NumpyArrayNodeRecorder
    """
    model = simple_linear_model
    otpt = model.node['Output']

    model.node['Input'].max_flow = 10.0
    otpt.cost = -2.0
    rec = NumpyArrayNodeRecorder(model, otpt)

    model.run()

    assert rec.data.shape == (365, 1)
    assert np.all((rec.data - 10.0) < 1e-12)


def test_csv_recorder(simple_linear_model, tmpdir):
    """
    Test the CSV Recorder

    """
    model = simple_linear_model
    otpt = model.node['Output']
    model.node['Input'].max_flow = 10.0
    otpt.cost = -2.0

    csvfile = tmpdir.join('output.csv')
    # By default the CSVRecorder saves all nodes in alphabetical order
    # and scenario index 0.
    rec = CSVRecorder(model, str(csvfile))

    model.run()

    import csv
    with open(str(csvfile), 'r') as fh:
        for irow, row in enumerate(csv.reader(fh)):
            if irow == 0:
                expected = ['Datetime', 'Input', 'Link', 'Output']
                actual = row
            else:
                dt = model.timestepper.start+(irow-1)*model.timestepper.delta
                expected = [dt.isoformat()]
                actual = [row[0]]
                assert np.all((np.array([float(v) for v in row[1:]]) - 10.0) < 1e-12)
            assert expected == actual




