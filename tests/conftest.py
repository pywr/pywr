import os
from pywr.model import Model


def pytest_report_header(config):
    headers = []
    solver_name = Model().solver.name
    headers.append("solver: {}".format(solver_name))
    return "\n".join(headers)
