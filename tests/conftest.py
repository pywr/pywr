
import pytest

def pytest_addoption(parser):
    parser.addoption("--solver", action="store", default="glpk",
        help="Solver to run the tests against.")

@pytest.fixture
def solver(request):
    return request.config.getoption("--solver")

def pytest_report_header(config):
    headers = []
    solver_name = config.getoption("--solver")
    headers.append('solver: {}'.format(solver_name))
    return '\n'.join(headers)
