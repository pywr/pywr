
import pytest

def pytest_addoption(parser):
    parser.addoption("--solver", action="store", default="glpk",
        help="Solver to run the tests against.")

@pytest.fixture
def solver(request):
    return request.config.getoption("--solver")
