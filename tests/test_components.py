"""
Tests for `pywr._component.Component` class behaviour and its use in `pywr.core.Model`.
"""

from pywr.core import Component, Model, ModelStructureError
from collections import defaultdict
from fixtures import simple_linear_model
import pytest


class DummyComponent(Component):
    def __init__(self, *args, **kwargs):
        super(DummyComponent, self).__init__(*args, **kwargs)
        self.func_counts = defaultdict(lambda: 0)

    def setup(self):
        self.func_counts["setup"] += 1

    def reset(self):
        self.func_counts["reset"] += 1

    def before(self):
        self.func_counts["before"] += 1
        # Check that all children have been called first.
        for c in self.children:
            assert c.func_counts["before"] == self.func_counts["before"]
        # Check that all parents have yet to called.
        for p in self.parents:
            assert p.func_counts["before"] + 1 == self.func_counts["before"]

    def after(self):
        self.func_counts["after"] += 1

    def finish(self):
        self.func_counts["finish"] += 1


def test_single_component(simple_linear_model):
    """Test that a component's methods are called the correct number of times"""
    m = simple_linear_model
    nt = len(m.timestepper)
    c = DummyComponent(m)

    m.run()

    assert c.func_counts["setup"] == 1
    assert c.func_counts["reset"] == 1
    assert c.func_counts["before"] == nt
    assert c.func_counts["after"] == nt
    assert c.func_counts["finish"] == 1


def test_shared_parent_component(simple_linear_model):
    """Test two components sharing the same parent"""

    m = simple_linear_model
    nt = len(m.timestepper)
    c1 = DummyComponent(m, name="c1")
    c2 = DummyComponent(m, name="c2")
    cp = DummyComponent(m, name="cp")

    c1.parents.add(cp)
    c2.parents.add(cp)

    assert cp in c1.parents
    assert cp in c2.parents
    assert c1 in cp.children
    assert c2 in cp.children

    m.run()

    assert len(m.components) == 3

    for c in m.components:
        assert c.func_counts["setup"] == 1
        assert c.func_counts["reset"] == 1
        assert c.func_counts["before"] == nt
        assert c.func_counts["after"] == nt
        assert c.func_counts["finish"] == 1

    # Simulate a change and re-run.
    m.setup()
    m.run()

    assert len(m.components) == 3

    for c in m.components:
        assert c.func_counts["setup"] == 2
        assert c.func_counts["reset"] == 3
        assert c.func_counts["before"] == 2 * nt
        assert c.func_counts["after"] == 2 * nt
        assert c.func_counts["finish"] == 2


def test_circular_components_error(simple_linear_model):
    """Test that circular components raise an error"""
    m = simple_linear_model
    c1 = DummyComponent(m)
    c2 = DummyComponent(m)

    c1.parents.add(c2)
    c2.parents.add(c1)

    with pytest.raises(ModelStructureError):
        m.run()


def test_selfloop_components_error(simple_linear_model):
    """Test that self-looping components raise an error"""
    m = simple_linear_model
    c1 = DummyComponent(m)
    c1.parents.add(c1)

    with pytest.raises(ModelStructureError):
        m.run()
