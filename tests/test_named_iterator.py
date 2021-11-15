import pytest

from pywr.model import NamedIterator


class Item:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def items():
    return [
        Item("Hello"),
        Item("World"),
        Item("River Thames"),
    ]


@pytest.fixture
def iterator(items):
    i = NamedIterator()
    for item in items:
        i[item.name] = item
    return i


def test_iterator_keys_and_values(items, iterator):
    expected_names = {"Hello", "World", "River Thames"}
    assert set(iterator.keys()) == expected_names
    assert set(iterator.values()) == set(items)


def test_iterator_contains(items, iterator):
    assert "Hello" in iterator
    assert "River Thames" in iterator
    assert items[0] in iterator
    assert items[-1] in iterator


def test_iterator_length(iterator):
    assert len(iterator) == 3


def test_iterator_delete(iterator):
    del iterator["World"]
    assert {i.name for i in iterator} == {"Hello", "River Thames"}


def test_iterator_append(iterator):
    new_item = Item("New Item")
    iterator.append(new_item)
    assert len(iterator) == 4
    assert new_item in iterator
