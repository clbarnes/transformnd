import pytest

from transformnd.util import chain_or, none_eq, same_or_none, window


def test_same_or_none():
    assert same_or_none(1, None) == 1
    assert same_or_none(None, 1) == 1
    with pytest.raises(ValueError):
        same_or_none(None)
    with pytest.raises(ValueError):
        same_or_none(1, 2)
    assert same_or_none(None, None, default=1) == 1


def test_none_eq():
    assert none_eq(None, 1)
    assert none_eq(1, None)
    assert none_eq(1, 1)
    assert not none_eq(1, 2)


def test_chain_or():
    assert chain_or(None, 1, 2) == 1
    assert chain_or(None, 2, 1) == 2
    with pytest.raises(ValueError):
        chain_or(None)
    assert chain_or(None, default=1) == 1


def test_window():
    test = list(window(range(5), 2))
    ref = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert test == ref
