from types import ModuleType
import pytest

from array_api_compat import is_array_api_obj

from transformnd.util import chain_or, none_eq, same_or_none, as_floats
import jax.numpy
import dask.array
import numpy


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


@pytest.mark.parametrize("to_ns", [numpy, dask.array, jax.numpy])
@pytest.mark.parametrize("from_ns", [None, numpy, dask.array, jax.numpy])
def test_as_floats(from_ns: ModuleType | None, to_ns: ModuleType):
    data = [[1, 2], [3, 4]]
    if from_ns is not None:
        data = from_ns.asarray(data)
    array = as_floats(data, namespace=to_ns)
    assert is_array_api_obj(array)
    # assert array_namespace(array) == to_ns
    assert array == pytest.approx(numpy.array(data))
