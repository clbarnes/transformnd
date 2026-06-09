import numpy as np
import dask.array as da

import pytest

from transformnd.util import as_floats
from transformnd.transforms.vector_field import Coordinates, Displacements


def test_coord_simple():
    # 2D inputs, 2D outputs = 3D vector field
    vf = as_floats([[[0, 0], [0, 10]], [[10, 0], [10, 10]]])
    t = Coordinates(vf, interpolation_order=1)
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    out = t.apply(coords)
    expected = as_floats([[0, 0], [0, 10], [10, 0], [10, 10]])
    assert out == pytest.approx(expected)


def test_coord_interp():
    # 2D inputs, 2D outputs = 3D vector field
    vf = as_floats([[[0, 0], [0, 10]], [[10, 0], [10, 10]]])
    t = Coordinates(vf, interpolation_order=1)
    coords = np.array([[0.5, 0.5]], float)
    out = t.apply(coords)
    expected = as_floats([[5, 5]])
    assert out == pytest.approx(expected)


def test_coord_random(rng):
    # 10x20 array of 6D vectors
    sh = (10, 20)
    nd = 6

    vf = rng.random(sh + (nd,))
    t = Coordinates(vf, interpolation_order=1)
    coords = (
        np.dstack(np.meshgrid(*(np.arange(s, dtype=int) for s in sh)))
        .reshape(-1, len(sh))
        .astype(float)
    )
    rng.shuffle(coords)
    assert coords.shape == (np.prod(sh), len(sh))
    out = t.apply(coords)
    assert out.shape == (np.prod(sh), nd)
    expected = []
    for coord in coords:
        idx = tuple(coord.astype(int))
        expected.append(vf[idx])
    assert out == pytest.approx(as_floats(expected))


def test_coord_dask(rng):
    # 10x20 array of 2D vectors
    sh = (10, 20)
    nd = 2

    vf = da.from_array(rng.random(sh + (nd,)), chunks=5)  # type:ignore
    t = Coordinates(vf, interpolation_order=1)
    coords = (
        np.dstack(np.meshgrid(*(np.arange(s, dtype=int) for s in sh)))
        .reshape(-1, len(sh))
        .astype(float)
    )

    rng.shuffle(coords)
    assert coords.shape == (np.prod(sh), len(sh))
    da_coords = da.from_array(coords)
    out = t.apply(da_coords).compute()
    assert out.shape == (np.prod(sh), nd)
    expected = []
    for coord in coords:
        idx = tuple(coord.astype(int))
        expected.append(vf[idx])
    assert out == pytest.approx(as_floats(expected))


def test_disp_simple():
    # 2D inputs, 2D outputs = 3D vector field
    vf = as_floats([[[0, 0], [0, 10]], [[10, 0], [10, 10]]])
    t = Displacements(vf, interpolation_order=1)
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    out = t.apply(coords)
    expected = as_floats([[0, 0], [0, 11], [11, 0], [11, 11]])
    assert out == pytest.approx(expected)


def test_disp_interp():
    # 2D inputs, 2D outputs = 3D vector field
    vf = as_floats([[[0, 0], [0, 10]], [[10, 0], [10, 10]]])
    t = Displacements(vf, interpolation_order=1)
    coords = np.array([[0.5, 0.5]], float)
    out = t.apply(coords)
    expected = as_floats([[5.5, 5.5]])
    assert out == pytest.approx(expected)


def test_disp_random(rng):
    # 10x20 array of 6D vectors
    sh = (10, 20)
    nd = len(sh)

    vf = rng.random(sh + (nd,))
    t = Displacements(vf, interpolation_order=1)
    coords = (
        np.dstack(np.meshgrid(*(np.arange(s, dtype=int) for s in sh)))
        .reshape(-1, len(sh))
        .astype(float)
    )
    rng.shuffle(coords)
    assert coords.shape == (np.prod(sh), len(sh))
    out = t.apply(coords)
    assert out.shape == (np.prod(sh), nd)
    expected = []
    for coord in coords:
        idx = tuple(coord.astype(int))
        expected.append(vf[idx] + coord)
    assert out == pytest.approx(as_floats(expected))
