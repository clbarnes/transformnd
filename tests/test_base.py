from copy import copy

import numpy as np
import pytest

from transformnd.base import TransformSequence, TransformWrapper
from transformnd.transforms.simple import Translate, Scale
from transformnd.transforms.affine import Affine
from transformnd.util import window

from .common import NullTransform


def noop(arg):
    return arg


def test_transform(coords5x3):
    t = TransformWrapper(noop, spaces=(1, 2))

    assert np.allclose(t.apply(coords5x3), coords5x3)


def test_sequence(coords5x3):
    ts = []
    last = 3
    for a, b in window(range(last + 1), 2):
        ts.append(TransformWrapper(noop, spaces=(a, b)))

    t = TransformSequence(ts)
    assert np.allclose(t.apply(coords5x3), coords5x3)
    assert t.source_space == 0
    assert t.target_space == last


def test_sequence_errors():
    with pytest.raises(ValueError):
        TransformSequence(
            [
                TransformWrapper(noop, spaces=(1, 2)),
                TransformWrapper(noop, spaces=(3, 4)),
            ]
        )


def test_sequence_does_not_split():
    t = TransformWrapper(noop)
    seq1 = t | copy(t)
    seq2 = TransformSequence([copy(t), seq1, copy(t)])
    assert len(seq2) == 3
    assert seq2[1] is seq1


def test_sequence_infers():
    t = TransformSequence(
        [
            TransformWrapper(noop, spaces=(0, None)),
            TransformWrapper(noop, spaces=(1, 2)),
        ]
    )
    assert t[0].target_space == 1


def test_add():
    t = [TransformWrapper(noop) for _ in range(5)]
    t12 = t[1] | t[2]
    assert isinstance(t12, TransformSequence)
    assert len(t12) == 2

    t123 = t12 | t[3]
    assert len(t123) == 3
    assert t123[2] is t[3]

    t4123 = t[4] | t123
    assert len(t4123) == 4
    assert t4123[0] is t[4]


def test_maths():
    t1 = Translate[np.ndarray](1)
    coords = np.zeros((5, 3))

    assert np.allclose((t1 | ~t1).apply(coords), coords)
    assert np.allclose((~t1).apply(t1.apply(coords)), coords)


def test_simplify_affine1(rng):
    s1 = Scale(np.array([5, 5, 5], float))
    s2 = Translate(np.array([2, 3, 4], float))
    s3 = NullTransform()
    s4 = Scale(6)
    coords = rng.random((10, 3))
    sequence = TransformSequence([s1, s2, s3, s4])
    expected = sequence.apply(coords)
    simplified = sequence.simplify()
    applied = simplified.apply(coords)
    assert expected == pytest.approx(applied)
    assert len(simplified) == 3
    assert isinstance(simplified[0], Affine)
    assert not isinstance(simplified[1], Affine)
    assert isinstance(simplified[2], Affine)


def test_simplify_affine2(rng):
    coords = rng.random((10, 3))
    dim = 3
    s1 = NullTransform()
    s2 = Scale(np.array([5, 5, 5], float))
    s3 = Translate(np.array([2, 3, 4], float))
    s4 = Scale(6)
    s2_affine = s2.to_affine()
    s3_affine = s3.to_affine()
    s4_affine = s4.to_affine(dim)
    assert s2_affine is not None
    assert s3_affine is not None
    assert s4_affine is not None
    seq = TransformSequence([s1, s2, s3, s4])
    coords2 = seq.apply(coords)
    simplified = seq.simplify()
    assert len(simplified) == 2
    assert isinstance(simplified[0], NullTransform)
    internal_affine = simplified[1]
    assert isinstance(internal_affine, Affine)
    coords3 = simplified.apply(coords)
    assert coords2 == pytest.approx(coords3)

    merged_affine = s4_affine @ s3_affine @ s2_affine
    assert merged_affine.matrix == pytest.approx(internal_affine.matrix)
