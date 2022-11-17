from copy import copy

import numpy as np
import pytest

from transformnd.base import TransformSequence, TransformWrapper
from transformnd.transforms.simple import Translate
from transformnd.util import window


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
    t1 = Translate(1)
    coords = np.zeros((5, 3))

    assert np.allclose((t1 | ~t1).apply(coords), coords)
    assert np.allclose((~t1).apply(t1.apply(coords)), coords)
