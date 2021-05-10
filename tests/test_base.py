from copy import copy

import pytest
import numpy as np

from transformnd.base import TransformSequence, TransformWrapper
from transformnd.util import window


def noop(arg):
    return arg


def test_transform(coords5x3):
    t = TransformWrapper(noop, source_space=1, target_space=2)

    assert np.allclose(t(coords5x3), coords5x3)


def test_sequence(coords5x3):
    ts = []
    last = 3
    for a, b in window(range(last + 1), 2):
        ts.append(TransformWrapper(noop, source_space=a, target_space=b))

    t = TransformSequence(ts)
    assert np.allclose(t(coords5x3), coords5x3)
    assert t.source_space == 0
    assert t.target_space == last


def test_sequence_errors():
    with pytest.raises(ValueError):
        TransformSequence([
            TransformWrapper(noop, source_space=1, target_space=2),
            TransformWrapper(noop, source_space=3, target_space=4),
        ])


def test_sequence_splits():
    t = TransformWrapper(noop)
    seq1 = t + copy(t)
    seq2 = TransformSequence([copy(t), seq1, copy(t)])
    assert len(seq2) == 4
    assert seq2[1] is t
    assert seq2[2] is not t


def test_sequence_infers():
    t = TransformSequence([
        TransformWrapper(noop, source_space=0),
        TransformWrapper(noop, source_space=1, target_space=2),
    ])
    assert t[0].target_space == 1


def test_add():
    t = [TransformWrapper(noop) for _ in range(5)]
    t12 = t[1] + t[2]
    assert isinstance(t12, TransformSequence)
    assert len(t12) == 2

    t123 = t12 + t[3]
    assert len(t123) == 3
    assert t123[2] is t[3]

    t4123 = t[4] + t123
    assert len(t4123) == 4
    assert t4123[0] is t[4]
