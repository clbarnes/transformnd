import transformnd as tnd
import pytest


def test_add_transforms():
    tnd.TransformGraph(
        [
            tnd.transforms.Scale([2, 2], spaces=tnd.Spaces("a", "b")),
            tnd.transforms.Translate([10, 20], spaces=tnd.Spaces("b", "c")),
        ]
    )


def test_path():
    t1 = tnd.transforms.Scale([2, 3], spaces=tnd.Spaces("a", "b"))
    t2 = tnd.transforms.Translate([10, 20], spaces=tnd.Spaces("b", "c"))
    transforms = [t1, t2]
    g = tnd.TransformGraph(transforms)
    spaces = ("a", "c")
    seq = g.get_sequence(*spaces, full=True)
    assert len(seq) == 2
    assert isinstance(seq[0], tnd.transforms.Scale)
    assert isinstance(seq[1], tnd.transforms.Translate)

    in_coords = tnd.util.as_floats([[0, 0], [1, 1], [2, 2]])
    res = g.transform("a", "c", in_coords)
    expected = tnd.TransformSequence(transforms).apply(in_coords)
    assert res == pytest.approx(expected)
