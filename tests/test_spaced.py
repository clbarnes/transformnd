import pytest
from transformnd.spaced import Spaced
from transformnd.transforms import Translate
from transformnd.util import as_floats


def test_invert_spaced(coords5x3):
    t1 = Translate(as_floats([1, 2, 3]))
    s1 = Spaced(t1, "a", "b")
    s2 = s1.invert()
    assert s2 is not None
    assert s2.spaces.source == s1.spaces.target
    assert s2.spaces.target == s1.spaces.source

    assert s2.transform.apply(s1.transform.apply(coords5x3)) == pytest.approx(coords5x3)
