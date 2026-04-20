import pandas as pd

from transformnd.adapters.pandas import PandasAdapter
from transformnd.transforms import Scale
import numpy as np

import pytest


def test_pandas():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "value": [1, 1, 1]}, dtype=float)
    s = Scale(np.array([10, 20], float))
    a = PandasAdapter(["x", "y"])
    df2 = a.apply(s, df)

    assert df2["x"].to_numpy() == pytest.approx([10, 20, 30])
    assert df2["y"].to_numpy() == pytest.approx([80, 100, 120])
    assert df2["value"].to_numpy() == pytest.approx([1, 1, 1])
