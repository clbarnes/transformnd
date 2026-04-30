import polars as pl

from transformnd.adapters.polars import PolarsAdapter
from transformnd.transforms import Scale
import numpy as np

import pytest


def test_pandas():
    df = pl.DataFrame(
        {"x": [1, 2, 3], "y": [4, 5, 6], "value": [1, 1, 1]},
        schema={"x": pl.Float64(), "y": pl.Float64(), "value": pl.UInt64()},
    )
    s = Scale(np.array([10, 20], float))
    a = PolarsAdapter(["x", "y"])
    df2 = a.apply(s, df)

    assert df2["x"].to_numpy() == pytest.approx([10, 20, 30])
    assert df2["y"].to_numpy() == pytest.approx([80, 100, 120])
    assert df2["value"].to_numpy() == pytest.approx([1, 1, 1])
