from typing import Hashable, List

import pandas as pd

from ..base import Transform
from .base import BaseAdapter


class DataFrameAdapter(BaseAdapter[pd.DataFrame]):
    def __init__(self, columns: List[Hashable]):
        self.columns = columns

    def __call__(self, transform: Transform, df: pd.DataFrame, in_place=False):
        coords = df[self.columns].to_numpy()
        transformed = transform(coords)
        if not in_place:
            df = df.copy()
        df[self.columns] = transformed
        return df
