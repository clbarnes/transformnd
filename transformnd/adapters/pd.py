from typing import Hashable, List

import pandas as pd

from ..base import Transform
from .base import BaseAdapter


class DataFrameAdapter(BaseAdapter[pd.DataFrame]):
    def __init__(self, columns: List[Hashable]):
        """Adapt transformation for coordinates stored in a pandas DataFrame.

        Parameters
        ----------
        columns : list of keys
            Keys for columns containing coordinates, e.g. ``["x", "y", "z"]``
        """
        self.columns = columns

    def __call__(
        self, transform: Transform, df: pd.DataFrame, in_place=False
    ) -> pd.DataFrame:
        """Transform the dataframe, optionally in-place.

        Parameters
        ----------
        transform : Transform
        df : pd.DataFrame

        in_place : bool, optional
            Whether to mutate the dataframe in place,
            by default False (i.e. make a copy of it).

        Returns
        -------
        pandas.DataFrame
        """
        coords = df[self.columns].to_numpy()
        transformed = transform(coords)
        if not in_place:
            df = df.copy()
        df[self.columns] = transformed
        return df
