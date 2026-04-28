"""Adapt polars DataFrames for transformation."""

import polars as pl
import numpy as np

from ..base import Transform
from .base import BaseAdapter


class PolarsAdapter(BaseAdapter[pl.DataFrame, np.ndarray]):
    def __init__(self, columns: list[str]):
        """Adapt transformation for coordinates stored in a polars DataFrame.

        Parameters
        ----------
        columns : list of keys
            Keys for columns containing coordinates, e.g. `["x", "y", "z"]`
        """
        self.columns = columns

    def apply(
        self, transform: Transform, df: pl.DataFrame, in_place: bool = False
    ) -> pl.DataFrame:
        """Transform the dataframe, optionally in-place.

        Parameters
        ----------
        transform : Transform
        df : pl.DataFrame

        in_place : bool, optional
            Whether to mutate the dataframe in place,
            by default False (i.e. make a copy of it).

        Returns
        -------
        pandas.DataFrame
        """
        coords = df[self.columns].to_numpy()
        transformed = transform.apply(coords)
        if not in_place:
            df = df.clone()
        df[self.columns] = transformed
        return df
