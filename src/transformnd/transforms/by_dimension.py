import numpy as np

from ..base import Transform
from ..util import SpaceTuple


class SubTransform(Transform):
    def __init__(
        self, input_axis: list[int], output_axis: list[int] | None, transform: Transform
    ):
        self.input_axis = input_axis
        if output_axis is None:
            # this needs to be adjusted if we want to support drop and add axis
            self.output_axis = input_axis
        else:
            self.output_axis = output_axis
        assert len(self.input_axis) == len(self.output_axis), "Input and output axes must have the same length"
        self.transform = transform

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Apply transformation to subset of coordinates."""
        for transform_step in self.transform:
            coords = transform_step.apply(coords)
        return coords


class ByDimension(Transform):
    """Map coordinates from one axis to another.
    Adapted from: https://ngff.openmicroscopy.org/specifications/dev/index.html#bydimension
    Access date: 15.04.2026
    """

    # ndim: Optional[Set[int]] = set(2)

    def __init__(
        self,
        sub_seq_transform: list[SubTransform],
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """Base class for transformations.

        Parameters
        ----------
        permutation: list[int]
            New order of column axis. For example, [1, 0] means x -> y and y -> x.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces
        """
        self.sub_seq_transform = sub_seq_transform
        self.spaces = spaces

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Apply transformation to subset of coordinates."""
        for sub_seq_transform in self.sub_seq_transform:
            coords[:, sub_seq_transform.output_axis] = sub_seq_transform.apply(
                (coords[:, sub_seq_transform.input_axis])
            )
        return coords

    def __invert__(self) -> Transform:
        """Invert transformation if possible.

        Returns
        -------
        Transform
            Inverted transformation.
        """
        inverted_transforms = [
            SubTransform(
                input_axis=t.output_axis,
                output_axis=t.input_axis,
                transform=[step.__invert__() for step in reversed(t.transform)],
            )
            for t in reversed(self.sub_seq_transform)
        ]
        return type(self)(
            sub_seq_transform=inverted_transforms,
            spaces=(self.spaces[1], self.spaces[0]),
        )
