import numpy as np

from ..base import Transform
from ..util import SpaceTuple


class SubTransform:
    def __init__(
        self, input_axes: list[int], output_axes: list[int] | None, transform: Transform
    ):

        self.input_axes = input_axes
        if output_axes is None:
            # this needs to be adjusted if we want to support drop and add axis
            self.output_axes = input_axes
        else:
            self.output_axes = output_axes
        assert len(set(self.input_axes)) == len(self.input_axes), (
            "Input axes must be unique and non-empty"
        )
        assert len(set(self.output_axes)) == len(self.output_axes), (
            "Output axes must be unique and non-empty"
        )
        assert len(self.input_axes) == len(self.output_axes), (
            "Input and output axes must have the same length"
        )
        self.transform = transform


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

        # check that input and output axes of sub transforms are disjoint
        all_input_axes = [ax for t in sub_seq_transform for ax in t.input_axes]
        assert len(all_input_axes) == len(set(all_input_axes)), (
            "Input axes of sub transforms must be disjoint"
        )
        all_output_axes = [ax for t in sub_seq_transform for ax in t.output_axes]
        assert len(all_output_axes) == len(set(all_output_axes)), (
            "Output axes of sub transforms must be disjoint"
        )

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Apply transformation to subset of coordinates."""
        coords_original = coords.copy()
        for sub_seq_transform in self.sub_seq_transform:
            coords[:, sub_seq_transform.output_axes] = (
                sub_seq_transform.transform.apply(
                    coords_original[:, sub_seq_transform.input_axes]
                )
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
                input_axes=t.output_axes,
                output_axes=t.input_axes,
                transform=t.transform.__invert__(),
            )
            for t in reversed(self.sub_seq_transform)
        ]
        return type(self)(
            sub_seq_transform=inverted_transforms,
            spaces=(self.spaces[1], self.spaces[0]),
        )
