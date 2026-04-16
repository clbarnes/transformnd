import numpy as np

from ..base import Transform
from ..util import SpaceTuple


class SubTransform:
    """Transformation to apply to subsets of the input dimensions and which output dimensions they calculate."""
    def __init__(
        self,
        transform: Transform,
        input_axes: list[int],
        output_axes: list[int] | None = None,
    ):

        self.input_axes = input_axes
        if output_axes is None:
            # this needs to be adjusted if we want to support drop and add axis
            self.output_axes = input_axes
        else:
            self.output_axes = output_axes

        in_ndim = len(self.input_axes)
        out_ndim = len(self.output_axes)

        if len(set(self.input_axes)) != in_ndim:
            raise ValueError("Input axes must be unique and non-empty")
        if len(set(self.output_axes)) != out_ndim:
            raise ValueError("Output axes must be unique and non-empty")

        if in_ndim != out_ndim:
            raise ValueError("Input and output axes must have the same length")

        if not transform.supports_ndim(in_ndim):
            raise ValueError(f"Given transform does not support {in_ndim} dimensions")

        self.ndim = {in_ndim}
        self.transform = transform


class ByDimension(Transform):
    """Apply transformations to subsets of the coordinates' dimensions.

    Adapted from: https://ngff.openmicroscopy.org/specifications/dev/index.html#bydimension
    """

    def __init__(
        self,
        subtransforms: list[SubTransform],
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
        self.subtransforms = subtransforms
        self.spaces = spaces

        # check that input and output axes of sub transforms are disjoint
        all_input_axes = [ax for t in subtransforms for ax in t.input_axes]
        if len(all_input_axes) != len(set(all_input_axes)):
            raise ValueError("Input axes of sub transforms must be disjoint")

        all_output_axes = [ax for t in subtransforms for ax in t.output_axes]
        if len(all_output_axes) != len(set(all_output_axes)):
            raise ValueError("Output axes of sub transforms must be disjoint")

        self.ndim = {len(all_input_axes)}

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Apply transformation to subset of coordinates."""
        coords_original = coords.copy()
        for sub_seq_transform in self.subtransforms:
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
                transform=~t.transform,
            )
            for t in reversed(self.subtransforms)
        ]
        return type(self)(
            subtransforms=inverted_transforms,
            spaces=(self.spaces[1], self.spaces[0]),
        )
