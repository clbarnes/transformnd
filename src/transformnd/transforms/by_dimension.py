from array_api_compat import array_namespace

from .simple import Identity

from ..base import Transform
from ..util import SpaceTuple, ArrayT, check_ndim


class SubTransform[ArrayT]:
    """Transformation to apply to subsets of the input dimensions and which output dimensions they calculate."""
    def __init__(
        self,
        transform: Transform[ArrayT],
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

        check_ndim(in_ndim, transform.ndim)

        self.ndim = in_ndim
        self.transform = transform


class ByDimension(Transform[ArrayT]):
    """Apply transformations to subsets of the coordinates' dimensions.

    Adapted from: https://ngff.openmicroscopy.org/specifications/dev/index.html#bydimension
    """

    def __init__(
        self,
        subtransforms: list[SubTransform[ArrayT]],
        fill_identity: int | None = None,
        *,
        spaces: SpaceTuple = (None, None),
    ):
        """
        Parameters
        ----------
        subtransforms: list[Subtransform]
            Transformations applying to subsets of the given coordinates.
        fill_identity: int | None
            If not None, fill any missing input and output axes with identity transforms in order, up to a maximum number of dimensions.
            e.g. if you have XYT imates which you only want to transform in XY, provide the XY subtransformations and `fill_identity=3`.
        spaces : tuple[SpaceRef, SpaceRef]
            Optional source and target spaces
        """
        self.spaces = spaces

        if fill_identity is not None:
            to_fill_in = set(range(fill_identity))
            to_fill_out = set(range(fill_identity))
            for t in subtransforms:
                for i in t.input_axes:
                    try:
                        to_fill_in.remove(i)
                    except KeyError:
                        pass
                for i in t.output_axes:
                    try:
                        to_fill_out.remove(i)
                    except KeyError:
                        pass
            subtransforms.append(
                SubTransform(Identity(), sorted(to_fill_in), sorted(to_fill_out))
            )

        # check that input and output axes of sub transforms are disjoint
        sorted_in = sorted(ax for t in subtransforms for ax in t.input_axes)
        if sorted_in != list(range(len(sorted_in))):
            raise ValueError("N-length input axes must go from 0 to N-1")

        sorted_out = sorted(ax for t in subtransforms for ax in t.output_axes)

        if sorted_out != list(range(len(sorted_out))):
            raise ValueError("N-length output axes must go from 0 to N-1")

        self.subtransforms = subtransforms
        self.ndim = {len(sorted_in)}

    def apply(self, coords: ArrayT) -> ArrayT:
        """Apply transformation to subset of coordinates."""
        coords = self._validate_coords(coords)
        xp = array_namespace(coords)
        output = xp.empty_like(coords)
        for t in self.subtransforms:
            transformed = t.transform.apply(xp.take(coords, t.input_axes, 1))
            for idx, o in enumerate(t.output_axes):
                output[:, o] = transformed[:, idx]  # type: ignore
        return output

    def __invert__(self) -> Transform[ArrayT]:
        """Invert transformation if possible.

        Returns
        -------
        Transform
            Inverted transformation.
        """
        inverted_transforms = [
            SubTransform[ArrayT](
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
