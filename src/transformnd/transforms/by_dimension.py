from array_api_compat import array_namespace

from .simple import Identity

from ..base import Transform
from ..util import ArrayT
from ..types import NDims


class SubTransform[ArrayT]:
    """Component of the `ByDimension` transformation.

    Transformation to apply to subsets of the input dimensions and which output dimensions they calculate.
    """

    def __init__(
        self,
        transform: Transform[ArrayT],
        input_axes: list[int],
        output_axes: list[int] | None = None,
    ):
        """
        Parameters
        ----------
        transform
            Transformation to apply to the subset of axes.
        input_axes
            Which axes to apply the transformation to, in order.
            The length must match the input dimensionality of `transform`.
        output_axes
            Which axes to apply the transformation to, in order.
            The length must match the input dimensionality of `transform`.
            If None, re-use the input axes.

        Raises
        ------
        ValueError
            `transform`'s dimensionality does not match the input/output axes.
        """

        self.input_axes = input_axes
        if output_axes is None:
            self.output_axes = input_axes
        else:
            self.output_axes = output_axes

        in_ndim = len(self.input_axes)
        out_ndim = len(self.output_axes)

        if transform.ndims.source != in_ndim:
            raise ValueError(
                f"Subtransform input dimensionality ({transform.ndims.source}) must match length of input_axes"
            )
        if transform.ndims.target != out_ndim:
            raise ValueError(
                f"Subtransform output dimensionality ({transform.ndims.target}) must match length of output_axes"
            )

        self.transform = transform


class ByDimension(Transform[ArrayT]):
    """Apply transformations to subsets of the coordinates' dimensions.

    Adapted from: https://ngff.openmicroscopy.org/specifications/dev/index.html#bydimension
    """

    def __init__(
        self,
        subtransforms: list[SubTransform[ArrayT]],
        fill_identity: int | None = None,
    ):
        """
        Parameters
        ----------
        subtransforms
            Transformations applying to subsets of the given coordinates.
        fill_identity
            If not None, fill any missing input and output axes with identity transforms in order, up to a maximum number of dimensions.
            e.g. if you have XYT imates which you only want to transform in XY, provide the XY subtransformations and `fill_identity=3`.

        Raises
        ------
        ValueError
            If input or output axes are not valid.
        """
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
                SubTransform(
                    Identity(len(to_fill_in)), sorted(to_fill_in), sorted(to_fill_out)
                )
            )

        # check that input and output axes of sub transforms are disjoint
        sorted_in = sorted(ax for t in subtransforms for ax in t.input_axes)
        if sorted_in != list(range(len(sorted_in))):
            raise ValueError("N-length input axes must go from 0 to N-1")

        sorted_out = sorted(ax for t in subtransforms for ax in t.output_axes)

        if sorted_out != list(range(len(sorted_out))):
            raise ValueError("N-length output axes must go from 0 to N-1")

        super().__init__(NDims(len(sorted_in), len(sorted_out)))
        self.subtransforms = subtransforms

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

    def invert(self) -> Transform[ArrayT] | None:
        try:
            inverted_transforms = [
                SubTransform[ArrayT](
                    input_axes=t.output_axes,
                    output_axes=t.input_axes,
                    transform=~t.transform,
                )
                for t in reversed(self.subtransforms)
            ]
        except NotImplementedError:
            return None

        return type(self)(
            subtransforms=inverted_transforms,
        )

    def is_identity(self) -> bool:
        for t in self.subtransforms:
            if t.input_axes != t.output_axes or not t.transform.is_identity():
                return False
        return True
