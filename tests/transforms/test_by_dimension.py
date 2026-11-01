import numpy as np
import pytest

from transformnd.base import TransformSequence
from transformnd.transforms import ByDimension, MapAxis, Scale, Translate
from transformnd.transforms.by_dimension import SubTransform

from ..common import NullTransform


def test_2d_scale():
    factor = [1.5]
    coords = np.array([[1, 2], [3, 4]], dtype=float)
    scale = Scale(factor)
    subseq = SubTransform(input_axes=[0], output_axes=[0], transform=scale)
    by_dim = ByDimension(subtransforms=[subseq], fill_identity=2)
    coords_transformed = by_dim.apply(coords.copy())
    assert coords_transformed[:, 0] == pytest.approx(coords[:, 0] * factor)
    assert coords_transformed[:, 1] == pytest.approx(coords[:, 1])


def test_3d_map_axis_and_scale():
    """Test 3D transformation with map_axis on columns 0,1 and scale on column 2."""
    s = 1.5
    coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

    # MapAxis: swap columns 0 and 1, keep column 2
    map_axis = MapAxis(permutation=[1, 0])
    map_axis_subseq = SubTransform(
        input_axes=[0, 1], output_axes=[0, 1], transform=map_axis
    )

    # Scale: apply scale to column 2
    scale = Scale([s])
    scale_subseq = SubTransform(input_axes=[2], output_axes=[2], transform=scale)

    # Apply both transformations (different order)
    by_dim_0 = ByDimension(subtransforms=[map_axis_subseq, scale_subseq])
    by_dim_1 = ByDimension(subtransforms=[scale_subseq, map_axis_subseq])

    coords_0 = by_dim_0.apply(coords.copy())
    coords_1 = by_dim_1.apply(coords.copy())

    # check that order of transformations does not matter
    assert np.array_equal(coords_0, coords_1), (
        "Order of transformations should not matter"
    )

    # Expected: columns 0 and 1 swapped, column 2 scaled by s
    expected = np.array([[2, 1, 3 * s], [5, 4, 6 * s]])
    print(coords_0, expected)
    assert np.allclose(coords_0, expected), (
        "Transformed coordinates do not match expected values"
    )

    ### test invert
    inverted = ~by_dim_0
    assert np.array_equal(inverted.apply(coords_0), coords)


def test_3d_transform_sequence():
    """Test multiple transformations (TransformSequence) for one subset of columns."""
    s = 1.5
    coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)

    # MapAxis: swap columns 0 and 1, keep column 2
    map_axis = MapAxis(permutation=[1, 0])
    scale_seq = Scale([s + 2] * 2)
    transform_sequence = TransformSequence([map_axis, scale_seq])
    map_axis_subseq = SubTransform(
        input_axes=[0, 1], output_axes=[0, 1], transform=transform_sequence
    )

    # Scale: apply scale to column 2
    scale = Scale([s])
    scale_subseq = SubTransform(input_axes=[2], output_axes=[2], transform=scale)

    # Apply both transformations
    by_dim = ByDimension(subtransforms=[map_axis_subseq, scale_subseq])
    coords_transformed = by_dim.apply(coords)

    # Expected: columns 0 and 1 swapped, column 2 scaled by s
    expected = np.array(
        [[2 * (s + 2), 1 * (s + 2), 3 * s], [5 * (s + 2), 4 * (s + 2), 6 * s]],
        dtype=float,
    )
    assert coords_transformed == pytest.approx(expected)

    ### test invert
    inverted = ~by_dim
    assert inverted.apply(coords_transformed) == pytest.approx(coords)


def test_non_unique_axes():
    """Test that non-unique axes raise an error."""
    scale = Scale([2])
    # non-unique input axes
    with pytest.raises(ValueError):
        SubTransform(input_axes=[0, 0], output_axes=[0, 1], transform=scale)
    # non-unique output axes
    with pytest.raises(ValueError):
        t_1 = SubTransform(input_axes=[0, 1], output_axes=[0, 1], transform=scale)
        t_2 = SubTransform(input_axes=[1, 2], output_axes=[1, 2], transform=scale)
        ByDimension(subtransforms=[t_1, t_2])


def test_cross_axes_transform():
    """Test for two subtransforms, where input axes of one subtransform overlap with output axes of the other subtransform.
    This should take apply both transformations on the original axis."""
    s_1, s_2 = 2, 3
    coords = np.array([[1, 2], [3, 4]], dtype=float)
    coords_transformed = np.array([[2 * s_2, 1 * s_1], [4 * s_2, 3 * s_1]])
    t_1 = SubTransform(input_axes=[0], output_axes=[1], transform=Scale([s_1]))
    t_2 = SubTransform(input_axes=[1], output_axes=[0], transform=Scale([s_2]))
    by_dim = ByDimension(subtransforms=[t_1, t_2])
    coords_byDim = by_dim.apply(coords.copy())
    assert np.array_equal(coords_byDim, coords_transformed)


@pytest.mark.parametrize(
    "subtransforms",
    [
        # scale, with identity filled in
        [SubTransform(Scale([1.5]), [1], [1])],
        # map axis and scale
        [
            SubTransform(MapAxis([1, 0]), [0, 1], [0, 1]),
            SubTransform(Scale([1.5]), [2], [2]),
        ],
        # sequence with translation, to swapped axes
        [
            SubTransform(
                TransformSequence([MapAxis([1, 0]), Translate([3.0, -4.0])]),
                [0, 2],
                [2, 0],
            ),
            SubTransform(Scale([2.0]), [1], [1]),
        ],
        # crossed axes
        [
            SubTransform(Scale([2.0]), [0], [1]),
            SubTransform(Translate([5.0]), [1], [2]),
            SubTransform(Scale([3.0]), [2], [0]),
        ],
    ],
)
def test_to_affine(subtransforms):
    by_dim = ByDimension(subtransforms=subtransforms, fill_identity=3)
    coords = np.random.default_rng(0).normal(size=(10, 3))
    affine = by_dim.to_affine()
    assert affine is not None
    assert affine.apply(coords.copy()) == pytest.approx(by_dim.apply(coords.copy()))


def test_to_affine_in_sequence():
    by_dim = ByDimension([SubTransform(Translate([2.0]), [0], [0])], fill_identity=2)
    affine = TransformSequence([by_dim, Scale([2.0, 3.0])]).to_affine()
    assert affine is not None
    assert affine.apply(np.array([[1.0, 1.0]])) == pytest.approx(np.array([[6.0, 3.0]]))


def test_to_affine_not_affineable():
    by_dim = ByDimension(
        [SubTransform(NullTransform(1, affineable=False), [0], [0])],
        fill_identity=2,
    )
    assert by_dim.to_affine() is None
