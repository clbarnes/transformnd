import numpy as np
import pytest

from transformnd.transforms.by_dimension import SubTransform
from transformnd.transforms import ByDimension, Scale, MapAxis
from transformnd.base import TransformSequence

@pytest.mark.parametrize(["s"], [[s] for s in range(2, 6)])
def test_2d_scale(s):
    coords = np.array([[1, 2], [3, 4]])
    scale = Scale(s)
    print(s)
    subseq = SubTransform(input_axes=[0], output_axes=[0], transform=scale)
    by_dim = ByDimension(sub_seq_transform=[subseq])
    coords_transformed = by_dim.apply(coords.copy())
    print(coords_transformed)
    print(coords[:, 0] * s)
    assert np.array_equal(coords_transformed[:, 0], coords[:, 0] * s)
    assert np.array_equal(coords_transformed[:, 1], coords[:, 1])


@pytest.mark.parametrize(["s"], [[s] for s in range(2, 4)])
def test_3d_map_axis_and_scale(s):
    """Test 3D transformation with map_axis on columns 0,1 and scale on column 2."""
    coords = np.array([[1, 2, 3], [4, 5, 6]])

    # MapAxis: swap columns 0 and 1, keep column 2
    map_axis = MapAxis(permutation=[1, 0])
    map_axis_subseq = SubTransform(
        input_axes=[0, 1], output_axes=[0, 1], transform=map_axis
    )

    # Scale: apply scale to column 2
    scale = Scale(s)
    scale_subseq = SubTransform(
        input_axes=[2], output_axes=[2], transform=scale
    )

    # Apply both transformations
    by_dim = ByDimension(sub_seq_transform=[map_axis_subseq, scale_subseq])
    coords_transformed = by_dim.apply(coords)

    # Expected: columns 0 and 1 swapped, column 2 scaled by s
    expected = np.array([[2, 1, 3 * s], [5, 4, 6 * s]])
    assert np.allclose(coords_transformed, expected)

    ### test invert
    inverted = ~by_dim
    assert np.array_equal(inverted.apply(coords_transformed), coords)


@pytest.mark.parametrize(["s"], [[s] for s in range(2, 4)])
def test_3d_transform_sequence(s):
    """Test multiple transformations (TransformSequence)for one subset of columns."""
    coords = np.array([[1, 2, 3], [4, 5, 6]])

    # MapAxis: swap columns 0 and 1, keep column 2
    map_axis = MapAxis(permutation=[1, 0])
    scale_seq = Scale(s + 2)
    transform_sequence = TransformSequence([map_axis, scale_seq])
    map_axis_subseq = SubTransform(
        input_axes=[0, 1], output_axes=[0, 1], transform=transform_sequence
    )

    # Scale: apply scale to column 2
    scale = Scale(s)
    scale_subseq = SubTransform(
        input_axes=[2], output_axes=[2], transform=scale
    )

    # Apply both transformations
    by_dim = ByDimension(sub_seq_transform=[map_axis_subseq, scale_subseq])
    coords_transformed = by_dim.apply(coords)

    # Expected: columns 0 and 1 swapped, column 2 scaled by s
    expected = np.array(
        [[2 * (s + 2), 1 * (s + 2), 3 * s], [5 * (s + 2), 4 * (s + 2), 6 * s]]
    )
    assert np.allclose(coords_transformed, expected)

    ### test invert
    inverted = ~by_dim
    assert np.array_equal(inverted.apply(coords_transformed), coords)
