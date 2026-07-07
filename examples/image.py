import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Image transformation with `transformnd`

    `transformnd` transforms coordinates, not images, but coordinate transformations can be used to transform images.
    Your output (transformed) and source images both have pixels with an `xy` coordinate in their respective image spaces,
    and image transformation is simply a case of finding which source pixel to use for each output pixel.

    Here we take a 2-channel fluorescence microscopy image of some cells in 3 dimensions, use scaling information to map those pixels into a real-world space, and then map the pixels of our viewport into the the same space.
    """)
    return


@app.cell
def _():
    from skimage.data import cells3d

    # ZCYX, (0.29um, membrane/nuclei channels, 0.26um, 0.26um)
    CELLS_SPACE = "cells"

    # CZYX, (membrane/nuclei, um, um, um)
    WORLD_SPACE = "world"

    # YXC image with RGB channels
    VIEWPORT_SPACE = "viewport"

    cells = cells3d()
    cells = cells.astype("float64")
    cells -= cells.min()
    cells /= cells.max()

    print(f"{cells.shape=}")
    print(f"{cells.dtype=}")
    print(f"{cells.min()=}")
    print(f"{cells.max()=}")
    return CELLS_SPACE, VIEWPORT_SPACE, WORLD_SPACE, cells


@app.cell
def _(CELLS_SPACE, VIEWPORT_SPACE, WORLD_SPACE):
    import transformnd as tnd
    from transformnd.transforms import ProjectAxis, MapAxis, Scale

    # Aligned at world origin.
    # This would be stored alongside the data.
    cells_to_world = tnd.base.TransformSequence(
        [
            # Move the color axis to the first position
            MapAxis([1, 0, 2, 3]),
            # Scale the space axes
            Scale([1, 0.29, 0.26, 0.26]),
        ],
    )
    print(cells_to_world)

    # Aligned at world origin.
    # This would be chosen by the viewing application.
    viewport_to_world = tnd.base.TransformSequence(
        [
            # Create a Z axis
            ProjectAxis(created={0}, source_ndim=3),
            # Move the color axis to the first position
            MapAxis([3, 0, 1, 2]),
            # Choose a spatial sampling frequency (here 0.2um isotropic)
            Scale([1, 0.2, 0.2, 0.2]),
        ],
    )
    print(viewport_to_world)
    return cells_to_world, viewport_to_world


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Both images know how to transform their array indices into the real world.

    We can invert one of those transforms to get a transformation between viewport-space and cell-space.
    We can also have a separate transformation to control moving the viewport (useful if we had an interactive viewer).
    """)
    return


@app.cell
def _(cells_to_world, viewport_to_world):
    from transformnd.transforms import Translate

    # Shift the viewport within the data, in world measurements.
    # This would be controlled by the user as they peruse the data.
    viewport_offset = Translate([0, 35 * 0.29, 64 * 0.26, 0.0])

    viewport_to_cells = viewport_to_world | viewport_offset | ~cells_to_world
    return (viewport_to_cells,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here we want to get all of the coordinates of our viewport, across all channels, in the shape needed by `transformnd` (number of coordinates x dimensionality of coordinates).

    We then transform that to get the positions of those coordinates within the cells image.
    """)
    return


@app.cell
def _(viewport_to_cells):
    import numpy as np

    # 2D YXC image
    viewport_shape = (128, 256, 3)

    indices = [np.arange(s, dtype=float) for s in viewport_shape]
    grids = np.meshgrid(*indices, indexing="ij")

    # Y, X, C, coords
    vp_coords_3d = np.stack(grids, -1)
    print(f"{vp_coords_3d.shape=}")

    # Y*X*C, coords
    vp_coords = vp_coords_3d.reshape((-1, len(viewport_shape)))
    print(f"{vp_coords.shape=}")

    # Z*C*Y*X, coords
    cells_coords = viewport_to_cells.apply(vp_coords)
    print(f"{cells_coords.shape=}")
    return cells_coords, viewport_shape


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `scipy.ndimage.map_coordinates` is where the magic happens; looking up our coordinates in the cells image to get the intensities. There's a dask version too!
    """)
    return


@app.cell
def _(cells, cells_coords, viewport_shape):
    from scipy.ndimage import map_coordinates

    # transformnd uses `NxD` coordinate arrays; map_coordinates uses `DxN`
    cells_vals = map_coordinates(cells, cells_coords.T).T
    print(f"{cells_vals.shape=}")
    viewport = cells_vals.reshape(viewport_shape)
    print(f"{viewport.shape=}")
    return (viewport,)


@app.cell
def _(viewport):
    from matplotlib import pyplot as plt

    plt.imshow(viewport)
    return


if __name__ == "__main__":
    app.run()
