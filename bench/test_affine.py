import pytest
import numpy as np


@pytest.mark.benchmark(group="pad-unpad")
@pytest.mark.parametrize(
    ("n_coords",), [(1,), (10,), (100,), (1_000,), (10_000,), (100_000,)]
)
class TestPadUnpadCoords:
    ndim = 3

    def scale_vector(self):
        return np.asarray([10, 20, 30], float)

    def translation_vector(self):
        return np.asarray([-0.5, 0.0, 0.5], float)

    def coords(self, rng: np.random.Generator, n_coords: int):
        return rng.random((n_coords, self.ndim))

    def affine(self, rng: np.random.Generator):
        m = rng.random((self.ndim + 1, self.ndim + 1))
        m[-1, :] = 0
        m[-1, -1] = 1
        return m

    def test_pad_unpad(self, n_coords, rng, benchmark):
        affine = self.affine(rng)
        coords = self.coords(rng, n_coords)

        def fn():
            c2 = np.concatenate(
                [coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)], axis=1
            )
            inter = c2 @ affine.T
            out = inter[:, :-1]  # noqa

        benchmark(fn)

    def test_lin_then_trans(self, n_coords, rng, benchmark):
        affine = self.affine(rng)
        coords = self.coords(rng, n_coords)

        def fn():
            t = affine.T
            inter = coords @ t[:-1, :-1]
            inter += t[-1, :-1]

        benchmark(fn)
