import pytest

import numpy as np
from numpy.testing import assert_almost_equal

from analysis import (
    get_ellipse_image,
    EllipseParams,
    fit_ellipse,
    get_cell_boundary,
    cart_to_pol,
    get_ellipse_fit_params,
)


def _get_ellipse_coords(
    input_ellipse: EllipseParams, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    image = get_ellipse_image([input_ellipse], width, height, npts=1000)
    coords = np.argwhere(image == 0)
    x = coords[:, 0]
    y = coords[:, 1]
    return x, y


def test_fit_circle():
    (
        x,
        y,
    ) = _get_ellipse_coords(EllipseParams(100, 100, 100, 100, 0, 0), 200, 200)
    coeffs = fit_ellipse(
        x - 100,
        y - 100,
        200,
        200,
        pbc=False,
    )
    coeffs_n = coeffs / (-coeffs[-1])
    assert np.allclose(coeffs_n, [1e-4, 0, 1e-4, 0, 0, -1], atol=5e-6)

    (
        x,
        y,
    ) = _get_ellipse_coords(
        EllipseParams(100, 100, 100, 50, 0, np.sqrt(1 - 0.25)), 200, 200
    )
    coeffs = fit_ellipse(
        x - 100,
        y - 100,
        200,
        200,
        pbc=False,
    )
    coeffs_n = coeffs / (-coeffs[-1])
    assert np.allclose(coeffs_n, [1e-4, 0, 4e-4, 0, 0, -1], atol=5e-6)


@pytest.mark.parametrize(
    "input_ellipse",
    [
        EllipseParams(100, 100, 100, 100, 0, 0),
        EllipseParams(100, 100, 100, 50, 0, np.sqrt(0.75)),
    ],
)
def test_fit_circle_conical(input_ellipse):
    # input_ellipse = EllipseParams(100, 100, 100, 50, 0, np.sqrt(0.75))

    x, y = _get_ellipse_coords(input_ellipse, 200, 200)
    ellipse = get_ellipse_fit_params(x, y)

    assert_almost_equal(input_ellipse.x0, ellipse.x0, decimal=1)
    assert_almost_equal(input_ellipse.y0, ellipse.y0, decimal=1)
    assert_almost_equal(input_ellipse.a, ellipse.a, decimal=1)
    assert_almost_equal(input_ellipse.b, ellipse.b, decimal=1)
    assert_almost_equal(input_ellipse.phi, ellipse.phi, decimal=1)
    assert_almost_equal(input_ellipse.e, ellipse.e, decimal=1)
