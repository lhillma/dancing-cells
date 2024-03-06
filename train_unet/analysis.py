from dataclasses import dataclass
import numpy as np
import torch

import numba as nb
from numba import njit


@njit
def fit_ellipse(
    x: np.ndarray, y: np.ndarray, width: int, height: int, pbc=True
) -> np.ndarray:
    """
    Fit the coefficient a, b, c, d, e, f, representing an ellipse described by
    the formula F(x, y) = axˆ2 + bxy + cyˆ2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1 , x2, . . . , xn] and y=[y1, y2, . . . , yn] .
    Based on the algorithm of Halir and Flusser, Numerically stable direct
    least squares fitting of ellipses.

    Taken from P C M Wielstra, Investigating the relationship between the nuclear and
    cellular shape and the dynamics in a confluent cell layer, Master thesis, 2023.
    """
    if pbc:
        apply_pbc_offset(x, y, width, height)

    # A = np.vstack([x**2, x * y, y**2, x, y]).T
    # B = np.ones_like(x)
    # coeffs = np.linalg.lstsq(A, B, rcond=None)[0].squeeze()
    # return coeffs

    D1 = np.vstack((x**2, x * y, y**2)).T
    D2 = np.vstack((x, y, np.ones(len(x)))).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=np.float64)
    M = np.linalg.inv(C) @ M

    _, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


spec = [
    ("x0", nb.float64),
    ("y0", nb.float64),
    ("a", nb.float64),
    ("b", nb.float64),
    ("phi", nb.float64),
    ("e", nb.float64),
]


@dataclass(eq=False)
class EllipseParams:
    x0: float
    y0: float
    a: float
    b: float
    phi: float
    e: float


del EllipseParams.__dataclass_params__  # type: ignore
del EllipseParams.__dataclass_fields__  # type: ignore
del EllipseParams.__match_args__  # type: ignore
del EllipseParams.__repr__  # type: ignore
EllipseParams = nb.experimental.jitclass(spec)(EllipseParams)  # type: ignore


@njit
def get_ellipse_fit_params(x: np.ndarray, y: np.ndarray) -> EllipseParams:
    x_com = x.mean()
    y_com = y.mean()

    coeffs = fit_ellipse(x - x_com, y - y_com, 200, 200, False)
    ellipse = cart_to_pol(coeffs)

    ellipse.x0 += x_com
    ellipse.y0 += y_com

    return ellipse


@njit
def cart_to_pol(coeffs: np.ndarray) -> EllipseParams:
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a * c
    if den > 0:
        raise ValueError("Could not be fit to ellipse")

    x0 = (c * d - b * f) / den
    y0 = (a * f - b * d) / den

    num = 2 * (a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g)
    fac = np.sqrt((a - c) ** 2 + 4 * b**2)

    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    a_gt_b = ap > bp
    if a_gt_b:
        ap = ap
        bp = bp
    else:
        ap = bp
        bp = ap

    r = (bp / ap) ** 2
    assert r <= 1, "r > 1"
    e = np.sqrt(1 - r)

    phi = 0.5 * np.arctan2(-2 * b, c - a)
    # phi = 0.5 * np.arctan(2 * b / (a - c)) if b != 0 else np.pi / 2 if a > c else 0
    # if a > c and b != 0:
    #     phi += np.pi / 2

    # if not a_gt_b:
    #     phi += np.pi / 2

    phi = phi % np.pi

    return EllipseParams(x0, y0, ap, bp, phi, e)


def get_ellipse_image(
    params: list[EllipseParams],
    width: int,
    height: int,
    npts=100,
    tmin=0,
    tmax=2 * np.pi,
) -> np.ndarray:
    image = np.ones((width, height), dtype=float)
    for param in params:
        t = np.linspace(tmin, tmax, npts)
        x = (
            param.x0
            + param.a * np.cos(t) * np.cos(param.phi)
            - param.b * np.sin(t) * np.sin(param.phi)
        )
        y = (
            param.y0
            + param.a * np.cos(t) * np.sin(param.phi)
            + param.b * np.sin(t) * np.cos(param.phi)
        )
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        x[x < 0] += width
        x[x >= width] -= width
        y[y < 0] += height
        y[y >= height] -= height
        image[x, y] = 0

    return image


@njit
def get_cell_boundary(
    image: np.ndarray, cell_ids: np.ndarray, cell_id: int
) -> np.ndarray:
    cell_boundary = np.zeros(image.shape, dtype=float)
    cell_boundary += np.where(cell_ids == cell_id, 1.0, 0.0) * (
        1 - image.astype(np.float64)
    )

    return cell_boundary.astype(np.uint8)


@njit
def apply_pbc_offset(x: np.ndarray, y: np.ndarray, width: int, height: int) -> None:
    delta_x = x.max() - x.min()
    delta_y = y.max() - y.min()

    if delta_x > width / 2:
        x[x > width / 2] -= width

    if delta_y > height / 2:
        y[y > height / 2] -= height


def hsv_to_rgb(
    h: torch.Tensor, s: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert HSV to RGB color space.

    Args:
        h: Hue
        s: Saturation
        v: Value

    Returns:
        r: Red
        g: Green
        b: Blue
    """
    h *= 360
    c = v * s
    x = c * (1 - torch.abs((h / 60) % 2 - 1))
    m = v - c

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    idx = (h >= 0) & (h < 60)
    r[idx] = c[idx]
    g[idx] = x[idx]

    idx = (h >= 60) & (h < 120)
    r[idx] = x[idx]
    g[idx] = c[idx]

    idx = (h >= 120) & (h < 180)
    g[idx] = c[idx]
    b[idx] = x[idx]

    idx = (h >= 180) & (h < 240)
    g[idx] = x[idx]
    b[idx] = c[idx]

    idx = (h >= 240) & (h < 300)
    r[idx] = x[idx]
    b[idx] = c[idx]

    idx = (h >= 300) & (h < 360)
    r[idx] = c[idx]
    b[idx] = x[idx]

    r += m
    g += m
    b += m

    return r, g, b
