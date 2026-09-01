import numpy as np
from numba import njit, prange
from scipy.interpolate import CubicSpline


def get_interp(x_bins, y_bins):
    """
    Get the interpolated profiles of the halo.

    Parameters:
    ----------
    x_bins: array
        Array of the x-param.
    y_bins: array
        Array of the y-param.

    Returns:
    -------
    interp: CubicSpline
        Interpolated profile of the halo.
    """

    _x_order = np.argsort(x_bins)
    _x_increasing_mask = np.append([True], np.diff(x_bins[_x_order]) > 0)

    x_bins = x_bins[_x_order][_x_increasing_mask]
    y_bins = y_bins[_x_order][_x_increasing_mask]

    _finite_mask = np.logical_and(np.isfinite(x_bins), np.isfinite(y_bins))

    return CubicSpline(x_bins[_finite_mask], y_bins[_finite_mask])


@njit(parallel=True)
def get_bias_factor(k_mag, b0=0.704, k0=0.0789, alpha=0.419, beta=2):
    """
    Compute the bias factor for a given k magnitude.

    Parameters:
    ----------
    k_mag: array
        Array of k magnitudes.
    b0: float
        Bias factor at k=0. Default is 0.704.
    k0: float
        Scale parameter for the bias factor. Default is 0.0789 cMpc^-1.
    alpha: float
        Power-law index for the bias factor. Default is 0.419.
    beta: float, optional
        Smoothing parameter for the bias factor. Default is 2.

    Returns:
    -------
    b_k: array
        Array of bias factors corresponding to the input k magnitudes.
    """

    _n = k_mag.size
    b_k = np.empty(_n)

    for i in prange(_n):
        b_k[i] = b0 / (1 + (k_mag[i] / k0) ** beta) ** (alpha / beta)

    return b_k
