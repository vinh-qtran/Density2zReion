import os

import h5py
import numpy as np
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


def check_file_exists(save_file):
    return save_file is not None and os.path.exists(save_file)  # noqa: PTH110


def load_or_compute(save_file, datasets, func, *args, **kwargs):
    if check_file_exists(save_file):
        with h5py.File(save_file, "r") as _f:
            return tuple(_f[_field_name][:] for _field_name in datasets)

    out = func(*args, **kwargs)
    if save_file is not None:
        with h5py.File(save_file, "w") as _f:
            for (_field_name, _dtype), _data in zip(datasets.items(), out, strict=True):
                _f.create_dataset(_field_name, data=_data, dtype=_dtype)

    return out if isinstance(out, tuple) else (out,)
