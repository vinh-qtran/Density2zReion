import os

import h5py
import numpy as np
from numba import njit, prange
from scipy.special import hyp2f1

from density2zreion.fourier import FFTWPlan, FourierTransform
from density2zreion.grid import CICDensityGrid
from density2zreion.utils import check_file_exists, load_or_compute


def _get_all_simulation_files(sim_dir, file_base):
    """
    Get all simulation files in a directory that match a given base name.

    Parameters:
    ----------
    sim_dir: str or Path
        Path to the simulation directory.
    file_base: str
        Base name of the simulation files.

    Returns:
    -------
    sim_files: list
        List of paths to the simulation files.
    """

    return sorted(
        [
            os.path.join(sim_dir, f)  # noqa: PTH118
            for f in os.listdir(sim_dir)  # noqa: PTH208
            if f.startswith(file_base) and f.endswith(".hdf5")
        ]
    )


def particles_to_grid(sim_dir, file_base, grid_file, N_cell=None, N_max_load=None):
    """
    Convert particle data to a density grid.

    Parameters:
    ----------
    sim_dir: str or Path
        Path to the simulation directory.
    file_base: str
        Base name of the simulation files.
    grid_file: str or Path
        Path to the output grid file.
    N_cell: int, optional
        Number of cells in each dimension of the grid. If None, it will be determined automatically.
    N_max_load: int, optional
        Maximum number of particles to load at a time. If None, it will be determined automatically.

    Returns:
    -------
    None
    """

    _sim_files = _get_all_simulation_files(sim_dir, file_base)

    _grid_name = os.path.basename(grid_file)  # noqa: PTH119
    _processed_dir = os.path.join(  # noqa: PTH118
        os.path.dirname(grid_file),  # noqa: PTH120
        f"{_grid_name.split('.')[0]}_processed",
    )
    os.makedirs(_processed_dir, exist_ok=True)  # noqa: PTH103

    _checkpoints = sorted(
        f.name
        for f in os.scandir(_processed_dir)
        if f.name.startswith(_grid_name) and f.name.endswith(".hdf5")
    )
    _processed_file = (
        os.path.join(_processed_dir, _checkpoints[-1]) if _checkpoints else None  # noqa: PTH118
    )
    _start = int(_checkpoints[-1].split(".")[-2]) if _checkpoints else -1

    for i, _sim_file in enumerate(_sim_files):
        if i <= _start:
            continue

        _new_processed_file = os.path.join(_processed_dir, f"{_grid_name}.{i:04d}.hdf5")  # noqa: PTH118

        _density_grid = CICDensityGrid(
            sim_file=_sim_file,
            grid_file=_processed_file
            if _processed_file is not None
            else _new_processed_file,
            N_cell=N_cell,
            N_max_load=N_max_load,
        )
        _density_grid.save_grid(_new_processed_file)

        if _processed_file is not None:
            os.remove(_processed_file)  # noqa: PTH107

        _processed_file = _new_processed_file

    if _processed_file is not None:
        os.rename(_processed_file, grid_file)  # noqa: PTH104


@njit(parallel=True)
def _get_bias_factor(k_mag, b0=0.704, k0=0.0789, alpha=0.419, beta=2):
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


def _get_fourier_transform(box_size, N_cell):
    _fftw_plan = FFTWPlan(N_cell=N_cell)
    _fftw_forward = _fftw_plan.build_forward_plan()
    _fftw_inverse = _fftw_plan.build_inverse_plan()

    return FourierTransform(
        box_size=box_size,
        N_cell=N_cell,
        fftw_forward=_fftw_forward,
        fftw_inverse=_fftw_inverse,
    )


def _get_density_grid(grid_file, double_precision=False):
    with h5py.File(grid_file, "r") as f:
        box_size = f["Header"].attrs["BoxSize"]
        N_cell = f["Header"].attrs["NumCells"]
        cell_size = f["Header"].attrs["CellSize"]
        a = f["Header"].attrs["Time"]

        _rho_grid = f["DensityDM"][:].astype(
            np.float64 if double_precision else np.float32
        )
        _rho_mean = np.mean(_rho_grid)

        delta_m_x = _rho_grid / _rho_mean - 1
        del _rho_grid, _rho_mean

    return delta_m_x, box_size, N_cell, cell_size, a


def _get_power_spectrum_scaler(a, z_reion_params, cosmology_params):
    def _D(a):
        return a * hyp2f1(
            1 / 3,
            1,
            11 / 6,
            -(a**3) * cosmology_params["Omega_L"] / cosmology_params["Omega_m"],
        )

    _a_z_reion_mean = 1 / (1 + z_reion_params["z_reion_mean"])

    return (_D(_a_z_reion_mean) / _D(a)) ** 2


def density_to_z_reion(
    density_grid_file,
    z_reion_grid_file,
    save_files=None,
    z_reion_params=None,
    cosmology_params=None,
    double_precision=False,
):
    _save_files = save_files or {
        "density_fourier_file": None,
        "conv_kernel_Ncell_file": None,
        "conv_kernel_Mpc_file": None,
        "z_reion_fourier_file": None,
    }

    _z_reion_params = z_reion_params or {
        "b0": 0.704,
        "k0": 0.0789,
        "alpha": 0.419,
        "z_reion_mean": 7.2,
    }

    _cosmology_params = cosmology_params or {
        "Omega_m": 0.3089,
        "Omega_L": 0.6911,
    }

    _real_dtype = np.float64 if double_precision else np.float32
    _complex_dtype = np.complex128 if double_precision else np.complex64

    # Get the density grid and its properties
    _delta_m_x, _box_size, _N_cell, _cell_size, _a = _get_density_grid(
        density_grid_file, double_precision=double_precision
    )

    # Get the Fourier transform utils
    _fourier_transform = _get_fourier_transform(_box_size, _N_cell)

    # Get the reionization redshift in Fourier space
    if not check_file_exists(_save_files["z_reion_fourier_file"]):
        # Get the density grid in Fourier space
        _k_mag, _delta_m_k = load_or_compute(
            save_file=_save_files["density_fourier_file"],
            datasets={
                "WaveVector": _real_dtype,
                "DensityDMFourier": _complex_dtype,
            },
            func=_fourier_transform.get_fourier_transform,
            delta_x=_delta_m_x,
        )
        del _delta_m_x

        _delta_m_k *= _get_power_spectrum_scaler(_a, _z_reion_params, _cosmology_params)

        # Apply convolution kernels if needed
        if _cell_size < np.pi:
            _conv_kernel_Ncell = load_or_compute(
                save_file=_save_files["conv_kernel_Ncell_file"],
                datasets={
                    "ConvKernelNcell": _complex_dtype,
                },
                func=_fourier_transform.get_conv_kernel,
                dx=_cell_size,
            )
            _delta_m_k /= _conv_kernel_Ncell
            del _conv_kernel_Ncell

            _conv_kernel_Mpc = load_or_compute(
                save_file=_save_files["conv_kernel_Mpc_file"],
                datasets={
                    "ConvKernelMpc": _complex_dtype,
                },
                func=_fourier_transform.get_conv_kernel,
                dx=np.pi,
            )
            _delta_m_k *= _conv_kernel_Mpc
            del _conv_kernel_Mpc

        # Get the reionization redshift in Fourier space
        _delta_z_k = _delta_m_k * _get_bias_factor(
            _k_mag.ravel(),
            b0=_z_reion_params["b0"],
            k0=_z_reion_params["k0"],
            alpha=_z_reion_params["alpha"],
        ).reshape(_k_mag.shape)
        del _delta_m_k, _k_mag

    else:
        # Get the reionization redshift in Fourier space
        with h5py.File(_save_files["z_reion_fourier_file"], "r") as f:
            _delta_z_k = f["ReionizationRedshiftFourier"][:].astype(_complex_dtype)

        del _delta_m_x

    # Get the reionization redshift in real space
    _delta_z_x = _fourier_transform.get_inv_fourier_transform(_delta_z_k)
    del _delta_z_k

    _z_reion_mean = _z_reion_params["z_reion_mean"]
    _z_reion_grid = _delta_z_x * (1 + _z_reion_mean) + _z_reion_mean
    del _delta_z_x

    with h5py.File(z_reion_grid_file, "w") as f:
        _header = f.create_group("Header")

        _header.attrs.create("BoxSize", _box_size, dtype=np.float64)
        _header.attrs.create("NumCells", _N_cell, dtype=np.int32)
        _header.attrs.create("CellSize", _cell_size, dtype=np.float64)

        f.create_dataset("ReionizationRedshift", data=_z_reion_grid, dtype=_real_dtype)
