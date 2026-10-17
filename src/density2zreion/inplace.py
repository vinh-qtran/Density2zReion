import os

import h5py
import numpy as np
import pyfftw
from numba import njit, prange
from scipy.special import hyp2f1
from tqdm import tqdm

from density2zreion.grid import CICDensityGrid


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


@njit
def _get_bias_factor(k_mag, b0=0.704, k0=0.0789, alpha=0.419, beta=2):
    """
    Compute the bias factor for a given k magnitude.

    Parameters:
    ----------
    k_mag: float
        Magnitude of the wavevector.
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
    b_k: float
        Bias factor corresponding to the input k magnitude.
    """

    return b0 / (1 + (k_mag / k0) ** beta) ** (alpha / beta)


@njit(parallel=True)
def _get_overdensity(rho_x, rho_mean):
    """
    Convert a density grid to an overdensity grid in place, delta = rho / rho_mean - 1.

    Parameters:
    ----------
    rho_x: 3D array
        Density grid in real space. Overwritten with the overdensity.
    rho_mean: float
        Mean density of the grid.
    """

    _N_x, _N_y, _N_z = rho_x.shape

    for i in prange(_N_x):
        for j in range(_N_y):
            for k in range(_N_z):
                rho_x[i, j, k] = rho_x[i, j, k] / rho_mean - 1


@njit(parallel=True)
def _apply_fourier_filter(
    delta_k, k_xy_1d, k_z_1d, w_xy_1d, w_z_1d, scaler, b0, k0, alpha, beta
):
    """
    Multiply the Fourier-space density field in place by the total filter
    F(k) = scaler * W(kx) * W(ky) * W(kz) * b(|k|), computed on the fly.

    Parameters:
    ----------
    delta_k: 3D complex array
        Density field in Fourier space, of shape (N_cell, N_cell, N_cell//2 + 1). Overwritten with the reionization redshift field in Fourier space.
    k_xy_1d: 1D array
        Wavenumbers along the x and y axes.
    k_z_1d: 1D array
        Wavenumbers along the z axis (half-space).
    w_xy_1d: 1D array
        Separable convolution kernel ratio along the x and y axes.
    w_z_1d: 1D array
        Separable convolution kernel ratio along the z axis.
    scaler: float
        Linear growth factor ratio D(z_reion_mean) / D(z).
    b0, k0, alpha, beta: float
        Bias factor parameters, see _get_bias_factor.
    """

    _N_x, _N_y, _N_z = delta_k.shape

    for i in prange(_N_x):
        for j in range(_N_y):
            for k in range(_N_z):
                _k_mag = np.sqrt(k_xy_1d[i] ** 2 + k_xy_1d[j] ** 2 + k_z_1d[k] ** 2)

                delta_k[i, j, k] *= (
                    scaler
                    * w_xy_1d[i]
                    * w_xy_1d[j]
                    * w_z_1d[k]
                    * _get_bias_factor(_k_mag, b0, k0, alpha, beta)
                )


@njit(parallel=True)
def _get_z_reion(delta_z_x, z_reion_mean):
    """
    Convert the reionization redshift fluctuation to the reionization redshift in place, z_reion = delta_z * (1 + z_reion_mean) + z_reion_mean.

    Parameters:
    ----------
    delta_z_x: 3D array
        Reionization redshift fluctuation in real space. Overwritten with the reionization redshift.
    z_reion_mean: float
        Mean reionization redshift.
    """

    _N_x, _N_y, _N_z = delta_z_x.shape

    for i in prange(_N_x):
        for j in range(_N_y):
            for k in range(_N_z):
                delta_z_x[i, j, k] = (
                    delta_z_x[i, j, k] * (1 + z_reion_mean) + z_reion_mean
                )


def _get_inplace_fftw(N_cell, fftw_effort, double_precision, n_threads=None):
    """
    Allocate a single padded buffer and build in-place real-to-complex FFTW plans on it. The buffer has shape (N_cell, N_cell, 2 * (N_cell//2 + 1)), so that the N_cell^3 real field and its N_cell^2 * (N_cell//2 + 1) complex transform share the same memory.

    The plans are built while the buffer is still empty, so FFTW_MEASURE and higher efforts are safe to use (they overwrite the buffer while planning).

    Parameters:
    ----------
    N_cell: int
        Number of cells in each dimension.
    fftw_effort: str
        Effort level for the FFTW planner.
    double_precision: bool
        Whether to use float64/complex128 (True) or float32/complex64 (False).
    n_threads: int, optional
        Number of threads for FFTW. If None, it will use the number of CPU cores available.

    Returns:
    -------
    field_x: 3D array
        Real-space view of the buffer, of shape (N_cell, N_cell, N_cell).
    field_k: 3D complex array
        Fourier-space view of the buffer, of shape (N_cell, N_cell, N_cell//2 + 1).
    fftw_forward: pyfftw.FFTW
        In-place forward plan, field_x -> field_k.
    fftw_inverse: pyfftw.FFTW
        In-place inverse plan, field_k -> field_x, normalized by 1/N_cell^3.
    """

    _real_dtype = np.float64 if double_precision else np.float32
    _complex_dtype = np.complex128 if double_precision else np.complex64
    _n_threads = n_threads or os.cpu_count()

    _buffer = pyfftw.empty_aligned(
        (N_cell, N_cell, 2 * (N_cell // 2 + 1)), dtype=_real_dtype
    )
    field_x = _buffer[..., :N_cell]
    field_k = _buffer.view(_complex_dtype)

    fftw_forward = pyfftw.FFTW(
        field_x,
        field_k,
        axes=(0, 1, 2),
        direction="FFTW_FORWARD",
        flags=(fftw_effort,),
        threads=_n_threads,
    )
    fftw_inverse = pyfftw.FFTW(
        field_k,
        field_x,
        axes=(0, 1, 2),
        direction="FFTW_BACKWARD",
        flags=(fftw_effort,),
        threads=_n_threads,
        normalise_idft=True,
    )

    return field_x, field_k, fftw_forward, fftw_inverse


def _load_density_grid(grid_file, field_x, N_slab):
    """
    Load the density grid into the real-space buffer slab by slab, and return the scale factor and mean density.

    Parameters:
    ----------
    grid_file: str or Path
        Path to the HDF5 density grid file.
    field_x: 3D array
        Real-space view of the buffer to load the density into.
    N_slab: int
        Number of x-planes to load at a time.

    Returns:
    -------
    a: float
        Scale factor of the snapshot.
    rho_mean: float
        Mean density of the grid.
    """

    _N_cell = field_x.shape[0]

    with h5py.File(grid_file, "r") as f:
        a = f["Header"].attrs["Time"]

        _dset = f["DensityDM"]
        if _dset.shape != field_x.shape:
            _msg = f"DensityDM has shape {_dset.shape}, expected {field_x.shape} from grid_params."
            raise ValueError(_msg)

        _rho_sum = 0.0
        for i in tqdm(range(0, _N_cell, N_slab), desc="Loading density"):
            field_x[i : i + N_slab] = _dset[i : i + N_slab]
            _rho_sum += np.sum(field_x[i : i + N_slab], dtype=np.float64)

    return a, _rho_sum / _N_cell**3


def _get_growth_factor_ratio(a, z_reion_params, cosmology_params):
    """
    Linear growth factor ratio D(z_reion_mean) / D(a), which rescales the density field (not its power spectrum) from the snapshot to the mean reionization redshift.
    """

    def _D(a):
        return a * hyp2f1(
            1 / 3,
            1,
            11 / 6,
            -(a**3) * cosmology_params["Omega_L"] / cosmology_params["Omega_m"],
        )

    _a_z_reion_mean = 1 / (1 + z_reion_params["z_reion_mean"])

    return _D(_a_z_reion_mean) / _D(a)


def _get_conv_kernel_ratio_1d(k_1d, cell_size, p=2):
    """
    Separable 1D ratio of the mass assignment kernels, W_pi(k) / W_cell(k), that deconvolves the CIC assignment at the grid cell size and re-smooths to pi cMpc. Returns ones if the cell size is not smaller than pi.

    Parameters:
    ----------
    k_1d: 1D array
        Wavenumbers along one axis.
    cell_size: float
        Grid cell size.
    p: int, optional
        Order of the mass assignment scheme. Default is 2 (CIC).

    Returns:
    -------
    w_1d: 1D array
        Kernel ratio along this axis.
    """

    if cell_size >= np.pi:
        return np.ones_like(k_1d)

    def _single_conv_kernel(dx):
        return np.sinc(k_1d * dx / (2 * np.pi)) ** p

    return _single_conv_kernel(np.pi) / _single_conv_kernel(cell_size)


def density_to_z_reion(
    density_grid_file,
    z_reion_grid_file,
    grid_params=None,
    z_reion_params=None,
    cosmology_params=None,
    fftw_effort="FFTW_ESTIMATE",
    double_precision=True,
    N_slab=64,
):
    """
    Convert a density grid to a reionization redshift grid, working in place on a single padded buffer. Peak memory is about one N_cell^3 array (8 bytes per cell in double precision, 4 in single) plus one slab of N_slab x-planes.

    Parameters:
    ----------
    density_grid_file: str or Path
        Path to the HDF5 density grid file.
    z_reion_grid_file: str or Path
        Path to the output HDF5 reionization redshift grid file.
    grid_params: dict, optional
        Grid parameters: box_size, N_cell, cell_size.
    z_reion_params: dict, optional
        Reionization parameters: b0, k0, alpha, beta, z_reion_mean.
    cosmology_params: dict, optional
        Cosmology parameters: Omega_m, Omega_L.
    fftw_effort: str, optional
        Effort level for the FFTW planner. Default is "FFTW_ESTIMATE".
    double_precision: bool, optional
        Whether to work in float64 (True) or float32 (False). Default is True.
    N_slab: int, optional
        Number of x-planes read or written at a time. Default is 64.
    """

    _grid_params = {
        "box_size": 500,
        "N_cell": 320,
        "cell_size": 500 / 320,
        **(grid_params or {}),
    }

    _z_reion_params = {
        "b0": 0.704,
        "k0": 0.0789,
        "alpha": 0.419,
        "beta": 2,
        "z_reion_mean": 7.2,
        **(z_reion_params or {}),
    }

    _cosmology_params = {
        "Omega_m": 0.3089,
        "Omega_L": 0.6911,
        **(cosmology_params or {}),
    }

    _N_cell = _grid_params["N_cell"]
    _cell_size = _grid_params["cell_size"]

    # Allocate the buffer and build the FFTW plans (before loading any data)
    _field_x, _field_k, _fftw_forward, _fftw_inverse = _get_inplace_fftw(
        _N_cell, fftw_effort=fftw_effort, double_precision=double_precision
    )

    # Load the density grid and convert it to overdensity in place
    _a, _rho_mean = _load_density_grid(density_grid_file, _field_x, N_slab)
    _get_overdensity(_field_x, _rho_mean)

    # Get the density grid in Fourier space
    _fftw_forward()

    # Apply growth rescaling, convolution kernels and bias in one pass
    _k_xy_1d = np.fft.fftfreq(_N_cell, d=_cell_size) * 2 * np.pi
    _k_z_1d = np.fft.rfftfreq(_N_cell, d=_cell_size) * 2 * np.pi

    _apply_fourier_filter(
        _field_k,
        _k_xy_1d,
        _k_z_1d,
        _get_conv_kernel_ratio_1d(_k_xy_1d, _cell_size),
        _get_conv_kernel_ratio_1d(_k_z_1d, _cell_size),
        _get_growth_factor_ratio(_a, _z_reion_params, _cosmology_params),
        _z_reion_params["b0"],
        _z_reion_params["k0"],
        _z_reion_params["alpha"],
        _z_reion_params["beta"],
    )

    # Get the reionization redshift in real space
    _fftw_inverse()
    _get_z_reion(_field_x, _z_reion_params["z_reion_mean"])

    with h5py.File(z_reion_grid_file, "w") as f:
        _header = f.create_group("Header")

        _header.attrs.create("BoxSize", _grid_params["box_size"], dtype=np.float64)
        _header.attrs.create("NumCells", _N_cell, dtype=np.int32)
        _header.attrs.create("CellSize", _cell_size, dtype=np.float64)
        for _key, _value in _z_reion_params.items():
            _header.attrs[_key] = _value

        _dset = f.create_dataset(
            "ReionizationRedshift",
            shape=_field_x.shape,
            dtype=_field_x.dtype,
        )
        for i in tqdm(range(0, _N_cell, N_slab), desc="Saving z_reion"):
            _dset[i : i + N_slab] = _field_x[i : i + N_slab]
