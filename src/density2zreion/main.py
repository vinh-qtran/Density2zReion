import os

import numpy as np

from density2zreion.fourier import FourierField
from density2zreion.grid import CICDensityGrid
from density2zreion.utils import get_grid_time, get_growth_factor


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
        Grid parameters: box_size, N_cell.
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

    _field = FourierField(
        _grid_params["box_size"],
        _grid_params["N_cell"],
        double_precision=double_precision,
        fftw_effort=fftw_effort,
    )

    # Load the density grid and convert it to overdensity
    _rho_mean = _field.load_field(density_grid_file, "DensityDM", N_slab=N_slab)
    _field.scale_field(1 / _rho_mean, -1)

    # Rescale to the mean reionization redshift, deconvolve CIC, smooth to pi cMpc and apply the bias
    _a = get_grid_time(density_grid_file)
    _a_z_reion_mean = 1 / (1 + _z_reion_params["z_reion_mean"])

    _field.forward()
    _field.apply_filter(
        scaler=get_growth_factor(_a_z_reion_mean, _cosmology_params)
        / get_growth_factor(_a, _cosmology_params),
        smoothing_scale=np.pi,
        bias_params=_z_reion_params,
    )
    _field.inverse()

    # Get the reionization redshift in real space
    _z_reion_mean = _z_reion_params["z_reion_mean"]
    _field.scale_field(1 + _z_reion_mean, _z_reion_mean)

    _field.save_field(
        z_reion_grid_file,
        "ReionizationRedshift",
        header={
            "BoxSize": np.float64(_field.box_size),
            "NumCells": np.int32(_field.N_cell),
            "CellSize": np.float64(_field.cell_size),
            **_z_reion_params,
        },
        N_slab=N_slab,
    )
