import os

import h5py
import numpy as np
from tqdm import tqdm

from density2zreion.fourier import FourierField
from density2zreion.utils import (
    check_file_exists,
    get_grid_time,
    get_growth_factor,
)


def density_to_smoothed_overdensity(
    density_grid_file,
    smoothed_grid_file,
    grid_params=None,
    smoothing_scale=np.pi,
    fftw_effort="FFTW_ESTIMATE",
    double_precision=True,
    N_slab=64,
):
    """
    Convert a density grid to an overdensity grid, deconvolving the CIC assignment and re-smoothing at smoothing_scale (if the cell size is smaller than it), and save it. This is the density field entering the 21cm brightness temperature, on the same smoothing scale as the reionization redshift field. Works in place on a single padded buffer.

    Parameters:
    ----------
    density_grid_file: str or Path
        Path to the HDF5 density grid file.
    smoothed_grid_file: str or Path
        Path to the output HDF5 file. The overdensity is saved as "OverDensityDM", with the box size, number of cells, cell size, scale factor of the snapshot, mean density and smoothing scale in the header.
    grid_params: dict, optional
        Grid parameters: box_size, N_cell.
    smoothing_scale: float, optional
        Smoothing scale in cMpc. Default is pi, matching density_to_z_reion.
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

    _field = FourierField(
        _grid_params["box_size"],
        _grid_params["N_cell"],
        double_precision=double_precision,
        fftw_effort=fftw_effort,
    )

    # Load the density grid and convert it to overdensity
    _rho_mean = _field.load_field(density_grid_file, "DensityDM", N_slab=N_slab)
    _field.scale_field(1 / _rho_mean, -1)

    # Deconvolve CIC and smooth to smoothing_scale
    _field.forward()
    _field.apply_filter(smoothing_scale=smoothing_scale)
    _field.inverse()

    _field.save_field(
        smoothed_grid_file,
        "OverDensityDM",
        header={
            "BoxSize": np.float64(_field.box_size),
            "NumCells": np.int32(_field.N_cell),
            "CellSize": np.float64(_field.cell_size),
            "Time": np.float64(get_grid_time(density_grid_file)),
            "MeanDensity": np.float64(_rho_mean),
            "SmoothingScale": np.float64(smoothing_scale),
        },
        N_slab=N_slab,
    )


def get_brightness_temperature_amplitude(z, cosmology_params=None):
    """
    Amplitude T0(z) of the 21cm differential brightness temperature, delta T_b = T0(z) x_HI (1 + delta), in mK (Furlanetto, Oh & Briggs 2006), assuming the spin temperature is much larger than the CMB temperature and neglecting redshift-space distortions.

    Parameters:
    ----------
    z: float
        Redshift.
    cosmology_params: dict, optional
        Cosmology parameters: Omega_m, Omega_b, h. Default is the Planck 2015 cosmology used by IllustrisTNG / MTNG.

    Returns:
    -------
    T0: float
        Brightness temperature amplitude in mK.
    """

    _cosmology_params = {
        "Omega_m": 0.3089,
        "Omega_b": 0.0486,
        "h": 0.6774,
        **(cosmology_params or {}),
    }
    _h2 = _cosmology_params["h"] ** 2

    return (
        27
        * (_cosmology_params["Omega_b"] * _h2 / 0.023)
        * np.sqrt(0.15 / (_cosmology_params["Omega_m"] * _h2) * (1 + z) / 10)
    )


def _read_21cm_results(output_file):
    """
    Read previously calculated redshifts from the output file, if it exists.

    Returns:
    -------
    k: 1D array or None
        Centers of the k bins.
    results: dict
        Results per redshift (rounded to 6 decimals): P_k, P_k_err, x_HI_mean, T_b_mean.
    """

    if not check_file_exists(output_file):
        return None, {}

    with h5py.File(output_file, "r") as f:
        k = f["WaveNumber"][:]
        results = {
            round(float(z), 6): {
                "P_k": P_k,
                "P_k_err": P_k_err,
                "x_HI_mean": x_HI_mean,
                "T_b_mean": T_b_mean,
            }
            for z, P_k, P_k_err, x_HI_mean, T_b_mean in zip(
                f["Redshift"][:],
                f["PowerSpectrum"][:],
                f["PowerSpectrumError"][:],
                f["NeutralFraction"][:],
                f["MeanBrightnessTemperature"][:],
                strict=True,
            )
        }

    return k, results


def _save_21cm_results(output_file, k, results, header):
    """
    Save the results of all calculated redshifts, sorted by redshift. The file is written to a temporary name and then renamed, so an interrupted write never corrupts the existing results.
    """

    _redshifts = sorted(results)
    _tmp_file = f"{output_file}.tmp"

    with h5py.File(_tmp_file, "w") as f:
        _header = f.create_group("Header")
        for _key, _value in header.items():
            _header.attrs[_key] = _value

        _P_k = np.array([results[z]["P_k"] for z in _redshifts])

        f.create_dataset("Redshift", data=_redshifts)
        f.create_dataset("WaveNumber", data=k)
        f.create_dataset("PowerSpectrum", data=_P_k)
        f.create_dataset(
            "PowerSpectrumError", data=[results[z]["P_k_err"] for z in _redshifts]
        )
        f.create_dataset(
            "DimensionlessPowerSpectrum", data=_P_k * k**3 / (2 * np.pi**2)
        )
        f.create_dataset(
            "NeutralFraction", data=[results[z]["x_HI_mean"] for z in _redshifts]
        )
        f.create_dataset(
            "MeanBrightnessTemperature",
            data=[results[z]["T_b_mean"] for z in _redshifts],
        )

    os.replace(_tmp_file, output_file)  # noqa: PTH105


def get_21cm_power_spectrum(
    smoothed_grid_file,
    z_reion_grid_file,
    redshifts,
    cosmology_params=None,
    rescale_density=True,
    n_k_bins=201,
    output_file=None,
    fftw_effort="FFTW_ESTIMATE",
    double_precision=True,
    N_slab=64,
):
    """
    Calculate the 21cm power spectrum at each redshift z from the smoothed overdensity and the reionization redshift field, with

        delta T_b(x, z) = T0(z) x_HI(x, z) (1 + delta(x, z)),    x_HI = 1 if z_reion(x) < z else 0,

    i.e. a cell is neutral until the universe reaches its reionization redshift. The buffer is allocated once and reused for every redshift; each redshift reads the two grid files once and runs one forward FFT. Peak memory is about one N_cell^3 array.

    If output_file is given and exists, redshifts already in it are read back instead of recalculated. The file is updated after every new redshift, so an interrupted run keeps the redshifts it finished. The settings are not checked against the file, so use a new output_file when changing any parameter.

    Parameters:
    ----------
    smoothed_grid_file: str or Path
        Path to the smoothed overdensity file from density_to_smoothed_overdensity.
    z_reion_grid_file: str or Path
        Path to the reionization redshift grid file from density_to_z_reion. Must have the same number of cells.
    redshifts: float or 1D array
        Redshift(s) at which to calculate the power spectrum.
    cosmology_params: dict, optional
        Cosmology parameters: Omega_m, Omega_L, Omega_b, h. Default is the Planck 2015 cosmology used by IllustrisTNG / MTNG.
    rescale_density: bool, optional
        Whether to rescale the overdensity from the snapshot to each redshift with the linear growth factor, delta(z) = D(z) / D(z_snap) delta(z_snap), as done for the reionization redshift field. Default is True.
    n_k_bins: int, optional
        Number of logarithmic k bin edges; there are n_k_bins - 1 bins. Default is 201.
    output_file: str or Path, optional
        If given, the HDF5 file the results are saved to and read back from.
    fftw_effort: str, optional
        Effort level for the FFTW planner. Default is "FFTW_ESTIMATE".
    double_precision: bool, optional
        Whether to work in float64 (True) or float32 (False). Default is True.
    N_slab: int, optional
        Number of x-planes read at a time. Default is 64.

    Returns:
    -------
    results: dict
        For the requested redshifts, in the order given:
        redshift: 1D array, (n_z,)
            Redshifts.
        k: 1D array, (n_k_bins - 1,)
            Centers of the k bins, in cMpc^-1.
        P_k: 2D array, (n_z, n_k_bins - 1)
            Power spectrum of delta T_b, in mK^2 cMpc^3.
        P_k_err: 2D array, (n_z, n_k_bins - 1)
            Gaussian uncertainty of P_k, in mK^2 cMpc^3.
        Delta2_k: 2D array, (n_z, n_k_bins - 1)
            Dimensionless power spectrum k^3 P_k / (2 pi^2), in mK^2.
        x_HI_mean: 1D array, (n_z,)
            Volume-weighted neutral fraction.
        T_b_mean: 1D array, (n_z,)
            Mean brightness temperature, in mK.
    """

    _cosmology_params = {
        "Omega_m": 0.3089,
        "Omega_L": 0.6911,
        "Omega_b": 0.0486,
        "h": 0.6774,
        **(cosmology_params or {}),
    }
    _redshifts = np.atleast_1d(np.asarray(redshifts, dtype=np.float64))

    with h5py.File(smoothed_grid_file, "r") as f:
        _box_size = f["Header"].attrs["BoxSize"]
        _N_cell = int(f["Header"].attrs["NumCells"])
        _a_snap = f["Header"].attrs["Time"]

    _header = {
        "BoxSize": np.float64(_box_size),
        "NumCells": np.int32(_N_cell),
        "RescaleDensity": rescale_density,
        **_cosmology_params,
    }

    # Previously calculated redshifts
    k, _results = _read_21cm_results(output_file)

    _field = None

    for _z in tqdm(_redshifts, desc="Redshifts"):
        if round(float(_z), 6) in _results:
            continue

        # Allocate the buffer only once there is something to calculate
        if _field is None:
            _field = FourierField(
                _box_size,
                _N_cell,
                double_precision=double_precision,
                fftw_effort=fftw_effort,
            )

        _growth_factor_ratio = (
            get_growth_factor(1 / (1 + _z), _cosmology_params)
            / get_growth_factor(_a_snap, _cosmology_params)
            if rescale_density
            else 1.0
        )
        _sums = {"x_HI": 0, "x_HI_rho": 0.0}

        def _get_neutral_density(
            delta, z_reion, z=_z, f=_growth_factor_ratio, sums=_sums
        ):
            _x_HI = z_reion < z
            _out = _x_HI * (1 + f * delta)

            sums["x_HI"] += np.count_nonzero(_x_HI)
            sums["x_HI_rho"] += np.sum(_out, dtype=np.float64)

            return _out

        # x_HI (1 + delta) in real space
        _field.load_field(smoothed_grid_file, "OverDensityDM", N_slab=N_slab)
        _field.combine_with_file(
            z_reion_grid_file,
            "ReionizationRedshift",
            _get_neutral_density,
            N_slab=N_slab,
        )

        # Power spectrum, scaled by T0^2 to mK^2 cMpc^3
        _field.forward()
        k, _P, _P_err = _field.get_power_spectrum(n_k_bins=n_k_bins)

        _T0 = get_brightness_temperature_amplitude(_z, _cosmology_params)
        _results[round(float(_z), 6)] = {
            "P_k": _P * _T0**2,
            "P_k_err": _P_err * _T0**2,
            "x_HI_mean": _sums["x_HI"] / _N_cell**3,
            "T_b_mean": _T0 * _sums["x_HI_rho"] / _N_cell**3,
        }

        # Save after every redshift, so an interrupted run keeps what it finished
        if output_file is not None:
            _save_21cm_results(output_file, k, _results, _header)

    # Requested redshifts, in the order given
    _rows = [_results[round(float(_z), 6)] for _z in _redshifts]

    results = {
        "redshift": _redshifts,
        "k": k,
        **{_key: np.array([_row[_key] for _row in _rows]) for _key in _rows[0]},
    }
    results["Delta2_k"] = results["P_k"] * k**3 / (2 * np.pi**2)

    return results
