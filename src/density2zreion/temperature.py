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


def field_to_smoothed_field(
    grid_file,
    smoothed_grid_file,
    field_name,
    smoothed_field_name,
    deviation=True,
    grid_params=None,
    smoothing_scale=np.pi,
    fftw_effort="FFTW_ESTIMATE",
    double_precision=True,
    N_slab=64,
):
    """
    Smooth a gridded field, deconvolving the CIC assignment and re-smoothing at smoothing_scale (if the cell size is smaller than it), and save it, either as the deviation from the mean, field / mean - 1 (e.g. the overdensity entering the 21cm brightness temperature), or as the field itself (e.g. a temperature). Works in place on a single padded buffer.

    The filter leaves the k = 0 mode unchanged, so the smoothed field and the smoothed deviation carry the same information: field = mean * (1 + deviation).

    Parameters:
    ----------
    grid_file: str or Path
        Path to the HDF5 grid file.
    smoothed_grid_file: str or Path
        Path to the output HDF5 file. The smoothed field is saved as smoothed_field_name, with the box size, number of cells, cell size, scale factor of the snapshot, mean of the field ("MeanDensity"), smoothing scale and whether it is a deviation ("Deviation") in the header.
    field_name: str
        Name of the field in grid_file to read.
    smoothed_field_name: str
        Name of the field in smoothed_grid_file to write.
    deviation: bool, optional
        Whether to save the smoothed deviation field / mean - 1 (True) or the smoothed field itself (False). Default is True.
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

    # Load the field, and convert it to the deviation from the mean if asked
    _mean = _field.load_field(grid_file, field_name, N_slab=N_slab)
    if deviation:
        _field.scale_field(1 / _mean, -1)

    # Deconvolve CIC and smooth to smoothing_scale
    _field.forward()
    _field.apply_filter(smoothing_scale=smoothing_scale)
    _field.inverse()

    _field.save_field(
        smoothed_grid_file,
        smoothed_field_name,
        header={
            "BoxSize": np.float64(_field.box_size),
            "NumCells": np.int32(_field.N_cell),
            "CellSize": np.float64(_field.cell_size),
            "Time": np.float64(get_grid_time(grid_file)),
            "MeanDensity": np.float64(_mean),
            "SmoothingScale": np.float64(smoothing_scale),
            "Deviation": deviation,
        },
        N_slab=N_slab,
    )


def get_brightness_temperature_amplitude(z, cosmology_params=None):
    """
    Amplitude T0(z) of the 21cm differential brightness temperature, delta T_b = T0(z) x_HI (1 + delta) (1 - T_gamma(z) / T_S), in mK (Furlanetto 2006, MNRAS 371, 867, Eq. 1; derived in Furlanetto, Oh & Briggs 2006, Eqs. 17-18), neglecting redshift-space distortions.

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
    temperature_grid_file=None,
    temperature_field_name="Temperature",
    n_k_bins=201,
    output_file=None,
    fftw_effort="FFTW_ESTIMATE",
    double_precision=True,
    N_slab=64,
):
    """
    Calculate the 21cm power spectrum at each redshift z from the smoothed overdensity and the reionization redshift field, with

        delta T_b(x, z) = T0(z) x_HI(x, z) (1 + delta(x, z)) (1 - T_gamma(z) / T_S(x)),    x_HI = 1 if z_reion(x) < z else 0,

    i.e. a cell is neutral until the universe reaches its reionization redshift. T_gamma(z) = T_CMB0 (1 + z) is the CMB temperature at the redshift of emission. The spin temperature T_S is read from temperature_grid_file, assuming the Lyman-alpha coupling is saturated so that T_S equals the gas temperature; without it, T_S >> T_gamma is assumed and the last factor is 1. With T_S < T_gamma the signal is in absorption (negative). The buffer is allocated once and reused for every redshift; each redshift reads the two grid files once and runs one forward FFT. Peak memory is about one N_cell^3 array.

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
        Cosmology parameters: Omega_m, Omega_L, Omega_b, h, T_CMB0 (in K). Default is the Planck 2015 cosmology used by IllustrisTNG / MTNG, with T_CMB0 = 2.7255 K.
    rescale_density: bool, optional
        Whether to rescale the overdensity from the snapshot to each redshift with the linear growth factor, delta(z) = D(z) / D(z_snap) delta(z_snap), as done for the reionization redshift field. Default is True.
    temperature_grid_file: str or Path, optional
        HDF5 file with the spin (gas) temperature in K on the same grid: the raw temperature, or the output of field_to_smoothed_field_deviation, either the smoothed temperature (deviation=False) or the smoothed deviation T / T_mean - 1 (deviation=True), told apart by its "Deviation" header attribute. The same temperature is used at every redshift. Default is None (T_S >> T_gamma).
    temperature_field_name: str, optional
        Name of the temperature dataset. Default is "Temperature".
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
        "T_CMB0": 2.7255,
        **(cosmology_params or {}),
    }
    _redshifts = np.atleast_1d(np.asarray(redshifts, dtype=np.float64))

    with h5py.File(smoothed_grid_file, "r") as f:
        _box_size = f["Header"].attrs["BoxSize"]
        _N_cell = int(f["Header"].attrs["NumCells"])
        _a_snap = f["Header"].attrs["Time"]

    # T = T_mean (1 + deviation) for a smoothed deviation file, T itself otherwise.
    # Files without the "Deviation" flag but with "MeanDensity" are deviations.
    _T_mean = None
    if temperature_grid_file is not None:
        with h5py.File(temperature_grid_file, "r") as f:
            _attrs = f["Header"].attrs if "Header" in f else {}
            if _attrs.get("Deviation", "MeanDensity" in _attrs):
                _T_mean = _attrs["MeanDensity"]

    _header = {
        "BoxSize": np.float64(_box_size),
        "NumCells": np.int32(_N_cell),
        "RescaleDensity": rescale_density,
        "SpinTemperature": temperature_grid_file is not None,
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
        _T_gamma = _cosmology_params["T_CMB0"] * (1 + _z)
        _sums = {"x_HI": 0, "x_HI_rho": 0.0}

        def _get_spin_temperature_factor(rho, T, T_gamma=_T_gamma):
            _T_S = T if _T_mean is None else _T_mean * (1 + T)
            return rho * (1 - T_gamma / _T_S)

        def _get_neutral_density(rho, z_reion, z=_z, sums=_sums):
            _x_HI = z_reion < z
            _out = _x_HI * rho

            sums["x_HI"] += np.count_nonzero(_x_HI)
            sums["x_HI_rho"] += np.sum(_out, dtype=np.float64)

            return _out

        # (1 + delta) in real space
        _field.load_field(smoothed_grid_file, "OverDensityDM", N_slab=N_slab)
        _field.scale_field(_growth_factor_ratio, 1.0)

        # (1 - T_gamma / T_S), if a temperature file is given
        if temperature_grid_file is not None:
            _field.combine_with_file(
                temperature_grid_file,
                temperature_field_name,
                _get_spin_temperature_factor,
                N_slab=N_slab,
            )

        # x_HI
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
