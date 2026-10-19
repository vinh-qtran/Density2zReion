import os

import h5py
import numpy as np
import pyfftw
from numba import njit, prange
from tqdm import tqdm


@njit
def _get_bias_factor(k_mag, b0, k0, alpha, beta):
    """
    Compute the bias factor for a given k magnitude, b(k) = b0 / (1 + (k / k0)^beta)^(alpha / beta).

    Parameters:
    ----------
    k_mag: float
        Magnitude of the wavevector.
    b0: float
        Bias factor at k=0.
    k0: float
        Scale parameter for the bias factor, in cMpc^-1.
    alpha: float
        Power-law index for the bias factor.
    beta: float
        Smoothing parameter for the bias factor.

    Returns:
    -------
    b_k: float
        Bias factor corresponding to the input k magnitude.
    """

    return b0 / (1 + (k_mag / k0) ** beta) ** (alpha / beta)


@njit(parallel=True)
def _scale_field(field_x, scale, shift):
    """
    Apply field = field * scale + shift in place.

    Parameters:
    ----------
    field_x: 3D array
        Field in real space. Overwritten.
    scale: float
        Multiplicative factor.
    shift: float
        Additive shift.
    """

    _N_x, _N_y, _N_z = field_x.shape

    for i in prange(_N_x):
        for j in range(_N_y):
            for k in range(_N_z):
                field_x[i, j, k] = field_x[i, j, k] * scale + shift


@njit(parallel=True)
def _apply_fourier_filter(
    field_k, k_xy_1d, k_z_1d, w_xy_1d, w_z_1d, scaler, use_bias, b0, k0, alpha, beta
):
    """
    Multiply the field in Fourier space in place by
    F(k) = scaler * W(kx) * W(ky) * W(kz) [* b(|k|)], computed on the fly.

    Parameters:
    ----------
    field_k: 3D complex array
        Field in Fourier space, of shape (N_cell, N_cell, N_cell//2 + 1). Overwritten.
    k_xy_1d: 1D array
        Wavenumbers along the x and y axes.
    k_z_1d: 1D array
        Wavenumbers along the z axis (half-space).
    w_xy_1d: 1D array
        Separable kernel along the x and y axes.
    w_z_1d: 1D array
        Separable kernel along the z axis.
    scaler: float
        Constant factor applied to every mode.
    use_bias: bool
        Whether to multiply by the bias factor b(|k|).
    b0, k0, alpha, beta: float
        Bias factor parameters, see _get_bias_factor. Ignored if use_bias is False.
    """

    _N_x, _N_y, _N_z = field_k.shape

    for i in prange(_N_x):
        for j in range(_N_y):
            for k in range(_N_z):
                _filter = scaler * w_xy_1d[i] * w_xy_1d[j] * w_z_1d[k]

                if use_bias:
                    _k_mag = np.sqrt(k_xy_1d[i] ** 2 + k_xy_1d[j] ** 2 + k_z_1d[k] ** 2)
                    _filter *= _get_bias_factor(_k_mag, b0, k0, alpha, beta)

                field_k[i, j, k] *= _filter


@njit(parallel=True)
def _bin_power_spectrum(
    field_k, k_xy_1d, k_z_1d, w_xy_1d, w_z_1d, N_cell, k_bins, log_k_min, dlog_k
):
    """
    Accumulate |field_k|^2 / W(k)^2 into logarithmic k bins, without building a k_mag array or sorting. Each half-space mode is weighted by the number of full-space modes it represents (1 on the kz=0 and Nyquist planes, 2 elsewhere), so the counts are full-space mode counts.

    Each x-plane accumulates into its own row, so the parallel loop has no write conflicts.

    Parameters:
    ----------
    field_k: 3D complex array
        Field in Fourier space, of shape (N_cell, N_cell, N_cell//2 + 1).
    k_xy_1d, k_z_1d: 1D array
        Wavenumbers along the x/y axes and the z axis.
    w_xy_1d, w_z_1d: 1D array
        Separable mass assignment kernel to deconvolve (ones for none).
    N_cell: int
        Number of cells in each dimension.
    k_bins: 1D array
        Logarithmically spaced bin edges. Bins are left-inclusive and the last bin is closed, as in np.histogram.
    log_k_min: float
        Natural log of k_bins[0], used for the initial bin guess.
    dlog_k: float
        Bin width in natural log k, used for the initial bin guess.

    Returns:
    -------
    sum_P: 1D array
        Weighted sum of |field_k|^2 / W(k)^2 in each bin.
    counts: 1D array
        Number of full-space modes in each bin.
    """

    _N_x, _N_y, _N_z = field_k.shape
    n_bins = k_bins.size - 1

    _sum_P = np.zeros((_N_x, n_bins))
    _counts = np.zeros((_N_x, n_bins))

    for i in prange(_N_x):
        for j in range(_N_y):
            for k in range(_N_z):
                _k_mag = np.sqrt(k_xy_1d[i] ** 2 + k_xy_1d[j] ** 2 + k_z_1d[k] ** 2)
                if _k_mag < k_bins[0] or _k_mag > k_bins[n_bins]:
                    continue

                # Guess the bin from log k, then correct against the exact edges
                _bin = int((np.log(_k_mag) - log_k_min) / dlog_k)
                _bin = min(max(_bin, 0), n_bins - 1)
                while _bin > 0 and _k_mag < k_bins[_bin]:
                    _bin -= 1
                while _bin < n_bins - 1 and _k_mag >= k_bins[_bin + 1]:
                    _bin += 1

                _weight = (
                    1.0 if (k == 0 or (N_cell % 2 == 0 and k == _N_z - 1)) else 2.0
                )
                _w = w_xy_1d[i] * w_xy_1d[j] * w_z_1d[k]
                _val = field_k[i, j, k]

                _sum_P[i, _bin] += _weight * (_val.real**2 + _val.imag**2) / _w**2
                _counts[i, _bin] += _weight

    return _sum_P.sum(axis=0), _counts.sum(axis=0)


def get_conv_kernel_1d(k_1d, dx, p=2):
    """
    Single-dimension mass assignment kernel, sinc(k dx / 2)^p. The 3D kernel is the product over the three axes.

    Parameters:
    ----------
    k_1d: 1D array
        Wavenumbers along one axis.
    dx: float
        Grid spacing (or smoothing scale) in real space.
    p: int, optional
        Order of the mass assignment scheme. Default is 2 (Cloud-in-Cell).

    Returns:
    -------
    w_1d: 1D array
        Kernel along this axis.
    """

    return np.sinc(k_1d * dx / (2 * np.pi)) ** p


class FourierField:
    """
    A 3D real field and its Fourier transform sharing one padded buffer, with in-place real-to-complex FFTW plans. Peak memory is about one N_cell^3 array (8 bytes per cell in double precision, 4 in single), plus one slab of N_slab x-planes during file I/O.

    The buffer holds either the real-space field (field_x) or its Fourier transform (field_k), never both. forward() and inverse() switch between them; each method checks it is called in the right space.
    """

    def __init__(
        self,
        box_size,
        N_cell,
        double_precision=True,
        n_threads=None,
        fftw_effort="FFTW_ESTIMATE",
    ):
        """
        Allocate the buffer and build the FFTW plans.

        Parameters:
        ----------
        box_size: float
            Size of the box in real space, in cMpc.
        N_cell: int
            Number of cells in each dimension.
        double_precision: bool, optional
            Whether to use float64/complex128 (True) or float32/complex64 (False). Default is True.
        n_threads: int, optional
            Number of threads for FFTW. If None, it will use the number of CPU cores available.
        fftw_effort: str, optional
            Effort level for the FFTW planner. The plans are built before any data is loaded, so "FFTW_MEASURE" and higher are safe, but they run many trial transforms. Default is "FFTW_ESTIMATE".
        """

        self._get_basic_params(box_size, N_cell)
        self._build_buffer_and_plans(
            double_precision, n_threads or os.cpu_count(), fftw_effort
        )

        self._in_fourier_space = False

    def _get_basic_params(self, box_size, N_cell):
        """
        Get the grid spacing, the k range, and the 1D wavenumbers along each axis.

        Parameters:
        ----------
        box_size: float
            Size of the box in real space.
        N_cell: int
            Number of cells in each dimension.
        """

        self.box_size = box_size
        self.N_cell = N_cell
        self.cell_size = box_size / N_cell

        self.k_min = 2 * np.pi / box_size
        self.k_max = np.pi / self.cell_size

        self._k_xy_1d = np.fft.fftfreq(N_cell, d=self.cell_size) * 2 * np.pi
        self._k_z_1d = np.fft.rfftfreq(N_cell, d=self.cell_size) * 2 * np.pi

    def _build_buffer_and_plans(self, double_precision, n_threads, fftw_effort):
        """
        Allocate a buffer of shape (N_cell, N_cell, 2 * (N_cell//2 + 1)), so that the N_cell^3 real field and its N_cell^2 * (N_cell//2 + 1) complex transform share the same memory, and build the in-place plans on it.

        Parameters:
        ----------
        double_precision: bool
            Whether to use float64/complex128 or float32/complex64.
        n_threads: int
            Number of threads for FFTW.
        fftw_effort: str
            Effort level for the FFTW planner.
        """

        _real_dtype = np.float64 if double_precision else np.float32
        _complex_dtype = np.complex128 if double_precision else np.complex64
        _N = self.N_cell

        _buffer = pyfftw.empty_aligned((_N, _N, 2 * (_N // 2 + 1)), dtype=_real_dtype)
        self.field_x = _buffer[..., :_N]
        self.field_k = _buffer.view(_complex_dtype)

        self._fftw_forward = pyfftw.FFTW(
            self.field_x,
            self.field_k,
            axes=(0, 1, 2),
            direction="FFTW_FORWARD",
            flags=(fftw_effort,),
            threads=n_threads,
        )
        self._fftw_inverse = pyfftw.FFTW(
            self.field_k,
            self.field_x,
            axes=(0, 1, 2),
            direction="FFTW_BACKWARD",
            flags=(fftw_effort,),
            threads=n_threads,
            normalise_idft=True,
        )

    def _check_space(self, fourier_space, action):
        if self._in_fourier_space != fourier_space:
            _expected = "Fourier" if fourier_space else "real"
            _msg = f"Cannot {action}: the field must be in {_expected} space."
            raise ValueError(_msg)

    def load_field(self, grid_file, dataset_name, N_slab=64):
        """
        Load a real-space field from an HDF5 file into the buffer, slab by slab.

        Parameters:
        ----------
        grid_file: str or Path
            Path to the HDF5 file.
        dataset_name: str
            Name of the dataset, e.g. "DensityDM".
        N_slab: int, optional
            Number of x-planes to read at a time. Default is 64.

        Returns:
        -------
        field_mean: float
            Mean of the loaded field, accumulated in double precision.
        """

        with h5py.File(grid_file, "r") as f:
            _dset = f[dataset_name]
            if _dset.shape != self.field_x.shape:
                _msg = f"{dataset_name} has shape {_dset.shape}, expected {self.field_x.shape}."
                raise ValueError(_msg)

            _field_sum = 0.0
            for i in tqdm(
                range(0, self.N_cell, N_slab), desc=f"Loading {dataset_name}"
            ):
                self.field_x[i : i + N_slab] = _dset[i : i + N_slab]
                _field_sum += np.sum(self.field_x[i : i + N_slab], dtype=np.float64)

        self._in_fourier_space = False

        return _field_sum / self.N_cell**3

    def save_field(self, grid_file, dataset_name, header=None, N_slab=64):
        """
        Save the real-space field to an HDF5 file, slab by slab.

        Parameters:
        ----------
        grid_file: str or Path
            Path to the output HDF5 file. Overwritten if it exists.
        dataset_name: str
            Name of the dataset, e.g. "ReionizationRedshift".
        header: dict, optional
            Attributes written to the "Header" group.
        N_slab: int, optional
            Number of x-planes to write at a time. Default is 64.
        """

        self._check_space(False, "save the field")

        with h5py.File(grid_file, "w") as f:
            _header = f.create_group("Header")
            for _key, _value in (header or {}).items():
                _header.attrs[_key] = _value

            _dset = f.create_dataset(
                dataset_name, shape=self.field_x.shape, dtype=self.field_x.dtype
            )
            for i in tqdm(range(0, self.N_cell, N_slab), desc=f"Saving {dataset_name}"):
                _dset[i : i + N_slab] = self.field_x[i : i + N_slab]

    def scale_field(self, scale, shift=0.0):
        """
        Apply field = field * scale + shift in real space, in place. For example, scale_field(1 / rho_mean, -1) turns a density into an overdensity.

        Parameters:
        ----------
        scale: float
            Multiplicative factor.
        shift: float, optional
            Additive shift. Default is 0.
        """

        self._check_space(False, "scale the field")
        _scale_field(self.field_x, scale, shift)

    def combine_with_file(self, grid_file, dataset_name, func, N_slab=64):
        """
        Combine the real-space field in place with another field stored in an HDF5 file, reading the other field slab by slab, so that only one N_cell^3 array is ever in memory: field[slab] = func(field[slab], other[slab]).

        For example, the 21cm brightness temperature field x_HI (1 + delta) at redshift z, with the cell neutral if it reionizes later (z_reion < z), from a smoothed overdensity in the buffer:

            field.combine_with_file(
                "z_reion.hdf5", "ReionizationRedshift",
                lambda delta, z_reion: (z_reion < z) * (1 + delta),
            )

        Parameters:
        ----------
        grid_file: str or Path
            Path to the HDF5 file with the other field.
        dataset_name: str
            Name of the other field's dataset. Must have shape (N_cell, N_cell, N_cell).
        func: callable
            func(field_slab, other_slab) returning the new field slab. Both slabs have shape (n, N_cell, N_cell) with n <= N_slab.
        N_slab: int, optional
            Number of x-planes to read at a time. Default is 64.
        """

        self._check_space(False, "combine the field")

        with h5py.File(grid_file, "r") as f:
            _dset = f[dataset_name]
            if _dset.shape != self.field_x.shape:
                _msg = f"{dataset_name} has shape {_dset.shape}, expected {self.field_x.shape}."
                raise ValueError(_msg)

            for i in tqdm(
                range(0, self.N_cell, N_slab), desc=f"Combining {dataset_name}"
            ):
                self.field_x[i : i + N_slab] = func(
                    self.field_x[i : i + N_slab], _dset[i : i + N_slab]
                )

    def forward(self):
        """
        Forward Fourier transform in place, field_x -> field_k (unnormalized, as numpy's rfftn).
        """

        self._check_space(False, "run the forward transform")
        self._fftw_forward()
        self._in_fourier_space = True

    def inverse(self):
        """
        Inverse Fourier transform in place, field_k -> field_x (normalized by 1 / N_cell^3, as numpy's irfftn).
        """

        self._check_space(True, "run the inverse transform")
        self._fftw_inverse()
        self._in_fourier_space = False

    def get_conv_kernel_ratio(self, smoothing_scale, p=2):
        """
        Separable 1D kernels W_smooth(k) / W_cell(k) that deconvolve the mass assignment at the grid cell size and re-smooth at smoothing_scale. Returns ones if smoothing_scale is not larger than the cell size.

        Parameters:
        ----------
        smoothing_scale: float
            Smoothing scale in real space, in cMpc.
        p: int, optional
            Order of the mass assignment scheme. Default is 2 (Cloud-in-Cell).

        Returns:
        -------
        w_xy_1d: 1D array
            Kernel ratio along the x and y axes.
        w_z_1d: 1D array
            Kernel ratio along the z axis.
        """

        if self.cell_size >= smoothing_scale:
            return np.ones_like(self._k_xy_1d), np.ones_like(self._k_z_1d)

        def _ratio(k_1d):
            return get_conv_kernel_1d(k_1d, smoothing_scale, p) / get_conv_kernel_1d(
                k_1d, self.cell_size, p
            )

        return _ratio(self._k_xy_1d), _ratio(self._k_z_1d)

    def apply_filter(self, scaler=1.0, smoothing_scale=np.pi, p=2, bias_params=None):
        """
        Multiply the field in Fourier space in place by scaler * W_smooth(k) / W_cell(k) [* b(|k|)], in a single pass over the modes.

        Without bias_params this only deconvolves the mass assignment and re-smooths, e.g. for the density field entering the 21cm brightness temperature. With bias_params it also applies the reionization bias, e.g. for the reionization redshift field.

        Parameters:
        ----------
        scaler: float, optional
            Constant factor, e.g. a growth factor ratio. Default is 1.
        smoothing_scale: float, optional
            If larger than the cell size, deconvolve the mass assignment and re-smooth at this scale. Default is pi cMpc.
        p: int, optional
            Order of the mass assignment scheme. Default is 2 (Cloud-in-Cell).
        bias_params: dict, optional
            If given, also multiply by b(|k|) with keys b0, k0, alpha, beta. Default is None (no bias).
        """

        self._check_space(True, "apply a Fourier filter")

        _w_xy_1d, _w_z_1d = self.get_conv_kernel_ratio(smoothing_scale, p=p)
        _bias_params = bias_params or {"b0": 1.0, "k0": 1.0, "alpha": 0.0, "beta": 1.0}

        _apply_fourier_filter(
            self.field_k,
            self._k_xy_1d,
            self._k_z_1d,
            _w_xy_1d,
            _w_z_1d,
            scaler,
            bias_params is not None,
            _bias_params["b0"],
            _bias_params["k0"],
            _bias_params["alpha"],
            _bias_params["beta"],
        )

    def get_power_spectrum(self, n_k_bins=201, deconvolve_p=None):
        """
        Calculate the power spectrum of the field, P(k) = <|field_k|^2> L^3 / N^6, in logarithmic bins between k_min and k_max. The field must be in Fourier space and is not modified.

        Parameters:
        ----------
        n_k_bins: int, optional
            Number of bin edges; there are n_k_bins - 1 bins. Default is 201.
        deconvolve_p: int, optional
            If given, divide |field_k|^2 by the mass assignment kernel of this order at the cell size squared (2 for Cloud-in-Cell). Default is None.

        Returns:
        -------
        k_bin_centers: 1D array
            Centers of the wavenumber bins.
        P_k: 1D array
            Power spectrum values in each bin.
        P_k_err: 1D array
            Gaussian uncertainties, sqrt(2 / N_modes) * P_k, with N_modes the number of full-space modes in the bin.
        """

        self._check_space(True, "compute the power spectrum")

        _k_bins = np.logspace(np.log10(self.k_min), np.log10(self.k_max), n_k_bins)
        k_bin_centers = 0.5 * (_k_bins[:-1] + _k_bins[1:])

        if deconvolve_p is None:
            _w_xy_1d, _w_z_1d = np.ones_like(self._k_xy_1d), np.ones_like(self._k_z_1d)
        else:
            _w_xy_1d = get_conv_kernel_1d(self._k_xy_1d, self.cell_size, deconvolve_p)
            _w_z_1d = get_conv_kernel_1d(self._k_z_1d, self.cell_size, deconvolve_p)

        _sum_P, _counts = _bin_power_spectrum(
            self.field_k,
            self._k_xy_1d,
            self._k_z_1d,
            _w_xy_1d,
            _w_z_1d,
            self.N_cell,
            _k_bins,
            np.log(_k_bins[0]),
            np.log(_k_bins[-1] / _k_bins[0]) / (n_k_bins - 1),
        )

        _mask = _counts > 0
        P_k = np.zeros_like(k_bin_centers)
        P_k_err = np.zeros_like(k_bin_centers)

        P_k[_mask] = _sum_P[_mask] / _counts[_mask] * self.box_size**3 / self.N_cell**6
        P_k_err[_mask] = np.sqrt(2 / _counts[_mask]) * P_k[_mask]

        return k_bin_centers, P_k, P_k_err
