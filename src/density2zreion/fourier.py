import os

import numpy as np
import pyfftw


class FFTWPlan:
    def __init__(self, N_cell, n_threads=None, fftw_effort="FFTW_MEASURE"):
        """
        Initialize the FFTW plan with the number of cells, number of threads, and effort level for optimization.

        Parameters:
        ----------
        N_cell: int
            Number of cells in each dimension for the FFT grid.
        n_threads: int, optional
            Number of threads to use for the FFTW library. If None, it will use the number of CPU cores available. The default value is None.
        fftw_effort: str, optional
            Effort level for the FFTW library, which determines the amount of time spent optimizing the FFT plan. The default value is "FFTW_MEASURE". The available options are "FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", and "FFTW_EXHAUSTIVE", with increasing levels of optimization and time spent on planning.
        """

        self.N_cell = N_cell
        self.n_threads = n_threads or os.cpu_count()
        self.fftw_effort = fftw_effort

    def build_forward_plan(self):
        """
        Build the FFTW plan for the forward Fourier transform. The input array will be a real-valued 3D array of shape (N_cell, N_cell, N_cell), and the output array will be a complex-valued 3D array of shape (N_cell, N_cell, N_cell//2 + 1) due to the use of the rfftn function for real-to-complex transforms.

        Returns:
        -------
        fftw_forward: pyfftw.FFTW object
            FFTW plan for the forward Fourier transform.
        """

        _in = pyfftw.empty_aligned(
            (self.N_cell, self.N_cell, self.N_cell), dtype="float64"
        )
        _out = pyfftw.empty_aligned(
            (self.N_cell, self.N_cell, self.N_cell // 2 + 1), dtype="complex128"
        )

        return pyfftw.FFTW(
            _in,
            _out,
            axes=(0, 1, 2),
            direction="FFTW_FORWARD",
            flags=(self.fftw_effort,),
            threads=self.n_threads,
        )

    def build_inverse_plan(self):
        """
        Build the FFTW plan for the inverse Fourier transform. The input array will be a complex-valued 3D array of shape (N_cell, N_cell, N_cell//2 + 1), and the output array will be a real-valued 3D array of shape (N_cell, N_cell, N_cell). The inverse transform will be normalized by 1/N to match the normalization convention used by numpy's FFT functions.

        Returns:
        -------
        fftw_inverse: pyfftw.FFTW object
            FFTW plan for the inverse Fourier transform, with normalization to match numpy's convention.
        """

        _in_inv = pyfftw.empty_aligned(
            (self.N_cell, self.N_cell, self.N_cell // 2 + 1), dtype="complex128"
        )
        _out_inv = pyfftw.empty_aligned(
            (self.N_cell, self.N_cell, self.N_cell), dtype="float64"
        )

        return pyfftw.FFTW(
            _in_inv,
            _out_inv,
            axes=(0, 1, 2),
            direction="FFTW_BACKWARD",
            flags=(self.fftw_effort,),
            threads=self.n_threads,
            normalise_idft=True,
        )


class FourierTransform:
    """
    Class for performing Fourier transform and calculating power spectrum from a 3D field.
    """

    def __init__(self, box_size, N_cell, fftw_forward, fftw_inverse=None):
        """
        Initialize the Fourier transform with the field, number of cells, and box size.

        Parameters:
        ----------
        box_size: float
            Size of the box in real space.
        N_cell: int
            Number of cells in each dimension.
        fftw_forward: pyfftw.FFTW
            FFTW plan for the forward Fourier transform.
        fftw_inverse: pyfftw.FFTW, optional
            FFTW plan for the inverse Fourier transform. The default value is None, with which no inverse transform will be performed.
        """

        self._get_basic_params(box_size, N_cell)

        self._fftw_forward = fftw_forward
        self._fftw_inverse = fftw_inverse

    def _get_basic_params(self, box_size, N_cell):
        """
        Get the basic parameters for the Fourier transform, including the grid spacing, minimum wavenumber, and maximum wavenumber.

        Parameters:
        ----------
        N_cell: int
            Number of cells in each dimension.
        box_size: float
            Size of the box in real space.

        Attributes:
        -------
        dx: float
            Grid spacing in real space.
        k_min: float
            Minimum wavenumber corresponding to the fundamental mode of the box.
        k_max: float
            Maximum wavenumber corresponding to the Nyquist frequency of the grid.
        """

        self._box_size = box_size
        self._N_cell = N_cell

        self.dx = box_size / N_cell
        self.k_min = 2 * np.pi / box_size
        self.k_max = np.pi / self.dx

    def get_fourier_transform(self, delta_x):
        """
        Get the Fourier transform of the input field and calculate the magnitude of the wavevector for each Fourier mode.

        Parameters:
        ----------
        delta_x: 3D array
            3D field in real space.

        Returns:
        -------
        k_mag: 3D array
            Magnitude of the wavevector corresponding to each Fourier mode.
        delta_k: 3D array
            Fourier transform of the input field.
        """

        self._fftw_forward.input_array[:] = delta_x
        self._fftw_forward()

        delta_k = self._fftw_forward.output_array.copy()

        _k_1d = np.fft.fftfreq(self._N_cell, d=self._dx) * 2 * np.pi
        _kz_1d = np.fft.rfftfreq(self._N_cell, d=self._dx) * 2 * np.pi

        _kx = _k_1d[:, None, None]
        _ky = _k_1d[None, :, None]
        _kz = _kz_1d[None, None, :]

        k_mag = np.sqrt(_kx**2 + _ky**2 + _kz**2)

        return k_mag, delta_k

    def inv_fourier_transform(self, delta_k):
        """
        Perform the inverse Fourier transform to get back the field in real space from its Fourier transform.

        Parameters:
        ----------
        delta_k: 3D array
            Fourier transform of the field.

        Returns:
        -------
        delta_x: 3D array
            Field in real space obtained from the inverse Fourier transform of the input Fourier transform.
        """

        if self._fftw_inverse is None:
            raise ValueError(  # noqa: TRY003
                "Inverse FFTW plan is not available. Please build the inverse plan first."  # noqa: EM101
            )

        self._fftw_inverse.input_array[:] = delta_k
        self._fftw_inverse()

        return self._fftw_inverse.output_array.copy()

    def get_conv_kernels(self, p=2, dx=None):
        """
        Get the convolution kernels for the Fourier transform, which includes both the shot noise correction and the deconvolution kernel for the mass assignment scheme.

        Parameters:
        ----------
        p: int, optional
            Order of the mass assignment scheme. The deconvolution kernel will be calculated as the sinc function raised to the power of p for each dimension. The default value is 2, corresponding to the Cloud-in-Cell (CIC) mass assignment scheme.
        dx: float, optional
            Grid spacing in real space. If None, it will be taken from the instance's dx attribute. The default value is None.

        Returns:
        -------
        conv_kernel: 3D array
            Convolution kernel in Fourier space, calculated as the product of the single-dimension convolution kernels for each dimension, which is the sinc function raised to the power of p.
        """
        _dx = dx or self._dx

        _k_1d = np.fft.fftfreq(self._N_cell, d=self._dx) * 2 * np.pi
        _kz_1d = np.fft.rfftfreq(self._N_cell, d=self._dx) * 2 * np.pi

        _kx = _k_1d[:, None, None]
        _ky = _k_1d[None, :, None]
        _kz = _kz_1d[None, None, :]

        def _single_conv_kernel(k_1d):
            return np.sinc(k_1d * _dx / (2 * np.pi)) ** p

        return (
            _single_conv_kernel(_kx)
            * _single_conv_kernel(_ky)
            * _single_conv_kernel(_kz)
        ).reshape(self._N_cell, self._N_cell, self._N_cell // 2 + 1)

    def get_power_spectrum(
        self,
        k_mag,
        delta_k,
        n_k_bins=201,
    ):
        """
        Calculate the power spectrum from the Fourier transform of the field. If a second Fourier transform is provided, calculate the cross-power spectrum between the two fields.

        Parameters:
        ----------
        k_mag: 3D array
            Magnitude of the wavevector corresponding to each Fourier mode, obtained from the Fourier transform of the field.
        delta_k: 3D array
            Fourier transform of the field.
        shot_noise_kernel: 3D array, optional
            Shot noise correction kernel in Fourier space. If provided, it will be used to correct the power spectrum for shot noise. If not provided, no shot noise correction will be applied.
        n_k_bins: int, optional
            Number of bins to use for the power spectrum. The wavenumber range will be divided into this many logarithmically spaced bins.

        Returns:
        -------
        k_bin_centers: 1D array
            Centers of the wavenumber bins used for the power spectrum.
        P_k: 1D array
            Power spectrum values corresponding to the wavenumber bins.
        P_k_err: 1D array
            Uncertainties in the power spectrum values, calculated as the standard error of the mean for each bin.
        """

        _k_mag = k_mag.ravel()
        _cov_k = np.real(delta_k * np.conj(delta_k)).ravel()

        _k_bins = np.logspace(np.log10(self._k_min), np.log10(self._k_max), n_k_bins)
        k_bin_centers = 0.5 * (_k_bins[:-1] + _k_bins[1:])

        _sum_cov_k, _ = np.histogram(_k_mag, bins=_k_bins, weights=_cov_k)
        _counts, _ = np.histogram(_k_mag, bins=_k_bins)

        _mask = _counts > 0
        P_k = np.zeros_like(k_bin_centers)
        P_k_err = np.zeros_like(k_bin_centers)

        P_k[_mask] = _sum_cov_k[_mask] / _counts[_mask]
        P_k *= (self._L_box**3) / (self._N_cell**6)

        P_k_err[_mask] = np.sqrt(2 / _counts[_mask]) * P_k[_mask]

        return k_bin_centers, P_k, P_k_err
