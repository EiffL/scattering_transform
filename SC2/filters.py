"""
Filter generation for the scattering transform.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple


class FiltersSet:
    """
    Generate filter banks for the scattering transform.
    """

    def __init__(self, M: int, N: int, J: Optional[int] = None, L: int = 4):
        """
        Initialize FiltersSet.

        Parameters
        ----------
        M : int
            Height of the image
        N : int
            Width of the image
        J : int, optional
            Number of scales (default: log2(min(M, N)) - 1)
        L : int, default=4
            Number of orientations
        """
        if J is None:
            J = int(np.log2(min(M, N))) - 1
        self.M = M
        self.N = N
        self.J = J
        self.L = L

    def generate_wavelets(
        self,
        wavelets: str = 'morlet',
        precision: str = 'single',
        l_oversampling: int = 1,
        frequency_factor: float = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Generate wavelet filter bank.

        Parameters
        ----------
        wavelets : str, default='morlet'
            Type of wavelets ('morlet', 'BS', 'gau', 'shannon')
        precision : str, default='single'
            Numerical precision ('single', 'double', 'half')
        l_oversampling : int, default=1
            Angular oversampling factor
        frequency_factor : float, default=1
            Frequency scaling factor

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing 'psi' and 'phi' filters
        """
        # Set dtype based on precision
        if precision == 'double':
            dtype = torch.float64
            dtype_np = np.float64
        elif precision == 'single':
            dtype = torch.float32
            dtype_np = np.float32
        else:  # 'half'
            dtype = torch.float16
            dtype_np = np.float16

        # Initialize wavelet tensor
        psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=dtype)

        for j in range(self.J):
            for l in range(self.L):
                k0 = frequency_factor * 3.0 / 4.0 * np.pi / 2**j
                theta0 = (int(self.L - self.L/2 - 1) - l) * np.pi / self.L

                if wavelets == 'morlet':
                    wavelet_spatial = self.morlet_2d(
                        M=self.M, N=self.N, xi=k0, theta=theta0,
                        sigma=0.8 * 2**j / frequency_factor,
                        slant=4.0 / self.L * l_oversampling,
                    )
                    wavelet_Fourier = np.fft.fft2(wavelet_spatial)
                elif wavelets == 'BS':
                    wavelet_Fourier = self.bump_steerable_2d(
                        M=self.M, N=self.N, k0=k0, theta0=theta0, L=self.L
                    )
                elif wavelets == 'gau':
                    wavelet_Fourier = self.gau_steerable_2d(
                        M=self.M, N=self.N, k0=k0, theta0=theta0, L=self.L
                    )
                elif wavelets == 'shannon':
                    wavelet_Fourier = self.shannon_2d(
                        M=self.M, N=self.N, kmin=k0 / 2**0.5, kmax=k0 * 2**0.5,
                        theta0=theta0, L=self.L
                    )
                else:
                    raise ValueError(f"Unknown wavelet type: {wavelets}")

                wavelet_Fourier[0, 0] = 0
                psi[j, l] = torch.from_numpy(wavelet_Fourier.real.astype(dtype_np))

        # Generate low-pass filter (phi)
        if wavelets == 'morlet':
            phi = torch.from_numpy(
                self.gabor_2d_mycode(
                    self.M, self.N, 0.8 * 2**(self.J-1) / frequency_factor, 0, 0
                ).real.astype(dtype_np)
            ) * (self.M * self.N)**0.5
        elif wavelets in ['BS', 'gau']:
            phi = torch.from_numpy(
                self.gabor_2d_mycode(
                    self.M, self.N,
                    2 * np.pi / (0.702*2**(-0.05)) * 2**(self.J-1) / frequency_factor,
                    0, 0
                ).real.astype(dtype_np)
            ) * (self.M * self.N)**0.5
        elif wavelets == 'shannon':
            phi = torch.from_numpy(
                self.shannon_2d(
                    M=self.M, N=self.N, kmin=-1,
                    kmax=frequency_factor * 0.375 * 2 * np.pi / 2**self.J * 2**0.5,
                    theta0=0, L=0.5
                ).real.astype(dtype_np)
            )
        else:
            raise ValueError(f"Unknown wavelet type: {wavelets}")

        return {'psi': psi, 'phi': phi}

    def morlet_2d(
        self,
        M: int,
        N: int,
        sigma: float,
        theta: float,
        xi: float,
        slant: float = 0.5,
        offset: float = 0,
        fft_shift: bool = False
    ) -> np.ndarray:
        """
        Generate 2D Morlet wavelet.

        Parameters
        ----------
        M : int
            Height
        N : int
            Width
        sigma : float
            Gaussian envelope std
        theta : float
            Orientation
        xi : float
            Central frequency
        slant : float, default=0.5
            Slant parameter
        offset : float, default=0
            Offset parameter
        fft_shift : bool, default=False
            Whether to apply fftshift

        Returns
        -------
        np.ndarray
            2D Morlet wavelet
        """
        x = np.arange(M) - M // 2
        y = np.arange(N) - N // 2
        Y, X = np.meshgrid(y, x)

        # Rotation and slant
        Xr = np.cos(theta) * X + np.sin(theta) * Y
        Yr = -np.sin(theta) * X + np.cos(theta) * Y + slant * Xr + offset

        # Gaussian envelope
        gaussian = np.exp(-(Xr**2 + Yr**2) / (2 * sigma**2))

        # Complex exponential
        complex_exp = np.exp(1j * xi * Xr)

        # Morlet wavelet
        morlet = gaussian * complex_exp

        # Normalization
        morlet = morlet - morlet.mean()
        morlet = morlet / np.abs(morlet).sum()

        if fft_shift:
            morlet = np.fft.fftshift(morlet)

        return morlet

    def gabor_2d_mycode(
        self,
        M: int,
        N: int,
        sigma: float,
        theta: float,
        xi: float,
        slant: float = 0
    ) -> np.ndarray:
        """
        Generate 2D Gabor filter in Fourier domain.

        Parameters
        ----------
        M : int
            Height
        N : int
            Width
        sigma : float
            Gaussian envelope std in spatial domain
        theta : float
            Orientation
        xi : float
            Central frequency
        slant : float, default=0
            Slant parameter

        Returns
        -------
        np.ndarray
            2D Gabor filter in Fourier domain
        """
        kx = np.arange(M).reshape(M, 1)
        ky = np.arange(N).reshape(1, N)

        # Center the frequencies
        kx = np.where(kx <= M//2, kx, kx - M)
        ky = np.where(ky <= N//2, ky, ky - N)

        kx = 2 * np.pi * kx / M
        ky = 2 * np.pi * ky / N

        # Apply rotation
        kx_rot = np.cos(theta) * kx + np.sin(theta) * ky
        ky_rot = -np.sin(theta) * kx + np.cos(theta) * ky

        # Gaussian in Fourier domain
        gabor_f = np.exp(-sigma**2 * ((kx_rot - xi)**2 + ky_rot**2) / 2)

        return gabor_f

    def bump_steerable_2d(
        self,
        M: int,
        N: int,
        k0: float,
        theta0: float,
        L: int
    ) -> np.ndarray:
        """
        Generate bump-steerable wavelet in Fourier domain.

        Parameters
        ----------
        M : int
            Height
        N : int
            Width
        k0 : float
            Central frequency
        theta0 : float
            Central orientation
        L : int
            Number of orientations

        Returns
        -------
        np.ndarray
            Bump-steerable wavelet in Fourier domain
        """
        kx = np.arange(M).reshape(M, 1)
        ky = np.arange(N).reshape(1, N)

        # Center the frequencies
        kx = np.where(kx <= M//2, kx, kx - M)
        ky = np.where(ky <= N//2, ky, ky - N)

        kx = 2 * np.pi * kx / M
        ky = 2 * np.pi * ky / N

        # Convert to polar coordinates
        k = np.sqrt(kx**2 + ky**2)
        theta = np.arctan2(ky, kx)

        # Radial part - bump function
        k_normalized = k / k0
        radial = np.zeros_like(k)
        mask = (k_normalized >= 0.5) & (k_normalized <= 2)
        radial[mask] = np.exp(1 - 1 / (1 - (k_normalized[mask] - 1)**2))

        # Angular part
        angular_diff = np.angle(np.exp(1j * L * (theta - theta0)))
        angular = np.cos(angular_diff / 2) ** (L - 1)

        return radial * angular

    def gau_steerable_2d(
        self,
        M: int,
        N: int,
        k0: float,
        theta0: float,
        L: int
    ) -> np.ndarray:
        """
        Generate Gaussian-steerable wavelet in Fourier domain.

        Parameters
        ----------
        M : int
            Height
        N : int
            Width
        k0 : float
            Central frequency
        theta0 : float
            Central orientation
        L : int
            Number of orientations

        Returns
        -------
        np.ndarray
            Gaussian-steerable wavelet in Fourier domain
        """
        kx = np.arange(M).reshape(M, 1)
        ky = np.arange(N).reshape(1, N)

        # Center the frequencies
        kx = np.where(kx <= M//2, kx, kx - M)
        ky = np.where(ky <= N//2, ky, ky - N)

        kx = 2 * np.pi * kx / M
        ky = 2 * np.pi * ky / N

        # Convert to polar coordinates
        k = np.sqrt(kx**2 + ky**2)
        theta = np.arctan2(ky, kx)

        # Radial part - Gaussian-like
        radial = (2 * k / k0)**2 * np.exp(-k**2 / (2 * (k0/1.4)**2))

        # Angular part
        angular_diff = np.angle(np.exp(1j * L * (theta - theta0)))
        angular = np.cos(angular_diff / 2) ** (L - 1)

        return radial * angular

    def shannon_2d(
        self,
        M: int,
        N: int,
        kmin: float,
        kmax: float,
        theta0: float,
        L: float
    ) -> np.ndarray:
        """
        Generate Shannon wavelet in Fourier domain.

        Parameters
        ----------
        M : int
            Height
        N : int
            Width
        kmin : float
            Minimum frequency
        kmax : float
            Maximum frequency
        theta0 : float
            Central orientation
        L : float
            Angular bandwidth parameter

        Returns
        -------
        np.ndarray
            Shannon wavelet in Fourier domain
        """
        kx = np.arange(M).reshape(M, 1)
        ky = np.arange(N).reshape(1, N)

        # Center the frequencies
        kx = np.where(kx <= M//2, kx, kx - M)
        ky = np.where(ky <= N//2, ky, ky - N)

        kx = 2 * np.pi * kx / M
        ky = 2 * np.pi * ky / N

        # Convert to polar coordinates
        k = np.sqrt(kx**2 + ky**2)
        theta = np.arctan2(ky, kx)

        # Radial part - top hat
        radial = ((k >= kmin) & (k <= kmax)).astype(float)

        # Angular part
        if L > 0:
            angular_diff = np.angle(np.exp(1j * (theta - theta0)))
            angular = (np.abs(angular_diff) <= np.pi / L).astype(float)
        else:
            angular = np.ones_like(theta)

        return radial * angular