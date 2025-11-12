"""
Polyspectra calculators for power spectrum and bispectrum.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union


def get_power_spectrum(
    image: Union[np.ndarray, torch.Tensor],
    k_range: Optional[Union[np.ndarray, torch.Tensor]] = None,
    bins: Optional[int] = None,
    bin_type: str = 'log',
    device: str = 'gpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the power spectrum of given images.

    Parameters
    ----------
    image : array-like
        Input images
    k_range : array-like, optional
        Frequency bins
    bins : int, optional
        Number of bins
    bin_type : str, default='log'
        Type of binning ('log' or 'linear')
    device : str, default='gpu'
        Device for computation

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Power spectrum and k_range
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    if k_range is not None and isinstance(k_range, np.ndarray):
        k_range = torch.from_numpy(k_range)

    if not torch.cuda.is_available():
        device = 'cpu'
    if device == 'gpu' and torch.cuda.is_available():
        image = image.cuda()

    M, N = image.shape[-2:]
    modulus = torch.fft.fftn(image, dim=(-2, -1), norm='ortho').abs()

    # Rearrange to have zero frequency at center
    modulus = torch.cat(
        (torch.cat((modulus[..., M//2:, N//2:], modulus[..., :M//2, N//2:]), -2),
         torch.cat((modulus[..., M//2:, :N//2], modulus[..., :M//2, :N//2]), -2)),
        -1
    )

    # Create frequency grid
    X = torch.arange(M)[:, None]
    Y = torch.arange(N)[None, :]
    Xgrid = X + Y * 0
    Ygrid = X * 0 + Y
    k = ((Xgrid - M/2)**2 + (Ygrid - N/2)**2)**0.5

    # Create k_range if not provided
    if k_range is None:
        if bin_type == 'linear':
            k_range = torch.linspace(1, M/2*1.415, bins+1)
        else:  # log binning
            k_range = torch.logspace(0, np.log10(M/2*1.415), bins+1)

    power_spectrum = torch.zeros(len(image), len(k_range)-1, dtype=image.dtype)

    if device == 'gpu' and torch.cuda.is_available():
        k = k.cuda()
        k_range = k_range.cuda()
        power_spectrum = power_spectrum.cuda()

    # Bin the power spectrum
    for i in range(len(k_range)-1):
        select = (k > k_range[i]) & (k <= k_range[i+1])
        if select.sum() > 0:
            power_spectrum[:, i] = ((modulus**2 * select[None, ...]).sum((-2, -1)) / select.sum()).log()

    return power_spectrum, k_range


class Bispectrum_Calculator:
    """
    Calculator for bispectrum computation.
    """

    def __init__(
        self,
        M: int,
        N: int,
        bins: int = 10,
        bin_type: str = 'log',
        device: str = 'gpu'
    ):
        """
        Initialize Bispectrum_Calculator.

        Parameters
        ----------
        M : int
            Height of images
        N : int
            Width of images
        bins : int, default=10
            Number of frequency bins
        bin_type : str, default='log'
            Type of binning ('log' or 'linear')
        device : str, default='gpu'
            Device for computation
        """
        self.M = M
        self.N = N
        self.bins = bins
        self.bin_type = bin_type
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Create frequency grid
        kx = torch.arange(M).reshape(M, 1)
        ky = torch.arange(N).reshape(1, N)

        # Center frequencies
        kx = torch.where(kx <= M//2, kx, kx - M)
        ky = torch.where(ky <= N//2, ky, ky - N)

        kx = 2 * np.pi * kx / M
        ky = 2 * np.pi * ky / N

        self.k = torch.sqrt(kx**2 + ky**2)
        self.theta = torch.atan2(ky, kx)

        # Create k bins
        if bin_type == 'linear':
            self.k_bins = torch.linspace(0, self.k.max(), bins+1)
        else:
            k_min = self.k[self.k > 0].min()
            self.k_bins = torch.logspace(np.log10(k_min), np.log10(self.k.max()), bins+1)

        if self.device == 'gpu' and torch.cuda.is_available():
            self.k = self.k.cuda()
            self.theta = self.theta.cuda()
            self.k_bins = self.k_bins.cuda()

    def forward(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute bispectrum of images.

        Parameters
        ----------
        image : array-like
            Input images

        Returns
        -------
        torch.Tensor
            Bispectrum coefficients
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if self.device == 'gpu' and torch.cuda.is_available():
            image = image.cuda()

        N_image = image.shape[0]

        # Compute Fourier transform
        image_f = torch.fft.fftn(image, dim=(-2, -1), norm='ortho')

        # Initialize bispectrum storage
        bispectrum = []

        # Compute bispectrum for different k1, k2 combinations
        for i in range(len(self.k_bins)-1):
            k1_mask = (self.k >= self.k_bins[i]) & (self.k < self.k_bins[i+1])

            for j in range(i, len(self.k_bins)-1):
                k2_mask = (self.k >= self.k_bins[j]) & (self.k < self.k_bins[j+1])

                # Simplified bispectrum calculation
                # B(k1, k2, theta) = <F(k1) * F(k2) * F*(k1+k2)>
                # Here we compute a simplified version

                if k1_mask.sum() > 0 and k2_mask.sum() > 0:
                    # Extract values at k1 and k2
                    F_k1 = image_f * k1_mask[None, :, :]
                    F_k2 = image_f * k2_mask[None, :, :]

                    # Compute bispectrum (simplified - real part only)
                    B = (F_k1 * F_k2).sum((-2, -1)) / (k1_mask.sum() * k2_mask.sum())**0.5
                    bispectrum.append(B.real)

        if bispectrum:
            bispectrum = torch.stack(bispectrum, dim=1)
        else:
            bispectrum = torch.zeros((N_image, 0), dtype=image.dtype)
            if self.device == 'gpu' and torch.cuda.is_available():
                bispectrum = bispectrum.cuda()

        return bispectrum


class Trispectrum_Calculator:
    """
    Calculator for trispectrum computation (placeholder for compatibility).
    """

    def __init__(self, M: int, N: int, device: str = 'gpu'):
        """
        Initialize Trispectrum_Calculator.

        Parameters
        ----------
        M : int
            Height of images
        N : int
            Width of images
        device : str, default='gpu'
            Device for computation
        """
        self.M = M
        self.N = N
        self.device = device if torch.cuda.is_available() else 'cpu'

    def forward(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute trispectrum (placeholder implementation).

        Parameters
        ----------
        image : array-like
            Input images

        Returns
        -------
        torch.Tensor
            Empty tensor (placeholder)
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        N_image = image.shape[0]

        # Placeholder - return empty tensor
        result = torch.zeros((N_image, 0), dtype=image.dtype)
        if self.device == 'gpu' and torch.cuda.is_available():
            result = result.cuda()

        return result