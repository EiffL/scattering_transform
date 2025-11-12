"""
Utility functions for the SC2 package.
"""

import numpy as np
import torch
from typing import Union, Dict, List, Tuple, Any


def to_numpy(data: Any) -> Any:
    """
    Converts a tensor/array/list to numpy array.
    Recurses over dictionaries and tuples. Values are left as-is.

    Parameters
    ----------
    data : Any
        Input data to convert

    Returns
    -------
    Any
        Converted data (numpy array if applicable)
    """
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    elif isinstance(data, tuple):
        return tuple(to_numpy(v) for v in data)
    return data


def cut_high_k_off(psi_f: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """
    Cut high frequency components in Fourier space.

    Parameters
    ----------
    psi_f : torch.Tensor
        Input in Fourier space
    dx : int
        x-direction cutoff
    dy : int
        y-direction cutoff

    Returns
    -------
    torch.Tensor
        Filtered Fourier coefficients
    """
    M, N = psi_f.shape[-2:]
    psi_f_small = torch.zeros((M//dx, N//dy), dtype=psi_f.dtype)

    if psi_f.is_cuda:
        psi_f_small = psi_f_small.cuda()

    # Copy low frequency components
    psi_f_small[:M//dx//2, :N//dy//2] = psi_f[:M//dx//2, :N//dy//2]
    psi_f_small[-M//dx//2:, :N//dy//2] = psi_f[-M//dx//2:, :N//dy//2]
    psi_f_small[:M//dx//2, -N//dy//2:] = psi_f[:M//dx//2, -N//dy//2:]
    psi_f_small[-M//dx//2:, -N//dy//2:] = psi_f[-M//dx//2:, -N//dy//2:]

    return psi_f_small


def get_edge_masks(M: int, N: int, J: int) -> Dict[int, torch.Tensor]:
    """
    Get edge masks for different scales.

    Parameters
    ----------
    M : int
        Height of the image
    N : int
        Width of the image
    J : int
        Number of scales

    Returns
    -------
    Dict[int, torch.Tensor]
        Dictionary of edge masks for each scale
    """
    edge_masks = {}

    for j in range(J):
        mask = torch.ones((M, N))
        edge_width = 2**j

        # Set edges to zero
        mask[:edge_width, :] = 0
        mask[-edge_width:, :] = 0
        mask[:, :edge_width] = 0
        mask[:, -edge_width:] = 0

        edge_masks[j] = mask

    return edge_masks


def compute_modulus(u: torch.Tensor) -> torch.Tensor:
    """
    Compute the modulus of a complex tensor.

    Parameters
    ----------
    u : torch.Tensor
        Complex input tensor

    Returns
    -------
    torch.Tensor
        Modulus of the input
    """
    if torch.is_complex(u):
        return u.abs()
    else:
        return u.abs()


def pad_signal(x: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
    """
    Pad a signal with reflection padding.

    Parameters
    ----------
    x : torch.Tensor
        Input signal
    pad_left : int
        Left padding size
    pad_right : int
        Right padding size

    Returns
    -------
    torch.Tensor
        Padded signal
    """
    if len(x.shape) == 2:
        return torch.nn.functional.pad(x, (pad_left, pad_right, pad_left, pad_right), mode='reflect')
    elif len(x.shape) == 3:
        return torch.nn.functional.pad(x, (pad_left, pad_right, pad_left, pad_right), mode='reflect')
    elif len(x.shape) == 4:
        return torch.nn.functional.pad(x, (pad_left, pad_right, pad_left, pad_right), mode='reflect')
    else:
        raise ValueError(f"Unsupported tensor dimension: {len(x.shape)}")


def subsample(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Subsample a signal by factor k.

    Parameters
    ----------
    x : torch.Tensor
        Input signal
    k : int
        Subsampling factor

    Returns
    -------
    torch.Tensor
        Subsampled signal
    """
    if k == 1:
        return x
    return x[..., ::k, ::k]