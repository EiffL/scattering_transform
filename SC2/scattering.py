"""
Core scattering transform implementation.
"""

import numpy as np
import torch
from typing import Dict, Optional, Union, Tuple, Any

from .filters import FiltersSet
from .utils import cut_high_k_off, get_edge_masks, to_numpy


class Scattering2d:
    """
    2D Scattering Transform implementation.
    """

    def __init__(
        self,
        M: int,
        N: int,
        J: int,
        L: int = 4,
        device: str = 'gpu',
        wavelets: str = 'morlet',
        filters_set: Optional[Dict] = None,
        weight: Optional[Union[np.ndarray, torch.Tensor]] = None,
        precision: str = 'single',
        ref: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ref_a: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ref_b: Optional[Union[np.ndarray, torch.Tensor]] = None,
        l_oversampling: int = 1,
        frequency_factor: float = 1
    ):
        """
        Initialize Scattering2d transform.

        Parameters
        ----------
        M : int
            Number of pixels along x direction
        N : int
            Number of pixels along y direction
        J : int
            Number of dyadic scales
        L : int, default=4
            Number of orientations
        device : str, default='gpu'
            Device to compute on ('gpu' or 'cpu')
        wavelets : str, default='morlet'
            Type of wavelets ('morlet', 'BS', 'gau', 'shannon')
        filters_set : dict, optional
            Pre-computed filter set
        weight : array-like, optional
            Weight array
        precision : str, default='single'
            Numerical precision
        ref : array-like, optional
            Reference image for normalization
        ref_a, ref_b : array-like, optional
            Reference images for 2-field covariance
        l_oversampling : int, default=1
            Angular oversampling factor
        frequency_factor : float, default=1
            Frequency scaling factor
        """
        if not torch.cuda.is_available():
            device = 'cpu'

        self.device = device
        self.M = M
        self.N = N
        self.J = J
        self.L = L
        self.frequency_factor = frequency_factor
        self.l_oversampling = l_oversampling
        self.wavelets = wavelets
        self.precision = precision

        # Generate or use provided filters
        if filters_set is None:
            filters_generator = FiltersSet(M=M, N=N, J=J, L=L)
            filters_set = filters_generator.generate_wavelets(
                wavelets=wavelets,
                precision=precision,
                l_oversampling=l_oversampling,
                frequency_factor=frequency_factor
            )

        # Store filters
        self.filters_set = filters_set['psi']
        self.phi = filters_set['phi']

        # Move filters to device if needed
        if device == 'gpu' and torch.cuda.is_available():
            self.filters_set = self.filters_set.cuda()
            self.phi = self.phi.cuda()

        # Handle weight
        self.weight = None
        self.weight_f = None
        self.weight_downsample_list = []
        if weight is not None:
            if isinstance(weight, np.ndarray):
                weight = torch.from_numpy(weight)
            self.weight = weight / weight.mean()
            self.weight_f = torch.fft.fftn(self.weight, dim=(-2, -1))

            for j in range(J):
                dx, dy = self.get_dxdy(j)
                weight_downsample = torch.fft.ifftn(
                    cut_high_k_off(self.weight_f, dx, dy),
                    dim=(-2, -1)
                ).real
                if device == 'gpu' and torch.cuda.is_available():
                    weight_downsample = weight_downsample.cuda()
                self.weight_downsample_list.append(
                    weight_downsample / weight_downsample.mean()
                )

        # Get edge masks
        self.edge_masks = get_edge_masks(M, N, J)

        # Store references if provided
        self.ref_scattering_cov = None
        self.ref_scattering_cov_2fields = None
        if ref is not None:
            self.add_ref(ref)
        if ref_a is not None and ref_b is not None:
            self.add_ref_ab(ref_a, ref_b)

    def get_dxdy(self, j: int) -> Tuple[int, int]:
        """Get downsampling factors for scale j."""
        dx = max(1, 2**(j - 1))
        dy = max(1, 2**(j - 1))
        return dx, dy

    def add_ref(self, ref: Union[np.ndarray, torch.Tensor]):
        """Add reference for covariance normalization."""
        self.ref_scattering_cov = self.scattering_cov(ref, if_large_batch=True)

    def add_ref_ab(self, ref_a: Union[np.ndarray, torch.Tensor], ref_b: Union[np.ndarray, torch.Tensor]):
        """Add references for 2-field covariance normalization."""
        self.ref_scattering_cov_2fields = self.scattering_cov_2fields(
            ref_a, ref_b, if_large_batch=True
        )

    def add_synthesis_P00(
        self,
        P00: Optional[torch.Tensor] = None,
        s_cov: Optional[torch.Tensor] = None,
        if_iso: bool = True
    ):
        """Add synthesis P00 for normalization."""
        self.ref_scattering_cov = {}
        if P00 is not None:
            self.ref_scattering_cov['P00'] = P00
        else:
            if if_iso:
                self.ref_scattering_cov['P00'] = torch.exp(s_cov[:, 1:1+self.J].reshape((-1, self.J, 1)))
            else:
                self.ref_scattering_cov['P00'] = torch.exp(s_cov[:, 1:1+self.J*self.L].reshape((-1, self.J, self.L)))

        if self.device == 'gpu' and torch.cuda.is_available():
            self.ref_scattering_cov['P00'] = self.ref_scattering_cov['P00'].cuda()

    def add_synthesis_P11(self, s_cov: torch.Tensor, if_iso: bool, C11_criteria: str = 'j2>=j1'):
        """Add synthesis P11 for normalization."""
        J = self.J
        L = self.L
        self.ref_scattering_cov = {}

        if if_iso:
            j1, j2, l2 = torch.meshgrid(torch.arange(J), torch.arange(J), torch.arange(L), indexing='ij')
            select_j12_iso = (j1 <= j2) * eval(C11_criteria)
            self.ref_scattering_cov['P11'] = torch.zeros(s_cov.shape[0], J, J, L, L)
            for i in range(select_j12_iso.sum()):
                self.ref_scattering_cov['P11'][:, j1[select_j12_iso][i], j2[select_j12_iso][i], :, l2[select_j12_iso][i]] = \
                    torch.exp(s_cov[:, 1+2*J+i, None])
        else:
            j1, j2, l1, l2 = torch.meshgrid(torch.arange(J), torch.arange(J), torch.arange(L), torch.arange(L), indexing='ij')
            select_j12 = (j1 <= j2) * eval(C11_criteria)
            self.ref_scattering_cov['P11'] = torch.zeros(s_cov.shape[0], J, J, L, L)
            for i in range(select_j12.sum()):
                self.ref_scattering_cov['P11'][
                    :, j1[select_j12][i], j2[select_j12][i], l1[select_j12][i], l2[select_j12][i]
                ] = torch.exp(s_cov[:, 1+2*J*L+i])

        if self.device == 'gpu' and torch.cuda.is_available():
            self.ref_scattering_cov['P11'] = self.ref_scattering_cov['P11'].cuda()

    def scattering_coef(
        self,
        data: Union[np.ndarray, torch.Tensor],
        if_large_batch: bool = False,
        j1j2_criteria: str = 'j2>=j1',
        algorithm: str = 'fast',
        pseudo_coef: float = 1,
        remove_edge: bool = False,
        flatten: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute scattering coefficients.

        Parameters
        ----------
        data : array-like
            Input images
        if_large_batch : bool, default=False
            Whether to process in chunks for large batches
        j1j2_criteria : str, default='j2>=j1'
            Criteria for j1, j2 selection
        algorithm : str, default='fast'
            Algorithm to use
        pseudo_coef : float, default=1
            Pseudo coefficient
        remove_edge : bool, default=False
            Whether to remove edge effects
        flatten : bool, default=False
            Whether to flatten the output

        Returns
        -------
        dict
            Dictionary containing scattering coefficients
        """
        # Convert input to tensor
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if self.precision == 'double':
            data = data.type(torch.DoubleTensor)
        else:
            data = data.type(torch.FloatTensor)

        if self.device == 'gpu' and torch.cuda.is_available():
            data = data.cuda()

        N_image = data.shape[0]
        M, N, J, L = self.M, self.N, self.J, self.L

        # Compute wavelet transforms
        U1_list = []
        for j in range(J):
            U1_j = []
            for l in range(L):
                psi_j_l = self.filters_set[j, l]
                U1_j_l = torch.fft.ifftn(
                    torch.fft.fftn(data, dim=(-2, -1)) * psi_j_l[None, :, :],
                    dim=(-2, -1)
                ).abs()
                U1_j.append(U1_j_l)
            U1_list.append(torch.stack(U1_j, dim=1))

        # Stack all U1
        U1_all = torch.stack(U1_list, dim=1)  # Shape: (N_image, J, L, M, N)

        # Compute mean
        mean = data.mean((-2, -1))

        # Compute P00 (zeroth order)
        P00 = U1_all.mean((-2, -1))  # Shape: (N_image, J, L)

        # Compute S1 (first order)
        S1 = torch.zeros((N_image, J, L), dtype=data.dtype)
        if self.device == 'gpu' and torch.cuda.is_available():
            S1 = S1.cuda()

        for j1 in range(J):
            dx, dy = self.get_dxdy(j1)
            for l1 in range(L):
                # Apply low-pass filter
                U1_j1_l1_smooth = torch.fft.ifftn(
                    torch.fft.fftn(U1_all[:, j1, l1], dim=(-2, -1)) * self.phi[None, :, :],
                    dim=(-2, -1)
                ).abs()
                # Downsample
                S1[:, j1, l1] = U1_j1_l1_smooth[:, ::dx, ::dy].mean((-2, -1))

        # Prepare output
        result = {
            'mean': mean,
            'P00': P00,
            'S1': S1,
        }

        if flatten:
            # Flatten for synthesis
            for_synthesis = torch.cat([
                mean[:, None],
                P00.reshape(N_image, -1),
                S1.reshape(N_image, -1)
            ], dim=1)

            for_synthesis_iso = torch.cat([
                mean[:, None],
                P00.mean(-1),
                S1.mean(-1)
            ], dim=1)

            result['for_synthesis'] = for_synthesis
            result['for_synthesis_iso'] = for_synthesis_iso

        return result

    def scattering_cov(
        self,
        data: Union[np.ndarray, torch.Tensor],
        if_large_batch: bool = False,
        C11_criteria: Optional[str] = None,
        use_ref: bool = False,
        normalization: str = 'P00',
        remove_edge: bool = False,
        pseudo_coef: float = 1,
        get_variance: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute scattering covariance.

        Parameters
        ----------
        data : array-like
            Input images
        if_large_batch : bool, default=False
            Whether to process in chunks
        C11_criteria : str, optional
            Criteria for C11 computation
        use_ref : bool, default=False
            Whether to use reference for normalization
        normalization : str, default='P00'
            Normalization type
        remove_edge : bool, default=False
            Whether to remove edge effects
        pseudo_coef : float, default=1
            Pseudo coefficient
        get_variance : bool, default=False
            Whether to compute variance

        Returns
        -------
        dict
            Dictionary containing scattering covariance coefficients
        """
        if C11_criteria is None:
            C11_criteria = 'j2>=j1'

        # Convert input to tensor
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        if self.precision == 'double':
            data = data.type(torch.DoubleTensor)
        else:
            data = data.type(torch.FloatTensor)

        if self.device == 'gpu' and torch.cuda.is_available():
            data = data.cuda()

        N_image = data.shape[0]
        M, N, J, L = self.M, self.N, self.J, self.L

        # Get reference if using
        if use_ref:
            if normalization == 'P00':
                ref_P00 = self.ref_scattering_cov['P00']
            else:
                ref_P11 = self.ref_scattering_cov['P11']

        # Compute wavelet transforms (U1)
        U1_list = []
        for j in range(J):
            U1_j = []
            for l in range(L):
                psi_j_l = self.filters_set[j, l]
                U1_j_l = torch.fft.ifftn(
                    torch.fft.fftn(data, dim=(-2, -1)) * psi_j_l[None, :, :],
                    dim=(-2, -1)
                ).abs()
                U1_j.append(U1_j_l)
            U1_list.append(torch.stack(U1_j, dim=1))

        U1_all = torch.stack(U1_list, dim=1)  # Shape: (N_image, J, L, M, N)

        # Mean
        mean = data.mean((-2, -1))

        # P00
        P00 = U1_all.mean((-2, -1))

        # S1
        S1 = torch.zeros((N_image, J, L), dtype=data.dtype)
        if self.device == 'gpu' and torch.cuda.is_available():
            S1 = S1.cuda()

        for j1 in range(J):
            dx, dy = self.get_dxdy(j1)
            for l1 in range(L):
                U1_j1_l1_smooth = torch.fft.ifftn(
                    torch.fft.fftn(U1_all[:, j1, l1], dim=(-2, -1)) * self.phi[None, :, :],
                    dim=(-2, -1)
                ).abs()
                S1[:, j1, l1] = U1_j1_l1_smooth[:, ::dx, ::dy].mean((-2, -1))

        # C01 and C11 (simplified version for minimal implementation)
        C01_list = []
        C11_list = []
        index_C01 = []
        index_C11 = []

        for j1 in range(J):
            for l1 in range(L):
                # C01: correlation between U1 and low-pass
                U0 = torch.fft.ifftn(
                    torch.fft.fftn(data, dim=(-2, -1)) * self.phi[None, :, :],
                    dim=(-2, -1)
                ).abs()

                C01_real = (U0 * U1_all[:, j1, l1]).mean((-2, -1))
                C01_imag = torch.zeros_like(C01_real)

                if use_ref and normalization == 'P00':
                    C01_real = C01_real / (ref_P00[:, j1, l1] + pseudo_coef)

                C01_list.append(torch.stack([C01_real, C01_imag], dim=-1))
                index_C01.append([j1, l1])

                # C11: correlation between pairs of U1
                for j2 in range(j1, J):
                    if not eval(C11_criteria.replace('j1', str(j1)).replace('j2', str(j2))):
                        continue
                    for l2 in range(L):
                        C11_real = (U1_all[:, j1, l1] * U1_all[:, j2, l2]).mean((-2, -1))
                        C11_imag = torch.zeros_like(C11_real)

                        if use_ref and normalization == 'P00':
                            normalization_factor = (ref_P00[:, j1, l1] * ref_P00[:, j2, l2]).sqrt() + pseudo_coef
                            C11_real = C11_real / normalization_factor

                        C11_list.append(torch.stack([C11_real, C11_imag], dim=-1))
                        index_C11.append([j1, l1, j2, l2])

        # Stack coefficients
        if C01_list:
            C01 = torch.stack(C01_list, dim=1)
        else:
            C01 = torch.zeros((N_image, 0, 2), dtype=data.dtype)
            if self.device == 'gpu' and torch.cuda.is_available():
                C01 = C01.cuda()

        if C11_list:
            C11 = torch.stack(C11_list, dim=1)
        else:
            C11 = torch.zeros((N_image, 0, 2), dtype=data.dtype)
            if self.device == 'gpu' and torch.cuda.is_available():
                C11 = C11.cuda()

        # Prepare flattened versions for synthesis
        for_synthesis = torch.cat([
            mean[:, None],
            P00.reshape(N_image, -1),
            S1.reshape(N_image, -1),
            C01[:, :, 0].reshape(N_image, -1),  # real part
            C01[:, :, 1].reshape(N_image, -1),  # imag part
            C11[:, :, 0].reshape(N_image, -1),  # real part
            C11[:, :, 1].reshape(N_image, -1),  # imag part
        ], dim=1)

        for_synthesis_iso = torch.cat([
            mean[:, None],
            P00.mean(-1),
            S1.mean(-1),
            C01[:, :, 0].mean(-1, keepdim=True) if C01.shape[1] > 0 else torch.zeros((N_image, 0)),
            C01[:, :, 1].mean(-1, keepdim=True) if C01.shape[1] > 0 else torch.zeros((N_image, 0)),
            C11[:, :, 0].mean(-1, keepdim=True) if C11.shape[1] > 0 else torch.zeros((N_image, 0)),
            C11[:, :, 1].mean(-1, keepdim=True) if C11.shape[1] > 0 else torch.zeros((N_image, 0)),
        ], dim=1)

        # Create index arrays
        # Type 0: mean, 1: P00, 2: S1, 3: C01_real, 4: C01_imag, 5: C11_real, 6: C11_imag
        index_for_synthesis = self._create_index_array(J, L, C11_criteria)
        index_for_synthesis_iso = self._create_index_array_iso(J, L, C11_criteria)

        return {
            'mean': mean,
            'P00': P00,
            'S1': S1,
            'C01': C01,
            'C11': C11,
            'for_synthesis': for_synthesis,
            'for_synthesis_iso': for_synthesis_iso,
            'index_for_synthesis': index_for_synthesis,
            'index_for_synthesis_iso': index_for_synthesis_iso,
        }

    def scattering_cov_2fields(
        self,
        data_a: Union[np.ndarray, torch.Tensor],
        data_b: Union[np.ndarray, torch.Tensor],
        if_large_batch: bool = False,
        C11_criteria: Optional[str] = None,
        use_ref: bool = False,
        normalization: str = 'P00',
        remove_edge: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute 2-field scattering covariance (simplified version).
        """
        # Simplified implementation - compute individual covariances
        s_cov_a = self.scattering_cov(
            data_a, if_large_batch, C11_criteria, False, normalization, remove_edge
        )
        s_cov_b = self.scattering_cov(
            data_b, if_large_batch, C11_criteria, False, normalization, remove_edge
        )

        # Combine results (simplified)
        return {
            'for_synthesis': torch.cat([s_cov_a['for_synthesis'], s_cov_b['for_synthesis']], dim=1),
            'for_synthesis_iso': torch.cat([s_cov_a['for_synthesis_iso'], s_cov_b['for_synthesis_iso']], dim=1),
            'index_for_synthesis': s_cov_a['index_for_synthesis'],
            'index_for_synthesis_iso': s_cov_a['index_for_synthesis_iso'],
        }

    def _create_index_array(self, J: int, L: int, C11_criteria: str) -> torch.Tensor:
        """Create index array for synthesis."""
        indices = []

        # Mean (type 0)
        indices.append([0, -1, -1, -1, -1, -1, -1])

        # P00 (type 1)
        for j in range(J):
            for l in range(L):
                indices.append([1, j, -1, -1, l, -1, -1])

        # S1 (type 2)
        for j in range(J):
            for l in range(L):
                indices.append([2, j, -1, -1, l, -1, -1])

        # C01 real (type 3) and imag (type 4)
        for j in range(J):
            for l in range(L):
                indices.append([3, -1, j, -1, -1, l, -1])  # real
                indices.append([4, -1, j, -1, -1, l, -1])  # imag

        # C11 real (type 5) and imag (type 6)
        for j1 in range(J):
            for j2 in range(j1, J):
                if not eval(C11_criteria):
                    continue
                for l1 in range(L):
                    for l2 in range(L):
                        indices.append([5, j1, j2, -1, l1, l2, -1])  # real
                        indices.append([6, j1, j2, -1, l1, l2, -1])  # imag

        return torch.tensor(indices).T

    def _create_index_array_iso(self, J: int, L: int, C11_criteria: str) -> torch.Tensor:
        """Create isotropic index array for synthesis."""
        indices = []

        # Mean (type 0)
        indices.append([0, -1, -1, -1, -1, -1, -1])

        # P00 (type 1) - averaged over angles
        for j in range(J):
            indices.append([1, j, -1, -1, -1, -1, -1])

        # S1 (type 2) - averaged over angles
        for j in range(J):
            indices.append([2, j, -1, -1, -1, -1, -1])

        # C01 real (type 3) and imag (type 4) - simplified
        for j in range(J):
            indices.append([3, -1, j, -1, -1, -1, -1])  # real
            indices.append([4, -1, j, -1, -1, -1, -1])  # imag

        # C11 real (type 5) and imag (type 6) - simplified
        for j1 in range(J):
            for j2 in range(j1, J):
                if not eval(C11_criteria):
                    continue
                for l_diff in range(L):
                    indices.append([5, j1, j2, -1, -1, l_diff, -1])  # real
                    indices.append([6, j1, j2, -1, -1, l_diff, -1])  # imag

        return torch.tensor(indices).T