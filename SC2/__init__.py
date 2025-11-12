"""
SC2: Minimal Scattering Synthesis Package

A lightweight implementation containing the essential components for scattering synthesis.
This package provides the minimal functionality required to run the synthesis function
from the original scattering transform package.
"""

import numpy as np
import torch
import time
from typing import Optional, Union, Callable, Tuple, Dict, Any

from .filters import FiltersSet
from .scattering import Scattering2d
from .utils import to_numpy


def synthesis(
    estimator_name: str,
    target: Union[np.ndarray, torch.Tensor],
    image_init: Optional[Union[np.ndarray, torch.Tensor]] = None,
    image_ref: Optional[Union[np.ndarray, torch.Tensor]] = None,
    image_b: Optional[Union[np.ndarray, torch.Tensor]] = None,
    J: Optional[int] = None,
    L: int = 4,
    M: Optional[int] = None,
    N: Optional[int] = None,
    l_oversampling: int = 1,
    frequency_factor: float = 1,
    mode: str = 'image',
    optim_algorithm: str = 'LBFGS',
    steps: int = 300,
    learning_rate: float = 0.2,
    device: str = 'gpu',
    wavelets: str = 'morlet',
    seed: Optional[int] = None,
    if_large_batch: bool = False,
    C11_criteria: Optional[str] = None,
    normalization: str = 'P00',
    precision: str = 'single',
    print_each_step: bool = False,
    s_cov_func: Optional[Callable] = None,
    s_cov_func_params: list = [],
    Fourier: bool = False,
    target_full: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ps: bool = False,
    ps_bins: Optional[int] = None,
    ps_bin_type: str = 'log',
    bi: bool = False,
    bispectrum_bins: Optional[int] = None,
    bispectrum_bin_type: str = 'log',
    phi4: bool = False,
    phi4_j: bool = False,
    hist: bool = False,
    hist_factor: float = 1,
    hist_j: bool = False,
    hist_j_factor: float = 1,
    ensemble: bool = False,
    N_ensemble: int = 1,
    reference_P00: Optional[torch.Tensor] = None,
    pseudo_coef: float = 1,
    remove_edge: bool = False
) -> np.ndarray:
    """
    Synthesize images using scattering transform statistics.

    Parameters
    ----------
    estimator_name : str
        The estimator name can be 's_mean', 's_mean_iso', 's_cov', 's_cov_iso', 'alpha_cov'
    target : array-like
        Target image or statistics to match
    image_init : array-like, optional
        Initial image for synthesis
    J : int, optional
        Number of dyadic scales
    L : int, default=4
        Number of orientations
    mode : str, default='image'
        Mode of operation: 'image' or 'estimator'
    optim_algorithm : str, default='LBFGS'
        Optimization algorithm to use
    steps : int, default=300
        Number of optimization steps
    learning_rate : float, default=0.2
        Learning rate for optimization
    device : str, default='gpu'
        Device to run computations on
    wavelets : str, default='morlet'
        Type of wavelets to use

    Returns
    -------
    np.ndarray
        Synthesized image(s)
    """
    if not torch.cuda.is_available():
        device = 'cpu'

    np.random.seed(seed)

    if C11_criteria is None:
        C11_criteria = 'j2>=j1'

    if mode == 'image':
        _, M, N = target.shape
        print('input_size: ', target.shape)

    # Set initial point of synthesis
    if image_init is None:
        if mode == 'image':
            if not ensemble:
                image_init = np.random.normal(
                    target.mean((-2, -1))[:, None, None],
                    target.std((-2, -1))[:, None, None],
                    (target.shape[0], M, N)
                )
            else:
                image_init = np.random.normal(
                    target.mean(),
                    target.std(),
                    (N_ensemble, M, N)
                )
        else:
            if M is None:
                print('please assign image size M and N.')
            if not ensemble:
                image_init = np.random.normal(0, 1, (target.shape[0], M, N))
            else:
                image_init = np.random.normal(0, 1, (N_ensemble, M, N))

    if J is None:
        J = int(np.log2(min(M, N))) - 1

    # Define calculator and estimator function
    if 's' in estimator_name:
        st_calc = Scattering2d(
            M, N, J, L, device, wavelets,
            l_oversampling=l_oversampling,
            frequency_factor=frequency_factor
        )

        if mode == 'image':
            if '2fields' not in estimator_name:
                st_calc.add_ref(ref=target)
            else:
                if image_b is None:
                    print('should provide a valid image_b.')
                else:
                    st_calc.add_ref_ab(ref_a=target, ref_b=image_b)

            if ensemble:
                ref_P00_mean = st_calc.ref_scattering_cov['P00'].mean(0)[None, :, :]
                if 'iso' in estimator_name:
                    ref_P00_mean = ref_P00_mean.mean(2)[:, :, None]
                st_calc.ref_scattering_cov['P00'] = ref_P00_mean

        if mode == 'estimator':
            if image_ref is None:
                if target_full is None:
                    temp = target
                else:
                    temp = target_full
                if normalization == 'P00':
                    if reference_P00 is None:
                        st_calc.add_synthesis_P00(s_cov=temp, if_iso='iso' in estimator_name)
                    else:
                        st_calc.add_synthesis_P00(P00=reference_P00)
                else:
                    st_calc.add_synthesis_P11(temp, 'iso' in estimator_name, C11_criteria)
            else:
                st_calc.add_ref(ref=image_ref)

        # Define estimator functions based on type
        if estimator_name == 's_mean_iso':
            func_s = lambda x: st_calc.scattering_coef(x, flatten=True)['for_synthesis_iso']
        elif estimator_name == 's_mean':
            func_s = lambda x: st_calc.scattering_coef(x, flatten=True)['for_synthesis']
        elif 's_cov_func' in estimator_name:
            def func_s(image):
                s_cov_set = st_calc.scattering_cov(
                    image, use_ref=True, if_large_batch=if_large_batch,
                    C11_criteria=C11_criteria,
                    normalization=normalization, pseudo_coef=pseudo_coef,
                    remove_edge=remove_edge
                )
                return s_cov_func(s_cov_set, s_cov_func_params)
        elif estimator_name == 's_cov_iso':
            func_s = lambda x: st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch,
                C11_criteria=C11_criteria,
                normalization=normalization, pseudo_coef=pseudo_coef,
                remove_edge=remove_edge
            )['for_synthesis_iso']
        elif estimator_name == 's_cov':
            func_s = lambda x: st_calc.scattering_cov(
                x, use_ref=True, if_large_batch=if_large_batch,
                C11_criteria=C11_criteria,
                normalization=normalization, pseudo_coef=pseudo_coef,
                remove_edge=remove_edge
            )['for_synthesis']
        else:
            raise ValueError(f"Unsupported estimator_name: {estimator_name}")
    else:
        raise ValueError(f"Unsupported estimator_name: {estimator_name}")

    # Power spectrum
    if ps:
        from .polyspectra import get_power_spectrum
        if ps_bins is None:
            ps_bins = J - 1
        def func_ps(image):
            ps_result, _ = get_power_spectrum(image, bins=ps_bins, bin_type=ps_bin_type)
            return torch.cat((
                (image.mean((-2, -1)) / image.std((-2, -1)))[:, None],
                image.var((-2, -1))[:, None],
                ps_result
            ), axis=-1)

    # Bispectrum
    if bi:
        from .polyspectra import Bispectrum_Calculator, get_power_spectrum
        if bispectrum_bins is None:
            bispectrum_bins = J - 1
        bi_calc = Bispectrum_Calculator(M, N, bins=bispectrum_bins, bin_type=bispectrum_bin_type, device=device)
        def func_bi(image):
            bi_result = bi_calc.forward(image)
            ps_result, _ = get_power_spectrum(image, bins=bispectrum_bins, bin_type=bispectrum_bin_type)
            return torch.cat((
                (image.mean((-2, -1)) / image.std((-2, -1)))[:, None],
                ps_result,
                bi_result
            ), axis=-1)

    # Histogram functions
    if hist:
        def func_hist(image):
            flat_image = image.reshape(len(image), -1)
            return flat_image.sort(dim=-1).values.reshape(len(image), -1, image.shape[-2]).mean(-1) / flat_image.std(-1)[:, None]

    if hist_j:
        def smooth(image, j):
            M, N = image.shape[-2:]
            X = torch.arange(M)[:, None]
            Y = torch.arange(N)[None, :]
            R2 = (X - M//2)**2 + (Y - N//2)**2
            weight_f = torch.fft.fftshift(torch.exp(-0.5 * R2 / (M//(2**j)//2)**2))
            if device == 'gpu':
                weight_f = weight_f.cuda()
            image_smoothed = torch.fft.ifftn(torch.fft.fftn(image, dim=(-2, -1)) * weight_f[None, :, :], dim=(-2, -1))
            return image_smoothed.real

        def func_hist_j(image, J):
            cumsum_list = []
            flat_image = image.reshape(len(image), -1)
            cumsum_list.append(
                flat_image.sort(dim=-1).values.reshape(len(image), -1, image.shape[-2]).mean(-1) / flat_image.std(-1)[:, None]
            )
            for j in range(J):
                subsample_rate = int(max(2**(j-1), 1))
                smoothed_image = smooth(image, j)[:, ::subsample_rate, ::subsample_rate]
                flat_image = smoothed_image.reshape(len(image), -1)
                cumsum_list.append(
                    flat_image.sort(dim=-1).values.reshape(len(image), -1, smoothed_image.shape[-2]).mean(-1) / flat_image.std(-1)[:, None]
                )
            return torch.cat(cumsum_list, dim=-1)

    if phi4:
        def func_phi4(image):
            return (image**4).mean((-2, -1))[..., None]

    if phi4_j:
        def func_phi4_j(image, J):
            cumsum_list = []
            cumsum_list.append(
                (image**4).mean((-2, -1))[..., None] / (image**2).mean((-2, -1))[..., None]**2
            )
            for j in range(J):
                subsample_rate = int(max(2**(j-1), 1))
                smoothed_image = smooth(image, j)[:, ::subsample_rate, ::subsample_rate]
                cumsum_list.append((smoothed_image**4).mean((-2, -1))[..., None] / (smoothed_image**2).mean((-2, -1))[..., None]**2)
            return torch.cat(cumsum_list, dim=-1)

    # Combined estimator function
    def func(image):
        coef_list = []
        if estimator_name != '':
            coef_list.append(func_s(image))
        if ps:
            coef_list.append(func_ps(image))
        if bi:
            coef_list.append(func_bi(image))
        if phi4:
            coef_list.append(func_phi4(image))
        if phi4_j:
            coef_list.append(func_phi4_j(image, J))
        if hist:
            coef_list.append(hist_factor * func_hist(image))
        if hist_j:
            coef_list.append(hist_j_factor * func_hist_j(image, J))
        return torch.cat(coef_list, axis=-1)

    # Define loss function
    def quadratic_loss(target, model):
        return ((target - model)**2).mean() * 1e8

    # Synthesis
    image_syn = synthesis_general(
        target, image_init, func, quadratic_loss,
        mode=mode,
        optim_algorithm=optim_algorithm, steps=steps, learning_rate=learning_rate,
        device=device, precision=precision, print_each_step=print_each_step,
        Fourier=Fourier, ensemble=ensemble,
    )
    return image_syn


def synthesis_general(
    target: Union[np.ndarray, torch.Tensor],
    image_init: Union[np.ndarray, torch.Tensor],
    estimator_function: Callable,
    loss_function: Callable,
    mode: str = 'image',
    optim_algorithm: str = 'LBFGS',
    steps: int = 100,
    learning_rate: float = 0.5,
    device: str = 'gpu',
    precision: str = 'single',
    print_each_step: bool = False,
    Fourier: bool = False,
    ensemble: bool = False,
) -> np.ndarray:
    """
    General synthesis function using gradient descent optimization.

    Parameters
    ----------
    target : array-like
        Target statistics to match
    image_init : array-like
        Initial image
    estimator_function : callable
        Function that computes statistics from image
    loss_function : callable
        Loss function for optimization
    mode : str, default='image'
        'image' or 'estimator' mode
    optim_algorithm : str, default='LBFGS'
        Optimization algorithm
    steps : int, default=100
        Number of optimization steps
    learning_rate : float, default=0.5
        Learning rate
    device : str, default='gpu'
        Device for computation
    precision : str, default='single'
        Numerical precision
    print_each_step : bool, default=False
        Whether to print progress
    Fourier : bool, default=False
        Whether to optimize in Fourier domain
    ensemble : bool, default=False
        Whether to use ensemble mode

    Returns
    -------
    np.ndarray
        Synthesized image(s)
    """
    # Get dimensions
    N_image = image_init.shape[0]
    M = image_init.shape[1]
    N = image_init.shape[2]

    # Format target and image_init (to tensor, to cuda)
    if type(target) == np.ndarray:
        target = torch.from_numpy(target)
    if type(image_init) == np.ndarray:
        image_init = torch.from_numpy(image_init)

    if precision == 'double':
        target = target.type(torch.DoubleTensor)
        image_init = image_init.type(torch.DoubleTensor)
    else:
        target = target.type(torch.FloatTensor)
        image_init = image_init.type(torch.FloatTensor)

    if device == 'gpu' and torch.cuda.is_available():
        target = target.cuda()
        image_init = image_init.cuda()

    # Calculate statistics for target images
    if mode == 'image':
        estimator_target = estimator_function(target)
    else:
        estimator_target = target
    print('# of estimators: ', estimator_target.shape[-1])

    # Define optimizable image model
    class OptimizableImage(torch.nn.Module):
        def __init__(self, input_init, Fourier=False):
            super().__init__()
            self.param = torch.nn.Parameter(input_init)

            if Fourier:
                self.image = torch.fft.ifftn(
                    self.param[0] + 1j * self.param[1],
                    dim=(-2, -1)
                ).real
            else:
                self.image = self.param

    if Fourier:
        temp = torch.fft.fftn(image_init, dim=(-2, -1))
        input_init = torch.cat((temp.real[None, ...], temp.imag[None, ...]), dim=0)
    else:
        input_init = image_init

    image_model = OptimizableImage(input_init, Fourier)

    # Define optimizer
    if optim_algorithm == 'Adam':
        optimizer = torch.optim.Adam(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm == 'NAdam':
        optimizer = torch.optim.NAdam(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm == 'SGD':
        optimizer = torch.optim.SGD(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm == 'Adamax':
        optimizer = torch.optim.Adamax(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm == 'LBFGS':
        optimizer = torch.optim.LBFGS(
            image_model.parameters(), lr=learning_rate,
            max_iter=steps, max_eval=None,
            tolerance_grad=1e-19, tolerance_change=1e-19,
            history_size=min(steps//2, 150), line_search_fn=None
        )
    else:
        raise ValueError(f"Unknown optimizer: {optim_algorithm}")

    # Closure function for optimization
    def closure():
        optimizer.zero_grad()
        loss = 0
        estimator_model = estimator_function(image_model.image)
        if ensemble:
            loss = loss_function(estimator_model.mean(0), estimator_target.mean(0))
        else:
            loss = loss_function(estimator_model, estimator_target)

        if print_each_step:
            if optim_algorithm == 'LBFGS' or (optim_algorithm != 'LBFGS' and (i % 100 == 0 or i % 100 == -1)):
                if not ensemble:
                    print('max residual: ',
                          np.max((estimator_model - estimator_target).abs().detach().cpu().numpy()),
                          ', mean residual: ',
                          np.mean((estimator_model - estimator_target).abs().detach().cpu().numpy()))
                else:
                    print('max residual: ',
                          np.max((estimator_model.mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()),
                          ', mean residual: ',
                          np.mean((estimator_model.mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()))

        loss.backward()
        return loss

    # Optimize
    t_start = time.time()

    # Print initial state
    if not ensemble:
        print('max residual: ',
              np.max((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()),
              ', mean residual: ',
              np.mean((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()))
    else:
        print('max residual: ',
              np.max((estimator_function(image_model.image).mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()),
              ', mean residual: ',
              np.mean((estimator_function(image_model.image).mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()))

    # Run optimization
    if optim_algorithm == 'LBFGS':
        i = 0
        optimizer.step(closure)
    else:
        for i in range(steps):
            optimizer.step(closure)

    # Print final state
    if not ensemble:
        print('max residual: ',
              np.max((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()),
              ', mean residual: ',
              np.mean((estimator_function(image_model.image) - estimator_target).abs().detach().cpu().numpy()))
    else:
        print('max residual: ',
              np.max((estimator_function(image_model.image).mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()),
              ', mean residual: ',
              np.mean((estimator_function(image_model.image).mean(0) - estimator_target.mean(0)).abs().detach().cpu().numpy()))

    t_end = time.time()
    print('time used: ', t_end - t_start, 's')

    return image_model.image.cpu().detach().numpy()


__all__ = ['synthesis', 'synthesis_general', 'FiltersSet', 'Scattering2d']