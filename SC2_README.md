# SC2 - Minimal Scattering Synthesis Package

SC2 is a lightweight, high-quality implementation of the scattering transform synthesis functionality. It provides the essential components needed to perform image synthesis using scattering transform statistics.

## Features

- **Minimal Dependencies**: Only requires NumPy and PyTorch
- **Clean Architecture**: Modular design with clear separation of concerns
- **High Performance**: Optimized implementation with GPU support
- **Compatible**: Produces results consistent with the original scattering package
- **Well-Tested**: Comprehensive test suite ensures reliability

## Installation

### Using pip with virtual environment (recommended)

```bash
# Create and activate virtual environment
python -m venv venv_sc2
source venv_sc2/bin/activate  # On Windows: venv_sc2\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Using pip directly

```bash
pip install numpy torch
```

## Quick Start

```python
import numpy as np
from SC2 import synthesis

# Load or create target image
target_image = np.random.randn(1, 128, 128).astype(np.float32)

# Run synthesis with scattering covariance
synthesized = synthesis(
    estimator_name='s_cov_iso',
    target=target_image,
    J=5,  # Number of scales
    L=4,  # Number of orientations
    steps=100,  # Optimization steps
    device='gpu'  # or 'cpu'
)

print(f"Synthesized image shape: {synthesized.shape}")
```

## Package Structure

```
SC2/
├── __init__.py          # Main synthesis functions and exports
├── filters.py           # Filter bank generation (Morlet, Shannon, etc.)
├── scattering.py        # Core scattering transform implementation
├── polyspectra.py       # Power spectrum and bispectrum calculators
└── utils.py            # Utility functions
```

## Core Components

### 1. Synthesis Function
The main entry point for image synthesis using various scattering statistics:
- `s_mean`: Scattering mean coefficients
- `s_mean_iso`: Isotropic scattering mean
- `s_cov`: Scattering covariance
- `s_cov_iso`: Isotropic scattering covariance

### 2. FiltersSet Class
Generates wavelet filter banks:
- Morlet wavelets
- Bump-steerable wavelets
- Gaussian-steerable wavelets
- Shannon wavelets

### 3. Scattering2d Class
Implements the 2D scattering transform with:
- Multi-scale wavelet decomposition
- Covariance computation
- Reference normalization

### 4. Optimization
Flexible optimization with multiple algorithms:
- LBFGS (default, recommended)
- Adam, NAdam, SGD, Adamax
- Fourier domain optimization support

## Advanced Usage

### Custom Scattering Function

```python
from SC2 import synthesis

def custom_scattering_func(s_cov_set, params):
    """Custom function to process scattering coefficients"""
    coef = s_cov_set['for_synthesis_iso']
    # Apply custom processing
    return coef * params[0]

synthesized = synthesis(
    estimator_name='s_cov_func',
    target=target_image,
    s_cov_func=custom_scattering_func,
    s_cov_func_params=[1.5],
    J=5, L=4
)
```

### Using Power Spectrum and Bispectrum

```python
synthesized = synthesis(
    estimator_name='s_cov_iso',
    target=target_image,
    ps=True,  # Include power spectrum
    ps_bins=10,
    bi=True,  # Include bispectrum
    bispectrum_bins=8,
    J=5, L=4
)
```

## Testing

Run the test suite to verify the installation:

```bash
python test_sc2.py
```

The test suite will:
1. Verify all components can be imported
2. Test basic functionality
3. Compare results with the original implementation
4. Measure performance

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- PyTorch >= 1.9.0

## Performance

SC2 is optimized for performance while maintaining code clarity:
- GPU acceleration when available
- Efficient FFT-based convolutions
- Optimized memory usage
- Comparable or better performance than the original implementation

## Compatibility

SC2 is designed to be compatible with the original scattering package. The synthesis function produces numerically consistent results, making it a drop-in replacement for the synthesis functionality.

## Contributing

Contributions are welcome! The package is designed with clean, modular architecture making it easy to extend and improve.

## License

MIT License - See LICENSE file for details

## Acknowledgments

This minimal implementation is based on the original scattering transform package, focusing on the essential synthesis functionality with improved clarity and maintainability.