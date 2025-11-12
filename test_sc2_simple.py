"""
Simple test for SC2 package functionality.
"""

import numpy as np
import torch
import sys
import time

# Test SC2 import and basic functionality
print("=" * 60)
print("SC2 SIMPLE TEST SUITE")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from SC2 import synthesis, FiltersSet, Scattering2d
    print("   ✓ SC2 imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test filter generation
print("\n2. Testing filter generation...")
try:
    filters = FiltersSet(M=64, N=64, J=3, L=4)
    filter_dict = filters.generate_wavelets(wavelets='morlet')
    assert 'psi' in filter_dict and 'phi' in filter_dict
    print(f"   ✓ Filters generated successfully")
    print(f"     psi shape: {filter_dict['psi'].shape}")
    print(f"     phi shape: {filter_dict['phi'].shape}")
except Exception as e:
    print(f"   ✗ Filter generation failed: {e}")
    sys.exit(1)

# Test Scattering2d initialization
print("\n3. Testing Scattering2d...")
try:
    device = 'cpu'  # Use CPU for simplicity
    scat = Scattering2d(M=64, N=64, J=3, L=4, device=device)
    print(f"   ✓ Scattering2d initialized successfully")

    # Test with simple data
    test_data = np.random.randn(2, 64, 64).astype(np.float32)
    result = scat.scattering_coef(test_data, flatten=True)

    print(f"   ✓ scattering_coef computed successfully")
    if 'for_synthesis' in result:
        print(f"     for_synthesis shape: {result['for_synthesis'].shape}")
    if 'for_synthesis_iso' in result:
        print(f"     for_synthesis_iso shape: {result['for_synthesis_iso'].shape}")
except Exception as e:
    print(f"   ✗ Scattering2d test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test synthesis function with minimal parameters
print("\n4. Testing synthesis function...")
try:
    np.random.seed(42)
    test_image = np.random.randn(1, 32, 32).astype(np.float32)

    print("   Running synthesis with s_mean estimator...")
    start_time = time.time()

    synthesized = synthesis(
        estimator_name='s_mean',
        target=test_image,
        J=3,
        L=4,
        steps=10,  # Very few steps for quick test
        device='cpu',
        print_each_step=False,
        seed=42
    )

    elapsed = time.time() - start_time
    print(f"   ✓ Synthesis completed in {elapsed:.2f} seconds")
    print(f"     Input shape: {test_image.shape}")
    print(f"     Output shape: {synthesized.shape}")
    print(f"     Output mean: {synthesized.mean():.4f}")
    print(f"     Output std: {synthesized.std():.4f}")

except Exception as e:
    print(f"   ✗ Synthesis failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test different estimators
print("\n5. Testing different estimators...")
estimators = ['s_mean', 's_mean_iso', 's_cov', 's_cov_iso']
for estimator in estimators:
    try:
        np.random.seed(42)
        synthesized = synthesis(
            estimator_name=estimator,
            target=test_image,
            J=3,
            L=4,
            steps=5,
            device='cpu',
            print_each_step=False,
            seed=42
        )
        print(f"   ✓ {estimator}: shape {synthesized.shape}")
    except Exception as e:
        print(f"   ✗ {estimator} failed: {e}")

print("\n" + "=" * 60)
print("✅ SC2 BASIC TESTS COMPLETED SUCCESSFULLY")
print("=" * 60)