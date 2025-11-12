"""
Minimal comparison test between SC2 and original scattering synthesis.
"""

import numpy as np
import sys

print("=" * 60)
print("COMPARISON TEST: SC2 vs Original Scattering")
print("=" * 60)

# Create test data
np.random.seed(42)
test_image = np.random.randn(1, 64, 64).astype(np.float32)
print(f"Test image shape: {test_image.shape}")

# Test parameters
params = {
    'J': 4,
    'L': 4,
    'steps': 20,
    'device': 'cpu',
    'seed': 42,
    'print_each_step': False
}

# Test SC2 implementation
print("\n1. Testing SC2 implementation...")
try:
    import SC2
    np.random.seed(params['seed'])
    sc2_result = SC2.synthesis(
        estimator_name='s_cov_iso',
        target=test_image,
        **params
    )
    print(f"   ✓ SC2 synthesis completed")
    print(f"     Shape: {sc2_result.shape}")
    print(f"     Mean: {sc2_result.mean():.4f}, Std: {sc2_result.std():.4f}")
except Exception as e:
    print(f"   ✗ SC2 failed: {e}")
    import traceback
    traceback.print_exc()
    sc2_result = None

# Test original implementation
print("\n2. Testing original implementation...")
try:
    import scattering
    np.random.seed(params['seed'])
    orig_result = scattering.synthesis(
        estimator_name='s_cov_iso',
        target=test_image,
        **params
    )
    print(f"   ✓ Original synthesis completed")
    print(f"     Shape: {orig_result.shape}")
    print(f"     Mean: {orig_result.mean():.4f}, Std: {orig_result.std():.4f}")
except Exception as e:
    print(f"   ✗ Original failed: {e}")
    import traceback
    traceback.print_exc()
    orig_result = None

# Compare results if both succeeded
if sc2_result is not None and orig_result is not None:
    print("\n3. Comparison:")

    # Check shapes
    if sc2_result.shape == orig_result.shape:
        print(f"   ✓ Shapes match: {sc2_result.shape}")
    else:
        print(f"   ✗ Shape mismatch: SC2 {sc2_result.shape} vs Original {orig_result.shape}")

    # Compute differences
    diff = np.abs(sc2_result - orig_result)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Relative error
    orig_scale = np.mean(np.abs(orig_result)) + 1e-10
    relative_error = mean_diff / orig_scale

    print(f"   Max absolute difference: {max_diff:.6e}")
    print(f"   Mean absolute difference: {mean_diff:.6e}")
    print(f"   Relative error: {relative_error:.2%}")

    # Check statistical properties
    print("\n   Statistical comparison:")
    print(f"     SC2  - Mean: {sc2_result.mean():.4f}, Std: {sc2_result.std():.4f}")
    print(f"     Orig - Mean: {orig_result.mean():.4f}, Std: {orig_result.std():.4f}")

    # Overall verdict
    if relative_error < 0.05:  # 5% tolerance
        print(f"\n   ✓ TEST PASSED - Results are consistent (relative error: {relative_error:.2%})")
    else:
        print(f"\n   ⚠ Results differ more than expected (relative error: {relative_error:.2%})")

print("\n" + "=" * 60)
print("Test completed")
print("=" * 60)