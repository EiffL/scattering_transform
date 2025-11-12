"""
Test suite for SC2 package to verify it matches the original scattering implementation.
"""

import numpy as np
import torch
import sys
import traceback
from pathlib import Path

# Import both implementations
import scattering
import SC2


def test_synthesis_consistency():
    """
    Test that SC2.synthesis produces the same results as scattering.synthesis.
    """
    print("=" * 80)
    print("Testing SC2.synthesis vs scattering.synthesis")
    print("=" * 80)

    # Load example data
    example_path = Path('scattering/example_fields.npy')
    if not example_path.exists():
        print(f"Warning: Example data not found at {example_path}")
        print("Creating synthetic test data...")
        # Create synthetic test data
        np.random.seed(42)
        test_image = np.random.randn(1, 128, 128).astype(np.float32)
    else:
        print(f"Loading example data from {example_path}")
        all_images = np.load(str(example_path))
        test_image = all_images[:1, :128, :128].astype(np.float32)  # Use smaller size for faster testing

    print(f"Test image shape: {test_image.shape}")

    # Test parameters
    test_params = {
        'J': 5,
        'L': 4,
        'steps': 50,  # Reduced for faster testing
        'seed': 42,
        'device': 'gpu' if torch.cuda.is_available() else 'cpu',
        'print_each_step': False,
    }

    print(f"Test parameters: {test_params}")
    print()

    # Test different estimator types
    estimator_names = ['s_mean', 's_mean_iso', 's_cov', 's_cov_iso']

    all_tests_passed = True

    for estimator_name in estimator_names:
        print(f"\nTesting estimator: {estimator_name}")
        print("-" * 40)

        try:
            # Run original implementation
            print("Running original scattering.synthesis...")
            np.random.seed(test_params['seed'])
            original_result = scattering.synthesis(
                estimator_name=estimator_name,
                target=test_image,
                **test_params
            )

            # Run SC2 implementation
            print("Running SC2.synthesis...")
            np.random.seed(test_params['seed'])
            sc2_result = SC2.synthesis(
                estimator_name=estimator_name,
                target=test_image,
                **test_params
            )

            # Compare results
            print("\nComparing results:")
            print(f"  Original shape: {original_result.shape}")
            print(f"  SC2 shape: {sc2_result.shape}")

            # Check shapes match
            assert original_result.shape == sc2_result.shape, \
                f"Shape mismatch: {original_result.shape} vs {sc2_result.shape}"

            # Compute statistics
            diff = np.abs(original_result - sc2_result)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            relative_error = mean_diff / (np.mean(np.abs(original_result)) + 1e-10)

            print(f"  Max absolute difference: {max_diff:.6e}")
            print(f"  Mean absolute difference: {mean_diff:.6e}")
            print(f"  Relative error: {relative_error:.6%}")

            # Check if results are close enough
            # Note: Due to numerical precision and optimization, exact match is not expected
            tolerance = 1e-2  # 1% tolerance
            if relative_error < tolerance:
                print(f"  ✓ Test PASSED for {estimator_name}")
            else:
                print(f"  ✗ Test FAILED for {estimator_name} - relative error too high")
                all_tests_passed = False

        except Exception as e:
            print(f"  ✗ Test FAILED for {estimator_name} with error:")
            print(f"    {str(e)}")
            traceback.print_exc()
            all_tests_passed = False

    print("\n" + "=" * 80)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)

    return all_tests_passed


def test_import_and_basic_functionality():
    """
    Test that SC2 package can be imported and has the expected interface.
    """
    print("\n" + "=" * 80)
    print("Testing SC2 package imports and interface")
    print("=" * 80)

    try:
        # Test imports
        print("Testing imports...")
        from SC2 import synthesis, synthesis_general, FiltersSet, Scattering2d
        print("  ✓ All main components can be imported")

        # Test FiltersSet
        print("\nTesting FiltersSet...")
        filters = FiltersSet(M=64, N=64, J=3, L=4)
        filter_dict = filters.generate_wavelets(wavelets='morlet')
        assert 'psi' in filter_dict and 'phi' in filter_dict
        assert filter_dict['psi'].shape == (3, 4, 64, 64)
        print("  ✓ FiltersSet works correctly")

        # Test Scattering2d
        print("\nTesting Scattering2d...")
        scat = Scattering2d(M=64, N=64, J=3, L=4, device='cpu')
        test_data = np.random.randn(2, 64, 64).astype(np.float32)
        result = scat.scattering_coef(test_data, flatten=True)
        assert 'for_synthesis' in result and 'for_synthesis_iso' in result
        print("  ✓ Scattering2d works correctly")

        print("\n✓ All basic functionality tests passed")
        return True

    except Exception as e:
        print(f"\n✗ Basic functionality test failed:")
        print(f"  {str(e)}")
        traceback.print_exc()
        return False


def run_performance_comparison():
    """
    Compare performance between original and SC2 implementations.
    """
    print("\n" + "=" * 80)
    print("Performance Comparison")
    print("=" * 80)

    import time

    # Create test data
    np.random.seed(42)
    test_image = np.random.randn(1, 64, 64).astype(np.float32)

    params = {
        'J': 4,
        'L': 4,
        'steps': 20,
        'seed': 42,
        'device': 'gpu' if torch.cuda.is_available() else 'cpu',
        'print_each_step': False,
    }

    # Time original implementation
    print("Timing original implementation...")
    start_time = time.time()
    _ = scattering.synthesis('s_cov_iso', test_image, **params)
    original_time = time.time() - start_time
    print(f"  Original: {original_time:.2f} seconds")

    # Time SC2 implementation
    print("Timing SC2 implementation...")
    start_time = time.time()
    _ = SC2.synthesis('s_cov_iso', test_image, **params)
    sc2_time = time.time() - start_time
    print(f"  SC2: {sc2_time:.2f} seconds")

    # Compare
    speedup = original_time / sc2_time
    print(f"\n  Performance ratio: {speedup:.2f}x")
    if speedup > 0.9:
        print("  ✓ SC2 performance is comparable or better")
    else:
        print("  ⚠ SC2 is slower than original")


def main():
    """
    Run all tests.
    """
    print("\n" + "🧪 SC2 PACKAGE TEST SUITE 🧪".center(80))
    print("=" * 80)

    # Test basic functionality first
    if not test_import_and_basic_functionality():
        print("\n⚠ Basic functionality tests failed. Skipping other tests.")
        return 1

    # Test synthesis consistency
    if not test_synthesis_consistency():
        print("\n⚠ Some synthesis consistency tests failed.")
        return 1

    # Run performance comparison
    try:
        run_performance_comparison()
    except Exception as e:
        print(f"\n⚠ Performance comparison failed: {e}")

    print("\n" + "=" * 80)
    print("✅ TEST SUITE COMPLETED SUCCESSFULLY".center(80))
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())