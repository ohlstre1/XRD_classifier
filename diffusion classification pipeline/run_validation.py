#!/usr/bin/env python3
"""
Simple script to run the diffusion model validation suite.

Usage:
    python run_validation.py

This will:
1. Test all model components
2. Test the diffusion process mathematics
3. Run a short training loop
4. Test sampling quality
5. Generate validation plots
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validation_suite import ValidationSuite
import torch

def main():
    print("🧪 DIFFUSION MODEL VALIDATION SUITE")
    print("="*50)

    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create validation suite
    validator = ValidationSuite(device=device)

    # Run validation
    try:
        success = validator.run_full_validation()

        if success:
            print("\n🎉 ALL VALIDATIONS PASSED!")
            print("The diffusion model implementation is working correctly.")
            print("\nNext steps:")
            print("1. Try with your real XRD data")
            print("2. Adjust model size for your dataset")
            print("3. Train for more epochs")
            print("4. Experiment with different conditioning")

            return 0
        else:
            print("\n⚠️  SOME VALIDATIONS FAILED!")
            print("Check the output above for specific issues.")
            print("The model may still work but needs attention.")

            return 1

    except KeyboardInterrupt:
        print("\n\n⏹️  Validation interrupted by user")
        return 1

    except Exception as e:
        print(f"\n\n💥 Validation failed with error: {e}")
        print("This indicates a serious issue with the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)