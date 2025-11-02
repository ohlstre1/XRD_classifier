#!/usr/bin/env python3
"""
Simple script to run the XRD diffusion model evaluation with standard deviation metrics.
This script provides a user-friendly interface to the comprehensive evaluation system.
"""

import os
import sys
import torch
from evaluate_diffusion_std import main as run_evaluation

def check_requirements():
    """
    Check if required files and dependencies are available.
    """
    # Check if model file exists
    model_path = "./models/xrd_diffusion/improved_diffusion_model_best.pth"
    data_path = "data/xrd_dataset_labeled_dtw_window.pt"
    diffusion_model_path = "./diffusion_model_0.1.5.py"

    missing_files = []

    if not os.path.exists(diffusion_model_path):
        missing_files.append(diffusion_model_path)

    if not os.path.exists(data_path):
        missing_files.append(data_path)

    # Model file is optional - will use random initialization if not found

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure these files are available before running the evaluation.")
        return False

    print("✅ All required files found!")
    return True

def print_usage_info():
    """
    Print information about how to use the evaluation script.
    """
    print("🔬 XRD Diffusion Model Evaluation with Standard Deviation")
    print("=" * 60)
    print()
    print("This script evaluates the performance of your trained XRD diffusion model")
    print("by running multiple evaluation passes and calculating comprehensive metrics")
    print("with standard deviation to assess model reliability and consistency.")
    print()
    print("📊 Metrics calculated:")
    print("  • Core Losses: Total, Diffusion, Reconstruction")
    print("  • Accuracy: MSE, MAE, Correlation coefficients")
    print("  • Peak-specific: Position error, Intensity error, Detection rate")
    print("  • Signal Quality: SNR improvement")
    print()
    print("📈 For each metric, you'll get:")
    print("  • Mean (μ) - average performance across runs")
    print("  • Standard deviation (σ) - consistency measure")
    print("  • Min/Max values - performance range")
    print()
    print("📁 Output files will be saved to './evaluation_results/'")
    print("  • evaluation_results_with_std.json - Detailed numerical results")
    print("  • metrics_with_std.png - Overview visualization")
    print("  • distribution_*.png - Distribution analysis plots")
    print()

def main():
    """
    Main function to orchestrate the evaluation process.
    """
    print_usage_info()

    # Check requirements
    if not check_requirements():
        return

    # Check device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")

    if device == 'cuda':
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Ask user to confirm
    print("\n" + "⚠️ " * 20)
    print("This evaluation will:")
    print("  • Run 10 evaluation passes (configurable in the script)")
    print("  • Process your entire test dataset for each pass")
    print("  • Generate comprehensive visualizations")
    print("  • Take approximately 2-5 minutes depending on your hardware")

    response = input("\nProceed with evaluation? (y/N): ").strip().lower()

    if response not in ['y', 'yes']:
        print("Evaluation cancelled.")
        return

    print("\n🚀 Starting comprehensive evaluation...")
    print("=" * 60)

    try:
        # Run the evaluation
        run_evaluation()

        print("\n" + "🎉 " * 20)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("🎉 " * 20)
        print()
        print("📂 Check the './evaluation_results/' directory for:")
        print("  • Detailed JSON results")
        print("  • Performance visualizations")
        print("  • Distribution analysis plots")
        print()
        print("💡 Use these metrics to:")
        print("  • Assess model reliability (low std = consistent performance)")
        print("  • Compare different model versions")
        print("  • Identify areas for improvement")
        print("  • Report performance in research/documentation")

    except Exception as e:
        print(f"\n❌ Evaluation failed with error:")
        print(f"   {str(e)}")
        print(f"\n🔧 Troubleshooting:")
        print(f"  • Ensure all required files are present")
        print(f"  • Check that you have sufficient memory")
        print(f"  • Verify the model file is not corrupted")
        print(f"  • Try reducing the number of evaluation runs in the script")

if __name__ == "__main__":
    main()