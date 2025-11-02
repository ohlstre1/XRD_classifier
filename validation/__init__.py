"""
XRD Diffusion Validation Suite

A modular validation framework for XRD diffusion models providing:
- Comprehensive stochasticity testing
- Timestep effect analysis
- DTW conditioning validation
- Real-world applicability assessment
- Interactive exploration tools

Usage:
    from validation import ValidationSuite

    # Quick validation
    suite = ValidationSuite()
    results = suite.run_all_tests(model, diffusion, test_data)

    # Individual components
    from validation.tests import test_model_stochasticity
    from validation.analysis import comprehensive_validation_suite
    from validation.visualization import plot_stochasticity_analysis
"""

from .core.data_loader import XRDDataLoader, load_xrd_data
from .core.model_loader import ModelLoader, load_diffusion_setup

from .analysis.validation_suite import (
    comprehensive_validation_suite,
    evaluate_test_set_performance,
    save_validation_results,
    load_validation_results
)

from .analysis.report_generator import (
    generate_summary_report,
    print_validation_summary,
    generate_quick_summary
)

__version__ = "1.0.0"
__author__ = "XRD Diffusion Team"

# Convenience class for easy usage
class ValidationSuite:
    """
    Main validation suite class for easy access to all validation functionality.
    """

    def __init__(self, device: str = 'auto'):
        """
        Initialize validation suite.

        Args:
            device: Device to use for computations
        """
        self.device = device
        self.data_loader = None
        self.model_loader = None
        self.model = None
        self.diffusion = None
        self.data_splits = None

    def load_data(self, dataset_path: str = "data/xrd_dataset_labeled_dtw_window.pt"):
        """Load XRD dataset."""
        self.data_splits, self.data_loader = load_xrd_data(dataset_path, self.device)
        return self.data_splits

    def load_model(self, model_path: str = "diffusion/models/xrd_diffusion/best_model.pth"):
        """Load diffusion model."""
        self.model, self.diffusion, self.model_loader = load_diffusion_setup(model_path, self.device)
        return self.model, self.diffusion

    def run_all_tests(self, model=None, diffusion=None, test_data=None, subset_size: int = 50):
        """Run comprehensive validation suite."""
        if model is None:
            model = self.model
        if diffusion is None:
            diffusion = self.diffusion
        if test_data is None and self.data_splits is not None:
            test_data = self.data_splits['test']

        if any(x is None for x in [model, diffusion, test_data]):
            raise ValueError("Model, diffusion, and test data must be provided or loaded first")

        return comprehensive_validation_suite(
            model, diffusion,
            test_data['synth'], test_data['real'], test_data['dtw'],
            subset_size=subset_size
        )

    def generate_report(self, validation_results, test_results=None):
        """Generate summary report."""
        return generate_summary_report(validation_results, test_results)

    def print_report(self, validation_results, test_results=None):
        """Print summary report."""
        print_validation_summary(validation_results, test_results)


__all__ = [
    'ValidationSuite',
    'XRDDataLoader', 'load_xrd_data',
    'ModelLoader', 'load_diffusion_setup',
    'comprehensive_validation_suite',
    'evaluate_test_set_performance',
    'save_validation_results', 'load_validation_results',
    'generate_summary_report', 'print_validation_summary', 'generate_quick_summary'
]