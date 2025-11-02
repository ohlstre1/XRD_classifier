"""
Helper script to reload validation modules and fix KeyError issues.
"""

import importlib
import sys

def reload_validation_modules():
    """Reload all validation modules to pick up latest changes."""

    # List of validation modules to reload
    modules_to_reload = [
        'validation',
        'validation.core.data_loader',
        'validation.core.model_loader',
        'validation.tests.stochasticity',
        'validation.tests.timestep_effects',
        'validation.tests.dtw_conditioning',
        'validation.analysis.validation_suite',
        'validation.analysis.report_generator',
        'validation.visualization.plotting',
        'validation.visualization.interactive',
        'validation.utils'
    ]

    print("Reloading validation modules...")

    for module_name in modules_to_reload:
        if module_name in sys.modules:
            print(f"  Reloading {module_name}")
            importlib.reload(sys.modules[module_name])
        else:
            print(f"  Module {module_name} not loaded yet")

    print("✓ Validation modules reloaded")
    return True

if __name__ == "__main__":
    reload_validation_modules()