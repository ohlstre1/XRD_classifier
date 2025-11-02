"""
Report generation functionality for XRD diffusion validation.
"""

import numpy as np
from typing import Dict, Optional


def generate_summary_report(validation_results: Dict, test_results: Optional[Dict] = None) -> str:
    """
    Generate a comprehensive summary report answering the key questions.

    Args:
        validation_results: Results from comprehensive validation suite
        test_results: Optional full test set results

    Returns:
        Formatted summary report as string
    """

    report_lines = []

    # Header
    report_lines.extend([
        "=" * 80,
        "XRD DIFFUSION MODEL VALIDATION SUMMARY REPORT",
        "=" * 80,
        "",
        "🎯 KEY QUESTIONS ANSWERED:",
        "-" * 50
    ])

    # Question 1: Model Stochasticity
    report_lines.append("\n1. IS THE MODEL STOCHASTIC?")
    if validation_results['stochasticity']['test_pass']:
        ratio = validation_results['stochasticity']['stochasticity_ratio']
        report_lines.extend([
            "   ✅ YES - The model exhibits clear stochastic behavior",
            f"   📊 Stochastic mode is {ratio:.1f}x more variable than deterministic",
            "   🔧 Controllable via model.set_stochastic_mode() and model.train()/eval()",
            "   💡 Stochastic features: dropout, stochastic depth, and variational components"
        ])
    else:
        ratio = validation_results['stochasticity']['stochasticity_ratio']
        report_lines.extend([
            "   ⚠️  UNCLEAR - Stochastic behavior may be limited",
            f"   📊 Stochasticity ratio: {ratio:.2f}"
        ])

    # Question 2: Timestep Effects
    report_lines.append("\n2. HOW DOES CHANGING TIMESTEP AFFECT THE MODEL?")
    if validation_results['timestep_effects']['test_pass']:
        corr = validation_results['timestep_effects']['timestep_recon_correlation']
        report_lines.extend([
            "   ✅ PROGRESSIVE EFFECTS - Timesteps systematically control augmentation strength",
            f"   📈 Correlation with reconstruction error: {corr:.3f}",
            "   🔄 Higher timesteps → More noise → Stronger augmentation",
            "   🧬 Includes physics-based peak broadening (Scherrer equation)",
            "   🎛️  t=0: Direct transformation, t>0: Diffusion with increasing noise"
        ])
    else:
        report_lines.append("   ⚠️  WEAK CORRELATION - Timestep effects may be inconsistent")

    # Question 3: DTW Distance Impact
    report_lines.append("\n3. HOW DOES DTW DISTANCE CHANGE THE MODEL?")
    if validation_results['dtw_conditioning']['test_pass']:
        effect_std = validation_results['dtw_conditioning']['effect_std']
        effect_range = validation_results['dtw_conditioning']['effect_range']
        report_lines.extend([
            "   ✅ SIGNIFICANT CONDITIONING - DTW distance effectively controls transformations",
            f"   📊 Effect standard deviation: {effect_std:.6f}",
            f"   📏 Effect range: {effect_range:.6f}",
            "   🎯 DTW values guide synthetic-to-real transformation strength",
            "   📐 Range: 0.0 (minimal transform) to 1.0 (maximal transform)"
        ])
    else:
        report_lines.append("   ⚠️  LIMITED EFFECT - DTW conditioning may not be working properly")

    # Question 4: Real-world Applicability
    report_lines.append("\n4. IS THIS AUGMENTATION SIMILAR TO REAL-WORLD DATA?")
    if validation_results['real_world_similarity']['pass']:
        mean_sim = validation_results['real_world_similarity']['mean_similarity']
        high_sim_frac = validation_results['real_world_similarity']['high_similarity_fraction']
        report_lines.extend([
            "   ✅ REALISTIC AUGMENTATION - Good similarity to real experimental data",
            f"   📊 Mean similarity to real patterns: {mean_sim:.3f}",
            f"   🎯 High similarity samples (>0.6): {high_sim_frac*100:.1f}%",
            "   🧪 Model learned realistic experimental transformations"
        ])
    else:
        mean_sim = validation_results['real_world_similarity']['mean_similarity']
        report_lines.extend([
            "   ⚠️  POOR SIMILARITY - Augmentations may not reflect real-world data",
            f"   📊 Mean similarity: {mean_sim:.3f}"
        ])

    # Question 5: Test Set Performance (if available)
    if test_results is not None:
        report_lines.append("\n5. DOES IT WORK ON THE TEST SET?")
        mean_sim = np.mean(test_results['real_similarities'])
        high_sim_frac = np.sum(test_results['real_similarities'] > 0.6) / len(test_results['real_similarities'])
        mean_transform_loss = np.mean(test_results['direct_transform_losses'])

        if mean_sim > 0.4 and high_sim_frac > 0.3:
            report_lines.append("   ✅ GOOD GENERALIZATION - Model performs well on unseen test data")
        else:
            report_lines.append("   ⚠️  LIMITED GENERALIZATION - Performance on test set needs improvement")

        report_lines.extend([
            f"   📊 Test set similarity: {mean_sim:.3f} ± {np.std(test_results['real_similarities']):.3f}",
            f"   🎯 Good similarity samples: {high_sim_frac*100:.1f}%",
            f"   💔 Transform loss: {mean_transform_loss:.4f} ± {np.std(test_results['direct_transform_losses']):.4f}",
            f"   📈 Peak correlation: {np.mean(test_results['peak_correlations']):.3f} ± {np.std(test_results['peak_correlations']):.3f}"
        ])

    # Technical Summary
    report_lines.extend([
        "\n\n🔧 TECHNICAL SUMMARY:",
        "-" * 50,
        "Architecture: U-Net style diffusion model with stochastic components",
        "Conditioning: DTW distance values for real-world similarity guidance",
        "Physics: Scherrer equation for realistic peak broadening",
        "Stochasticity: Controlled via dropout, stochastic depth, and training mode",
        "Timesteps: Progressive augmentation from t=0 (clean) to t=999 (noisy)"
    ])

    # Add test results info if available
    if test_results is not None:
        report_lines.append(f"Test samples: {len(test_results['real_similarities'])} samples evaluated")

    # Recommendations
    report_lines.extend([
        "\n\n💡 RECOMMENDATIONS:",
        "-" * 50
    ])

    if validation_results['stochasticity']['test_pass']:
        report_lines.append("✅ Stochastic augmentation is working - use model.train() for data augmentation")
    else:
        report_lines.append("🔧 Consider increasing dropout rates or stochastic depth probability")

    if validation_results['timestep_effects']['test_pass']:
        report_lines.append("✅ Use varying timesteps (0-500) for different augmentation strengths")
    else:
        report_lines.append("🔧 Review timestep schedule and diffusion process parameters")

    if validation_results['dtw_conditioning']['test_pass']:
        report_lines.append("✅ DTW conditioning is effective - use original DTW values for best results")
    else:
        report_lines.append("🔧 Consider adjusting DTW conditioning architecture or normalization")

    if validation_results['real_world_similarity']['pass']:
        report_lines.extend([
            "✅ Model produces realistic augmentations suitable for training",
            "📈 Consider using augmented data for downstream classification tasks"
        ])
    else:
        report_lines.extend([
            "🔧 May need additional training or architecture improvements",
            "🧪 Validate with domain experts on augmentation realism"
        ])

    # Validation Summary
    total_tests = validation_results['total_tests']
    passed_tests = validation_results['passed_tests']

    report_lines.extend([
        "\n\n📋 VALIDATION SUMMARY:",
        "-" * 50,
        f"Overall validation: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)"
    ])

    for test_name, result in validation_results.items():
        if isinstance(result, dict) and 'pass' in result:
            status = "✅ PASS" if result['pass'] else "❌ FAIL"
            display_name = test_name.replace('_', ' ').title()
            report_lines.append(f"  {display_name:20s}: {status}")

    if passed_tests == total_tests:
        report_lines.append("\n🎉 ALL VALIDATIONS PASSED - Model is ready for production use!")
    else:
        report_lines.append(f"\n⚠️  {total_tests - passed_tests} validations failed - Review recommendations above")

    report_lines.append("\n" + "=" * 80)

    return "\n".join(report_lines)


def print_validation_summary(validation_results: Dict, test_results: Optional[Dict] = None):
    """
    Print a comprehensive validation summary.

    Args:
        validation_results: Results from comprehensive validation suite
        test_results: Optional full test set results
    """
    report = generate_summary_report(validation_results, test_results)
    print(report)


def generate_quick_summary(validation_results: Dict) -> str:
    """
    Generate a quick one-page summary of validation results.

    Args:
        validation_results: Results from comprehensive validation suite

    Returns:
        Quick summary as string
    """
    total_tests = validation_results['total_tests']
    passed_tests = validation_results['passed_tests']

    summary_lines = [
        "XRD DIFFUSION MODEL - QUICK VALIDATION SUMMARY",
        "=" * 50,
        f"Overall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)",
        ""
    ]

    # Quick test results
    tests = [
        ("Stochasticity", validation_results['stochasticity']['test_pass']),
        ("Timestep Effects", validation_results['timestep_effects']['test_pass']),
        ("DTW Conditioning", validation_results['dtw_conditioning']['test_pass']),
        ("Real Similarity", validation_results['real_world_similarity']['pass']),
        ("Model Stability", validation_results['stability']['pass'])
    ]

    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        summary_lines.append(f"{test_name:15s}: {status}")

    if passed_tests == total_tests:
        summary_lines.append("\n🎉 Model ready for production!")
    else:
        summary_lines.append(f"\n⚠️  {total_tests - passed_tests} issues need attention")

    return "\n".join(summary_lines)