#!/bin/bash
# ============================================================
# Show All Results — no training, just prints the table
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Results Comparison"
echo "============================================================"

python3 -c "
import json, os

results_dirs = {
    'Baseline':            './models/baseline_comparison',
    'Smart Aug':           './models/smart_aug_comparison',
    'Smart Diff Aug':      './models/smart_diffusion_aug_classifier',
}

# Print header
print(f'{\"Method\":<18} {\"Top-1 (%)\":>10} {\"Top-5 (%)\":>10} {\"Top-10 (%)\":>11} {\"Best Val\":>10} {\"Train Size\":>12}')
print('-' * 75)

for name, d in results_dirs.items():
    path = os.path.join(d, 'training_results.json')
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        top1 = r.get('test_acc_top1', 0)
        top5 = r.get('test_acc_top5', 0)
        top10 = r.get('test_acc_top10', 0)
        best_val = r.get('best_val_acc_top1', 0)
        train_size = r.get('train_samples', 'N/A')
        print(f'{name:<18} {top1:>10.2f} {top5:>10.2f} {top10:>11.2f} {best_val:>10.2f} {str(train_size):>12}')
    else:
        print(f'{name:<18} {\"—\":>10} {\"—\":>10} {\"—\":>11} {\"—\":>10} {\"—\":>12}  (not found)')

# Add retrieval baseline
retrieval_path = os.path.join('$PROJECT_ROOT', 'retrieval', 'evaluation_results.json')
if os.path.exists(retrieval_path):
    with open(retrieval_path) as f:
        ret = json.load(f)
    print(f'{\"Retrieval\":<18} {ret[\"top1_accuracy\"]:>10.2f} {ret[\"top5_accuracy\"]:>10.2f} {ret[\"top10_accuracy\"]:>11.2f} {\"—\":>10} {\"—\":>12}')
"
