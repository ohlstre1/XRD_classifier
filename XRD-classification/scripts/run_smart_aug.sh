#!/bin/bash
set -e

# ============================================================
# Run Smart Class-Aware Augmentation + Print Comparison Table
# ============================================================
# Runs smart aug training (lower intensity + class-aware),
# then prints a 4-way comparison table with existing results.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$SCRIPT_DIR"
source "$PROJECT_ROOT/.venv/bin/activate"

# Shared hyperparameters (must match run_comparison.sh)
DATA_PATH="$PROJECT_ROOT/data/xrd_dataset_labeled_dtw_window.pt"
EPOCHS=200
BATCH_SIZE=32
LR=1e-4
DROPOUT=0.3
SEED=42
NUM_VARIATIONS=3
AUG_THRESHOLD=10

echo "============================================================"
echo "  Smart Class-Aware Augmentation Training"
echo "============================================================"
echo "Dataset:       $DATA_PATH"
echo "Epochs:        $EPOCHS"
echo "Batch size:    $BATCH_SIZE"
echo "LR:            $LR"
echo "Dropout:       $DROPOUT"
echo "Seed:          $SEED"
echo "Variations:    $NUM_VARIATIONS (underrepresented classes only)"
echo "Aug threshold: $AUG_THRESHOLD (classes with fewer samples get augmented)"
echo "============================================================"
echo ""

# ----------------------------------------------------------
# Run: Smart Augmentation ResNet
# ----------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: Smart Class-Aware Augmentation ResNet"
mkdir -p ./models/smart_aug_comparison

python train_resnet_smart_aug.py \
  --data_path "$DATA_PATH" \
  --num_variations $NUM_VARIATIONS \
  --aug_threshold $AUG_THRESHOLD \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --dropout $DROPOUT \
  --seed $SEED \
  --save_dir ./models/smart_aug_comparison \
  --disable_wandb \
  2>&1 | tee ./models/smart_aug_comparison/train.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Smart augmentation training complete."
echo ""

# ----------------------------------------------------------
# 4-Way Comparison Table
# ----------------------------------------------------------
echo "============================================================"
echo "  Results Comparison (4-Way)"
echo "============================================================"

python3 -c "
import json, os

results_dirs = {
    'Baseline':       './models/baseline_comparison',
    'Classical Aug':  './models/classical_aug_comparison',
    'Smart Aug':      './models/smart_aug_comparison',
    'Diffusion Aug':  './models/diffusion_aug_comparison',
}

metrics = []
for name, d in results_dirs.items():
    path = os.path.join(d, 'training_results.json')
    if os.path.exists(path):
        with open(path) as f:
            r = json.load(f)
        metrics.append((name, r))
    else:
        print(f'WARNING: {path} not found')

if not metrics:
    print('No results found!')
    exit(1)

# Print header
print(f'{\"Method\":<18} {\"Top-1 (%)\":>10} {\"Top-5 (%)\":>10} {\"Top-10 (%)\":>11} {\"Best Val\":>10} {\"Train Size\":>12}')
print('-' * 75)

for name, r in metrics:
    top1 = r.get('test_acc_top1', 0)
    top5 = r.get('test_acc_top5', 0)
    top10 = r.get('test_acc_top10', 0)
    best_val = r.get('best_val_acc_top1', 0)
    train_size = r.get('train_samples', 'N/A')
    print(f'{name:<18} {top1:>10.2f} {top5:>10.2f} {top10:>11.2f} {best_val:>10.2f} {str(train_size):>12}')

# Print smart aug details if available
smart_path = './models/smart_aug_comparison/training_results.json'
if os.path.exists(smart_path):
    with open(smart_path) as f:
        sr = json.load(f)
    if 'aug_strategy' in sr:
        s = sr['aug_strategy']
        print()
        print('Smart Aug Details:')
        print(f'  Classes augmented: {s[\"classes_augmented\"]} / {s[\"classes_augmented\"] + s[\"classes_skipped\"]}')
        print(f'  Augmented copies:  {s[\"augmented_copies\"]}')
        print(f'  Total synthetic:   {s[\"total_synthetic\"]}')

# Add retrieval baseline
retrieval_path = os.path.join('$PROJECT_ROOT', 'retrieval', 'evaluation_results.json')
if os.path.exists(retrieval_path):
    with open(retrieval_path) as f:
        ret = json.load(f)
    print(f'{\"Retrieval\":<18} {ret[\"top1_accuracy\"]:>10.2f} {ret[\"top5_accuracy\"]:>10.2f} {ret[\"top10_accuracy\"]:>11.2f} {\"—\":>10} {\"—\":>12}')
"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
