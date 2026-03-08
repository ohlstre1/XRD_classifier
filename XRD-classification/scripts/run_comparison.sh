#!/bin/bash
set -e

# ============================================================
# Fair 3-Way Classifier Comparison
# ============================================================
# Runs baseline, classical-aug, and diffusion-aug ResNet
# classifiers sequentially with identical hyperparameters.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$SCRIPT_DIR"
source "$PROJECT_ROOT/.venv/bin/activate"

# Shared hyperparameters
DATA_PATH="$PROJECT_ROOT/data/xrd_dataset_labeled_dtw_window.pt"
EPOCHS=200
BATCH_SIZE=32
LR=1e-4
DROPOUT=0.3
SEED=42
NUM_VARIATIONS=5
DIFFUSION_MODEL="$PROJECT_ROOT/diffusion/models/xrd_diffusion/best_model.pth"

echo "============================================================"
echo "  3-Way Classifier Comparison"
echo "============================================================"
echo "Dataset:    $DATA_PATH"
echo "Epochs:     $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "LR:         $LR"
echo "Dropout:    $DROPOUT"
echo "Seed:       $SEED"
echo "Variations: $NUM_VARIATIONS (classical & diffusion)"
echo "============================================================"
echo ""

# ----------------------------------------------------------
# Run 1: Baseline ResNet (no augmentation)
# ----------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Run 1/3: Baseline ResNet"
mkdir -p ./models/baseline_comparison

python train_resnet_classifier.py \
  --data_path "$DATA_PATH" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --dropout $DROPOUT \
  --seed $SEED \
  --save_dir ./models/baseline_comparison \
  --disable_wandb \
  2>&1 | tee ./models/baseline_comparison/train.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run 1/3 complete: Baseline ResNet"
echo ""

# ----------------------------------------------------------
# Run 2: Classical Augmentation ResNet
# ----------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Run 2/3: Classical Augmentation ResNet"
mkdir -p ./models/classical_aug_comparison

python train_resnet_classical_aug.py \
  --data_path "$DATA_PATH" \
  --num_variations $NUM_VARIATIONS \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --dropout $DROPOUT \
  --seed $SEED \
  --save_dir ./models/classical_aug_comparison \
  --disable_wandb \
  2>&1 | tee ./models/classical_aug_comparison/train.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run 2/3 complete: Classical Augmentation ResNet"
echo ""

# ----------------------------------------------------------
# Run 3: Diffusion Augmentation ResNet
# ----------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Run 3/3: Diffusion Augmentation ResNet"
mkdir -p ./models/diffusion_aug_comparison

python train_resnet_diffusion_aug.py \
  --data_path "$DATA_PATH" \
  --diffusion_model_path "$DIFFUSION_MODEL" \
  --num_variations $NUM_VARIATIONS \
  --diffusion_batch_size 32 \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --dropout $DROPOUT \
  --seed $SEED \
  --save_dir ./models/diffusion_aug_comparison \
  --disable_wandb \
  2>&1 | tee ./models/diffusion_aug_comparison/train.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run 3/3 complete: Diffusion Augmentation ResNet"
echo ""

# ----------------------------------------------------------
# Comparison Table
# ----------------------------------------------------------
echo "============================================================"
echo "  Results Comparison"
echo "============================================================"

python3 -c "
import json, os

results_dirs = {
    'Baseline':       './models/baseline_comparison',
    'Classical Aug':  './models/classical_aug_comparison',
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

# Add retrieval baseline
retrieval_path = os.path.join('$PROJECT_ROOT', 'retrieval', 'evaluation_results.json')
if os.path.exists(retrieval_path):
    with open(retrieval_path) as f:
        ret = json.load(f)
    print(f'{\"Retrieval\":<18} {ret[\"top1_accuracy\"]:>10.2f} {ret[\"top5_accuracy\"]:>10.2f} {ret[\"top10_accuracy\"]:>11.2f} {\"—\":>10} {\"—\":>12}')
"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All runs complete."
