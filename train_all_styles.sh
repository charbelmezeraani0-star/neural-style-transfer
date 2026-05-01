#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Train one Fast-NST model per style image in NST/images/styles/
#
#  Usage:
#    bash train_all_styles.sh           # train all styles
#    bash train_all_styles.sh watercolor # train only watercolor
# ─────────────────────────────────────────────────────────────

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="$PROJECT_DIR/train2014"
MANIFEST="$PROJECT_DIR/valid_images.txt"
STYLES_DIR="$PROJECT_DIR/NST/images/styles"
FILTER="${1:-}"   # optional: only train styles matching this name

echo "========================================"
echo "  Fast NST — Multi-Style Training"
echo "========================================"
echo ""

# ── Validate dataset ──────────────────────────────────────────
if [ ! -d "$DATASET" ]; then
  echo "ERROR: Dataset not found at $DATASET"
  echo "Download MS-COCO train2014 first."
  exit 1
fi

# ── Preprocess if manifest missing ───────────────────────────
if [ ! -f "$MANIFEST" ]; then
  echo "First-time setup: validating dataset images..."
  python3 "$PROJECT_DIR/preprocess.py" \
    --dataset "$DATASET" --out "$MANIFEST" --workers 8
  echo ""
fi

# ── Collect style images ──────────────────────────────────────
STYLES=()
for f in "$STYLES_DIR"/*.jpg "$STYLES_DIR"/*.jpeg "$STYLES_DIR"/*.png; do
  [ -f "$f" ] || continue
  BASE=$(basename "$f")
  NAME="${BASE%.*}"
  if [ -n "$FILTER" ] && [[ "$NAME" != *"$FILTER"* ]]; then
    continue
  fi
  STYLES+=("$f")
done

if [ ${#STYLES[@]} -eq 0 ]; then
  echo "No style images found in $STYLES_DIR"
  echo "Add .jpg or .png files there to train new styles."
  exit 1
fi

echo "Found ${#STYLES[@]} style(s) to train:"
for f in "${STYLES[@]}"; do echo "  • $(basename "$f")"; done
echo ""

# ── Train each style ──────────────────────────────────────────
TOTAL=${#STYLES[@]}
IDX=0

for STYLE_FILE in "${STYLES[@]}"; do
  IDX=$((IDX + 1))
  BASE=$(basename "$STYLE_FILE")
  MODEL_NAME="${BASE%.*}"
  FINAL_MODEL="$PROJECT_DIR/models/${MODEL_NAME}.pth"

  echo "════════════════════════════════════════"
  echo "  [$IDX/$TOTAL]  $MODEL_NAME"
  echo "════════════════════════════════════════"

  if [ -f "$FINAL_MODEL" ]; then
    echo "  Already trained — $FINAL_MODEL"
    echo "  Delete it to retrain."
    echo ""
    continue
  fi

  python3 "$PROJECT_DIR/train.py" \
    --style          "$STYLE_FILE" \
    --dataset        "$DATASET" \
    --manifest       "$MANIFEST" \
    --name           "$MODEL_NAME" \
    --epochs         4 \
    --batch-size     4 \
    --image-size     256 \
    --content-weight 1e5 \
    --style-weight   5e10 \
    --tv-weight      1e-6 \
    --checkpoint-every 500 \
    --sample-every   500

  echo ""
  echo "  Done → $FINAL_MODEL"
  echo ""
done

echo "========================================"
echo "  All styles trained."
echo "  Open http://localhost:5000 → Fast NST"
echo "========================================"
