#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Fast NST Training
#
#  First run:   bash start_training.sh
#  Resume:      bash start_training.sh --resume
# ─────────────────────────────────────────────────────────────

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="$PROJECT_DIR/train2014"
MANIFEST="$PROJECT_DIR/valid_images.txt"
ZIPFILE="$HOME/Downloads/train2014.zip"
STYLE="$PROJECT_DIR/NST/images/styles/pencil_style_img1.jpg"
MODEL_NAME="pencil_sketch"
RESUME_CKPT="$PROJECT_DIR/models/${MODEL_NAME}_resume.pth"

echo "========================================"
echo "  Fast NST Training — NST Studio"
echo "========================================"
echo ""

# ── Unzip if needed ───────────────────────────────────────────
if [ ! -d "$DATASET" ]; then
  if [ ! -f "$ZIPFILE" ]; then
    echo "ERROR: $ZIPFILE not found."
    echo "Make sure the COCO download finished and is in ~/Downloads/"
    exit 1
  fi
  echo "Unzipping dataset (this may take a few minutes)..."
  unzip -q "$ZIPFILE" -d "$PROJECT_DIR/"
  echo "Done."
fi

COUNT=$(find "$DATASET" -maxdepth 1 -name "*.jpg" | wc -l)
echo "Dataset : $DATASET  ($COUNT images)"
echo ""

# ── Preprocess if manifest doesn't exist ─────────────────────
if [ ! -f "$MANIFEST" ]; then
  echo "First-time setup: validating dataset images..."
  echo "(Removes corrupt/too-small images. Runs once, takes ~2 min)"
  echo ""
  python3 "$PROJECT_DIR/preprocess.py" \
    --dataset "$DATASET" \
    --out     "$MANIFEST" \
    --min-size 256        \
    --workers 8
  echo ""
else
  VALID_COUNT=$(wc -l < "$MANIFEST")
  echo "Manifest : $MANIFEST  ($VALID_COUNT valid images)"
  echo ""
fi

# ── Controls reminder ─────────────────────────────────────────
echo "Controls:"
echo "  Pause  →  touch \"$PROJECT_DIR/training.pause\""
echo "  Resume →  rm    \"$PROJECT_DIR/training.pause\""
echo "  Stop   →  Ctrl+C  (checkpoint auto-saved)"
echo ""
echo "Estimated time on RTX 4050: ~3 hours"
echo ""

# ── Build base command ────────────────────────────────────────
CMD="python3 train.py \
  --style          \"$STYLE\" \
  --dataset        \"$DATASET\" \
  --manifest       \"$MANIFEST\" \
  --name           $MODEL_NAME \
  --epochs         2 \
  --batch-size     8 \
  --image-size     256 \
  --content-weight 1e5 \
  --style-weight   1e10 \
  --stats-weight   1e6 \
  --tv-weight      1e-6 \
  --checkpoint-every 500 \
  --sample-every     500"

# ── Append --resume if checkpoint exists and flag passed ──────
if [ "$1" = "--resume" ]; then
  if [ ! -f "$RESUME_CKPT" ]; then
    echo "ERROR: No resume checkpoint found at $RESUME_CKPT"
    echo "Start training first with:  bash start_training.sh"
    exit 1
  fi
  CMD="$CMD --resume \"$RESUME_CKPT\""
  echo "Resuming from checkpoint: $RESUME_CKPT"
  echo ""
fi

# ── Run ───────────────────────────────────────────────────────
cd "$PROJECT_DIR"
eval $CMD

echo ""
echo "========================================"
echo "  Done! Open http://localhost:5000"
echo "  → Fast NST tab → select pencil_sketch"
echo "========================================"
