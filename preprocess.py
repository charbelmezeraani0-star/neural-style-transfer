"""
Dataset preprocessing for Fast NST training.

What this does:
  1. Scans the dataset directory for all images
  2. Validates each image (catches corruption, truncation, decode errors)
  3. Filters images below a minimum resolution
  4. Writes a clean manifest (valid_images.txt) for fast training startup
  5. Reports statistics + size distribution

Usage:
    python preprocess.py --dataset ~/Downloads/train2014
    python preprocess.py --dataset ~/Downloads/train2014 --min-size 256 --workers 16
"""

import os
import sys
import time
import argparse
import concurrent.futures
from pathlib import Path
from collections import Counter

from PIL import Image

EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


# ── Per-image validation ──────────────────────────────────────────────────────

def validate(args_tuple):
    """
    Returns (path_str, status, (w, h)).
    status: 'ok' | 'small' | 'corrupt'
    """
    path, min_size = args_tuple
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w < min_size or h < min_size:
                return str(path), 'small', (w, h)
            img.load()          # force full pixel decode — catches truncated jpegs
            if img.mode not in ('RGB', 'RGBA', 'L', 'P'):
                return str(path), 'corrupt', (w, h)
        return str(path), 'ok', (w, h)
    except Exception:
        return str(path), 'corrupt', (0, 0)


# ── Size-bucket helper ────────────────────────────────────────────────────────

def size_bucket(w, h):
    shorter = min(w, h)
    for threshold in (128, 256, 384, 512, 768, 1024):
        if shorter < threshold:
            return f'<{threshold}px'
    return '≥1024px'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Validate a dataset and write a clean image manifest.')
    ap.add_argument('--dataset',   required=True,
                    help='Root directory of the image dataset')
    ap.add_argument('--min-size',  type=int, default=256,
                    help='Minimum width AND height in pixels (default: 256)')
    ap.add_argument('--workers',   type=int, default=8,
                    help='Parallel validation workers (default: 8)')
    ap.add_argument('--out',       default=None,
                    help='Output manifest path (default: <dataset-parent>/valid_images.txt)')
    args = ap.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.exists():
        print(f'ERROR: dataset directory not found: {dataset}')
        sys.exit(1)

    out_manifest = Path(args.out) if args.out else dataset.parent / 'valid_images.txt'
    out_corrupt  = dataset.parent / 'corrupt_images.txt'

    # ── Scan ─────────────────────────────────────────────────
    print(f'\n{"═"*60}')
    print(f'  NST Dataset Preprocessor')
    print(f'{"═"*60}')
    print(f'  Dataset  : {dataset}')
    print(f'  Min size : {args.min_size}×{args.min_size} px')
    print(f'  Workers  : {args.workers}')
    print(f'  Output   : {out_manifest}')
    print(f'{"═"*60}\n')

    print('Scanning for images...')
    all_paths = sorted([p for p in dataset.rglob('*') if p.suffix.lower() in EXTS])
    total = len(all_paths)
    if total == 0:
        print(f'ERROR: no images found under {dataset}')
        sys.exit(1)
    print(f'Found {total:,} images. Validating...\n')

    # ── Validate in parallel ──────────────────────────────────
    valid, small, corrupt = [], [], []
    buckets = Counter()
    done = 0
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(validate, (p, args.min_size)): p
            for p in all_paths
        }
        for future in concurrent.futures.as_completed(futures):
            path_str, status, (w, h) = future.result()
            done += 1

            if status == 'ok':
                valid.append(path_str)
                buckets[size_bucket(w, h)] += 1
            elif status == 'small':
                small.append(path_str)
            else:
                corrupt.append(path_str)

            if done % 2000 == 0 or done == total:
                elapsed = time.time() - t0
                pct  = done / total * 100
                rate = done / elapsed if elapsed > 0 else 0
                eta  = (total - done) / rate if rate > 0 else 0
                print(
                    f'  {done:>6}/{total}  ({pct:5.1f}%)  '
                    f'valid={len(valid):,}  '
                    f'small={len(small):,}  '
                    f'corrupt={len(corrupt):,}  '
                    f'eta={eta:>4.0f}s  '
                    f'{rate:>5.0f} img/s',
                    end='\r'
                )

    elapsed = time.time() - t0
    print()

    # ── Report ────────────────────────────────────────────────
    print(f'\n{"═"*60}')
    print(f'  RESULTS')
    print(f'{"═"*60}')
    print(f'  Total scanned   : {total:,}')
    print(f'  Valid           : {len(valid):,}  ({len(valid)/total*100:.1f}%)')
    print(f'  Too small       : {len(small):,}  (< {args.min_size}px on shortest side)')
    print(f'  Corrupt/unread  : {len(corrupt):,}')
    print(f'  Time            : {elapsed:.1f}s  ({total/elapsed:.0f} img/s)')
    print()

    if buckets:
        print('  Size distribution of valid images:')
        for bucket in ['<128px', '<256px', '<384px', '<512px', '<768px', '<1024px', '≥1024px']:
            count = buckets.get(bucket, 0)
            if count:
                bar = '█' * min(30, count * 30 // len(valid))
                print(f'    {bucket:>9}  {bar:<30}  {count:>6,}')
    print(f'{"═"*60}')

    # ── Write manifest ────────────────────────────────────────
    valid.sort()
    with open(out_manifest, 'w') as f:
        f.write('\n'.join(valid) + '\n')
    print(f'\n  Manifest → {out_manifest}  ({len(valid):,} paths)')

    if corrupt:
        with open(out_corrupt, 'w') as f:
            f.write('\n'.join(sorted(corrupt)) + '\n')
        print(f'  Corrupt  → {out_corrupt}  ({len(corrupt):,} paths)')

    print(f'\n  Run training with:')
    print(f'    bash start_training.sh')
    print(f'  (manifest is picked up automatically)\n')


if __name__ == '__main__':
    main()
