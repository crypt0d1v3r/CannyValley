"""
prepare_dataset.py
------------------
Filters the bitmind/AI-vs-Real-Dataset-Images-Proper HuggingFace dataset,
removing thumbnails and low-resolution images, and saves the remaining
images to a local directory structured for PyTorch ImageFolder:

    datasets/bitmind_filtered/
        train/
            AI/
                00000.jpg
                ...
            Real/
                00000.jpg
                ...

Usage:
    python prepare_dataset.py [options]

Options:
    --min-width      Minimum image width  (default: 256)
    --min-height     Minimum image height (default: 256)
    --output-dir     Output root directory (default: datasets/bitmind_filtered)
    --split          Dataset split to process (default: train)
    --num-proc       Number of parallel workers (default: 4)
    --dry-run        Report stats only, do not save images
"""

import os
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_WIDTH  = 256
DEFAULT_MIN_HEIGHT = 256
DEFAULT_OUTPUT_DIR = os.path.join("datasets", "bitmind_filtered")
DEFAULT_SPLIT      = "train"
DEFAULT_NUM_PROC   = 4
DATASET_ID         = "bitmind/AI-vs-Real-Dataset-Images-Proper"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter low-resolution images from the bitmind dataset "
                    "and export to an ImageFolder-compatible directory."
    )
    parser.add_argument("--min-width",  type=int, default=DEFAULT_MIN_WIDTH,
                        help=f"Minimum image width in pixels (default: {DEFAULT_MIN_WIDTH})")
    parser.add_argument("--min-height", type=int, default=DEFAULT_MIN_HEIGHT,
                        help=f"Minimum image height in pixels (default: {DEFAULT_MIN_HEIGHT})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--split",      type=str, default=DEFAULT_SPLIT,
                        help=f"Dataset split to process (default: {DEFAULT_SPLIT})")
    parser.add_argument("--num-proc",   type=int, default=DEFAULT_NUM_PROC,
                        help=f"Number of parallel workers (default: {DEFAULT_NUM_PROC})")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Report filtering stats only; do not save any files.")
    return parser.parse_args()


def is_valid_image(pil_image: Image.Image, min_w: int, min_h: int) -> bool:
    """Return True if the image meets the minimum size requirements."""
    w, h = pil_image.size
    return w >= min_w and h >= min_h


def save_image(args_tuple):
    """
    Worker function: checks an image and saves it to disk if it passes the
    resolution filter.

    Returns a dict with 'kept', 'skipped', and 'error' counts.
    """
    idx, pil_image, class_name, out_class_dir, min_w, min_h, dry_run = args_tuple
    result = {"kept": 0, "skipped": 0, "error": 0}

    try:
        # Convert to RGB — handles palette / RGBA / grayscale images
        img = pil_image.convert("RGB")

        if not is_valid_image(img, min_w, min_h):
            result["skipped"] += 1
            return result

        if dry_run:
            result["kept"] += 1
            return result

        filename = f"{idx:06d}.jpg"
        save_path = os.path.join(out_class_dir, filename)
        img.save(save_path, format="JPEG", quality=95)
        result["kept"] += 1

    except Exception as e:
        print(f"\n  [WARN] Index {idx} ({class_name}): {e}", file=sys.stderr)
        result["error"] += 1

    return result


def main():
    args = parse_args()

    min_w      = args.min_width
    min_h      = args.min_height
    output_dir = args.output_dir
    split      = args.split
    num_proc   = args.num_proc
    dry_run    = args.dry_run

    print("=" * 60)
    print(f"  Dataset  : {DATASET_ID}")
    print(f"  Split    : {split}")
    print(f"  Min size : {min_w} x {min_h} px")
    print(f"  Output   : {output_dir}")
    print(f"  Workers  : {num_proc}")
    print(f"  Dry run  : {dry_run}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load the dataset (cached after first download)
    # ------------------------------------------------------------------
    print(f"\n[1/3] Loading dataset '{DATASET_ID}' split='{split}' ...")
    ds = load_dataset(DATASET_ID, split=split, trust_remote_code=False)
    label_names = ds.features["label"].names   # e.g. ["AI", "Real"]
    total = len(ds)
    print(f"      Loaded {total:,} rows. Classes: {label_names}")

    # ------------------------------------------------------------------
    # 2. Create output directories
    # ------------------------------------------------------------------
    class_dirs = {}
    if not dry_run:
        for class_name in label_names:
            class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(class_dir, exist_ok=True)
            class_dirs[class_name] = class_dir
        print(f"\n[2/3] Output directories created under: {output_dir}")
    else:
        print(f"\n[2/3] Dry-run mode — no directories will be created.")

    # ------------------------------------------------------------------
    # 3. Filter & save
    # ------------------------------------------------------------------
    print(f"\n[3/3] Processing {total:,} images with {num_proc} workers ...\n")

    kept_counts    = {name: 0 for name in label_names}
    skipped_counts = {name: 0 for name in label_names}
    error_count    = 0

    # Build task list: (idx, pil_image, class_name, out_dir, min_w, min_h, dry_run)
    # We iterate lazily to avoid loading the whole dataset into RAM at once.
    def task_generator():
        for idx, row in enumerate(ds):
            class_name  = label_names[row["label"]]
            out_class_dir = class_dirs.get(class_name, "")
            yield (idx, row["image"], class_name, out_class_dir, min_w, min_h, dry_run)

    with ThreadPoolExecutor(max_workers=num_proc) as executor:
        futures = {executor.submit(save_image, task): task[2]   # task[2] == class_name
                   for task in task_generator()}

        with tqdm(total=total, unit="img", desc="Filtering") as pbar:
            for future in as_completed(futures):
                class_name = futures[future]
                res = future.result()
                kept_counts[class_name]    += res["kept"]
                skipped_counts[class_name] += res["skipped"]
                error_count                += res["error"]
                pbar.update(1)
                pbar.set_postfix(
                    kept=sum(kept_counts.values()),
                    skipped=sum(skipped_counts.values()),
                )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    total_kept    = sum(kept_counts.values())
    total_skipped = sum(skipped_counts.values())

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Class':<10}  {'Kept':>8}  {'Filtered Out':>14}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*14}")
    for name in label_names:
        print(f"  {name:<10}  {kept_counts[name]:>8,}  {skipped_counts[name]:>14,}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*14}")
    print(f"  {'TOTAL':<10}  {total_kept:>8,}  {total_skipped:>14,}")
    if error_count:
        print(f"\n  Errors/corrupt images skipped: {error_count}")
    print(f"\n  Filter threshold : >= {min_w} x {min_h} px")
    print(f"  Reduction        : {total_skipped / total * 100:.1f}% of images removed")
    if not dry_run:
        print(f"\n  Output saved to  : {Path(output_dir).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
