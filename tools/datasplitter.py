#!/usr/bin/env python3
"""
Strict builder for block_splits_by_image_all.pth (ZERO-INDEXED)

ASSUMPTIONS (hard-coded, validated):
- Train/val images: 0..16539 have exactly 4 repetitions each
- Test images:      16540..16739 have exactly 80 repetitions each
- Dataset is ordered contiguously by image ID (blocks)
- Splits are done at IMAGE-ID level, expanded to DATASET INDICES

If ANY assumption fails → script exits with error.
"""

import argparse
import random
import sys
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eeg_dataset", type=str, required=True, help="Path to eeg_dataset.pth")
    p.add_argument(
        "--out",
        type=str,
        default="./data/block/block_splits_by_image_all.pth",
        help="Output .pth path",
    )
    p.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of train/val images to use for val")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def main():
    args = parse_args()
    random.seed(args.seed)

    loaded = torch.load(args.eeg_dataset, map_location="cpu",weights_only=False)
    if "dataset" not in loaded:
        die("eeg_dataset does not contain key 'dataset'")

    dataset = loaded["dataset"]

    # ---- HARD-CODED DATASET STRUCTURE (ZERO-INDEXED IMAGE IDS) ----
    TRAINVAL_IMAGE_START = 0
    TRAINVAL_IMAGE_END = 16539  # inclusive
    TEST_IMAGE_START = 16540
    TEST_IMAGE_END = 16739      # inclusive

    TRAINVAL_REPS = 4
    TEST_REPS = 80

    n_trainval_images = TRAINVAL_IMAGE_END - TRAINVAL_IMAGE_START + 1
    n_test_images = TEST_IMAGE_END - TEST_IMAGE_START + 1

    expected_len = (n_trainval_images * TRAINVAL_REPS) + (n_test_images * TEST_REPS)
    if len(dataset) != expected_len:
        die(f"Dataset length mismatch. Expected {expected_len}, got {len(dataset)}")

    # ---- VALIDATE CONTIGUOUS IMAGE BLOCKS ----
    print("Validating dataset ordering and repetitions...")

    idx = 0

    # Train/val block: image 0..16539 each repeated 4 times
    for img_id in range(TRAINVAL_IMAGE_START, TRAINVAL_IMAGE_END + 1):
        for _ in range(TRAINVAL_REPS):
            got = int(dataset[idx]["image"])
            if got != img_id:
                die(f"Ordering violation at dataset index {idx}: expected image {img_id}, got {got}")
            idx += 1

    # Test block: image 16540..16739 each repeated 80 times
    for img_id in range(TEST_IMAGE_START, TEST_IMAGE_END + 1):
        for _ in range(TEST_REPS):
            got = int(dataset[idx]["image"])
            if got != img_id:
                die(f"Ordering violation at dataset index {idx}: expected image {img_id}, got {got}")
            idx += 1

    if idx != len(dataset):
        die(f"Internal error: validation consumed {idx} samples, dataset has {len(dataset)}")

    print("OK: Dataset ordering and repetition counts validated.")

    # ---- IMAGE-LEVEL TRAIN/VAL SPLIT ----
    trainval_image_ids = list(range(TRAINVAL_IMAGE_START, TRAINVAL_IMAGE_END + 1))
    random.shuffle(trainval_image_ids)

    n_val = int(round(args.val_ratio * len(trainval_image_ids)))
    val_image_ids = set(trainval_image_ids[:n_val])
    train_image_ids = set(trainval_image_ids[n_val:])

    # ---- EXPAND IMAGE IDS → DATASET INDICES ----
    train_idx, val_idx, test_idx = [], [], []

    def add_block(start_idx: int, reps: int, target_list: list[int]):
        target_list.extend(range(start_idx, start_idx + reps))

    # Train/val blocks (start index = img_id * 4)
    for img_id in range(TRAINVAL_IMAGE_START, TRAINVAL_IMAGE_END + 1):
        start = img_id * TRAINVAL_REPS
        if img_id in train_image_ids:
            add_block(start, TRAINVAL_REPS, train_idx)
        elif img_id in val_image_ids:
            add_block(start, TRAINVAL_REPS, val_idx)
        else:
            die(f"Image {img_id} not assigned to train or val (should never happen)")

    # Test blocks start right after all train/val samples
    test_block_start = n_trainval_images * TRAINVAL_REPS  # = 16540*4 = 66160
    for img_id in range(TEST_IMAGE_START, TEST_IMAGE_END + 1):
        start = test_block_start + (img_id - TEST_IMAGE_START) * TEST_REPS
        add_block(start, TEST_REPS, test_idx)

    # ---- FINAL SANITY CHECKS ----
    set_train, set_val, set_test = set(train_idx), set(val_idx), set(test_idx)
    if set_train & set_val or set_train & set_test or set_val & set_test:
        die("Dataset index leakage between splits")

    for name, lst in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if not lst:
            die(f"{name} split is empty")
        if min(lst) < 0 or max(lst) >= len(dataset):
            die(f"{name} indices out of bounds: min={min(lst)}, max={max(lst)}, len(dataset)={len(dataset)}")

    # ---- SAVE ----
    splits = {"splits": [{"train": train_idx, "val": val_idx, "test": test_idx}]}
    torch.save(splits, args.out)

    print("Saved:", args.out)
    print(f"Train images: {len(train_image_ids)} → {len(train_idx)} samples")
    print(f"Val images:   {len(val_image_ids)} → {len(val_idx)} samples")
    print(f"Test images:  {n_test_images} → {len(test_idx)} samples")
    print("DONE")


if __name__ == "__main__":
    main()
