#!/usr/bin/env python3
"""
Strict builder for block_splits_by_image_all.pth

ASSUMPTIONS (hard-coded, validated):
- Images 1..16540 have exactly 4 repetitions each
- Images 16541..16740 have exactly 80 repetitions each
- Dataset is ordered contiguously by image ID
- Split is done at IMAGE-ID level, expanded to DATASET INDICES

If ANY assumption fails → script exits with error.
"""

import argparse
import random
import sys
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eeg_dataset", type=str, required=True)
    p.add_argument("--out", type=str, default="./data/block/block_splits_by_image_all.pth")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    loaded = torch.load(args.eeg_dataset, map_location="cpu")
    if "dataset" not in loaded:
        print("ERROR: eeg_dataset does not contain key 'dataset'", file=sys.stderr)
        sys.exit(1)

    dataset = loaded["dataset"]

    # ---- HARD-CODED DATASET STRUCTURE ----
    TRAINVAL_IMAGE_START = 1
    TRAINVAL_IMAGE_END = 16540
    TEST_IMAGE_START = 16541
    TEST_IMAGE_END = 16740

    TRAINVAL_REPS = 4
    TEST_REPS = 80

    expected_len = (TRAINVAL_IMAGE_END * TRAINVAL_REPS) + (
        (TEST_IMAGE_END - TEST_IMAGE_START + 1) * TEST_REPS
    )

    if len(dataset) != expected_len:
        print(
            f"ERROR: Dataset length mismatch. "
            f"Expected {expected_len}, got {len(dataset)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- VALIDATE CONTIGUOUS IMAGE BLOCKS ----
    print("Validating dataset ordering and repetitions...")

    idx = 0

    # Train / val images
    for img_id in range(TRAINVAL_IMAGE_START, TRAINVAL_IMAGE_END + 1):
        for r in range(TRAINVAL_REPS):
            if int(dataset[idx]["image"]) != img_id:
                print(
                    f"ERROR: Dataset ordering violation at index {idx}. "
                    f"Expected image {img_id}, got {dataset[idx]['image']}",
                    file=sys.stderr,
                )
                sys.exit(1)
            idx += 1

    # Test images
    for img_id in range(TEST_IMAGE_START, TEST_IMAGE_END + 1):
        for r in range(TEST_REPS):
            if int(dataset[idx]["image"]) != img_id:
                print(
                    f"ERROR: Dataset ordering violation at index {idx}. "
                    f"Expected image {img_id}, got {dataset[idx]['image']}",
                    file=sys.stderr,
                )
                sys.exit(1)
            idx += 1

    print("OK: Dataset ordering and repetition counts validated.")

    # ---- IMAGE-LEVEL TRAIN / VAL SPLIT ----
    trainval_image_ids = list(range(TRAINVAL_IMAGE_START, TRAINVAL_IMAGE_END + 1))
    random.shuffle(trainval_image_ids)

    n_val = int(round(args.val_ratio * len(trainval_image_ids)))
    val_image_ids = set(trainval_image_ids[:n_val])
    train_image_ids = set(trainval_image_ids[n_val:])
    test_image_ids = set(range(TEST_IMAGE_START, TEST_IMAGE_END + 1))

    # ---- EXPAND IMAGE IDS → DATASET INDICES ----
    train_idx, val_idx, test_idx = [], [], []

    def add_block(start_idx, reps, target_list):
        for i in range(start_idx, start_idx + reps):
            target_list.append(i)

    # Train / val blocks
    for img_id in range(TRAINVAL_IMAGE_START, TRAINVAL_IMAGE_END + 1):
        start = (img_id - 1) * TRAINVAL_REPS
        if img_id in train_image_ids:
            add_block(start, TRAINVAL_REPS, train_idx)
        elif img_id in val_image_ids:
            add_block(start, TRAINVAL_REPS, val_idx)
        else:
            print(f"ERROR: Image {img_id} not assigned to train or val", file=sys.stderr)
            sys.exit(1)

    # Test blocks
    test_block_start = TRAINVAL_IMAGE_END * TRAINVAL_REPS
    for img_id in range(TEST_IMAGE_START, TEST_IMAGE_END + 1):
        start = test_block_start + (img_id - TEST_IMAGE_START) * TEST_REPS
        add_block(start, TEST_REPS, test_idx)

    # ---- FINAL SANITY CHECKS ----
    if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
        print("ERROR: Dataset index leakage between splits", file=sys.stderr)
        sys.exit(1)

    # ---- SAVE ----
    splits = {
        "splits": [
            {
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            }
        ]
    }

    torch.save(splits, args.out)

    print("Saved:", args.out)
    print(f"Train images: {len(train_image_ids)} → {len(train_idx)} samples")
    print(f"Val images:   {len(val_image_ids)} → {len(val_idx)} samples")
    print(f"Test images:  {len(test_image_ids)} → {len(test_idx)} samples")
    print("DONE")


if __name__ == "__main__":
    main()
