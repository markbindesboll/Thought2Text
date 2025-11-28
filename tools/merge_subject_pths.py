#!/usr/bin/env python3
"""
Merge multiple per-subject Thought2Text .pth files into a single dataset and
assign subject IDs based on filename order.

Usage:
  python tools/merge_subject_pths.py --input-dir /path/to/converted --pattern "sub-*.pth" --output data/all_subjects.pth

The script expects each input .pth to contain a dict with key "dataset" (list of sample dicts),
and will set each sample['subject'] to the subject index (1-based) according to sorted filenames.
It will also concatenate labels and keep images as ["placeholder"].
"""
import argparse
import glob
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="sub-*.pth")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not paths:
        raise SystemExit(f"No files matched in {args.input_dir} with pattern {args.pattern}")

    combined_dataset = []
    combined_labels = []
    images = ["placeholder"]

    for idx, p in enumerate(paths, start=1):
        print(f"Loading {p} as subject {idx}")
        data = torch.load(p)
        ds = data.get("dataset", None)
        if ds is None:
            raise SystemExit(f"File {p} does not contain 'dataset' key")
        for sample in ds:
            sample["subject"] = idx
            combined_dataset.append(sample)
        lbls = data.get("labels", None)
        if lbls is not None:
            combined_labels.extend(lbls)

    out = {"dataset": combined_dataset, "labels": combined_labels, "images": images}
    print(f"Saving combined dataset with {len(combined_dataset)} samples to {args.output}")
    torch.save(out, args.output)
    print("Done.")


if __name__ == '__main__':
    main()
