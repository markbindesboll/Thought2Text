#!/usr/bin/env python3
"""
Convert a preprocessed EEG file into the Thought2Text torch dataset format.

Expected output format (saved with torch.save):
  {
    "dataset": [ {"eeg": Tensor(channels, time), "label": int, "image": int, "subject": int}, ... ],
    "labels": [...],
    "images": [...]
  }

This script handles inputs saved as NumPy (.npz/.npy) or Torch (.pt/.pth).
It looks for a key named like `preprocessed_eeg_data` (case-insensitive) and
expects shape (trials, reps, channels, time). You can choose to flatten reps
as separate samples or average across reps.

Usage examples (on HPC):
  python tools/convert_eeg_to_thought2text.py \
    --input /path/to/my_data.pth \
    --output /path/to/data/my_eeg_dataset.pth \
    --treat-reps separate \
    --trials-per-subject 1654

If you have a per-trial subject index array (length==trials), provide
`--subjects-file subjects.npy` (NumPy .npy or .npz or torch .pth) to tag each original trial.
If not provided, subject will default to 0 (meaning 'all subjects' in repo args).

"""
import argparse
import os
import sys
import json
from typing import Optional

import numpy as np
import torch


def _find_key(d, candidates):
    if not isinstance(d, dict):
        return None
    keys = list(d.keys())
    lower = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_input(path: str):
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".npz", ".npz"):
        arr = np.load(path, allow_pickle=True)
        return dict(arr)
    if ext in (".npy",):
        arr = np.load(path, allow_pickle=True)
        return {"arr": arr}
    # otherwise try torch.load
    try:
        data = torch.load(path)
        # If it's a tensor or ndarray, wrap
        if isinstance(data, (torch.Tensor, np.ndarray)):
            return {"arr": data}
        return data
    except Exception as e:
        raise RuntimeError(f"Could not load input file {path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert EEG to Thought2Text torch dataset")
    parser.add_argument("--input", required=True, help="Input .pth/.pt/.npz/.npy file")
    parser.add_argument("--output", required=True, help="Output torch .pth file to write")
    parser.add_argument(
        "--treat-reps",
        choices=("separate", "average"),
        default="separate",
        help="If input has reps dimension, either treat each rep as a separate sample or average across reps",
    )
    parser.add_argument(
        "--subjects-file",
        default=None,
        help="Optional .npy/.npz/.pth file containing per-trial subject ids (length == trials)",
    )
    parser.add_argument(
        "--trials-per-subject",
        type=int,
        default=0,
        help="If trials are concatenated by subject and all subjects have equal trials, use this to auto-assign subject ids",
    )
    parser.add_argument(
        "--labels-file",
        default=None,
        help="Optional per-trial labels file (npy/pth) matching trials dimension",
    )
    args = parser.parse_args()

    print(f"Loading input: {args.input}")
    raw = load_input(args.input)

    # Find eeg array key
    eeg_key = _find_key(raw, ["preprocessed_eeg_data", "preprocessed", "arr", "data"])
    if eeg_key is None:
        raise SystemExit("Could not find EEG data array in the input file. Expected key like 'preprocessed_eeg_data' or 'arr'.")

    eeg_arr = raw[eeg_key]
    # Ensure numpy array
    if isinstance(eeg_arr, torch.Tensor):
        eeg_arr = eeg_arr.cpu().numpy()

    if eeg_arr.ndim != 4:
        raise SystemExit(f"Expected EEG array with 4 dims (trials, reps, channels, time). Got shape {eeg_arr.shape}")

    trials, reps, channels, time = eeg_arr.shape
    print(f"Detected EEG array shape: trials={trials}, reps={reps}, channels={channels}, time={time}")

    # Load optional subjects array
    subjects = None
    if args.subjects_file:
        subj_raw = load_input(args.subjects_file)
        # pick first array-like
        key = _find_key(subj_raw, ["arr", "subjects", "subject_ids", "subs"]) or (list(subj_raw.keys())[0] if isinstance(subj_raw, dict) else None)
        if key is None:
            raise SystemExit("Could not find subject array inside subjects-file")
        subjects = np.array(subj_raw[key])
        if subjects.shape[0] != trials:
            print(f"Warning: subjects array length {subjects.shape[0]} != trials {trials}.")
            # allow if subjects length equals trials*reps when treating reps separate
            if args.treat_reps == "separate" and subjects.shape[0] == trials * reps:
                subjects = subjects.reshape(trials, reps)[:, 0]
            else:
                raise SystemExit("subjects array length mismatch")

    # If trials_per_subject provided, generate subjects sequentially
    if args.trials_per_subject > 0:
        tps = args.trials_per_subject
        ns = int(np.ceil(trials / tps))
        subjects = np.repeat(np.arange(1, ns + 1), tps)[:trials]
        print(f"Assigned subjects 1..{ns} with trials-per-subject={tps}")

    # Load optional labels
    labels_arr = None
    if args.labels_file:
        lab_raw = load_input(args.labels_file)
        key = _find_key(lab_raw, ["arr", "labels", "label"]) or (list(lab_raw.keys())[0] if isinstance(lab_raw, dict) else None)
        labels_arr = np.array(lab_raw[key])
        if labels_arr.shape[0] != trials:
            raise SystemExit("labels array length != trials")

    dataset_list = []
    images_list = ["placeholder"]
    labels_list = []

    sample_count = 0
    for ti in range(trials):
        subj_id = int(subjects[ti]) if subjects is not None else 0
        lab = int(labels_arr[ti]) if labels_arr is not None else 0

        if args.treat_reps == "separate":
            for ri in range(reps):
                arr = eeg_arr[ti, ri]
                # convert to float32
                a = np.asarray(arr, dtype=np.float32)
                # ensure shape (channels, time)
                if a.shape != (channels, time):
                    # try transpose if incoming shape is (time, channels)
                    if a.shape == (time, channels):
                        a = a.T
                    else:
                        raise SystemExit(f"Unexpected inner shape at trial {ti} rep {ri}: {a.shape}")
                tensor = torch.from_numpy(a)
                sample = {"eeg": tensor, "label": lab, "image": 0, "subject": subj_id}
                dataset_list.append(sample)
                labels_list.append(lab)
                sample_count += 1
        else:  # average
            arr = eeg_arr[ti].mean(axis=0)
            a = np.asarray(arr, dtype=np.float32)
            if a.shape != (channels, time):
                if a.shape == (time, channels):
                    a = a.T
                else:
                    raise SystemExit(f"Unexpected averaged shape at trial {ti}: {a.shape}")
            tensor = torch.from_numpy(a)
            sample = {"eeg": tensor, "label": lab, "image": 0, "subject": subj_id}
            dataset_list.append(sample)
            labels_list.append(lab)
            sample_count += 1

    print(f"Created {sample_count} samples. Saving to {args.output}")
    out = {"dataset": dataset_list, "labels": labels_list, "images": images_list}
    torch.save(out, args.output)
    print("Saved output successfully.")


if __name__ == "__main__":
    main()
