#!/usr/bin/env python3
"""
Plots:
1) Training loss per step (dense curve)
2) Val/Test loss per epoch (separate plot)

Assumes:
  data/runs/<run_name>/checkpoint-*/trainer_state.json
"""

import argparse
import json
import os
import glob
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name under data/runs/ (e.g. sub01_encode_only)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.join("data", "runs", args.run_name)

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # ---- find latest checkpoint ----
    checkpoints = sorted(
        glob.glob(os.path.join(run_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
    )
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir}")

    ckpt_dir = checkpoints[-1]
    state_path = os.path.join(ckpt_dir, "trainer_state.json")

    with open(state_path, "r") as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    if not logs:
        raise RuntimeError("No log_history found in trainer_state.json")

    # ------------------------------------------------------------------
    # Plot 1: TRAINING LOSS PER STEP (original behavior)
    # ------------------------------------------------------------------
    train_steps = []
    train_loss = []

    for entry in logs:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])

    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, label="Train loss", linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training curves – {args.run_name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_train = os.path.join(ckpt_dir, "training_curves.png")
    plt.savefig(out_train, dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: VAL + TEST PER EPOCH ONLY
    # ------------------------------------------------------------------
    val_epochs, val_loss = [], []
    test_epochs, test_loss = [], []

    for entry in logs:
        if "eval_loss" in entry and "epoch" in entry:
            val_epochs.append(int(entry["epoch"]))
            val_loss.append(entry["eval_loss"])

        if "test_loss" in entry and "epoch" in entry:
            test_epochs.append(int(entry["epoch"]))
            test_loss.append(entry["test_loss"])

    plt.figure(figsize=(10, 6))

    if val_epochs:
        plt.plot(val_epochs, val_loss, marker="o", linestyle="--", label="Val loss")

    if test_epochs:
        plt.plot(test_epochs, test_loss, marker="s", linestyle=":", label="Test loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Val/Test curves per epoch – {args.run_name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_valtest = os.path.join(ckpt_dir, "val_test_curves_per_epoch.png")
    plt.savefig(out_valtest, dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    print(f"Latest checkpoint: {ckpt_dir}")
    print(f"Saved training plot  -> {out_train}")
    print(f"Saved val/test plot  -> {out_valtest}")


if __name__ == "__main__":
    main()
