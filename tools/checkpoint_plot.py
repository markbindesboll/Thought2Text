#!/usr/bin/env python3
"""Parse training log and plot train / eval / test loss curves.

Usage:
  python tools/checkpoint_plot.py --log logs/train_sub08_run5.log --outdir logs

Saves a PNG file named <logfile>_loss_plot.png into --outdir.
"""
import argparse
import ast
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path):
    train_epochs = []
    train_loss = []
    eval_epochs = []
    eval_loss = []
    test_epochs = []
    test_loss = []

    last_epoch = 0.0
    with open(path, "r") as f:
        for line in f:
            # find first {...} substring
            m = re.search(r"\{.*\}", line)
            if not m:
                continue
            s = m.group(0)
            try:
                d = ast.literal_eval(s)
            except Exception:
                continue

            # training step (has 'loss' and usually 'epoch')
            if "loss" in d and "epoch" in d:
                try:
                    e = float(d.get("epoch", last_epoch))
                    l = float(d.get("loss"))
                except Exception:
                    continue
                last_epoch = e
                train_epochs.append(e)
                train_loss.append(l)

            # sometimes eval/test use different keys
            if "eval_loss" in d:
                try:
                    l = float(d.get("eval_loss"))
                except Exception:
                    continue
                # place eval point at the most recent epoch
                eval_epochs.append(last_epoch)
                eval_loss.append(l)

            if "test_loss" in d:
                try:
                    l = float(d.get("test_loss"))
                except Exception:
                    continue
                test_epochs.append(last_epoch)
                test_loss.append(l)

    return {
        "train_epochs": train_epochs,
        "train_loss": train_loss,
        "eval_epochs": eval_epochs,
        "eval_loss": eval_loss,
        "test_epochs": test_epochs,
        "test_loss": test_loss,
    }


def plot_losses(parsed, outpath, title=None):
    plt.figure(figsize=(8, 5))
    if parsed["train_epochs"]:
        plt.plot(parsed["train_epochs"], parsed["train_loss"], label="train loss", color="C0", alpha=0.8)
    if parsed["eval_epochs"]:
        plt.scatter(parsed["eval_epochs"], parsed["eval_loss"], label="eval loss", color="C1", s=12)
        plt.plot(parsed["eval_epochs"], parsed["eval_loss"], color="C1", alpha=0.6)
    if parsed["test_epochs"]:
        plt.scatter(parsed["test_epochs"], parsed["test_loss"], label="test loss", color="C2", s=12)
        plt.plot(parsed["test_epochs"], parsed["test_loss"], color="C2", alpha=0.6)

    # mark best (lowest) test loss with a vertical line matching test color (no legend entry)
    if parsed["test_epochs"] and parsed["test_loss"]:
        try:
            best_idx = min(range(len(parsed["test_loss"])), key=lambda i: parsed["test_loss"][i])
            best_test_epoch = parsed["test_epochs"][best_idx]
            plt.axvline(best_test_epoch, color="C2", linestyle="--", linewidth=1.2, label="_nolegend_")
        except Exception:
            pass

    # mark best (lowest) eval loss with a vertical line matching eval color (no legend entry)
    if parsed["eval_epochs"] and parsed["eval_loss"]:
        try:
            best_idx = min(range(len(parsed["eval_loss"])), key=lambda i: parsed["eval_loss"][i])
            best_eval_epoch = parsed["eval_epochs"][best_idx]
            plt.axvline(best_eval_epoch, color="C1", linestyle="--", linewidth=1.2, label="_nolegend_")
        except Exception:
            pass

    plt.xlabel("epoch")
    plt.ylabel("loss")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to training log")
    ap.add_argument("--outdir", default="logs", help="output directory for PNG")
    ap.add_argument("--name", default=None, help="output filename (without ext)")
    args = ap.parse_args()

    parsed = parse_log(args.log)
    os.makedirs(args.outdir, exist_ok=True)
    base = args.name or os.path.splitext(os.path.basename(args.log))[0]
    outpath = os.path.join(args.outdir, f"{base}_loss_plot.png")
    title = f"Loss curves — {base}"
    plot_losses(parsed, outpath, title=title)

    # print a brief summary
    print(f"Wrote plot: {outpath}")
    print(f"Train points: {len(parsed['train_loss'])}, eval points: {len(parsed['eval_loss'])}, test points: {len(parsed['test_loss'])}")


if __name__ == "__main__":
    main()
