#!/usr/bin/env python3
"""ACL figures from ARR + memory reports (numbers frozen in this file)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
C_FULL = "#1f4e79"
C_OURS = "#c45911"
C_K10 = "#7f7f7f"
C_NONE = "#9e9e9e"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def acc_bars():
    tasks = ["GSM8K", "MATH-500", "GPQA", "AIME24", "AIME25"]
    full = np.array([96.3, 85.7, 53.7, 36.7, 35.0])
    ours = np.array([95.7, 83.7, 48.7, 32.2, 31.7])
    x = np.arange(len(tasks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.bar(x - w / 2, full, w, label="LatentMAS $K{=}40$ (full KV)", color=C_FULL)
    ax.bar(x + w / 2, ours, w, label="Same $K{=}40$ + eviction", color=C_OURS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Eviction after a $K{=}40$ rollout (3-seed mean; AIME25: 2 seeds)")
    save(fig, "acc_evict")


def memory_batch():
    B = np.array([1, 8, 16, 32, 64])
    full = np.array([27.9, 30.1, 32.5, 37.3, 47.1])
    ours = np.array([27.7, 28.2, 28.8, 29.9, 32.2])
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.plot(B, full, "o-", color=C_FULL, label="Full relay")
    ax.plot(B, ours, "s-", color=C_OURS, label="Evict keep-64")
    ax.set_xlabel("Judger batch size $B$")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title("GSM8K donor cache, 16-token decode, Qwen3-14B on H200")
    ax.legend(frameon=False)
    ax.set_xticks(B)
    save(fig, "memory_batch")


def latency_agents():
    labels = ["Planner", "Critic", "Refiner", "Judger"]
    gsm_full = [1.01, 1.01, 1.00, 14.4]
    gsm_ours = [1.01, 1.01, 1.00, 13.0]
    gpq_full = [1.04, 1.04, 1.04, 87.6]
    gpq_ours = [1.04, 1.04, 1.04, 65.8]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5), sharey=False)
    x = np.arange(len(labels))
    w = 0.36
    for ax, full, ours, title in (
        (axes[0], gsm_full, gsm_ours, "GSM8K"),
        (axes[1], gpq_full, gpq_ours, "GPQA"),
    ):
        ax.bar(x - w / 2, full, w, color=C_FULL, label="Full KV")
        ax.bar(x + w / 2, ours, w, color=C_OURS, label="Evicted")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Seconds / item")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Per-agent wall-clock (CUDA-synced). Upstream is unchanged.", y=1.03)
    save(fig, "latency_agents")


def swap_budget():
    """MedQA 14B greedy cache-swap (n=40), from diagnostics."""
    budgets = ["1024", "2048", "4096"]
    real = [67.5, 77.5, 80.0]
    shuf = [62.5, 77.5, 77.5]
    none = [35.0, 70.0, 75.0]
    x = np.arange(len(budgets))
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.bar(x - w, real, w, label="Real cache", color=C_FULL)
    ax.bar(x, shuf, w, label="Shuffled (wrong-item)", color=C_OURS)
    ax.bar(x + w, none, w, label="No cache", color=C_NONE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tmax={b}" for b in budgets])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MedQA, Qwen3-14B, greedy: content swap vs presence")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0, 100)
    save(fig, "swap_budget")


if __name__ == "__main__":
    acc_bars()
    memory_batch()
    latency_agents()
    swap_budget()
    print("wrote", list(OUT.glob("*")))
