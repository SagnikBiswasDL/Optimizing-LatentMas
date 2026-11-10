#!/usr/bin/env python3
"""Accuracy-vs-cost plot for the Looped-LatentMAS gates (depth sweep + MAS gate).

  python scripts/looped_report.py --root artifacts/looped
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/looped")
    args = ap.parse_args()
    sweep = load(os.path.join(args.root, "depth_sweep", "summary.json"))
    gate = load(os.path.join(args.root, "mas_gate_r4", "summary.json"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if sweep:
        xs, ys, lo, hi = [], [], [], []
        for r, s in sorted(sweep["depths"].items(), key=lambda kv: int(kv[0])):
            xs.append(s["mean_loop_tokens"] / 1e3)
            ys.append(s["acc"])
            lo.append(s["acc"] - s["acc_ci95"][0])
            hi.append(s["acc_ci95"][1] - s["acc"])
            ax.annotate(f"r={r}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(5, -10), fontsize=8)
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label="single solver, depth r")
    if gate:
        arms = gate["arms"]
        sc = sorted((k for k in arms if k.startswith("sc@")), key=lambda k: int(k[3:]))
        if sc:
            ax.plot([arms[k]["mean_loop_tokens"] / 1e3 for k in sc], [arms[k]["acc"] for k in sc],
                    marker="s", linestyle="--", label="self-consistency@m (r=4)")
            for k in sc:
                ax.annotate(k[3:], (arms[k]["mean_loop_tokens"] / 1e3, arms[k]["acc"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
        for name, mk in (("single", "o"), ("selfrev", "^"), ("mas", "*")):
            a = arms[name]
            ax.errorbar([a["mean_loop_tokens"] / 1e3], [a["acc"]],
                        yerr=[[a["acc"] - a["acc_ci95"][0]], [a["acc_ci95"][1] - a["acc"]]],
                        marker=mk, markersize=11 if mk == "*" else 7, capsize=3, linestyle="none",
                        label={"single": "single (r=4)", "selfrev": "self-revision (r=4)",
                               "mas": "planner→critic→solver (r=4)"}[name])
    ax.set_xlabel("mean executed loop-tokens per question (thousands)")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("Ouro-1.4B-Thinking: accuracy vs executed loop-token work")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(args.root, "acc_vs_cost.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
