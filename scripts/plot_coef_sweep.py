"""Token count against steering coefficient, both signs, on the screen items.

The point of the figure: the curve is not monotonic through zero, it is a valley
with its minimum AT zero. Steering the Judger in *either* direction along the
SEAL axis makes it emit more tokens, which says the effect is a magnitude effect
(a perturbation off the model's distribution) rather than the semantic
execution-vs-reflection effect the vector is supposed to encode.

Usage:
  python scripts/plot_coef_sweep.py --rows artifacts/aime_localize/rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM = re.compile(r"^real(?:_seal(?P<coef>-?\d+(?:\.\d+)?))?__(?P<tag>\w+)$")


def collect(rows_path: str, tag: str, items: List[int]) -> Dict[int, List[Tuple]]:
    by_item: Dict[int, List[Tuple]] = {i: [] for i in items}
    for line in open(rows_path):
        if not line.strip():
            continue
        r = json.loads(line)
        m = ARM.match(str(r.get("method", "")))
        if not m or m.group("tag") != tag:
            continue
        idx = int(r["idx"])
        if idx not in by_item:
            continue
        coef = float(m.group("coef") or 0.0)
        by_item[idx].append((coef, int(r["tokens"]), bool(r.get("correct")),
                             bool(r.get("eos"))))
    for i in by_item:
        by_item[i].sort(key=lambda t: t[0])
    return by_item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="artifacts/aime_localize/rows.jsonl")
    ap.add_argument("--tag", default="b16k")
    ap.add_argument("--items", default="10,18")
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--out", default="artifacts/aime_localize/coef_sweep.png")
    args = ap.parse_args()

    items = [int(x) for x in args.items.split(",") if x]
    data = collect(args.rows, args.tag, items)

    fig, axes = plt.subplots(1, len(items), figsize=(5.2 * len(items), 4.4),
                             sharey=True)
    if len(items) == 1:
        axes = [axes]

    for ax, idx in zip(axes, items):
        pts = data[idx]
        if not pts:
            continue
        base = next((t for t in pts if t[0] == 0.0), None)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", color="0.65", zorder=1, linewidth=1.4)
        for coef, tok, correct, eos in pts:
            # Censored runs are lower bounds, so they get an upward arrow rather
            # than a point: the true token count is unknown and larger.
            if not eos:
                ax.annotate("", xy=(coef, tok * 1.075), xytext=(coef, tok),
                            arrowprops=dict(arrowstyle="-|>", color="#b3261e",
                                            linewidth=1.5))
            ax.scatter([coef], [tok], s=95, zorder=3,
                       marker="o" if correct else "X",
                       color="#1b7f3b" if correct else "#b3261e",
                       edgecolor="white", linewidth=1.1)
        if base:
            ax.axhline(base[1], color="#1b7f3b", linestyle=":", linewidth=1.2)
            ax.axvline(0, color="0.8", linewidth=1.0, zorder=0)
        ax.axhline(args.cap, color="#b3261e", linestyle="--", linewidth=1.0,
                   alpha=0.55)
        ax.text(min(xs), args.cap * 0.975, " token cap", color="#b3261e",
                fontsize=8, va="top")
        ax.set_title(f"AIME item {idx}"
                     + (f"  (baseline {base[1]:,} tokens)" if base else ""),
                     fontsize=11)
        ax.set_xlabel("steering coefficient")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("tokens emitted by the Judger")

    handles = [
        plt.Line2D([], [], marker="o", color="#1b7f3b", linestyle="", label="correct"),
        plt.Line2D([], [], marker="X", color="#b3261e", linestyle="", label="incorrect"),
        plt.Line2D([], [], marker="$\\uparrow$", color="#b3261e", linestyle="",
                   label="hit cap (lower bound)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Steering in either direction costs tokens; the minimum is at zero",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"wrote {args.out}")

    for idx in items:
        base = next((t for t in data[idx] if t[0] == 0.0), None)
        if not base:
            continue
        worse = sum(1 for c, t, _, _ in data[idx] if c != 0 and t > base[1])
        n = sum(1 for c, _, _, _ in data[idx] if c != 0)
        print(f"item {idx}: {worse}/{n} steered settings emit MORE than baseline "
              f"({base[1]} tokens)")


if __name__ == "__main__":
    main()
