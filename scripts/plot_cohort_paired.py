"""Per-item paired token change for one steering coefficient over the cohort.

The figure exists to show two things at once: the effect is centred on zero, and
the two items the screen was built on (10 and 18) sit at the extremes of the
distribution. The screen items were originally chosen as "the items coefficient 40
broke" -- selected on the outcome -- which is why they looked dramatic and why
two-item screens kept producing results that did not survive the cohort.

Usage:
  python scripts/plot_cohort_paired.py --json artifacts/aime_localize/paired_coef5.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="artifacts/aime_localize/paired_coef5.json")
    ap.add_argument("--highlight", default="10,18",
                    help="Items to mark as the historical screen set.")
    ap.add_argument("--out", default="artifacts/aime_localize/cohort_paired.png")
    args = ap.parse_args()

    d = json.load(open(args.json))
    items = sorted(d["items"], key=lambda r: r["pct"])
    hi = {int(x) for x in args.highlight.split(",") if x.strip()}

    pcts = [r["pct"] for r in items]
    labels = [str(r["idx"]) for r in items]
    logs = [math.log(1 + p / 100) for p in pcts]
    n = len(logs)
    mean, sd = st.mean(logs), st.stdev(logs)
    se = sd / math.sqrt(n)
    lo = 100 * (math.exp(mean - 1.96 * se) - 1)
    hi_ci = 100 * (math.exp(mean + 1.96 * se) - 1)
    med = 100 * (math.exp(st.median(logs)) - 1)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [2.5, 1]})

    colors = ["#b3261e" if p > 0 else "#1b7f3b" for p in pcts]
    edges = ["#000000" if int(l) in hi else "none" for l in labels]
    widths = [1.8 if int(l) in hi else 0.0 for l in labels]
    ax.bar(range(n), pcts, color=colors, edgecolor=edges, linewidth=widths)
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.axhline(med, color="#1f4e9c", linestyle="--", linewidth=1.3,
               label=f"median {med:+.1f}%")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("AIME item (sorted by change; boxed = the two screen items)")
    ax.set_ylabel("token change vs its own baseline")
    ax.set_title(f"Paired token change at coefficient {d['coef']}  "
                 f"(n={n} items, both arms correct and terminating)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    # The interval is the point of the right panel: it straddles zero, and it
    # excludes the savings that would have been worth shipping.
    ax2.axvline(0, color="0.2", linewidth=1.0)
    ax2.errorbar([100 * (math.exp(mean) - 1)], [0],
                 xerr=[[100 * (math.exp(mean) - 1) - lo],
                       [hi_ci - 100 * (math.exp(mean) - 1)]],
                 fmt="o", color="#1f4e9c", capsize=6, markersize=9, linewidth=2)
    ax2.axvspan(-100, -15, color="#1b7f3b", alpha=0.10)
    ax2.text(-16, 0.33, "saving worth\nshipping", fontsize=8.5, ha="right",
             color="#1b7f3b")
    ax2.set_yticks([])
    ax2.set_xlim(-45, 45)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_xlabel("mean token change (95% CI)")
    ax2.set_title(f"mean {100 * (math.exp(mean) - 1):+.1f}%  "
                  f"[{lo:+.1f}%, {hi_ci:+.1f}%]", fontsize=10.5)
    ax2.grid(axis="x", alpha=0.25, linewidth=0.6)

    fig.suptitle("Steering at the best available coefficient does not reduce "
                 "tokens; the old screen items were the two extremes",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"wrote {args.out}")

    rank = {r["idx"]: i for i, r in enumerate(items)}
    for i in sorted(hi):
        if i in rank:
            print(f"  item {i}: rank {rank[i] + 1}/{n} "
                  f"({items[rank[i]]['pct']:+.1f}%)")


if __name__ == "__main__":
    main()
