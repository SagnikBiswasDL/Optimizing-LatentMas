#!/usr/bin/env python3
"""The loop-routing pilot gate (docs/looped_serving/LOOP_ROUTING.md).

Reads the graded pilot runs (depths 1-4, run seeds 0 and 1), checks that both runs
cover the same questions, and writes gate.json / gate.md with:

  - accuracy, loop-tokens, cap-hit share and `<think>` share per depth and run
  - Check 1: kappa of "needs the next loop" across runs, steps 1->2, 2->3 (3->4 reported)
  - Check 2: the oracle routing read off one run, scored on the other, against the
    best question-independent mixture at the same cost

Grade first (gold stays local):
    for s in 0 1; do uv run ... python scripts/grade_looped_profile.py \
        --runs artifacts/looped_serving/loop_routing/pilot/seed$s \
        --gold artifacts/looped_serving/splits/math_train_calibration.gold.jsonl; done
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from looped import loop_select as ls  # noqa: E402
from scripts.eval_loop_selectors import run_labels  # noqa: E402


def arm_dir(base: str, seed: int, d: int) -> str:
    hits = glob.glob(f"{base}/seed{seed}/d{d}_w4*cap8192*")
    if len(hits) != 1:
        raise SystemExit(f"expected one depth-{d} arm under {base}/seed{seed}, found {hits}")
    return hits[0]


def arm_stats(path: str) -> dict:
    n = capped = think = 0
    for line in open(os.path.join(path, "candidates.jsonl")):
        r = json.loads(line)
        n += 1
        capped += r["finish_reason"] in ("length", "length_cap")
        think += bool(r.get("think"))
    s = json.load(open(os.path.join(path, "summary.json")))
    return {"candidates": n, "cap_hit": capped / n, "think_share": think / n, "wall_s": s.get("wall_s"),
            "questions_per_min": s.get("questions_per_min"), "mean_gen_tokens": s.get("mean_gen_tokens")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_dir", default="artifacts/looped_serving/loop_routing/pilot")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    labels, stats = {}, defaultdict(dict)
    for s in (0, 1):
        for d in ls.DEPTHS:
            p = arm_dir(args.pilot_dir, s, d)
            labels[s, d] = run_labels(p)
            stats[s][d] = arm_stats(p)
    qsets = {k: set(v) for k, v in labels.items()}
    common = set.intersection(*qsets.values())
    if any(len(v) != len(common) for v in qsets.values()):
        raise SystemExit("the runs do not cover the same questions")
    qids = sorted(common)
    C = {s: np.array([[labels[s, d][q][0] for d in ls.DEPTHS] for q in qids], float) for s in (0, 1)}
    cost = {s: np.array([[labels[s, d][q][1] for d in ls.DEPTHS] for q in qids], float) for s in (0, 1)}

    gate = ls.repeatability_gate(C[0], C[1], cost[0], cost[1], steps=(0, 1), n_boot=args.n_boot)
    extra = ls.repeatability_gate(C[0], C[1], cost[0], cost[1], steps=(2,), n_boot=args.n_boot)["steps"]
    same_depth = {str(d): int((C[0][:, j] != C[1][:, j]).sum()) for j, d in enumerate(ls.DEPTHS)}
    per_run = {}
    for s in (0, 1):
        fr = ls.mixture_frontier(C[s].mean(0), cost[s].mean(0))
        orc = ls.oracle_policy(C[s], cost[s])
        per_run[str(s)] = {
            "fixed": {str(d): {"acc": float(C[s][:, j].mean()), "loop_tokens": float(cost[s][:, j].mean()),
                               **stats[s][d]} for j, d in enumerate(ls.DEPTHS)},
            "frontier": fr,
            "oracle_same_run": {"acc": ls.evaluate(orc, C[s], cost[s])[0], "cost": ls.evaluate(orc, C[s], cost[s])[1],
                                "depths": ls.depth_counts(orc)}}
    out = {"n": len(qids), "gate": gate, "step_3_4_reported": extra,
           "same_depth_disagreements": same_depth, "runs": per_run}
    with open(os.path.join(args.pilot_dir, "gate.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    L = [f"# Loop-routing pilot gate ({len(qids)} questions, two runs)", "",
         "| depth | run 0 accuracy | run 1 accuracy | loop-tokens (run 0) | cap hit (run 0) | think share (run 0) | votes that differ between runs |",
         "|---|---|---|---|---|---|---|"]
    for j, d in enumerate(ls.DEPTHS):
        a0, a1 = per_run["0"]["fixed"][str(d)], per_run["1"]["fixed"][str(d)]
        L.append(f"| {d} | {100*a0['acc']:.1f}% | {100*a1['acc']:.1f}% | {a0['loop_tokens']:.0f} | "
                 f"{100*a0['cap_hit']:.1f}% | {100*a0['think_share']:.1f}% | {same_depth[str(d)]} |")
    L += ["", "**Check 1: does \"needs the next loop\" repeat across runs?**", "",
          "| step | run 0 | run 1 | both | kappa (95% interval) | gated |", "|---|---|---|---|---|---|"]
    for name, st in {**gate["steps"], **extra}.items():
        gated = name in gate["steps"]
        L.append(f"| {name} | {st['needs_next_A']} | {st['needs_next_B']} | {st['both']} | "
                 f"{st['kappa']:.2f} ({st['kappa_lo']:.2f} to {st['kappa_hi']:.2f}) | "
                 f"{('pass' if st['pass'] else 'fail') if gated else 'reported'} |")
    t = gate["transfer"]
    L += ["", "**Check 2: does the ideal routing from one run still win on the other?**", "",
          f"Transferred oracle minus best mixture at the same cost: {100*t['margin']:+.1f} points "
          f"(95% interval {100*t['margin_lo']:+.1f} to {100*t['margin_hi']:+.1f}); "
          f"run 0 to run 1: {100*t['A_to_B_acc']:.1f}% at {t['A_to_B_cost']:.0f} loop-tokens, depths {t['A_to_B_depths']}. "
          f"{'pass' if t['pass'] else 'fail'}", "",
          f"**Gate: {'PASS' if gate['pass'] else 'FAIL'}**"]
    open(os.path.join(args.pilot_dir, "gate.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
