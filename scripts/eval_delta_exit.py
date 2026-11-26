#!/usr/bin/env python3
"""First look at Jiayi's token-level exit, on a few MATH questions.

Compares:

  fixed     every loop, last-loop readout
  gate      every loop, Ouro's cumulative gate chooses the readout
  kv_delta  a generated token stops once its KV delta falls below epsilon
  diagonal  that halt rule on the convergence diagonal (recompute on a miss)
  wfd       diagonal wavefront, full depth, keep a draft only when the ids match

Prompt tokens always run every loop.

    .venv/bin/python scripts/eval_delta_exit.py --n 2 --max_new_tokens 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped import prompts as P  # noqa: E402
from looped.delta_exit import generate_exit  # noqa: E402
from looped.loopdelta import grade_math, load_math_train_stratified  # noqa: E402
from looped.ouro_adapter import OURO_BASE_ID, OURO_BASE_REVISION, OuroRunner  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_level", type=int, default=8)
    ap.add_argument("--levels", default="1,2,3,4,5",
                    help="Comma-separated MATH levels to keep. Same seed-0 draw as a full run.")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--epsilons", default="",
                    help="Comma-separated KV-delta cutoffs. Repeats kv_delta and diagonal once each.")
    ap.add_argument("--modes", default="fixed_4,fixed_3,gate_0.5,kv_delta")
    ap.add_argument("--out", default="artifacts/looped_serving/delta_exit/gpu_merit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    keep = {int(part) for part in args.levels.split(",") if part.strip()}
    items = [it for it in load_math_train_stratified(args.per_level, 0) if it["level"] in keep]
    attn = "sdpa"
    runner = OuroRunner(
        model_id=OURO_BASE_ID, revision=OURO_BASE_REVISION, think=False, attn_implementation=attn,
    )
    catalog = {
        "fixed_4": ("fixed_4", {"mode": "fixed", "depth": 4}),
        "fixed_3": ("fixed_3", {"mode": "fixed", "depth": 3}),
        "gate_0.5": ("gate_0.5", {"mode": "gate", "threshold": 0.5, "depth": args.depth}),
        "kv_delta": ("kv_delta", {"mode": "kv_delta", "epsilon": args.epsilon, "depth": args.depth}),
        "diagonal": ("diagonal", {"mode": "diagonal", "epsilon": args.epsilon, "depth": args.depth}),
        "wfd": ("wfd", {"mode": "wfd", "depth": args.depth}),
        "b6_accel": ("b6_accel", {"mode": "kv_delta", "epsilon": args.epsilon, "depth": args.depth, "rule": "accel"}),
        "b7_random": ("b7_random", {"mode": "kv_delta", "epsilon": args.epsilon, "depth": args.depth, "rule": "random"}),
        "b8_recompute": ("b8_recompute", {"mode": "kv_delta", "epsilon": args.epsilon, "depth": args.depth, "reuse": False}),
    }
    modes = []
    cutoffs = [float(part) for part in args.epsilons.split(",") if part.strip()]
    for name in [part.strip() for part in args.modes.split(",") if part.strip()]:
        if name not in catalog:
            raise SystemExit(f"unknown mode {name}")
        label, kw = catalog[name]
        if cutoffs and name in ("kv_delta", "diagonal"):
            for cutoff in cutoffs:
                modes.append((f"{label}_e{cutoff:g}", {**kw, "epsilon": cutoff}))
        else:
            modes.append((label, kw))
    out_path = os.path.join(args.out, "rows.jsonl")
    rows = []
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if not line.strip():
                continue
            try:
                saved = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(saved)
            done.add((saved["idx"], saved["mode"]))
    for item in items:
        prompt = runner.chat_ids(P.msgs(P.solver(item["question"])), think=False)
        for name, kw in modes:
            if (item["idx"], name) in done:
                continue
            print(f"q{item['idx']} {name}", flush=True)
            out = generate_exit(
                runner, prompt, max_new_tokens=args.max_new_tokens, **kw,
            )
            graded = grade_math(out["text"], item["gold"])
            row = {
                "idx": item["idx"],
                "level": item["level"],
                "mode": name,
                "correct": graded["correct"],
                "pred": graded["pred"],
                "gold": item["gold"],
                "hit_eos": out["hit_eos"],
                "new_tokens": len(out["token_ids"]),
                "mean_chosen_loop": out["mean_chosen_loop"],
                "mean_executed_loops": out["mean_executed_loops"],
                "loop_tokens": out["work"]["loop_tokens"],
                "wall_s": out["wall_s"],
                "rejected_loops": out.get("rejected_loops", 0),
                "wavefront_steps": out.get("wavefront_steps", 0),
                "drafts_checked": out.get("drafts_checked", 0),
                "drafts_accepted": out.get("drafts_accepted", 0),
                "text": out["text"],
            }
            rows.append(row)
            print(
                f"  correct={row['correct']} tokens={row['new_tokens']} "
                f"chosen={row['mean_chosen_loop']} executed={row['mean_executed_loops']} "
                f"loop_tokens={row['loop_tokens']} {row['wall_s']:.0f}s",
                flush=True,
            )
            with open(out_path, "w") as f:
                for saved in rows:
                    f.write(json.dumps(saved) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
