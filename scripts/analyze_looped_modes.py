#!/usr/bin/env python
"""How the answer mode (``looped/modes.py``) shapes saved candidate pools.

For every pool this reports, separately for answers that opened a reasoning
block (``<think>``) and answers that did not: their share, how often they hit
the token cap, how often they are correct, how long they are, and what share of
all generated tokens they consumed. It also reports how the mode is distributed
within a question, compares the two modes on questions that produced both, and
gives the chance that a candidate still running after L tokens finishes before
the cap.

Pools are graded arm directories or ``stage0b:<dir>`` (see
``replay_looped_policies.py``). Gold-derived fields are used; this is offline
analysis only.

    python scripts/analyze_looped_modes.py \
        --pools "smoke d3 x8=natural:3=artifacts/looped_serving/profile_smoke/d3_w8_cap2048" \
        --out artifacts/looped_serving/mode_audit/modes.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import replay_looped_policies as R  # noqa: E402

CHECKPOINTS = (300, 500, 700, 900, 1100, 1300, 1500, 1800)


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def side(rows, total_tokens) -> dict:
    fin = [r for r in rows if r["finish_reason"] == "eos"]
    return {
        "candidates": len(rows),
        "cap_rate": mean(r["finish_reason"] == "length_cap" for r in rows),
        "answered": mean(r["answer_key"] is not None for r in rows),
        "accuracy": mean(r["correct"] for r in rows),
        "accuracy_when_finished": mean(r["correct"] for r in fin),
        "mean_gen_tokens": mean(r["gen_tokens"] for r in rows),
        "mean_gen_tokens_when_finished": mean(r["gen_tokens"] for r in fin),
        "share_of_tokens": sum(r["gen_tokens"] for r in rows) / total_tokens if total_tokens else None,
    }


def analyze(pool: dict, levels=None) -> dict:
    by_q = defaultdict(list)
    for (qid, sample), r in pool["rows"].items():
        if sample >= 0:                                    # sampled candidates only; greedy is reported separately
            by_q[qid].append(r)
    rows = [r for q in by_q.values() for r in q]
    greedy = [r for (qid, sample), r in pool["rows"].items() if sample < 0]
    total = sum(r["gen_tokens"] for r in rows)
    think = [r for r in rows if r["think"]]
    direct = [r for r in rows if not r["think"]]
    width = max(len(q) for q in by_q.values())
    hist = Counter(sum(r["think"] for r in q) for q in by_q.values())
    mixed = [q for q in by_q.values() if 0 < sum(r["think"] for r in q) < len(q)]
    mixed_finished = [q for q in mixed if any(r["think"] and r["finish_reason"] == "eos" for r in q)]
    out = {
        "source": pool["source"], "cap": pool["cap"], "questions": len(by_q), "sampled_candidates": len(rows),
        "width": width, "think_share": len(think) / len(rows),
        "think": side(think, total), "direct": side(direct, total),
        "capped_share_of_tokens": sum(r["gen_tokens"] for r in rows if r["finish_reason"] == "length_cap") / total,
        "questions_by_think_count": [hist.get(k, 0) for k in range(width + 1)],
        "mixed_mode_questions": {
            "questions": len(mixed),
            "direct_accuracy": mean(np.mean([r["correct"] for r in q if not r["think"]]) for q in mixed),
            "think_accuracy": mean(np.mean([r["correct"] for r in q if r["think"]]) for q in mixed),
            "with_a_finished_think_answer": len(mixed_finished),
            "finished_think_accuracy": mean(np.mean([r["correct"] for r in q if r["think"] and r["finish_reason"] == "eos"])
                                            for q in mixed_finished),
            "direct_accuracy_same_questions": mean(np.mean([r["correct"] for r in q if not r["think"]])
                                                   for q in mixed_finished),
        },
        "all_direct_questions": {"questions": hist.get(0, 0),
                                 "candidate_accuracy": mean(r["correct"] for q in by_q.values()
                                                            if not any(r["think"] for r in q) for r in q)},
        "all_think_questions": {"questions": hist.get(width, 0),
                                "candidate_accuracy": mean(r["correct"] for q in by_q.values()
                                                           if all(r["think"] for r in q) for r in q)},
        "still_running_after": {
            str(L): {m: {"share_of_candidates": mean(r["gen_tokens"] > L for r in sel),
                         "finish_before_cap": mean(r["finish_reason"] == "eos" for r in sel if r["gen_tokens"] > L),
                         "correct": mean(r["correct"] for r in sel if r["gen_tokens"] > L)}
                     for m, sel in (("think", think), ("direct", direct))}
            for L in CHECKPOINTS if L < pool["cap"]},
    }
    if greedy:
        out["greedy"] = {"think_share": mean(r["think"] for r in greedy), "accuracy": mean(r["correct"] for r in greedy),
                         "cap_rate": mean(r["finish_reason"] == "length_cap" for r in greedy),
                         "direct_accuracy": mean(r["correct"] for r in greedy if not r["think"]),
                         "think_accuracy": mean(r["correct"] for r in greedy if r["think"])}
    if levels:
        per = defaultdict(list)
        for (qid, sample), r in pool["rows"].items():
            if sample >= 0:
                per[levels[qid]].append(r)
        out["by_level"] = {str(lv): {"think_share": mean(r["think"] for r in rs),
                                     "direct_accuracy": mean(r["correct"] for r in rs if not r["think"]),
                                     "think_accuracy": mean(r["correct"] for r in rs if r["think"]),
                                     "think_cap_rate": mean(r["finish_reason"] == "length_cap" for r in rs if r["think"])}
                           for lv, rs in sorted(per.items())}
    return out


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def markdown(results: dict) -> str:
    lines = ["| pool | questions | think % of candidates | think: capped % | think: correct % | think: correct % when finished | "
             "think: mean tokens | think: % of all tokens | direct: capped % | direct: correct % | direct: mean tokens |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        t, d = r["think"], r["direct"]
        lines.append(f"| {name} | {r['questions']} | {pct(r['think_share'])} | {pct(t['cap_rate'])} | {pct(t['accuracy'])} | "
                     f"{pct(t['accuracy_when_finished'])} | {t['mean_gen_tokens']:.0f} | {pct(t['share_of_tokens'])} | "
                     f"{pct(d['cap_rate'])} | {pct(d['accuracy'])} | {d['mean_gen_tokens']:.0f} |")
    lines += ["", "| pool | questions with both modes | direct correct % | think correct % | of which a think answer finished | "
                  "finished think correct % | direct correct % (same questions) |", "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        m = r["mixed_mode_questions"]
        lines.append(f"| {name} | {m['questions']} | {pct(m['direct_accuracy'])} | {pct(m['think_accuracy'])} | "
                     f"{m['with_a_finished_think_answer']} | {pct(m['finished_think_accuracy'])} | "
                     f"{pct(m['direct_accuracy_same_questions'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True, action="append",
                    help="<label>=<mode>:<depth>=<arm dir | stage0b:<dir>>; repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results = {}
    for spec in args.pools:
        label, _, rest = spec.partition("=")
        (key, pool), = R.parse_pools(rest).items()
        levels = None
        if "stage0b:" in rest:
            root = rest.split("stage0b:")[1]
            levels = {f"idx{r['idx']}": r["level"] for r in R.read_jsonl(os.path.join(root, f"scored_r{key[1]}.jsonl"))}
        results[label] = analyze(pool, levels)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"think_token_id": R.THINK_TOKEN_ID, "pools": results}, open(args.out, "w"), indent=1)
    md = markdown(results)
    with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
