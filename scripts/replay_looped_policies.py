#!/usr/bin/env python
"""Replay allocation policies on saved, graded candidate pools.

The policies and the driver are the ones that run live
(``looped/allocation.py``, ``serving/looped/driver.py``); only the backend is
replaced by ``ReplayBackend``, which hands back saved candidates one token per
step. Every question runs alone with all the slots it wants, so a policy's
"stop step" is the number of decode steps of an ideal batch between launching
its first candidates and fixing the answer. It is a length proxy: use it to
choose policy parameters, never as a latency.

A pool is one graded arm directory written by ``looped_profile.py`` and
``grade_looped_profile.py``, addressed by the mode and depth its candidates
were generated with:

    python scripts/replay_looped_policies.py \
        --pools natural:3=artifacts/looped_serving/profile_smoke/d3_w8_cap2048 \
        --policies fixed:3x4,cert:3x4,quorum:3x4,quorum:3x8 --baseline fixed:3x4

``stage0b:<dir>`` in place of a directory reads the historical Stage-0b pool at
that depth (natural mode, four samples and a greedy candidate).

A policy asking for sample ``s`` of a mode gets the pool's sample ``s % 100``,
so ``FastThenThink``'s reasoning candidates (samples 100, 101, ...) read the
think pool's samples 0, 1, ... . A candidate cut by a shorter cap than it was
generated with casts no vote.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped import allocation as al  # noqa: E402
from looped import answer_identity as ai  # noqa: E402
from looped import experiment_manifest as em  # noqa: E402
from serving.looped.backend import GenerationRequest  # noqa: E402
from serving.looped.driver import run_closed_loop  # noqa: E402
from serving.looped.replay_backend import ReplayBackend  # noqa: E402

THINK_TOKEN_ID = 3          # <think> in the pinned Ouro tokenizer


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_arm(path: str) -> dict:
    """{(question_id, sample): row} plus the cap the arm was generated with."""
    graded = {r["candidate_id"]: r for r in read_jsonl(os.path.join(path, "graded.jsonl"))}
    rows = {}
    for r in read_jsonl(os.path.join(path, "candidates.jsonl")):
        if r["finish_reason"] not in ("eos", "length_cap"):
            continue
        think = r.get("think")
        if think is None:
            think = bool(r["generated_token_ids"]) and r["generated_token_ids"][0] == THINK_TOKEN_ID
        rows[(r["question_id"], r["sample"])] = {
            "gen_tokens": r["gen_tokens"], "prefix_tokens": r.get("prefix_tokens", 0), "prompt_tokens": r["prompt_tokens"],
            "finish_reason": r["finish_reason"], "answer_key": r["answer_key"],
            "correct": bool(graded[r["candidate_id"]]["correct"]), "think": bool(think),
        }
    cap = json.load(open(os.path.join(path, em.MANIFEST_FILE)))["decode_config"]["cap"]
    return {"rows": rows, "cap": cap, "source": path}


def load_stage0b(root: str, depth: int) -> dict:
    gen = read_jsonl(os.path.join(root, f"gen_r{depth}.jsonl"))
    scored = read_jsonl(os.path.join(root, f"scored_r{depth}.jsonl"))
    rows = {}
    for g, r in zip(gen, scored):
        if g["sample"] != r["sample"] or len(g["ids"]) != r["gen_tokens"]:
            raise ValueError("gen and scored rows of the Stage-0b pool do not line up")
        rows[(f"idx{r['idx']}", r["sample"])] = {
            "gen_tokens": r["gen_tokens"], "prefix_tokens": 0, "prompt_tokens": r["prompt_tokens"],
            "finish_reason": "eos" if r["hit_eos"] else "length_cap", "answer_key": ai.answer_key(r["pred"]),
            "correct": bool(r["correct"]), "think": bool(g["ids"]) and g["ids"][0] == THINK_TOKEN_ID,
        }
    return {"rows": rows, "cap": 2048, "source": f"{root} depth {depth}"}


def parse_pools(spec: str) -> dict:
    pools = {}
    for part in [p for p in spec.split(",") if p]:
        key, _, path = part.partition("=")
        mode, _, depth = key.partition(":")
        if path.startswith("stage0b:"):
            pools[(mode, int(depth))] = load_stage0b(path[len("stage0b:"):], int(depth))
        else:
            pools[(mode, int(depth))] = load_arm(path)
    return pools


def boot_ci(values, iters: int, seed: int):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(iters, len(v)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def replay(policy: al.Policy, pools: dict, qids, default_cap: int) -> dict:
    """Per-question outcome of ``policy`` on the pools."""
    def pool_row(mode, depth, qid, sample):
        try:
            pool = pools[(mode, depth)]
        except KeyError:
            raise SystemExit(f"policy {policy.describe()} needs a {mode} pool at depth {depth}") from None
        s = sample if sample < 0 else sample % 100
        try:
            return pool, pool["rows"][(qid, s)]
        except KeyError:
            raise SystemExit(f"the {mode} depth-{depth} pool has no sample {s} for {qid}") from None

    meta = {}

    def make_request(qid, prompt, l: al.Launch) -> GenerationRequest:
        pool, row = pool_row(l.mode, l.depth, qid, l.sample)
        rid = f"{qid}|{l.mode}|d{l.depth}|s{l.sample}"
        meta[rid] = (pool, row)
        return GenerationRequest(rid, (0,) * (row["prompt_tokens"] + row["prefix_tokens"]), l.depth,
                                 (l.cap or default_cap) - row["prefix_tokens"])

    be = ReplayBackend(
        lookup=lambda req: (meta[req.request_id][1]["gen_tokens"] - meta[req.request_id][1]["prefix_tokens"],
                            meta[req.request_id][1]["finish_reason"], meta[req.request_id][1]),
        pool_cap=lambda req: meta[req.request_id][0]["cap"] - meta[req.request_id][1]["prefix_tokens"],
    )
    out = {q: {"tokens": 0, "loop_tokens": 0, "launched": 0, "cancelled": 0} for q in qids}
    reasons = {}

    def on_candidate(q, sample, comp, cancel_time):
        o = out[q.question_id]
        o["tokens"] += len(comp.token_ids)
        o["loop_tokens"] += comp.logical_loop_tokens
        o["launched"] += 1
        o["cancelled"] += cancel_time is not None

    def on_decision(q, decision, now, free):
        if decision.done:
            reasons[q.question_id] = decision.reason

    def on_question(q, chosen, now):
        o = out[q.question_id]
        o["steps"] = now - q.arrival
        o["correct"] = bool(chosen is not None and meta[q.candidates[chosen.sample]["request_id"]][1]["correct"])
        o["answered"] = chosen is not None

    run_closed_loop(
        be, policy, [(q, (0,)) for q in qids], slots=policy.max_candidates * len(qids) + 1,
        make_request=make_request,
        keyer=lambda comp: None if comp.detail["cut_by_cap"] else be.payload[comp.request_id]["answer_key"],
        think_of=lambda comp, l: be.payload[comp.request_id]["think"],
        on_candidate=on_candidate, on_decision=on_decision, on_question=on_question, clock=be.clock,
    )
    for q in qids:
        out[q]["reason"] = reasons[q]
    return out


def summarize(spec: str, per_q: dict, qids, args, base=None) -> dict:
    correct = np.array([per_q[q]["correct"] for q in qids], dtype=float)
    steps = np.array([per_q[q]["steps"] for q in qids], dtype=float)
    row = {
        "policy": spec, "questions": len(qids), "accuracy": float(correct.mean()),
        "accuracy_ci95": boot_ci(correct, args.iters, args.seed),
        "answered": float(np.mean([per_q[q]["answered"] for q in qids])),
        "stop_step": {"mean": float(steps.mean()), "p50": float(np.percentile(steps, 50)),
                      "p90": float(np.percentile(steps, 90)), "p95": float(np.percentile(steps, 95))},
        "decode_tokens_per_question": float(np.mean([per_q[q]["tokens"] for q in qids])),
        "loop_tokens_per_question": float(np.mean([per_q[q]["loop_tokens"] for q in qids])),
        "candidates_per_question": float(np.mean([per_q[q]["launched"] for q in qids])),
        "cancelled_per_question": float(np.mean([per_q[q]["cancelled"] for q in qids])),
        "decision_reasons": dict(Counter(per_q[q]["reason"] for q in qids)),
    }
    if base is not None:
        b = np.array([base[q]["correct"] for q in qids], dtype=float)
        row["vs_baseline"] = {
            "accuracy_diff": float((correct - b).mean()), "accuracy_diff_ci95": boot_ci(correct - b, args.iters, args.seed),
            "only_this_correct": int(((correct == 1) & (b == 0)).sum()),
            "only_baseline_correct": int(((correct == 0) & (b == 1)).sum()),
        }
    return row


def markdown(rows) -> str:
    lines = ["| policy | accuracy % (95% CI) | vs baseline pts (95% CI; wins/losses) | stop step mean | p50 | p95 | "
             "decode tokens/q | loop-tokens/q | candidates/q |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        vb = r.get("vs_baseline")
        cmp_ = "-" if vb is None else (f"{100 * vb['accuracy_diff']:+.1f} ({100 * vb['accuracy_diff_ci95'][0]:+.1f}, "
                                       f"{100 * vb['accuracy_diff_ci95'][1]:+.1f}; {vb['only_this_correct']}/"
                                       f"{vb['only_baseline_correct']})")
        lines.append(
            f"| {r['policy']} | {100 * r['accuracy']:.1f} ({100 * r['accuracy_ci95'][0]:.1f}, {100 * r['accuracy_ci95'][1]:.1f}) | "
            f"{cmp_} | {r['stop_step']['mean']:.0f} | {r['stop_step']['p50']:.0f} | {r['stop_step']['p95']:.0f} | "
            f"{r['decode_tokens_per_question']:.0f} | {r['loop_tokens_per_question']:.0f} | {r['candidates_per_question']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True, help="comma list of <mode>:<depth>=<arm dir | stage0b:<dir>>")
    ap.add_argument("--policies", required=True)
    ap.add_argument("--baseline", default="", help="policy spec the others are compared with, question by question")
    ap.add_argument("--cap", type=int, default=2048, help="cap of a launch that does not carry its own")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    pools = parse_pools(args.pools)
    qids = sorted(set.intersection(*[{q for q, _ in p["rows"]} for p in pools.values()]))
    specs = [s for s in args.policies.split(",") if s]
    if args.baseline and args.baseline not in specs:
        specs.insert(0, args.baseline)
    per = {s: replay(al.parse_policy(s), pools, qids, args.cap) for s in specs}
    rows = [summarize(s, per[s], qids, args, per[args.baseline] if args.baseline and s != args.baseline else None)
            for s in specs]
    md = markdown(rows)
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"clock": "decode steps of an ideal batch; a length proxy, not a latency",
                   "pools": {f"{m}:{d}": {"source": p["source"], "cap": p["cap"], "candidates": len(p["rows"])}
                             for (m, d), p in pools.items()},
                   "questions": len(qids), "default_cap": args.cap, "baseline": args.baseline or None,
                   "bootstrap": {"iters": args.iters, "seed": args.seed}, "policies": rows,
                   "per_question_correct": {s: {q: per[s][q]["correct"] for q in qids} for s in specs}},
                  open(args.out, "w"), indent=1)
        with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
