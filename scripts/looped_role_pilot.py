#!/usr/bin/env python3
"""Stage 3: does the frozen backbone do the planner's and the critic's jobs?

Planner vs generic solver on the same questions: does the planner produce a plan
and withhold the answer (instruction adherence), where the solver answers?

Critic vs generic solver on the same (question, candidate) pairs, each question
with its reference solution and a corrupted copy with one known first error:
verdict accuracy, first-error localisation, and corrected-answer accuracy. The
solver control sees the same candidate without the critic role; its implied
verdict is "its own answer equals the candidate's stated answer".

  python scripts/looped_role_pilot.py --n 50 --out artifacts/looped/role_pilot
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from looped import prompts as P  # noqa: E402
from looped.harness import environment, run_agent, warmup, write_json, write_jsonl  # noqa: E402
from looped.ouro_adapter import OuroRunner  # noqa: E402
from looped.stats import paired  # noqa: E402
from looped.tasks import (  # noqa: E402
    correct_solution, corrupt_solution, extract_answer, load_gsm8k_subset, subset_fingerprint,
)

STEP_LINE = re.compile(r"^\s*(?:\d+[.)]|step\s*\d+|[-*•])", re.I | re.M)
VERDICT = re.compile(r"verdict\s*[:\-]?\s*\**\s*(correct|incorrect)", re.I)
FIRST_ERR = re.compile(r"first\s+error\s*[:\-]?\s*\**\s*(?:step\s*)?(\d+|none)", re.I)


def mentions(text: str, value) -> bool:
    return value is not None and re.search(rf"(?<![\d.]){re.escape(str(value))}(?![\d.])", text.replace(",", "")) is not None


def plan_metrics(rec, gold):
    a = rec["answer_text"] if rec["closed_think"] else ""
    steps = len(STEP_LINE.findall(a))
    boxed = "\\boxed" in a
    return {
        "closed_think": rec["closed_think"],
        "n_step_lines": steps,
        "boxed": boxed,
        "mentions_gold": mentions(a, gold),
        "adherent": rec["closed_think"] and 1 <= steps <= 8 and not boxed,
    }


def critic_parse(rec):
    a = rec["answer_text"] if rec["closed_think"] else rec["full_text"][-1500:]
    v = VERDICT.findall(a)
    e = FIRST_ERR.findall(a)
    return {
        "verdict": v[-1].upper() if v else None,
        "first_error": (None if not e else (None if e[-1].lower() == "none" else int(e[-1]))),
        "first_error_raw": e[-1] if e else None,
        "answer": extract_answer(a) if "\\boxed" in a else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=3072)
    ap.add_argument("--model", default=None, help="default: Ouro-1.4B-Thinking")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="artifacts/looped/role_pilot")
    args = ap.parse_args()

    items = load_gsm8k_subset(args.n, args.seed)
    kw = dict(device=args.device, attn_implementation=args.attn)
    runner = OuroRunner(model_id=args.model, **kw) if args.model else OuroRunner(**kw)
    warmup(runner, [args.depth])
    env = environment(runner, args)
    env.update(subset=subset_fingerprint(items), indices=[it["idx"] for it in items])
    write_json(os.path.join(args.out, "env.json"), env)
    kw = dict(depth=args.depth, max_new_tokens=args.max_new, batch_size=args.bs)

    # ---- planner vs solver
    plan = run_agent(runner, [P.planner(it["question"]) for it in items], **kw)
    solve = run_agent(runner, [P.solver(it["question"]) for it in items], **kw)
    prow = []
    for it, pr, sr in zip(items, plan["records"], solve["records"]):
        prow.append({
            "idx": it["idx"], "gold": it["gold"],
            "planner": {**plan_metrics(pr, it["gold"]), **pr},
            "solver": {**plan_metrics(sr, it["gold"]), "pred": extract_answer(sr["full_text"]), **sr},
        })
    write_jsonl(os.path.join(args.out, "planner_rows.jsonl"), prow)

    def rate(rows, arm, key):
        return round(sum(bool(r[arm][key]) for r in rows) / len(rows), 4)

    planner_sum = {
        arm: {k: rate(prow, arm, k) for k in ("closed_think", "adherent", "boxed", "mentions_gold")}
        for arm in ("planner", "solver")
    }
    for arm in ("planner", "solver"):
        planner_sum[arm]["mean_step_lines"] = round(sum(r[arm]["n_step_lines"] for r in prow) / len(prow), 2)
        planner_sum[arm]["mean_gen_tokens"] = round(sum(r[arm]["gen_tokens"] for r in prow) / len(prow), 1)
    planner_sum["solver"]["acc"] = round(sum(r["solver"]["pred"] == r["gold"] for r in prow) / len(prow), 4)

    # ---- critic vs solver-with-candidate
    cases = []
    for it in items:
        cases.append((it, "correct", correct_solution(it)))
        bad = corrupt_solution(it, args.seed)
        if bad is not None:
            cases.append((it, "corrupted", bad))
    crit = run_agent(runner, [P.solution_critic(it["question"], c["text"]) for it, _, c in cases], **kw)
    ctrl = run_agent(runner, [P.solver_with_candidate(it["question"], c["text"]) for it, _, c in cases], **kw)
    crow = []
    for (it, kind, cand), cr, sr in zip(cases, crit["records"], ctrl["records"]):
        cp = critic_parse(cr)
        s_ans = extract_answer(sr["full_text"])
        truth = "CORRECT" if kind == "correct" else "INCORRECT"
        crow.append({
            "idx": it["idx"], "kind": kind, "gold": it["gold"], "candidate": cand,
            "critic": {**cp, "verdict_ok": cp["verdict"] == truth,
                       "localised": kind == "corrupted" and cp["first_error"] == cand["error_step"],
                       "answer_ok": cp["answer"] == it["gold"],
                       "copied_candidate": cp["answer"] == cand["stated_answer"], **cr},
            "solver": {"answer": s_ans,
                       "implied_verdict": None if s_ans is None else ("CORRECT" if s_ans == cand["stated_answer"] else "INCORRECT"),
                       "verdict_ok": s_ans is not None and (("CORRECT" if s_ans == cand["stated_answer"] else "INCORRECT") == truth),
                       "answer_ok": s_ans == it["gold"],
                       "copied_candidate": s_ans == cand["stated_answer"], **sr},
        })
    write_jsonl(os.path.join(args.out, "critic_rows.jsonl"), crow)

    def crit_sum(kind):
        rows = [r for r in crow if r["kind"] == kind]
        out = {"n": len(rows)}
        for arm in ("critic", "solver"):
            out[arm] = {k: round(sum(bool(r[arm][k]) for r in rows) / max(1, len(rows)), 4)
                        for k in ("verdict_ok", "answer_ok", "copied_candidate")}
        out["critic"]["verdict_parsed"] = round(sum(r["critic"]["verdict"] is not None for r in rows) / max(1, len(rows)), 4)
        out["critic"]["first_error_parsed"] = round(sum(r["critic"]["first_error_raw"] is not None for r in rows) / max(1, len(rows)), 4)
        if kind == "corrupted":
            out["critic"]["localised"] = round(sum(r["critic"]["localised"] for r in rows) / max(1, len(rows)), 4)
            out["paired_answer_ok_critic_minus_solver"] = paired(
                [r["critic"]["answer_ok"] for r in rows], [r["solver"]["answer_ok"] for r in rows])
            out["paired_verdict_ok_critic_minus_solver"] = paired(
                [r["critic"]["verdict_ok"] for r in rows], [r["solver"]["verdict_ok"] for r in rows])
        return out

    summary = {
        "subset": env["subset"], "depth": args.depth,
        "planner_vs_solver": planner_sum,
        "critic_on_correct": crit_sum("correct"),
        "critic_on_corrupted": crit_sum("corrupted"),
        "cost": {name: {"wall_s": round(r["wall_s"], 1), "executed_token_loops": r["executed_token_loops"]}
                 for name, r in (("planner", plan), ("solver", solve), ("critic", crit), ("solver_with_candidate", ctrl))},
    }
    write_json(os.path.join(args.out, "summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
