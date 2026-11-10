#!/usr/bin/env python3
"""Stage 4 (main gate): text-handoff MAS vs single model at fixed depth.

Arms, all on the same questions, same extractor and scorer, one frozen backbone:

  single   greedy solver
  selfrev  single's attempt + one greedy self-revision pass (same model, no roles)
  mas      planner -> critic -> final solver, text handoffs (post-</think> text)
  sc       k sampled solver attempts; majority vote over the first m, for every m

Every arm's cost counts all prompt processing, handoff processing and generation
(``work.loop_tokens``, ``work.attn_kv_loop``) plus measured batched wall time.
The analysis picks the self-consistency size whose mean loop-token work is
closest to MAS, so the single model gets a comparable budget.

  python scripts/looped_mas_gate.py --n 200 --depth 4 --out artifacts/looped/mas_gate_r4
  python scripts/looped_mas_gate.py --analyze_only --out artifacts/looped/mas_gate_r4
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from looped import prompts as P  # noqa: E402
from looped.harness import (  # noqa: E402
    add_work, environment, handoff, read_jsonl, run_agent, warmup, write_json, write_jsonl,
)
from looped.stats import mean, paired, wilson  # noqa: E402
from looped.tasks import extract_answer, load_gsm8k_subset, subset_fingerprint  # noqa: E402


def majority(answers):
    votes = [a for a in answers if a is not None]
    if not votes:
        return None
    c = collections.Counter(votes)
    top = max(c.values())
    return next(a for a in votes if c[a] == top)


def run(args) -> None:
    from looped.ouro_adapter import OuroRunner

    items = load_gsm8k_subset(args.n, args.seed)
    kw = dict(device=args.device, attn_implementation=args.attn)
    runner = OuroRunner(model_id=args.model, **kw) if args.model else OuroRunner(**kw)
    warmup(runner, [args.depth])
    env = environment(runner, args)
    env.update(subset=subset_fingerprint(items), indices=[it["idx"] for it in items])
    write_json(os.path.join(args.out, "env.json"), env)
    kw = dict(depth=args.depth, batch_size=args.bs)
    timing = read_json_opt(os.path.join(args.out, "timing.json")) or {}

    def go(name, texts, max_new, **extra):
        path = os.path.join(args.out, f"arm_{name}.jsonl")
        if os.path.exists(path):
            saved = read_jsonl(path)
            if len(saved) == len(items):
                print(f"[{name}] resumed {len(saved)} rows", flush=True)
                return saved
        res = run_agent(runner, texts, max_new_tokens=max_new, **kw, **extra)
        timing[name] = {"wall_s": round(res["wall_s"], 2), "executed_token_loops": res["executed_token_loops"],
                        "peak_mem_gb": res["peak_mem_gb"]}
        write_jsonl(path, res["records"])
        write_json(os.path.join(args.out, "timing.json"), timing)
        print(f"[{name}] {timing[name]}", flush=True)
        return res["records"]

    single = go("single", [P.solver(it["question"]) for it in items], args.max_new_solver)
    rev = go("selfrev_pass2", [P.self_revision(it["question"], handoff(s)) for it, s in zip(items, single)],
             args.max_new_solver)
    plan = go("mas_planner", [P.planner(it["question"]) for it in items], args.max_new_planner)
    crit = go("mas_critic", [P.plan_critic(it["question"], handoff(p)) for it, p in zip(items, plan)],
              args.max_new_critic)
    fin = go("mas_final", [P.final_solver(it["question"], handoff(p), handoff(c))
                           for it, p, c in zip(items, plan, crit)], args.max_new_solver)
    samples = []
    for s in range(args.sc_k):
        samples.append(go(f"sc_sample{s}", [P.solver(it["question"]) for it in items], args.max_new_solver,
                          temperature=args.temperature, top_p=args.top_p, seed=1000 * (s + 1)))

    rows = []
    for i, it in enumerate(items):
        rows.append({
            "idx": it["idx"], "gold": it["gold"], "question": it["question"],
            "single": single[i], "selfrev_pass2": rev[i],
            "mas": {"planner": plan[i], "critic": crit[i], "final": fin[i]},
            "sc": [smp[i] for smp in samples],
        })
    write_jsonl(os.path.join(args.out, "rows.jsonl"), rows)
    write_json(os.path.join(args.out, "timing.json"), timing)


def analyze(args) -> None:
    rows = read_jsonl(os.path.join(args.out, "rows.jsonl"))
    timing = read_json_opt(os.path.join(args.out, "timing.json"))
    n = len(rows)
    per = []
    for r in rows:
        g = r["gold"]
        single = extract_answer(r["single"]["full_text"])
        rev = extract_answer(r["selfrev_pass2"]["full_text"])
        mas = extract_answer(r["mas"]["final"]["full_text"])
        sc_ans = [extract_answer(s["full_text"]) for s in r["sc"]]
        rec = {
            "idx": r["idx"], "gold": g,
            "single": {"pred": single, "ok": single == g, "work": r["single"]["work"]},
            "selfrev": {"pred": rev, "ok": rev == g,
                        "work": add_work(r["single"]["work"], r["selfrev_pass2"]["work"])},
            "mas": {"pred": mas, "ok": mas == g,
                    "work": add_work(*(r["mas"][k]["work"] for k in ("planner", "critic", "final")))},
            "sc": [],
        }
        for m in range(1, len(sc_ans) + 1):
            v = majority(sc_ans[:m])
            rec["sc"].append({"m": m, "pred": v, "ok": v == g,
                              "work": add_work(*(s["work"] for s in r["sc"][:m]))})
        per.append(rec)
    write_jsonl(os.path.join(args.out, "per_question.jsonl"), per)

    def arm(name, pick):
        oks = [pick(p)["ok"] for p in per]
        works = [pick(p)["work"] for p in per]
        return {
            "acc": round(sum(oks) / n, 4), "acc_ci95": wilson(sum(oks), n),
            "mean_loop_tokens": round(mean(w["loop_tokens"] for w in works), 1),
            "mean_attn_kv_loop": round(mean(w["attn_kv_loop"] for w in works), 1),
            "mean_gen_tokens": round(mean(w["gen_tokens"] for w in works), 1),
            "mean_prompt_tokens": round(mean(w["prompt_tokens"] for w in works), 1),
        }

    arms = {"single": arm("single", lambda p: p["single"]),
            "selfrev": arm("selfrev", lambda p: p["selfrev"]),
            "mas": arm("mas", lambda p: p["mas"])}
    k = len(per[0]["sc"]) if per else 0
    for m in range(1, k + 1):
        arms[f"sc@{m}"] = arm(f"sc@{m}", lambda p, m=m: p["sc"][m - 1])
    mas_w = arms["mas"]["mean_loop_tokens"]
    m_star = min(range(1, k + 1), key=lambda m: abs(arms[f"sc@{m}"]["mean_loop_tokens"] - mas_w)) if k else None

    def ok(name):
        if name.startswith("sc@"):
            m = int(name[3:])
            return [p["sc"][m - 1]["ok"] for p in per]
        return [p[name]["ok"] for p in per]

    comps = {f"mas_minus_{b}": paired(ok("mas"), ok(b)) for b in ("single", "selfrev")}
    if m_star:
        comps[f"mas_minus_sc@{m_star}"] = paired(ok("mas"), ok(f"sc@{m_star}"))
    comps["selfrev_minus_single"] = paired(ok("selfrev"), ok("single"))
    wall = {}
    if timing:
        wall["single"] = timing["single"]["wall_s"]
        wall["selfrev"] = timing["single"]["wall_s"] + timing["selfrev_pass2"]["wall_s"]
        wall["mas"] = sum(timing[f"mas_{x}"]["wall_s"] for x in ("planner", "critic", "final"))
        for m in range(1, k + 1):
            wall[f"sc@{m}"] = sum(timing[f"sc_sample{s}"]["wall_s"] for s in range(m))
    for a, w in wall.items():
        arms[a]["batched_wall_s_total"] = round(w, 2)
    summary = {"n": n, "arms": arms, "sc_matched_m": m_star, "paired": comps,
               "note": "wall_s is batched (same batch size for every arm): throughput, not request latency"}
    write_json(os.path.join(args.out, "summary.json"), summary)
    print(json.dumps(summary, indent=2))


def read_json_opt(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_new_solver", type=int, default=3072)
    ap.add_argument("--max_new_planner", type=int, default=2048)
    ap.add_argument("--max_new_critic", type=int, default=2048)
    ap.add_argument("--sc_k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.7)
    ap.add_argument("--model", default=None, help="default: Ouro-1.4B-Thinking")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default=None)
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--out", default="artifacts/looped/mas_gate_r4")
    args = ap.parse_args()
    if not args.analyze_only:
        run(args)
    analyze(args)


if __name__ == "__main__":
    main()
