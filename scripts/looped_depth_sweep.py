#!/usr/bin/env python3
"""Stage 2: single-model GSM8K accuracy/cost at each recurrence depth.

Same questions, prompt, extraction and greedy decoding at every depth. Batched
arms give accuracy, tokens, executed loop-token work and throughput; an
unbatched probe on the first ``--latency_n`` items gives per-request latency.

  python scripts/looped_depth_sweep.py --n 200 --depths 1,2,3,4 --out artifacts/looped/depth_sweep
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from looped import prompts as P  # noqa: E402
from looped.harness import environment, run_agent, warmup, write_json, write_jsonl  # noqa: E402
from looped.ouro_adapter import OuroRunner  # noqa: E402
from looped.stats import mean, wilson  # noqa: E402
from looped.tasks import extract_answer, is_correct, load_gsm8k_subset, subset_fingerprint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depths", default="1,2,3,4")
    ap.add_argument("--max_new", type=int, default=3072)
    ap.add_argument("--bs", type=int, default=40, help="batch size at full depth")
    ap.add_argument("--bs_cap", type=int, default=100)
    ap.add_argument("--latency_n", type=int, default=10)
    ap.add_argument("--latency_max_new", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="artifacts/looped/depth_sweep")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    items = load_gsm8k_subset(args.n, args.seed)
    t0 = time.perf_counter()
    runner = OuroRunner(device=args.device)
    load_s = time.perf_counter() - t0
    warm_s = warmup(runner, depths)
    env = environment(runner, args)
    env.update(load_s=round(load_s, 1), warmup_s=round(warm_s, 1),
               subset=subset_fingerprint(items), indices=[it["idx"] for it in items])
    write_json(os.path.join(args.out, "env.json"), env)

    texts = [P.solver(it["question"]) for it in items]
    summary = {"subset": env["subset"], "n": len(items), "depths": {}}
    for r in depths:
        # cache bytes per token scale with depth, so shallower runs fit larger batches
        bs = min(args.bs_cap, args.bs * runner.max_depth // r)
        res = run_agent(runner, texts, depth=r, max_new_tokens=args.max_new, batch_size=bs)
        rows = []
        for it, rec in zip(items, res["records"]):
            rows.append({
                "idx": it["idx"], "depth": r, "gold": it["gold"],
                "pred": extract_answer(rec["full_text"]),
                "correct": is_correct(rec["full_text"], it["gold"]),
                **rec,
            })
        write_jsonl(os.path.join(args.out, f"rows_r{r}.jsonl"), rows)

        lat = []
        for it, t in list(zip(items, texts))[: args.latency_n]:
            g = runner.generate([runner.chat_ids(P.msgs(t))], depth=r, max_new_tokens=args.latency_max_new)
            lat.append({"idx": it["idx"], "s": g.wall_s, "gen_tokens": len(g.token_ids[0]),
                        "s_per_token": g.wall_s / max(1, len(g.token_ids[0]))})
        write_jsonl(os.path.join(args.out, f"latency_r{r}.jsonl"), lat)

        k = sum(x["correct"] for x in rows)
        gen = sum(x["gen_tokens"] for x in rows)
        summary["depths"][str(r)] = {
            "acc": round(k / len(rows), 4), "acc_ci95": wilson(k, len(rows)),
            "mean_prompt_tokens": round(mean(x["prompt_tokens"] for x in rows), 1),
            "mean_gen_tokens": round(mean(x["gen_tokens"] for x in rows), 1),
            "censored": sum(not x["hit_eos"] for x in rows),
            "mean_loop_tokens": round(mean(x["work"]["loop_tokens"] for x in rows), 1),
            "mean_attn_kv_loop": round(mean(x["work"]["attn_kv_loop"] for x in rows), 1),
            "executed_token_loops_incl_padding": res["executed_token_loops"],
            "batch_size": bs,
            "batched_wall_s": round(res["wall_s"], 2),
            "batched_tok_per_s": round(gen / res["wall_s"], 1),
            "peak_mem_gb": res["peak_mem_gb"], "peak_mem_kind": res["peak_mem_kind"],
            "latency_probe": {
                "n": len(lat), "max_new": args.latency_max_new,
                "median_s_per_token": sorted(x["s_per_token"] for x in lat)[len(lat) // 2] if lat else None,
                "mean_s": round(mean(x["s"] for x in lat), 3) if lat else None,
            },
        }
        print(f"[r={r}] {summary['depths'][str(r)]}", flush=True)
        write_json(os.path.join(args.out, "summary.json"), summary)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
