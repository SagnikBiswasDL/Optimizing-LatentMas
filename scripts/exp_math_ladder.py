#!/usr/bin/env python3
"""Math ladder: one MATH-train cache, optional small-K TTC, then eviction.

  build  Collect N MATH-train silent-agent tapes, Mean-Replay, freeze KV.
  eval   Same items: none / frozen / frozen_k{2,5} / real, each ± evict.

Headline: match LatentMAS accuracy at lower end-to-end latency.

Examples
--------
  # Once, on GPU: 1000 MATH-train donors.
  python scripts/exp_math_ladder.py --mode build --stat_n 1000 \\
      --out_dir artifacts/math_ladder/math1k

  # Jiayi gate: is the frozen cache enough?
  python scripts/exp_math_ladder.py --mode eval \\
      --cache artifacts/math_ladder/math1k/cache.pt \\
      --task gsm8k --n 100 --arms none,frozen,real \\
      --out_dir artifacts/math_ladder/gate_gsm8k

  # If not: residual K, then light eviction.
  python scripts/exp_math_ladder.py --mode eval \\
      --cache artifacts/math_ladder/math1k/cache.pt \\
      --task math --n 100 --k_ttc 0,2,5 --evict_budget 0,64 \\
      --out_dir artifacts/math_ladder/math_kttc
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import load_aime2024, load_aime2025, load_gsm8k, load_math  # noqa: E402
from methods import default_agents  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.latent_eval import (  # noqa: E402
    build_upstream_timed,
    decode_batch,
    deep_clone,
    graded,
    make_ns,
    mean_ci,
    peak_mb,
    pred_short,
    reset_peak,
    split_past,
    stack_past,
    sync,
    to_cpu,
    to_dev,
)
from seal.math_cache import (  # noqa: E402
    UNIFIED_MATH_PROMPT,
    append_donor,
    cache_as_past,
    cache_path,
    donor_checkpoint_path,
    empty_donor_state,
    expand_ladder_arms,
    gate_verdict,
    load_donor_state_lists,
    load_unified_cache,
    mean_tapes,
    n_kept,
    parse_arm,
    parse_int_list,
    save_donor_state,
    save_unified_cache,
)
from seal.cache_bank import kv_mb, num_positions  # noqa: E402
from seal.relay_compress import RelayCompressor  # noqa: E402
from utils import auto_device, set_seed  # noqa: E402


def load_task(task: str, split: str):
    if task == "aime2024":
        return list(load_aime2024(split="train"))
    if task == "aime2025":
        return list(load_aime2025(split="train"))
    if task == "math":
        return list(load_math(split=split))
    return list(load_gsm8k(split=split))


def maybe_load_model(args, ns):
    from models import ModelWrapper

    print(f"[load] {args.model_name} device={args.device}", flush=True)
    return ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)


def evict_one(past, budget: int, sink: int, importance: str):
    if past is None or int(budget) <= 0:
        return past, None
    comp = RelayCompressor(mode="evict", budget=int(budget), sink=int(sink),
                           importance=importance)
    out, st = comp.compress(past)
    return out, st.as_dict()


def run_build(args) -> str:
    ns = make_ns(args, task="math")
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = donor_checkpoint_path(args.out_dir)
    out_cache = cache_path(args.out_dir)
    want = int(args.stat_n)
    train_pool = list(load_math(split="train"))
    if args.stat_pool > 0:
        train_pool = train_pool[: int(args.stat_pool)]
    print(
        f"[build] MATH-train pool={len(train_pool)} want={want} k={args.k} "
        f"ckpt={ckpt}",
        flush=True,
    )
    if os.path.isfile(ckpt):
        state = load_donor_state_lists(ckpt)
        print(
            f"[build] resume kept={n_kept(state)} tried={state['n_tried']} "
            f"next_idx={state['next_idx']}",
            flush=True,
        )
        if int(state.get("k") or 0) != int(args.k):
            raise RuntimeError(f"ckpt k={state.get('k')} != --k {args.k}")
    else:
        state = empty_donor_state(args.k, args.model_name)

    wrapper = None
    up_agents = [a for a in default_agents() if a.role != "judger"]
    t0 = time.perf_counter()
    idx = int(state["next_idx"])
    while n_kept(state) < want:
        if idx >= len(train_pool):
            raise RuntimeError(
                f"MATH-train exhausted at {n_kept(state)}/{want} kept "
                f"(tried {state['n_tried']})"
            )
        if wrapper is None:
            if not torch.cuda.is_available():
                raise SystemExit("CUDA required for --mode build")
            wrapper = maybe_load_model(args, ns)
        it = train_pool[idx]
        got: Dict[str, torch.Tensor] = {}
        past, times, peaks = build_upstream_timed(
            wrapper, [it["question"]], args.k, ns, up_agents,
            collect_latents=True, out_latents=got,
        )
        row = {
            "idx": idx,
            "subject": it.get("subject") or "",
            "level": it.get("level_int"),
            "planner_s": times.get("planner", 0.0),
            "critic_s": times.get("critic", 0.0),
            "refiner_s": times.get("refiner", 0.0),
            "upstream_s": times["upstream"],
            "pos": int(num_positions(past)),
            "mb": float(kv_mb(past)),
            "peak_mb_planner": peaks.get("planner", 0.0),
            "peak_mb_critic": peaks.get("critic", 0.0),
            "peak_mb_refiner": peaks.get("refiner", 0.0),
        }
        append_donor(state, latents=got, row=row, next_idx=idx + 1)
        kept = n_kept(state)
        print(
            f"[build] donor {kept}/{want} idx={idx} "
            f"P={row['planner_s']:.2f}s C={row['critic_s']:.2f}s R={row['refiner_s']:.2f}s "
            f"pos={row['pos']}",
            flush=True,
        )
        del past
        torch.cuda.empty_cache()
        idx += 1
        if kept % int(args.ckpt_every) == 0 or kept >= want:
            save_donor_state(state, ckpt)
            print(f"[build] checkpoint {ckpt} kept={kept}", flush=True)

    forced = mean_tapes(state)
    for role, emb in forced.items():
        print(
            f"[replay] {role} K={emb.shape[0]} D={emb.shape[1]} "
            f"l2={float(emb.norm(dim=-1).mean()):.3f} n={n_kept(state)}",
            flush=True,
        )
    if wrapper is None:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required to replay Mean-Replay KV")
        wrapper = maybe_load_model(args, ns)
    print(f"[replay] type prefill: {UNIFIED_MATH_PROMPT[:80]}...", flush=True)
    past_r, _, _ = build_upstream_timed(
        wrapper, [UNIFIED_MATH_PROMPT], args.k, ns, up_agents,
        forced_by_role=forced,
    )
    synth = to_cpu(past_r)
    del past_r
    torch.cuda.empty_cache()
    t_pre = time.perf_counter() - t0
    meta = {
        "scaffold": "replay",
        "donor_task": "math",
        "n_donors": n_kept(state),
        "n_tried": int(state["n_tried"]),
        "k": int(args.k),
        "model_name": args.model_name,
        "type_question": UNIFIED_MATH_PROMPT,
        "t_precompute_s": float(t_pre),
        "replay_norms": {
            r: {
                "k": int(forced[r].shape[0]),
                "d": int(forced[r].shape[1]),
                "mean_l2": float(forced[r].norm(dim=-1).mean()),
                "n_donors": n_kept(state),
            }
            for r in forced
        },
    }
    save_unified_cache(out_cache, past=synth, meta=meta, mean_latents=forced)
    with open(os.path.join(args.out_dir, "build_report.json"), "w") as f:
        json.dump(
            {**meta, "cache": out_cache, "n_pos": int(num_positions(synth)),
             "mb": float(kv_mb(synth))},
            f, indent=2,
        )
    print(
        f"[build] wrote {out_cache} pos={num_positions(synth)} "
        f"{kv_mb(synth):.1f}MB in {t_pre:.1f}s",
        flush=True,
    )
    print("MATH_LADDER_BUILD_DONE", flush=True)
    return out_cache


def _acc_str(rows, method):
    rs = [r for r in rows if r["method"] == method]
    if not rs:
        return "na"
    return f"{float(np.mean([r['correct'] for r in rs])):.3f}(n={len(rs)})"


def run_eval(args):
    if not args.cache:
        args.cache = cache_path(args.out_dir)
    if not os.path.isfile(args.cache):
        raise SystemExit(f"missing cache {args.cache}; run --mode build first")
    payload = load_unified_cache(args.cache)
    frozen_cpu = cache_as_past(payload, device="cpu")
    pre_meta = dict(payload.get("meta") or {})
    print(
        f"[eval] cache={args.cache} pos={num_positions(frozen_cpu)} "
        f"{kv_mb(frozen_cpu):.1f}MB donors={pre_meta.get('n_donors')}",
        flush=True,
    )

    k_ttc = parse_int_list(args.k_ttc, (0,))
    evict_b = parse_int_list(args.evict_budget, (0,))
    requested = [a.strip() for a in str(args.arms).split(",") if a.strip()] if args.arms else None
    arms = expand_ladder_arms(k_ttc, evict_b, requested)
    specs = [parse_arm(a) for a in arms]
    print(f"[eval] arms={arms}", flush=True)

    items = load_task(args.task, args.split)[: int(args.n)]
    for i, it in enumerate(items):
        it["idx"] = i
    ns = make_ns(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for --mode eval")
    wrapper = maybe_load_model(args, ns)
    up_agents = [a for a in default_agents() if a.role != "judger"]
    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass

    dtype = next(wrapper.model.parameters()).dtype
    frozen_cpu = cache_as_past(payload, device="cpu", dtype=dtype)

    os.makedirs(args.out_dir, exist_ok=True)
    dump_path = os.path.join(args.out_dir, "latency_rows.jsonl")
    rows: List[Dict[str, Any]] = []
    t_wall = time.time()
    bs = max(1, int(args.generate_bs))
    ours_specs = [s for s in specs if s["kind"] == "ours"]
    live_specs = [s for s in specs if s["kind"] == "live"]
    k_needed = sorted({int(s["k_ttc"]) for s in ours_specs})
    k_live = sorted({int(s["k_ttc"]) for s in live_specs})
    want_real = any(s["kind"] == "real" for s in specs)
    want_none = any(s["kind"] == "none" for s in specs)

    dump_f = open(dump_path, "w")
    for start in range(0, len(items), bs):
        batch = items[start : start + bs]
        B = len(batch)
        questions = [it["question"] for it in batch]
        golds = [it["gold"] for it in batch]
        jmsgs = [
            build_agent_message_sequential_latent_mas(
                role="judger", question=q, context="", method="latent_mas", args=ns)
            for q in questions
        ]
        _, jids, jmask, _ = wrapper.prepare_chat_batch(jmsgs, add_generation_prompt=True)
        batch_rows: List[Dict[str, Any]] = []
        t_real_up = t_real_j = 0.0

        if want_real:
            past, up_t, up_peak = build_upstream_timed(
                wrapper, questions, args.k, ns, up_agents)
            full_pos, full_mb = num_positions(past), kv_mb(past)
            per_p = up_t.get("planner", 0.0) / B
            per_c = up_t.get("critic", 0.0) / B
            per_r = up_t.get("refiner", 0.0) / B
            per_up = up_t["upstream"] / B
            t_real_up = up_t["upstream"]
            reset_peak()
            sync()
            td0 = time.perf_counter()
            texts, ntoks, eoss = decode_batch(
                wrapper, jids, jmask, split_past(deep_clone(past), B),
                args.judger_budget, temperature=args.temperature, top_p=args.top_p)
            sync()
            t_real_j = time.perf_counter() - td0
            per_j = t_real_j / B
            for b, it in enumerate(batch):
                batch_rows.append({
                    "idx": it.get("idx", start + b),
                    "method": "real",
                    "k_ttc": int(args.k),
                    "evict": 0,
                    "batch_size": B,
                    "planner_s": per_p,
                    "critic_s": per_c,
                    "refiner_s": per_r,
                    "upstream_s": per_up,
                    "cache_load_s": 0.0,
                    "judger_s": per_j,
                    "e2e_s": per_up + per_j,
                    "tokens": int(ntoks[b]),
                    "eos": bool(eoss[b]),
                    "correct": bool(graded(texts[b], golds[b], args.task)),
                    "pred": pred_short(texts[b]),
                    "cache_pos": int(full_pos),
                    "cache_mb": float(full_mb) / B,
                    "upstream_forwards": 3 * (int(args.k) + 1),
                    "peak_mb_planner": up_peak.get("planner", 0.0),
                    "peak_mb_critic": up_peak.get("critic", 0.0),
                    "peak_mb_refiner": up_peak.get("refiner", 0.0),
                })
            del past
            torch.cuda.empty_cache()

        prefix_by_k: Dict[int, Any] = {}
        times_by_k: Dict[int, Dict[str, float]] = {}
        for k_res in k_needed:
            sync()
            tl0 = time.perf_counter()
            clones = [to_dev(deep_clone(frozen_cpu), wrapper.device) for _ in range(B)]
            t_load = time.perf_counter() - tl0
            if k_res <= 0:
                prefix_by_k[k_res] = [to_cpu(c) for c in clones]
                times_by_k[k_res] = {
                    "planner": 0.0, "critic": 0.0, "refiner": 0.0,
                    "upstream": 0.0, "load": t_load,
                }
                del clones
            else:
                stacked = stack_past(clones)
                past, up_t, _ = build_upstream_timed(
                    wrapper, questions, k_res, ns, up_agents,
                    start_past=stacked, latent_steps=k_res,
                )
                prefix_by_k[k_res] = [to_cpu(p) for p in split_past(past, B)]
                times_by_k[k_res] = {
                    "planner": up_t.get("planner", 0.0),
                    "critic": up_t.get("critic", 0.0),
                    "refiner": up_t.get("refiner", 0.0),
                    "upstream": up_t["upstream"],
                    "load": t_load,
                }
                del past, stacked, clones
            torch.cuda.empty_cache()

        for spec in ours_specs:
            k_res = int(spec["k_ttc"])
            cpu_caches = prefix_by_k[k_res]
            evict_stats = None
            used = cpu_caches
            if int(spec["evict"]) > 0:
                used = []
                for c in cpu_caches:
                    out, st = evict_one(c, spec["evict"], args.sink, args.importance)
                    used.append(out)
                    evict_stats = st
            ut = times_by_k[k_res]
            per_up = ut["upstream"] / B
            per_load = ut["load"] / B
            per_p = ut["planner"] / B
            per_c = ut["critic"] / B
            per_r = ut["refiner"] / B
            reset_peak()
            sync()
            td1 = time.perf_counter()
            texts, ntoks, eoss = decode_batch(
                wrapper, jids, jmask,
                [to_dev(deep_clone(c), wrapper.device) for c in used],
                args.judger_budget, temperature=args.temperature, top_p=args.top_p)
            sync()
            t_j = time.perf_counter() - td1
            per_j = t_j / B
            fwds = 0 if k_res <= 0 else 3 * (k_res + 1)
            for b, it in enumerate(batch):
                batch_rows.append({
                    "idx": it.get("idx", start + b),
                    "method": spec["name"],
                    "k_ttc": k_res,
                    "evict": int(spec["evict"]),
                    "batch_size": B,
                    "planner_s": per_p,
                    "critic_s": per_c,
                    "refiner_s": per_r,
                    "upstream_s": per_up,
                    "cache_load_s": per_load,
                    "judger_s": per_j,
                    "e2e_s": per_load + per_up + per_j,
                    "tokens": int(ntoks[b]),
                    "eos": bool(eoss[b]),
                    "correct": bool(graded(texts[b], golds[b], args.task)),
                    "pred": pred_short(texts[b]),
                    "cache_pos": int(num_positions(used[b])),
                    "cache_mb": float(kv_mb(used[b])),
                    "upstream_forwards": fwds,
                    "evict_stats": evict_stats,
                })
            del texts
            torch.cuda.empty_cache()

        live_by_k: Dict[int, Any] = {}
        live_times: Dict[int, Dict[str, float]] = {}
        for k_res in k_live:
            past, up_t, _ = build_upstream_timed(
                wrapper, questions, k_res, ns, up_agents,
                latent_steps=k_res,
            )
            live_by_k[k_res] = [to_cpu(p) for p in split_past(past, B)]
            live_times[k_res] = {
                "planner": up_t.get("planner", 0.0),
                "critic": up_t.get("critic", 0.0),
                "refiner": up_t.get("refiner", 0.0),
                "upstream": up_t["upstream"],
            }
            del past
            torch.cuda.empty_cache()

        for spec in live_specs:
            k_res = int(spec["k_ttc"])
            used = live_by_k[k_res]
            ut = live_times[k_res]
            per_up = ut["upstream"] / B
            per_p = ut["planner"] / B
            per_c = ut["critic"] / B
            per_r = ut["refiner"] / B
            reset_peak()
            sync()
            td_live = time.perf_counter()
            texts, ntoks, eoss = decode_batch(
                wrapper, jids, jmask,
                [to_dev(deep_clone(c), wrapper.device) for c in used],
                args.judger_budget, temperature=args.temperature, top_p=args.top_p)
            sync()
            t_j = time.perf_counter() - td_live
            per_j = t_j / B
            fwds = 3 * (k_res + 1)
            for b, it in enumerate(batch):
                batch_rows.append({
                    "idx": it.get("idx", start + b),
                    "method": spec["name"],
                    "k_ttc": k_res,
                    "evict": 0,
                    "batch_size": B,
                    "planner_s": per_p,
                    "critic_s": per_c,
                    "refiner_s": per_r,
                    "upstream_s": per_up,
                    "cache_load_s": 0.0,
                    "judger_s": per_j,
                    "e2e_s": per_up + per_j,
                    "tokens": int(ntoks[b]),
                    "eos": bool(eoss[b]),
                    "correct": bool(graded(texts[b], golds[b], args.task)),
                    "pred": pred_short(texts[b]),
                    "cache_pos": int(num_positions(used[b])),
                    "cache_mb": float(kv_mb(used[b])),
                    "upstream_forwards": fwds,
                })
            del texts
            torch.cuda.empty_cache()

        if want_none:
            reset_peak()
            sync()
            td2 = time.perf_counter()
            texts, ntoks, eoss = decode_batch(
                wrapper, jids, jmask, [None] * B, args.judger_budget,
                temperature=args.temperature, top_p=args.top_p)
            sync()
            t_n = time.perf_counter() - td2
            per_j = t_n / B
            for b, it in enumerate(batch):
                batch_rows.append({
                    "idx": it.get("idx", start + b),
                    "method": "none",
                    "k_ttc": 0,
                    "evict": 0,
                    "batch_size": B,
                    "planner_s": 0.0,
                    "critic_s": 0.0,
                    "refiner_s": 0.0,
                    "upstream_s": 0.0,
                    "cache_load_s": 0.0,
                    "judger_s": per_j,
                    "e2e_s": per_j,
                    "tokens": int(ntoks[b]),
                    "eos": bool(eoss[b]),
                    "correct": bool(graded(texts[b], golds[b], args.task)),
                    "pred": pred_short(texts[b]),
                    "cache_pos": 0,
                    "cache_mb": 0.0,
                    "upstream_forwards": 0,
                })

        rows.extend(batch_rows)
        for rec in batch_rows:
            dump_f.write(json.dumps(rec) + "\n")
        dump_f.flush()
        methods = sorted({r["method"] for r in batch_rows})
        accs = " ".join(f"{m}={_acc_str(rows, m)}" for m in methods)
        print(
            f"[eval] items {start+1}-{start+B}/{len(items)} B={B} "
            f"real_up={t_real_up:.2f}s real_J={t_real_j:.2f}s {accs} "
            f"elapsed={time.time()-t_wall:.0f}s",
            flush=True,
        )
    dump_f.close()

    report = summarize(rows, args, pre_meta, len(items))
    with open(os.path.join(args.out_dir, "latency_rows.json"), "w") as f:
        json.dump(rows, f)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print_table(report)
    print_gate(report)
    print(f"[done] {os.path.join(args.out_dir, 'report.json')}", flush=True)
    print("MATH_LADDER_EVAL_DONE", flush=True)
    return report


def summarize(rows, args, pre_meta, n_test):
    methods = []
    seen = set()
    for r in rows:
        if r["method"] not in seen:
            seen.add(r["method"])
            methods.append(r["method"])
    out: Dict[str, Any] = {
        "config": {
            "task": args.task, "n": n_test, "k": args.k,
            "k_ttc": args.k_ttc, "evict_budget": args.evict_budget,
            "generate_bs": args.generate_bs, "judger_budget": args.judger_budget,
            "cache": args.cache, "seed": args.seed,
        },
        "precompute": {
            k: pre_meta.get(k) for k in (
                "n_donors", "k", "scaffold", "donor_task", "t_precompute_s",
                "n_pos", "mb", "type_question", "model_name",
            )
        },
        "arms": {},
    }
    keys = [
        "planner_s", "critic_s", "refiner_s", "upstream_s", "cache_load_s",
        "judger_s", "e2e_s", "tokens", "cache_mb", "cache_pos", "upstream_forwards",
    ]
    by = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        by[m] = {r["idx"]: r for r in rs}
        blk = {
            "n": len(rs),
            "acc": float(np.mean([r["correct"] for r in rs])),
            "k_ttc": int(rs[0].get("k_ttc") or 0),
            "evict": int(rs[0].get("evict") or 0),
        }
        for k in keys:
            if k in rs[0]:
                blk[k] = mean_ci([r[k] for r in rs], seed=args.seed)
        out["arms"][m] = blk

    paired = {}
    if "real" in by:
        real_idx = set(by["real"])
        for m, idxmap in by.items():
            if m == "real":
                continue
            common = sorted(real_idx & set(idxmap))
            for k in ("e2e_s", "tokens", "correct", "upstream_s", "cache_mb"):
                diffs = [idxmap[i][k] - by["real"][i][k] for i in common]
                paired[f"{m}_minus_real_{k}"] = mean_ci(diffs, seed=args.seed)
    out["paired"] = paired
    amort = float(pre_meta.get("t_precompute_s") or 0.0) / max(n_test, 1)
    out["precompute_amortized_per_query_s"] = amort
    out["gate"] = gate_verdict(out)
    return out


def print_table(report):
    print("\n=== MATH LADDER (mean / item) ===", flush=True)
    hdr = (
        f"{'arm':<22}{'acc':>8}{'e2e':>8}{'up':>8}{'J':>8}"
        f"{'tok':>8}{'MB':>8}{'pos':>8}{'fwd':>8}"
    )
    print(hdr, flush=True)
    for name, a in report.get("arms", {}).items():
        def m(key):
            blk = a.get(key) or {}
            return float(blk.get("mean") or 0.0)
        print(
            f"{name:<22}{a.get('acc', 0):8.3f}{m('e2e_s'):8.2f}{m('upstream_s'):8.2f}"
            f"{m('judger_s'):8.2f}{m('tokens'):8.1f}{m('cache_mb'):8.1f}"
            f"{m('cache_pos'):8.0f}{m('upstream_forwards'):8.0f}",
            flush=True,
        )


def print_gate(report):
    g = report.get("gate") or {}
    print("\n=== JIAYI GATE ===", flush=True)
    print(
        f"  frozen={g.get('frozen_acc')}  real={g.get('real_acc')}  none={g.get('none_acc')}",
        flush=True,
    )
    print(f"  recommend={g.get('recommend')}", flush=True)
    print(f"  {g.get('reason')}", flush=True)


def default_n(task: str, smoke: bool) -> int:
    if smoke:
        return 2
    if task.startswith("aime"):
        return 30
    return 100


def default_budget(task: str) -> int:
    if task.startswith("aime"):
        return 8192
    if task == "math":
        return 2048
    return 1024


def default_bs(task: str) -> int:
    if task.startswith("aime"):
        return 1
    return 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="eval", choices=["build", "eval", "both"])
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--task", default="gsm8k",
                    choices=["gsm8k", "math", "aime2024", "aime2025"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--stat_n", type=int, default=1000,
                    help="MATH-train donors for the unified cache (Jiayi: 1000).")
    ap.add_argument("--stat_pool", type=int, default=0,
                    help="Max MATH-train items to scan (0 = all).")
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--k", type=int, default=10, help="Silent-agent K for donors and Real.")
    ap.add_argument("--k_ttc", default="0",
                    help="Residual silent-agent K on the test item, comma-separated. 0 = frozen only.")
    ap.add_argument("--evict_budget", default="0",
                    help="Plain-H keep-budgets. 0 = no eviction arm.")
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--importance", default="key_norm",
                    choices=["key_norm", "value_norm", "recency"])
    ap.add_argument("--n", type=int, default=0, help="Eval items. 0 -> task default.")
    ap.add_argument("--generate_bs", type=int, default=0, help="0 -> 20, or 1 on AIME.")
    ap.add_argument("--judger_budget", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--arms", default="",
                    help="Subset of expanded arms, e.g. none,frozen,real")
    ap.add_argument("--cache", default="",
                    help="Path to cache.pt from --mode build.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/math_ladder/run")
    args = ap.parse_args()

    if args.n <= 0:
        args.n = default_n(args.task, args.smoke)
    if args.generate_bs <= 0:
        args.generate_bs = 2 if args.smoke else default_bs(args.task)
    if args.judger_budget <= 0:
        args.judger_budget = 64 if args.smoke else default_budget(args.task)
    if args.smoke:
        args.k = min(args.k, 4)
        args.stat_n = min(args.stat_n, 2)
        args.generate_bs = min(args.generate_bs, args.n)
        print(
            f"[smoke] k={args.k} n={args.n} donors={args.stat_n} "
            f"bs={args.generate_bs} T={args.judger_budget}",
            flush=True,
        )

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode in ("build", "both"):
        path = run_build(args)
        if not args.cache:
            args.cache = path
    if args.mode in ("eval", "both"):
        run_eval(args)


if __name__ == "__main__":
    main()
