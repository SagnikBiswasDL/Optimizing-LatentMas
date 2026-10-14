#!/usr/bin/env python3
"""Task-banked latent prefixes: skip Planner/Critic/Refiner at test time.

Mental model
------------
Offline, run the silent agents on a few donor items *per problem type* and save
the resulting KV relays. Online, look up the bank by type, give that prefix to
the Judger, and never run the upstream agents on the test item.

That is the method. Eviction is not involved. The Judger still sees the *test*
question in its prompt; only the latent prefix is recycled.

Arms
----
  none       Judger-only (no prefix)
  bank       this task's precomputed donor cache (0 upstream forwards)
  wrong_bank a *different* type's donor cache (type-sensitivity control)
  synth      Gaussian matched to this type's donor stats (no real sequence)
  real       full LatentMAS on the test item (optional; expensive)

Examples
--------
  # Build one bank file covering GSM8K + MedQA + MATH (K=40).
  python scripts/exp_cache_bank.py --mode build \\
      --model_name Qwen/Qwen3-14B --k 40 \\
      --tasks gsm8k,medqa,math --n_donors 4 \\
      --out_dir artifacts/cache_bank/qwen3-14b_k40

  # Eval: skip the agents. Paper sampling: add --temperature 0.6 --top_p 0.95
  python scripts/exp_cache_bank.py --mode eval \\
      --model_name Qwen/Qwen3-14B --task gsm8k --n 100 --k 40 \\
      --bank artifacts/cache_bank/qwen3-14b_k40/bank.pt \\
      --arms none,bank,wrong_bank,synth --wrong_bank_key medqa \\
      --judger_budget 2048 --out_dir artifacts/cache_bank/eval_gsm8k_s42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import (  # noqa: E402
    load_gsm8k,
    load_math,
    load_medqa,
)
try:
    from data import load_gpqa_diamond
except ImportError:
    load_gpqa_diamond = None
try:
    from data import load_aime2024
except ImportError:
    load_aime2024 = None

from methods import default_agents  # noqa: E402
from models import ModelWrapper  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.cache_bank import (  # noqa: E402
    CacheBank,
    clone_past,
    kv_mb,
    num_positions,
    pool_stats,
    problem_type,
    to_cpu_legacy,
)
from utils import (  # noqa: E402
    auto_device,
    extract_boxed_answer,
    extract_gsm8k_answer,
    normalize_answer,
    normalize_math_answer,
    set_seed,
)


TASKS = ("gsm8k", "medqa", "math", "gpqa", "aime2024")


def load_task(task: str, split: str) -> List[Dict]:
    if task == "medqa":
        return list(load_medqa(split=split))
    if task == "math":
        return list(load_math(split=split))
    if task == "gpqa":
        if load_gpqa_diamond is None:
            raise RuntimeError("load_gpqa_diamond missing")
        return list(load_gpqa_diamond(split="test"))
    if task == "aime2024":
        if load_aime2024 is None:
            raise RuntimeError("load_aime2024 missing")
        return list(load_aime2024(split="train"))
    return list(load_gsm8k(split=split))


def donor_pool(task: str, donor_split: str, n_donors: int, exclude_qs=None):
    """Prefer a disjoint split; fall back to holding out the first n items."""
    exclude_qs = set(exclude_qs or [])
    holdout = False
    try:
        items = load_task(task, donor_split)
    except Exception:
        items = []
    if not items and donor_split != "test":
        items = load_task(task, "test")
        holdout = True
    filtered = [it for it in items if it.get("question") not in exclude_qs]
    if not filtered:
        filtered = list(items)
        holdout = True
    return filtered, holdout


def make_ns(args, task: str):
    return SimpleNamespace(
        model_name=args.model_name, task=task, prompt="sequential", think=False,
        latent_only=False, sequential_info_only=False, agents=None, use_vllm=False,
        device=args.device, device2="cuda:1",
        max_new_tokens=int(getattr(args, "judger_budget", 2048)),
        text_mas_context_length=-1,
        temperature=float(getattr(args, "temperature", 0.0)),
        top_p=float(getattr(args, "top_p", 1.0)), seed=args.seed,
        seal=False, kvsteer=False, ces=False, capture_acts=None,
        planner_steps=None, critic_steps=None, refiner_steps=None,
        latent_steps=int(args.k),
        latent_space_realign=bool(getattr(args, "latent_space_realign", False)),
    )


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def graded(text, gold, task):
    if task == "math":
        pred = normalize_math_answer(extract_boxed_answer(text) or text)
        return bool(pred) and bool(gold) and pred == gold
    if task == "gpqa":
        boxed = extract_boxed_answer(text) or extract_gsm8k_answer(text) or ""
        pred = normalize_answer(boxed)
        g = normalize_answer(gold)
        if pred and g and (pred == g or pred[:1] == g[:1]):
            return True
        import re
        letters = re.findall(r"\b([A-Da-d])\b", text[-200:])
        return bool(letters) and g and normalize_answer(letters[-1]) == g[:1]
    pred = normalize_answer(extract_boxed_answer(text) or extract_gsm8k_answer(text))
    g = normalize_answer(gold)
    return bool(pred) and bool(g) and pred == g


def bootstrap_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def paired_ci(a, b, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    m, lo, hi = bootstrap_ci(d, seed=seed)
    return {"mean_diff": m, "ci_lo": lo, "ci_hi": hi,
            "credible_positive": bool(lo > 0), "credible_negative": bool(hi < 0)}


def judger_inputs(wrapper, question, ns):
    messages = build_agent_message_sequential_latent_mas(
        role="judger", question=question, context="", method="latent_mas", args=ns)
    _, ids, mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
    return ids, mask


def build_upstream(wrapper, question, k, ns, agents):
    past = None
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for agent in agents:
        messages = build_agent_message_sequential_latent_mas(
            role=agent.role, question=question, context="", method="latent_mas", args=ns)
        _, ids, mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
        past = wrapper.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=int(k),
            past_key_values=past, role=agent.role)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return past, time.perf_counter() - t0


def decode(wrapper, ids, mask, past, budget, temperature, top_p):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    gens, _ = wrapper.generate_text_batch(
        ids, mask, max_new_tokens=int(budget),
        temperature=float(temperature), top_p=float(top_p),
        past_key_values=past, role="judger")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    toks = list(getattr(wrapper, "last_gen_token_counts", [0]))
    return gens[0], (toks[0] if toks else 0), elapsed


def _load_wrapper(args, task: str):
    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(2)
    ns = make_ns(args, task)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    return wrapper, ns


def _group_donors(items, task, granularity, n_donors):
    buckets = defaultdict(list)
    for it in items:
        key = problem_type(it, task, granularity)
        if len(buckets[key]) < n_donors:
            buckets[key].append(it)
    return dict(buckets)


def cmd_build(args):
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in TASKS:
            raise SystemExit(f"unknown task {t}; choose from {TASKS}")
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    wrapper, _ = _load_wrapper(args, tasks[0])
    up_agents = [a for a in default_agents() if a.role != "judger"]
    bank = CacheBank(meta={
        "model_name": args.model_name,
        "k": int(args.k),
        "n_donors": int(args.n_donors),
        "granularity": args.granularity,
        "donor_split": args.donor_split,
        "seed": int(args.seed),
        "latent_space_realign": bool(args.latent_space_realign),
        "tasks": tasks,
    })
    t_wall = time.time()
    for task in tasks:
        ns = make_ns(args, task)
        pool, holdout = donor_pool(task, args.donor_split, args.n_donors)
        buckets = _group_donors(pool, task, args.granularity, args.n_donors)
        print(f"[bank:build] task={task} keys={list(buckets)} holdout={holdout} "
              f"pool={len(pool)}", flush=True)
        for key, donors in buckets.items():
            legacies = []
            meta_donors = []
            for it in donors:
                past, dt = build_upstream(wrapper, it["question"], args.k, ns, up_agents)
                cpu = to_cpu_legacy(past)
                legacies.append(cpu)
                meta_donors.append({
                    "idx": it.get("idx"),
                    "subject": it.get("subject") or "",
                    "gold": it.get("gold"),
                    "question": (it.get("question") or "")[:160],
                    "upstream_s": dt,
                    "n_pos": num_positions(cpu),
                    "mb": kv_mb(cpu),
                })
                del past
                print(f"[bank:build]   {key} donor n_pos={meta_donors[-1]['n_pos']} "
                      f"mb={meta_donors[-1]['mb']:.1f} upstream_s={dt:.2f}", flush=True)
            stats, med_len = pool_stats(legacies)
            # Reuse the first donor's real cache (on-manifold). Stats feed the synth arm.
            bank.add(key, legacy=legacies[0], donors=meta_donors, stats=stats, synth_len=med_len,
                     extra={"task": task, "holdout_donors": holdout})
        # If subtype granularity never created a task-level key, also store one
        # from the first bucket so eval --granularity task still works.
        if args.granularity == "subtype" and task not in bank.entries and buckets:
            first_key = next(iter(buckets))
            src = bank.entries[first_key]
            bank.entries[task] = dict(src)
            bank.entries[task]["key"] = task
    path = os.path.join(args.out_dir, "bank.pt")
    bank.save(path)
    summary = {
        "path": path,
        "meta": bank.meta,
        "keys": {
            k: {"n_pos": e["n_pos"], "mb": e["mb"], "n_donors": e["n_donors"],
                "task": e.get("task")}
            for k, e in bank.entries.items()
        },
        "elapsed_s": time.time() - t_wall,
    }
    with open(os.path.join(args.out_dir, "build_report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[bank:build] saved {path} keys={bank.keys()} elapsed={summary['elapsed_s']:.0f}s",
          flush=True)
    print("BANK_BUILD_DONE", flush=True)


def _pick_wrong_key(bank: CacheBank, want: Optional[str], eval_key: str) -> Optional[str]:
    if want:
        try:
            return bank.resolve(want)
        except KeyError:
            print(f"[bank:eval] --wrong_bank_key {want} not in bank {bank.keys()}", flush=True)
            return None
    for k in bank.keys():
        if k != eval_key and not k.startswith(eval_key + ":"):
            return k
    return None


def cmd_eval(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    legal = {"none", "bank", "wrong_bank", "synth", "real"}
    bad = [a for a in arms if a not in legal]
    if bad:
        raise SystemExit(f"unknown arms {bad}; legal={sorted(legal)}")

    bank = CacheBank.load(args.bank)
    items = load_task(args.task, args.split)
    # Drop items whose question was used as a donor for this task.
    donor_qs = set()
    for e in bank.entries.values():
        if e.get("task") == args.task or e.get("key") == args.task:
            for d in e.get("donors") or []:
                q = (d.get("question") or "").strip()
                if q:
                    donor_qs.add(q)
    if donor_qs:
        before = len(items)
        items = [it for it in items if not any(
            (it.get("question") or "").startswith(q) or q.startswith((it.get("question") or "")[:160])
            for q in donor_qs
        )]
        print(f"[bank:eval] dropped {before - len(items)} donor-overlapping items", flush=True)
    offset = int(getattr(args, "offset", 0) or 0)
    if offset:
        items = items[offset:]
    if args.n > 0:
        items = items[: args.n]
    n = len(items)
    print(f"[bank:eval] task={args.task} n={n} k={args.k} arms={arms} "
          f"bank_keys={bank.keys()} temp={args.temperature} budget={args.judger_budget}",
          flush=True)

    wrapper, ns = _load_wrapper(args, args.task)
    up_agents = [a for a in default_agents() if a.role != "judger"]
    device = wrapper.device
    dtype = next(wrapper.model.parameters()).dtype

    # Materialize prefixes once. Same object for every test item.
    sample_key = problem_type(items[0], args.task, args.granularity) if items else args.task
    try:
        home_key = bank.resolve(sample_key)
    except KeyError:
        home_key = bank.resolve(args.task)
    wrong_key = _pick_wrong_key(bank, args.wrong_bank_key, home_key) if "wrong_bank" in arms else None

    prefixes = {}
    if "bank" in arms:
        prefixes["bank"] = bank.donor_cache(home_key, device, dtype=dtype)
        print(f"[bank:eval] bank key={home_key} n_pos={num_positions(prefixes['bank'])} "
              f"mb={kv_mb(prefixes['bank']):.1f}", flush=True)
    if "wrong_bank" in arms:
        if wrong_key is None:
            print("[bank:eval] no wrong_bank key; dropping arm", flush=True)
            arms = [a for a in arms if a != "wrong_bank"]
        else:
            prefixes["wrong_bank"] = bank.donor_cache(wrong_key, device, dtype=dtype)
            print(f"[bank:eval] wrong_bank key={wrong_key} n_pos={num_positions(prefixes['wrong_bank'])}",
                  flush=True)
    if "synth" in arms:
        prefixes["synth"] = bank.synth_cache(home_key, device, dtype, seed=args.seed)
        print(f"[bank:eval] synth key={home_key} n_pos={num_positions(prefixes['synth'])}", flush=True)

    rows: List[Dict] = []
    t0 = time.time()
    for i, item in enumerate(items):
        q, gold = item["question"], item["gold"]
        ptype = problem_type(item, args.task, args.granularity)
        ids, mask = judger_inputs(wrapper, q, ns)
        # Per-item bank lookup (subtype granularity may differ across items).
        item_prefixes = dict(prefixes)
        if "bank" in arms and args.granularity == "subtype":
            try:
                item_prefixes["bank"] = bank.donor_cache(ptype, device, dtype=dtype)
            except KeyError:
                item_prefixes["bank"] = prefixes["bank"]
        for arm in arms:
            past = None
            up_s = 0.0
            if arm == "none":
                past = None
            elif arm == "real":
                past, up_s = build_upstream(wrapper, q, args.k, ns, up_agents)
            else:
                # clone: HF generate appends Judger tokens onto the cache
                past = clone_past(item_prefixes[arm])
            prefix_pos = 0 if past is None else num_positions(past)
            text, ntok, j_s = decode(
                wrapper, ids, mask, past, args.judger_budget,
                args.temperature, args.top_p)
            ok = graded(text, gold, args.task)
            rows.append({
                "idx": item.get("idx", i),
                "arm": arm,
                "problem_type": ptype,
                "correct": bool(ok),
                "tokens": int(ntok),
                "judger_s": float(j_s),
                "upstream_s": float(up_s),
                "prefix_pos": prefix_pos,
            })
            if past is not None:
                del past
        if (i + 1) % 5 == 0 or i == 0:
            done = [r for r in rows if r["arm"] == arms[0]]
            acc = float(np.mean([r["correct"] for r in done])) if done else 0.0
            print(f"[bank:eval] {i+1}/{n} elapsed={time.time()-t0:.0f}s "
                  f"{arms[0]}_acc={acc:.3f}", flush=True)

    by_arm = {}
    maps = {}
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm]
        accs = [1.0 if r["correct"] else 0.0 for r in ar]
        m, lo, hi = bootstrap_ci(accs, seed=args.seed)
        maps[arm] = {r["idx"]: (1.0 if r["correct"] else 0.0) for r in ar}
        by_arm[arm] = {
            "acc": m, "ci": [lo, hi], "n": len(ar),
            "mean_tokens": float(np.mean([r["tokens"] for r in ar])) if ar else 0.0,
            "mean_judger_s": float(np.mean([r["judger_s"] for r in ar])) if ar else 0.0,
            "mean_upstream_s": float(np.mean([r["upstream_s"] for r in ar])) if ar else 0.0,
            "mean_prefix_pos": float(np.mean([r["prefix_pos"] for r in ar])) if ar else 0.0,
        }
    for arm in arms:
        if arm == "none":
            continue
        for base in ("none", "real", "bank"):
            if base not in maps or base == arm:
                continue
            common = sorted(set(maps[arm]) & set(maps[base]))
            if common:
                by_arm[arm][f"minus_{base}"] = paired_ci(
                    [maps[arm][j] for j in common],
                    [maps[base][j] for j in common],
                    seed=args.seed)

    report = {
        "config": vars(args),
        "bank_meta": bank.meta,
        "bank_keys": bank.keys(),
        "home_key": home_key,
        "wrong_key": wrong_key,
        "n": n,
        "arms": by_arm,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out_dir, "rows.json"), "w") as f:
        json.dump(rows, f)

    print(f"BANK_EVAL n={n} task={args.task} home={home_key} wrong={wrong_key}", flush=True)
    for arm in arms:
        e = by_arm[arm]
        extra = ""
        mr = e.get("minus_none") or {}
        if mr:
            extra = (f" minus_none={mr['mean_diff']:+.3f}"
                     f"[{mr['ci_lo']:+.3f},{mr['ci_hi']:+.3f}]")
        print(f"BANK_EVAL   {arm:<12} acc={e['acc']:.3f} tok={e['mean_tokens']:.0f} "
              f"judger_s={e['mean_judger_s']:.2f} upstream_s={e['mean_upstream_s']:.2f}"
              f"{extra}", flush=True)
    print("BANK_EVAL_DONE", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["build", "eval"])
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=40, help="latent steps used to *build* donor caches")
    ap.add_argument("--latent_space_realign", action="store_true")
    ap.add_argument("--granularity", default="task", choices=["task", "subtype"],
                    help="task = one cache per dataset; subtype = MATH subject / GPQA domain")
    ap.add_argument("--out_dir", default="artifacts/cache_bank/run")

    # build
    ap.add_argument("--tasks", default="gsm8k,medqa",
                    help="comma-separated tasks to bank (build mode)")
    ap.add_argument("--n_donors", type=int, default=4)
    ap.add_argument("--donor_split", default="train")

    # eval
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--bank", default="", help="path to bank.pt (eval)")
    ap.add_argument("--arms", default="none,bank,wrong_bank,synth")
    ap.add_argument("--wrong_bank_key", default="",
                    help="type key for the wrong-bank control (default: another key in the file)")
    ap.add_argument("--judger_budget", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy. Paper sampling is 0.6")
    ap.add_argument("--top_p", type=float, default=1.0,
                    help="Paper sampling is 0.95")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny build+eval knobs (still needs GPU)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.k = min(args.k, 2)
        args.n = min(args.n, 2)
        args.n_donors = 1
        args.judger_budget = min(args.judger_budget, 64)
        args.tasks = args.tasks.split(",")[0]
        print(f"[bank] SMOKE k={args.k} n={args.n} budget={args.judger_budget}", flush=True)
    if args.mode == "build":
        cmd_build(args)
        return
    if not args.bank:
        raise SystemExit("--bank path/to/bank.pt is required for eval")
    cmd_eval(args)


if __name__ == "__main__":
    main()
