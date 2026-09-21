#!/usr/bin/env python3
"""Frozen MATH-1k + Judger SEAL. Gated latency campaign. No training.

  python scripts/exp_frozen_seal.py --mode eval --task gsm8k --n 40
  python scripts/exp_frozen_seal.py --mode eval --task math --n 40
  python scripts/exp_frozen_seal.py --mode eval --task aime2024 --n 6

Arms: frozen vs frozen_seal (coef 40, layer 28). Kill on accuracy drop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import load_aime2024, load_aime2025, load_gsm8k, load_math  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.cache_bank import kv_mb, num_positions  # noqa: E402
from seal.frozen_seal import (  # noqa: E402
    DEFAULT_COEF,
    DEFAULT_LAYER,
    DEFAULT_VECTOR,
    FALLBACK_VECTOR,
    aime_latency_verdict,
    iso_acc_token_gate,
    render_checkin,
)
from seal.latent_eval import (  # noqa: E402
    attach_judger_seal,
    decode_batch_maybe_seal,
    deep_clone,
    graded,
    make_ns,
    mean_ci,
    pred_short,
    sync,
    to_dev,
)
from seal.math_cache import cache_as_past, cache_path, load_unified_cache  # noqa: E402
from seal.one_role import AIME6_IDX, AIME12_IDX, parse_indices, select_items  # noqa: E402
from utils import auto_device, set_seed  # noqa: E402


def load_task(task: str, split: str):
    if task == "aime2024":
        return list(load_aime2024(split="train"))
    if task == "aime2025":
        return list(load_aime2025(split="train"))
    if task == "math":
        return list(load_math(split=split))
    return list(load_gsm8k(split=split))


def resolve_vector(path: str) -> str:
    if path and os.path.isfile(path):
        return path
    for cand in (os.path.join(ROOT, DEFAULT_VECTOR), os.path.join(ROOT, FALLBACK_VECTOR)):
        if os.path.isfile(cand):
            return cand
    raise SystemExit(f"missing SEAL vector; tried {path!r}, {DEFAULT_VECTOR}, {FALLBACK_VECTOR}")


def maybe_load_model(args, ns):
    from models import ModelWrapper

    print(f"[load] {args.model_name} device={args.device}", flush=True)
    return ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)


def load_frozen(args, dtype=None):
    cache = args.cache or cache_path(os.path.join(ROOT, "artifacts/math_ladder/math1k"))
    if not os.path.isfile(cache):
        raise SystemExit(f"missing cache {cache}")
    payload = load_unified_cache(cache)
    past = cache_as_past(payload, device="cpu", dtype=dtype)
    print(
        f"[cache] {cache} pos={num_positions(past)} {kv_mb(past):.1f}MB",
        flush=True,
    )
    return payload, past


def judger_tensors(wrapper, questions, ns):
    jmsgs = [
        build_agent_message_sequential_latent_mas(
            role="judger", question=q, context="", method="latent_mas", args=ns)
        for q in questions
    ]
    _, jids, jmask, _ = wrapper.prepare_chat_batch(jmsgs, add_generation_prompt=True)
    return jids, jmask


def _done_idx_arms(jsonl_path: str):
    found = defaultdict(set)
    if not os.path.isfile(jsonl_path):
        return found
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            found[int(rec["idx"])].add(str(rec["method"]))
    return found


def load_jsonl(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def default_n(task: str) -> int:
    if task.startswith("aime"):
        return 6
    return 40


def default_budget(task: str) -> int:
    if task.startswith("aime"):
        return 8192
    if task == "math":
        return 2048
    return 1024


def default_bs(task: str) -> int:
    return 1 if task.startswith("aime") else 4


def run_eval(args):
    ns = make_ns(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    wrapper = maybe_load_model(args, ns)
    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass
    vec = resolve_vector(args.seal_vector)
    attach_judger_seal(wrapper, vec, coef=args.seal_coef, layer_index=args.seal_layer)
    dtype = next(wrapper.model.parameters()).dtype
    payload, frozen_cpu = load_frozen(args, dtype=dtype)
    frozen_cpu = cache_as_past(payload, device="cpu", dtype=dtype)

    pool = load_task(args.task, args.split)
    idxs = parse_indices(args.indices)
    if args.task.startswith("aime") and not idxs:
        if int(args.n) <= 6:
            idxs = list(AIME6_IDX)
        elif int(args.n) <= 12:
            idxs = list(AIME12_IDX)
    items = select_items(pool, args.n, idxs or None)
    arms = ["frozen", "frozen_seal"]
    print(
        f"[eval] task={args.task} n={len(items)} coef={args.seal_coef} idx={[it['idx'] for it in items]}",
        flush=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    dump = os.path.join(args.out_dir, "latency_rows.jsonl")
    if args.force and os.path.isfile(dump):
        os.remove(dump)
    done = _done_idx_arms(dump)
    t_wall = time.time()
    bs = max(1, int(args.generate_bs))

    for start in range(0, len(items), bs):
        batch = items[start : start + bs]
        for method, seal_on in (("frozen", False), ("frozen_seal", True)):
            need = [it for it in batch if method not in done.get(int(it["idx"]), set())]
            if not need:
                continue
            questions = [it["question"] for it in need]
            golds = [it["gold"] for it in need]
            B = len(need)
            jids, jmask = judger_tensors(wrapper, questions, ns)
            caches = [to_dev(deep_clone(frozen_cpu), wrapper.device) for _ in range(B)]
            sync()
            t0 = time.perf_counter()
            texts, ntoks, eoss = decode_batch_maybe_seal(
                wrapper, jids, jmask, caches, args.judger_budget,
                temperature=args.temperature, top_p=args.top_p, seal_on=seal_on,
            )
            sync()
            t_j = time.perf_counter() - t0
            recs = []
            for b, it in enumerate(need):
                rec = {
                    "idx": int(it["idx"]),
                    "method": method,
                    "seal": bool(seal_on),
                    "coef": float(args.seal_coef) if seal_on else 0.0,
                    "batch_size": B,
                    "judger_s": t_j / B,
                    "e2e_s": t_j / B,
                    "tokens": int(ntoks[b]),
                    "eos": bool(eoss[b]),
                    "correct": bool(graded(texts[b], golds[b], args.task)),
                    "pred": pred_short(texts[b]),
                    "cache_pos": int(num_positions(frozen_cpu)),
                    "upstream_forwards": 0,
                }
                recs.append(rec)
                done[int(it["idx"])].add(method)
            with open(dump, "a") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
            n_ok = sum(r["correct"] for r in recs)
            print(
                f"[eval] {method} items {[it['idx'] for it in need]} "
                f"{n_ok}/{B} tok~{float(np.mean(ntoks)):.0f} J={t_j:.1f}s "
                f"elapsed={time.time()-t_wall:.0f}s",
                flush=True,
            )
            del caches, texts
            torch.cuda.empty_cache()

    rows = load_jsonl(dump)
    report = summarize(rows, args, items)
    with open(os.path.join(args.out_dir, "latency_rows.json"), "w") as f:
        json.dump(rows, f)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print_table(report)
    write_gate(args, report)
    print("FROZEN_SEAL_EVAL_DONE", flush=True)
    return report


def summarize(rows, args, items):
    methods = []
    seen = set()
    for r in rows:
        if r["method"] not in seen:
            seen.add(r["method"])
            methods.append(r["method"])
    out: Dict[str, Any] = {
        "config": {
            "task": args.task, "n": len(items), "seal_coef": args.seal_coef,
            "seal_layer": args.seal_layer, "judger_budget": args.judger_budget,
            "seed": args.seed, "indices": [it["idx"] for it in items],
        },
        "arms": {},
    }
    by = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        by[m] = {r["idx"]: r for r in rs}
        blk = {
            "n": len(rs),
            "acc": float(np.mean([r["correct"] for r in rs])) if rs else None,
        }
        for k in ("judger_s", "e2e_s", "tokens"):
            if rs:
                blk[k] = mean_ci([r[k] for r in rs], seed=args.seed)
        out["arms"][m] = blk

    f = out["arms"].get("frozen") or {}
    s = out["arms"].get("frozen_seal") or {}
    f_tok = (f.get("tokens") or {}).get("mean")
    s_tok = (s.get("tokens") or {}).get("mean")
    if args.task.startswith("aime") and "frozen" in by and "frozen_seal" in by:
        common = sorted(set(by["frozen"]) & set(by["frozen_seal"]))
        stage = "aime6" if len(common) <= 6 else ("aime12" if len(common) <= 12 else "aime30")
        out["gate"] = aime_latency_verdict(
            [bool(by["frozen"][i]["correct"]) for i in common],
            [bool(by["frozen_seal"][i]["correct"]) for i in common],
            [by["frozen"][i]["tokens"] for i in common],
            [by["frozen_seal"][i]["tokens"] for i in common],
            stage=stage,
        )
    else:
        out["gate"] = iso_acc_token_gate(f.get("acc"), s.get("acc"), f_tok, s_tok)
    return out


def print_table(report):
    print("\n=== FROZEN + SEAL ===", flush=True)
    for name, a in report.get("arms", {}).items():
        tok = (a.get("tokens") or {}).get("mean")
        print(
            f"  {name:<14} acc={a.get('acc')} tok={tok} n={a.get('n')}",
            flush=True,
        )
    g = report.get("gate") or {}
    print(f"  recommend={g.get('recommend')} {g.get('reason')}", flush=True)


def gate_path(args) -> str:
    return os.path.join(args.root_dir, "gate.json")


def checkin_path(args) -> str:
    return os.path.join(args.root_dir, "CHECKIN.md")


def load_gate(args) -> Dict[str, Any]:
    p = gate_path(args)
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {"recommend": "running", "reason": "no stages yet"}


def write_gate(args, report: Dict[str, Any]):
    os.makedirs(args.root_dir, exist_ok=True)
    gate = load_gate(args)
    g = report.get("gate") or {}
    task = args.task
    key = task if not task.startswith("aime") else (
        "aime6" if int(report.get("config", {}).get("n") or 0) <= 6
        else ("aime12" if int(report.get("config", {}).get("n") or 0) <= 12 else "aime30")
    )
    blk = dict(g)
    blk["frozen_acc"] = (report.get("arms") or {}).get("frozen", {}).get("acc")
    blk["seal_acc"] = (report.get("arms") or {}).get("frozen_seal", {}).get("acc")
    blk["frozen_tok"] = ((report.get("arms") or {}).get("frozen", {}).get("tokens") or {}).get("mean")
    blk["seal_tok"] = ((report.get("arms") or {}).get("frozen_seal", {}).get("tokens") or {}).get("mean")
    gate[key] = blk
    gate["recommend"] = g.get("recommend") or "unknown"
    gate["reason"] = g.get("reason") or ""
    if gate["recommend"] == "go":
        gate["next"] = {
            "gsm8k": "MATH n=40 Frozen vs Frozen+SEAL",
            "math": "AIME n=6 Frozen vs Frozen+SEAL",
            "aime6": "AIME n=12",
            "aime12": "AIME n=30",
            "aime30": "write the table; stop the pod",
        }.get(key, "next stage")
    elif gate["recommend"] == "borderline":
        gate["next"] = "AIME n=12, then stop if still net ≤ 0"
    else:
        gate["next"] = "stop the pod — do not hunt AIME accuracy"
    with open(gate_path(args), "w") as f:
        json.dump(gate, f, indent=2)
    with open(checkin_path(args), "w") as f:
        f.write(render_checkin(gate))
    print(open(checkin_path(args)).read(), flush=True)


def run_gate(args):
    gate = load_gate(args)
    os.makedirs(args.root_dir, exist_ok=True)
    with open(checkin_path(args), "w") as f:
        f.write(render_checkin(gate))
    print(open(checkin_path(args)).read(), flush=True)
    if gate.get("recommend") == "stop":
        raise SystemExit("FROZEN_SEAL_GATE_STOP")
    print("FROZEN_SEAL_GATE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="eval", choices=["eval", "gate"])
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--task", default="gsm8k",
                    choices=["gsm8k", "math", "aime2024", "aime2025"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--generate_bs", type=int, default=0)
    ap.add_argument("--judger_budget", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--indices", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--seal_vector", default="")
    ap.add_argument("--seal_coef", type=float, default=DEFAULT_COEF)
    ap.add_argument("--seal_layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--root_dir", default="artifacts/frozen_seal")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    if args.n <= 0:
        args.n = default_n(args.task)
    if args.generate_bs <= 0:
        args.generate_bs = default_bs(args.task)
    if args.judger_budget <= 0:
        args.judger_budget = default_budget(args.task)
    if not args.out_dir:
        args.out_dir = os.path.join(args.root_dir, f"{args.task}_n{args.n}")
    if not args.cache:
        args.cache = cache_path(os.path.join(ROOT, "artifacts/math_ladder/math1k"))

    set_seed(args.seed)
    os.makedirs(args.root_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "eval":
        run_eval(args)
    else:
        run_gate(args)


if __name__ == "__main__":
    main()
