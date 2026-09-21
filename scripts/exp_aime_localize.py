#!/usr/bin/env python3
"""Localize AIME work vs waste in the LatentMAS tape, then evict by segment.

One growing Real K=10 pass writes Planner/Critic/Refiner into one KV tape.
We then decode the Judger from slices of that tape (Jiayi's c1 / c2 / c3 /
c1+c2+c3) plus shuffle, latents-only, uniform SnapKV, and role-budgeted
eviction — without re-running the silent agents.

A second upstream pass isolates each role (Jiayi line 1 / line 4): the agent
does not read the previous live cache; the Judger still sees the concat of
writes. If {4,10,18} survive, inter-agent *read* is wasteful.

  python scripts/exp_aime_localize.py --mode smoke
  python scripts/exp_aime_localize.py --mode collect --n 6 --indices 0,1,2,4,10,18
  python scripts/exp_aime_localize.py --mode views
  python scripts/exp_aime_localize.py --mode isolated
  python scripts/exp_aime_localize.py --mode isolated_frozen --cache artifacts/math_ladder/math1k/cache.pt
  python scripts/exp_aime_localize.py --mode report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import load_aime2024, load_aime2025, load_gpqa_diamond, load_humanevalplus  # noqa: E402
from methods import default_agents  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.cache_bank import kv_mb, num_positions  # noqa: E402
from seal.latent_eval import (  # noqa: E402
    attach_judger_seal,
    build_upstream_segments,
    decode_batch,
    decode_batch_maybe_seal,
    deep_clone,
    graded,
    make_ns,
    mean_ci,
    pred_short,
    reset_peak,
    sync,
    to_cpu,
    to_dev,
)
from seal.math_cache import cache_as_past, load_unified_cache  # noqa: E402
from seal.segment_kv import (  # noqa: E402
    concat_seq,
    evict_by_segment,
    evict_uniform,
    take_latents,
    take_roles,
)
from utils import auto_device, set_seed  # noqa: E402

CRITICAL = (4, 10, 18)
FOCUS = (0, 1, 2, 4, 10, 18)
VIEW_ARMS = (
    "none",
    "real",
    "c1",
    "c2",
    "c3",
    "c23",
    "c12",
    "latents",
    "shuf",
    "evict_uniform",
    "evict_seg",
)


def parse_indices(raw: str, n: int) -> List[int]:
    if not raw or not str(raw).strip():
        return list(range(int(n)))
    out: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    # unique, stable
    seen, ordered = set(), []
    for i in out:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


def load_pool(task: str) -> List[Dict]:
    if task == "aime2025":
        return list(load_aime2025(split="train"))
    if task == "gpqa":
        return list(load_gpqa_diamond(split="test"))
    if task == "humanevalplus":
        return list(load_humanevalplus(split="test"))
    return list(load_aime2024(split="train"))


def maybe_load_model(args, ns):
    from models import ModelWrapper

    print(f"[load] {args.model_name} device={args.device}", flush=True)
    return ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)


def tape_path(root: str, idx: int) -> str:
    return os.path.join(root, "tapes", f"{int(idx):04d}.pt")


def rows_path(root: str) -> str:
    return os.path.join(root, "rows.jsonl")


def load_rows(root: str) -> List[Dict]:
    path = rows_path(root)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_row(root: str, row: Dict) -> None:
    os.makedirs(root, exist_ok=True)
    with open(rows_path(root), "a") as f:
        f.write(json.dumps(row) + "\n")


def seen_keys(rows: Sequence[Dict]) -> set:
    return {(int(r["idx"]), str(r["method"])) for r in rows}


def judger_tensors(wrapper, questions, ns):
    messages = [
        build_agent_message_sequential_latent_mas(
            role="judger", question=q, context="", method="latent_mas", args=ns)
        for q in questions
    ]
    _, ids, mask, _ = wrapper.prepare_chat_batch(messages, add_generation_prompt=True)
    return ids, mask


def load_frozen(path: str, device="cpu"):
    payload = load_unified_cache(path)
    return cache_as_past(payload, device=device), payload.get("meta") or {}


def save_tape(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def load_tape(path: str) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def view_cache(name: str, tape: Dict, others: Sequence[Dict], args) -> Any:
    past = tape["past"]
    spans = tape["spans"]
    k = int(tape.get("k") or args.k)
    if name == "none":
        return None, None
    if name == "real":
        return past, None
    if name == "c1":
        return take_roles(past, spans, ["planner"]), None
    if name == "c2":
        return take_roles(past, spans, ["critic"]), None
    if name == "c3":
        return take_roles(past, spans, ["refiner"]), None
    if name == "c23":
        return take_roles(past, spans, ["critic", "refiner"]), None
    if name == "c12":
        return take_roles(past, spans, ["planner", "critic"]), None
    if name == "latents":
        return take_latents(past, spans, k), None
    if name == "shuf":
        if len(others) < 2:
            return past, {"note": "n<2, shuffle is identity"}
        donor = others[(tape["idx"] + 1) % len(others)]
        return donor["past"], {"donor_idx": int(donor["idx"])}
    if name == "evict_uniform":
        out, st = evict_uniform(past, int(args.evict_budget), sink=int(args.sink))
        return out, st
    if name == "evict_seg":
        out, st = evict_by_segment(
            past, spans, budget_per_role=int(args.seg_budget), sink=int(args.seg_sink))
        return out, st
    raise ValueError(name)


def run_collect(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for --mode collect")
    ns = make_ns(args)
    ns.temperature = float(args.temperature)
    ns.top_p = float(args.top_p)
    pool = load_pool(args.task)
    idxs = parse_indices(args.indices, args.n)
    idxs = [i for i in idxs if i < len(pool)][: int(args.n)]
    wrapper = maybe_load_model(args, ns)
    up_agents = [a for a in default_agents() if a.role != "judger"]
    for i in idxs:
        path = tape_path(args.out_dir, i)
        if os.path.isfile(path) and not args.overwrite:
            print(f"[collect] skip idx={i}", flush=True)
            continue
        it = pool[i]
        past, times, peaks, spans, _writes = build_upstream_segments(
            wrapper, [it["question"]], args.k, ns, up_agents,
        )
        cpu = to_cpu(past)
        payload = {
            "idx": i,
            "task": args.task,
            "k": int(args.k),
            "question": it["question"],
            "gold": it.get("gold") or it.get("solution") or "",
            "spans": spans,
            "past": cpu,
            "times": times,
            "peaks": peaks,
            "pos": int(num_positions(cpu)),
            "mb": float(kv_mb(cpu)),
        }
        save_tape(path, payload)
        print(
            f"[collect] idx={i} pos={payload['pos']} mb={payload['mb']:.1f} "
            f"up={times['upstream']:.2f}s spans={spans}",
            flush=True,
        )
        del past, cpu
        torch.cuda.empty_cache()


def list_tapes(root: str) -> List[Dict]:
    d = os.path.join(root, "tapes")
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".pt"):
            out.append(load_tape(os.path.join(d, name)))
    return out


def decode_one(wrapper, ns, args, it, cache, method: str, extra: Optional[Dict]) -> Dict:
    jids, jmask = judger_tensors(wrapper, [it["question"]], ns)
    caches = [None if cache is None else to_dev(deep_clone(cache), wrapper.device)]
    reset_peak()
    sync()
    t0 = time.perf_counter()
    texts, ntoks, eoss = decode_batch_maybe_seal(
        wrapper, jids, jmask, caches, args.judger_budget,
        temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k if float(args.temperature) > 0 else None,
        seal_on=bool(args.seal_on and method.endswith("_seal")),
    )
    sync()
    t_j = time.perf_counter() - t0
    gold = it.get("gold") or ""
    row = {
        "idx": int(it["idx"]),
        "method": method,
        "task": args.task,
        "correct": bool(graded(texts[0], gold, args.task)),
        "pred": pred_short(texts[0]),
        "tokens": int(ntoks[0]),
        "eos": bool(eoss[0]),
        "judger_s": float(t_j),
        "cache_pos": 0 if cache is None else int(num_positions(cache)),
        "cache_mb": 0.0 if cache is None else float(kv_mb(cache)),
        "critical": int(it["idx"]) in CRITICAL,
    }
    if extra:
        row["extra"] = extra
    print(
        f"[decode] idx={row['idx']} {method} acc={int(row['correct'])} "
        f"tok={row['tokens']} pos={row['cache_pos']} {t_j:.1f}s pred={row['pred']!r}",
        flush=True,
    )
    return row


def run_views(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for --mode views")
    tapes = list_tapes(args.out_dir)
    if not tapes:
        raise SystemExit(f"no tapes in {args.out_dir}/tapes — run --mode collect")
    ns = make_ns(args)
    wrapper = maybe_load_model(args, ns)
    if args.seal_vector:
        attach_judger_seal(wrapper, args.seal_vector, coef=args.seal_coef,
                           layer_index=args.seal_layer)
    already = seen_keys(load_rows(args.out_dir))
    arms = [a.strip() for a in args.view_arms.split(",") if a.strip()]
    for tape in tapes:
        it = {"idx": tape["idx"], "question": tape["question"], "gold": tape["gold"]}
        for name in arms:
            cache, extra = view_cache(name, tape, tapes, args)
            method = name
            if float(args.temperature) > 0:
                method = f"{name}_t{str(args.temperature).replace('.', '')}"
            key = (int(tape["idx"]), method)
            if key in already and not args.overwrite:
                print(f"[views] skip {key}", flush=True)
                continue
            row = decode_one(wrapper, ns, args, it, cache, method, extra)
            append_row(args.out_dir, row)
            already.add(key)
            torch.cuda.empty_cache()


def run_isolated(args, *, use_frozen: bool) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for isolated pass")
    ns = make_ns(args)
    wrapper = maybe_load_model(args, ns)
    up_agents = [a for a in default_agents() if a.role != "judger"]
    prefix = None
    method = "isolated"
    if use_frozen:
        if not args.cache or not os.path.isfile(args.cache):
            raise SystemExit("--cache required for isolated_frozen")
        prefix, meta = load_frozen(args.cache, device="cpu")
        method = "isolated_frozen"
        print(f"[isolated] frozen pos={num_positions(prefix)} meta={meta}", flush=True)
        if args.seal_vector:
            attach_judger_seal(wrapper, args.seal_vector, coef=args.seal_coef,
                               layer_index=args.seal_layer)
            method = "isolated_frozen_seal"
            args.seal_on = True
    pool = load_pool(args.task)
    tapes = list_tapes(args.out_dir)
    idxs = [t["idx"] for t in tapes] if tapes else parse_indices(args.indices, args.n)
    already = seen_keys(load_rows(args.out_dir))
    for i in idxs:
        if (int(i), method) in already and not args.overwrite:
            print(f"[isolated] skip idx={i} {method}", flush=True)
            continue
        it = pool[i]
        start = None if prefix is None else to_dev(deep_clone(prefix), wrapper.device)
        past, times, peaks, spans, writes = build_upstream_segments(
            wrapper, [it["question"]], args.k, ns, up_agents,
            reset_between=True, prefix_each=start,
        )
        cpu = to_cpu(past)
        row = decode_one(
            wrapper, ns, args,
            {"idx": i, "question": it["question"], "gold": it.get("gold") or ""},
            cpu, method,
            {"spans": spans, "upstream_s": times.get("upstream"),
             "writes_pos": {r: int(num_positions(w)) if w is not None else 0
                            for r, w in writes.items()}},
        )
        row["upstream_s"] = times.get("upstream")
        append_row(args.out_dir, row)
        tag = os.path.join(args.out_dir, "tapes_isolated", f"{int(i):04d}_{method}.pt")
        save_tape(tag, {"idx": i, "method": method, "spans": spans, "past": cpu,
                        "times": times})
        del past, cpu, start
        torch.cuda.empty_cache()


def _arm_table(rows: List[Dict]) -> Dict[str, Dict]:
    by: Dict[str, List[Dict]] = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r)
    out = {}
    for m, rs in by.items():
        acc = [float(x["correct"]) for x in rs]
        toks = [float(x["tokens"]) for x in rs]
        pos = [float(x.get("cache_pos") or 0) for x in rs]
        crit = [x for x in rs if x.get("critical")]
        out[m] = {
            "n": len(rs),
            "acc": float(np.mean(acc)) if acc else 0.0,
            "tokens": mean_ci(toks),
            "cache_pos": mean_ci(pos),
            "correct_idx": sorted(int(x["idx"]) for x in rs if x["correct"]),
            "critical_kept": sorted(int(x["idx"]) for x in crit if x["correct"]),
            "critical_lost": sorted(int(x["idx"]) for x in crit if not x["correct"]),
        }
    return out


def write_report(args) -> Dict:
    rows = load_rows(args.out_dir)
    arms = _arm_table(rows)
    real = set((arms.get("real") or {}).get("correct_idx") or [])
    none = set((arms.get("none") or {}).get("correct_idx") or [])
    pairing = {}
    for m, blk in arms.items():
        got = set(blk.get("correct_idx") or [])
        pairing[m] = {
            "recovered_vs_none": sorted(got - none),
            "lost_vs_real": sorted(real - got),
            "kept_vs_real": sorted(real & got),
        }
    rec = "collect_views"
    reason = "need real / none / c3 before calling a hop wasteful"
    if "real" in arms and "c3" in arms:
        lost_c3 = set(pairing["c3"]["lost_vs_real"])
        if not (set(CRITICAL) & lost_c3):
            rec = "drop_planner_critic"
            reason = "c3 keeps the Real-only AIME items — Planner/Critic writes look wasteful"
        elif "c23" in arms and not (set(CRITICAL) & set(pairing["c23"]["lost_vs_real"])):
            rec = "drop_planner"
            reason = "c23 keeps Real-only items — Planner write looks wasteful"
        else:
            rec = "keep_concat"
            reason = "no single later role recovers {4,10,18}; do not drop writes yet"
    if "isolated" in arms and "real" in arms:
        lost_iso = set(pairing["isolated"]["lost_vs_real"])
        if set(CRITICAL) & lost_iso:
            rec = "keep_growing_read"
            reason = "isolated (no inter-agent read) dropped critical items — the growing channel is doing work"
        elif rec == "keep_concat":
            rec = "isolate_then_evict"
            reason = "isolated keeps critical items — communication read is wasteful; evict inside writes"
    report = {
        "task": args.task,
        "critical": list(CRITICAL),
        "n_rows": len(rows),
        "arms": arms,
        "pairing": pairing,
        "recommend": rec,
        "reason": reason,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    lines = [
        "# AIME localization check-in",
        "",
        f"**recommend:** `{rec}`",
        "",
        reason,
        "",
        f"Critical items (Real-only vs Frozen in the locked ladder): {list(CRITICAL)}",
        "",
        "| Arm | n | acc | critical kept | lost vs Real |",
        "|---|---:|---:|---|---|",
    ]
    for m, blk in sorted(arms.items()):
        p = pairing.get(m) or {}
        lines.append(
            f"| {m} | {blk['n']} | {blk['acc']:.3f} | "
            f"{blk.get('critical_kept')} | {p.get('lost_vs_real')} |"
        )
    lines.extend(["", "Jiayi map: `real` = judger(c1+c2+c3). `c1`/`c2`/`c3` = private writes. "
                  "`isolated` = a1(c1')→… with empty/frozen c', judger still gets concat writes. "
                  "`evict_seg` = SnapKV budget per role span.", ""])
    with open(os.path.join(args.out_dir, "CHECKIN.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"recommend": rec, "reason": reason, "arms": list(arms)}, indent=2), flush=True)
    return report


def run_smoke(args) -> None:
    """No model: segment slice / evict roundtrip on a fake tape."""
    from seal.cache_bank import from_legacy

    g = torch.Generator().manual_seed(0)
    seq = 90
    layers = []
    for _ in range(3):
        k = torch.randn(1, 2, seq, 4, generator=g)
        v = torch.randn(1, 2, seq, 4, generator=g)
        layers.append((k, v))
    past = from_legacy(layers)
    spans = [
        {"role": "planner", "start": 0, "end": 30},
        {"role": "critic", "start": 30, "end": 60},
        {"role": "refiner", "start": 60, "end": 90},
    ]
    c3 = take_roles(past, spans, ["refiner"])
    assert num_positions(c3) == 30
    lat = take_latents(past, spans, 10)
    assert num_positions(lat) == 30
    ev, st = evict_by_segment(past, spans, budget_per_role=8, sink=2)
    assert num_positions(ev) == 24, st
    uni, _ = evict_uniform(past, 24, sink=4)
    assert num_positions(uni) == 24
    print("[smoke] segment_kv ok", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "smoke.json"), "w") as f:
        json.dump({"ok": True, "evict_seg": st}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="report",
                    choices=["smoke", "collect", "views", "isolated", "isolated_frozen", "report"])
    ap.add_argument("--out_dir", default="artifacts/aime_localize")
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default="aime2024")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--indices", default="0,1,2,4,10,18")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--generate_bs", type=int, default=1)
    ap.add_argument("--judger_budget", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--evict_budget", type=int, default=64)
    ap.add_argument("--seg_budget", type=int, default=24)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--seg_sink", type=int, default=2)
    ap.add_argument("--view_arms", default=",".join(VIEW_ARMS))
    ap.add_argument("--cache", default="")
    ap.add_argument("--seal_vector", default="")
    ap.add_argument("--seal_coef", type=float, default=40.0)
    ap.add_argument("--seal_layer", type=int, default=28)
    ap.add_argument("--seal_on", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "collect":
        run_collect(args)
    elif args.mode == "views":
        run_views(args)
    elif args.mode == "isolated":
        run_isolated(args, use_frozen=False)
    elif args.mode == "isolated_frozen":
        run_isolated(args, use_frozen=True)
    else:
        write_report(args)


if __name__ == "__main__":
    main()
