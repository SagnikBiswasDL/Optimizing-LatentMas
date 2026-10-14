#!/usr/bin/env python3
"""Frozen MATH-1k + one live K=10 agent. Gated. No training.

  python scripts/exp_one_role.py --mode smoke
  python scripts/exp_one_role.py --mode eval --task math --n 20 \\
      --arms frozen,frozen_planner_k10,frozen_refiner_k10,frozen_critic_k10
  python scripts/exp_one_role.py --mode eval --task aime2024 --n 6 \\
      --arms frozen,frozen_planner_k10
  python scripts/exp_one_role.py --mode gate

Kill after MATH n=20 or AIME n=6/12 if paired results are unpromising.
`--seal_coef` is refused until aime30 GO.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import load_aime2024, load_aime2025, load_gsm8k, load_math  # noqa: E402
from methods import agents_from_spec, default_agents  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.cache_bank import kv_mb, num_positions  # noqa: E402
from seal.latent_eval import (  # noqa: E402
    build_upstream_timed,
    decode_batch,
    deep_clone,
    graded,
    make_ns,
    mean_ci,
    pred_short,
    reset_peak,
    split_past,
    stack_past,
    sync,
    to_cpu,
    to_dev,
)
from seal.math_cache import cache_as_past, cache_path, load_unified_cache  # noqa: E402
from seal.one_role import (  # noqa: E402
    AIME6_IDX,
    AIME12_IDX,
    K_DEFAULT,
    aime_subset_verdict,
    expand_one_role_arms,
    paired_flags,
    parse_indices,
    parse_one_role_arm,
    pick_best_role,
    render_checkin,
    select_items,
    smoke_match,
    upstream_forwards,
)
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


def default_n(task: str, smoke: bool) -> int:
    if smoke:
        return 2
    if task.startswith("aime"):
        return 6
    return 20


def default_budget(task: str) -> int:
    if task.startswith("aime"):
        return 8192
    if task == "math":
        return 2048
    return 1024


def default_bs(task: str) -> int:
    if task.startswith("aime"):
        return 1
    return 4


def load_frozen(args, dtype=None):
    cache = args.cache or cache_path(os.path.join(ROOT, "artifacts/math_ladder/math1k"))
    if not os.path.isfile(cache):
        raise SystemExit(
            f"missing cache {cache}; scp from the pod or "
            "run: bash scripts/run_math_ladder.sh build"
        )
    payload = load_unified_cache(cache)
    past = cache_as_past(payload, device="cpu", dtype=dtype)
    print(
        f"[cache] {cache} pos={num_positions(past)} "
        f"{kv_mb(past):.1f}MB donors={(payload.get('meta') or {}).get('n_donors')}",
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


def live_agents(spec: Dict[str, Any]):
    if spec["kind"] == "frozen":
        return []
    if spec["kind"] == "real":
        return [a for a in default_agents() if a.role != "judger"]
    return [a for a in agents_from_spec(spec["role"]) if a.role != "judger"]


def run_arm_batch(wrapper, ns, questions, frozen_cpu, spec, k_real: int):
    """Build the prefix for one arm, then return CPU caches + timing."""
    B = len(questions)
    times = {
        "planner": 0.0, "critic": 0.0, "refiner": 0.0,
        "upstream": 0.0, "load": 0.0,
    }
    if spec["kind"] == "real":
        agents = live_agents(spec)
        past, up_t, _ = build_upstream_timed(
            wrapper, questions, k_real, ns, agents, latent_steps=int(spec["k"]),
        )
        times.update({
            "planner": up_t.get("planner", 0.0),
            "critic": up_t.get("critic", 0.0),
            "refiner": up_t.get("refiner", 0.0),
            "upstream": up_t["upstream"],
        })
        caches = [to_cpu(p) for p in split_past(past, B)]
        del past
        return caches, times

    sync()
    t0 = time.perf_counter()
    clones = [to_dev(deep_clone(frozen_cpu), wrapper.device) for _ in range(B)]
    times["load"] = time.perf_counter() - t0
    agents = live_agents(spec)
    if not agents:
        caches = [to_cpu(c) for c in clones]
        del clones
        return caches, times
    stacked = stack_past(clones)
    past, up_t, _ = build_upstream_timed(
        wrapper, questions, spec["k"], ns, agents,
        start_past=stacked, latent_steps=int(spec["k"]),
    )
    times.update({
        "planner": up_t.get("planner", 0.0),
        "critic": up_t.get("critic", 0.0),
        "refiner": up_t.get("refiner", 0.0),
        "upstream": up_t["upstream"],
        "load": times["load"],
    })
    caches = [to_cpu(p) for p in split_past(past, B)]
    del past, stacked, clones
    return caches, times


def dump_rows(rows: List[dict], path: str, mode: str = "a"):
    with open(path, mode) as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")


def load_jsonl(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_eval(args):
    if float(getattr(args, "seal_coef", 0.0) or 0.0) > 0:
        raise SystemExit(
            "Judger SEAL is refused until aime30 GO. "
            "Find the live K=10 block first (see scripts/run_one_role.sh)."
        )
    ns = make_ns(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for --mode eval / smoke")
    wrapper = maybe_load_model(args, ns)
    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass
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
    requested = [a.strip() for a in str(args.arms).split(",") if a.strip()] if args.arms else None
    arms = expand_one_role_arms(requested, k=args.k, include_real=bool(requested and "real" in requested))
    specs = [parse_one_role_arm(a) for a in arms]
    print(f"[eval] task={args.task} n={len(items)} idx={[it['idx'] for it in items]} arms={arms}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    dump = os.path.join(args.out_dir, "latency_rows.jsonl")
    if args.force and os.path.isfile(dump):
        os.remove(dump)
    done = _done_idx_arms(dump)
    t_wall = time.time()
    bs = max(1, int(args.generate_bs))

    for start in range(0, len(items), bs):
        batch = items[start : start + bs]
        for spec in specs:
            need = [it for it in batch if spec["name"] not in done.get(int(it["idx"]), set())]
            if not need:
                continue
            questions = [it["question"] for it in need]
            golds = [it["gold"] for it in need]
            B = len(need)
            jids, jmask = judger_tensors(wrapper, questions, ns)
            caches, ut = run_arm_batch(wrapper, ns, questions, frozen_cpu, spec, args.k)
            reset_peak()
            sync()
            td0 = time.perf_counter()
            texts, ntoks, eoss = decode_batch(
                wrapper, jids, jmask,
                [to_dev(deep_clone(c), wrapper.device) for c in caches],
                args.judger_budget, temperature=args.temperature, top_p=args.top_p,
            )
            sync()
            t_j = time.perf_counter() - td0
            fwds = upstream_forwards(spec)
            recs = []
            for b, it in enumerate(need):
                rec = {
                    "idx": int(it["idx"]),
                    "method": spec["name"],
                    "role": spec["role"],
                    "k": int(spec["k"]),
                    "batch_size": B,
                    "planner_s": ut["planner"] / B,
                    "critic_s": ut["critic"] / B,
                    "refiner_s": ut["refiner"] / B,
                    "upstream_s": ut["upstream"] / B,
                    "cache_load_s": ut["load"] / B,
                    "judger_s": t_j / B,
                    "e2e_s": (ut["load"] + ut["upstream"] + t_j) / B,
                    "tokens": int(ntoks[b]),
                    "eos": bool(eoss[b]),
                    "correct": bool(graded(texts[b], golds[b], args.task)),
                    "pred": pred_short(texts[b]),
                    "cache_pos": int(num_positions(caches[b])),
                    "cache_mb": float(kv_mb(caches[b])),
                    "upstream_forwards": fwds,
                }
                recs.append(rec)
                done[int(it["idx"])].add(spec["name"])
            dump_rows(recs, dump, "a")
            n_ok = sum(r["correct"] for r in recs)
            print(
                f"[eval] {spec['name']} items {[it['idx'] for it in need]} "
                f"{n_ok}/{B} tok~{float(np.mean(ntoks)):.0f} "
                f"up={ut['upstream']:.2f}s J={t_j:.1f}s "
                f"elapsed={time.time()-t_wall:.0f}s",
                flush=True,
            )
            del caches, texts
            torch.cuda.empty_cache()

    rows = load_jsonl(dump)
    report = summarize(rows, args, payload.get("meta") or {}, items, specs)
    with open(os.path.join(args.out_dir, "latency_rows.json"), "w") as f:
        json.dump(rows, f)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print_table(report)
    write_gate(args, report)
    print(f"[done] {os.path.join(args.out_dir, 'report.json')}", flush=True)
    print("ONE_ROLE_EVAL_DONE", flush=True)
    return report


def run_smoke(args):
    """Two Frozen passes on n=2 MATH items must match exactly."""
    args.task = "math"
    args.n = 2
    args.arms = "frozen"
    args.indices = ""
    args.generate_bs = 1
    if args.judger_budget <= 0:
        args.judger_budget = 2048
    ns = make_ns(args, task="math")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for --mode smoke")
    wrapper = maybe_load_model(args, ns)
    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass
    dtype = next(wrapper.model.parameters()).dtype
    payload, frozen_cpu = load_frozen(args, dtype=dtype)
    frozen_cpu = cache_as_past(payload, device="cpu", dtype=dtype)
    items = select_items(load_task("math", args.split), 2, None)
    spec = parse_one_role_arm("frozen")
    os.makedirs(args.out_dir, exist_ok=True)

    def one_pass(tag: str):
        recs = []
        for it in items:
            q = it["question"]
            jids, jmask = judger_tensors(wrapper, [q], ns)
            caches, ut = run_arm_batch(wrapper, ns, [q], frozen_cpu, spec, args.k)
            texts, ntoks, eoss = decode_batch(
                wrapper, jids, jmask,
                [to_dev(deep_clone(caches[0]), wrapper.device)],
                args.judger_budget, temperature=0.0, top_p=1.0,
            )
            recs.append({
                "idx": int(it["idx"]),
                "pass": tag,
                "correct": bool(graded(texts[0], it["gold"], "math")),
                "pred": pred_short(texts[0]),
                "tokens": int(ntoks[0]),
                "eos": bool(eoss[0]),
                "cache_pos": int(num_positions(caches[0])),
                "load_s": ut["load"],
            })
            print(
                f"[smoke] {tag} idx={it['idx']} correct={recs[-1]['correct']} "
                f"pred={recs[-1]['pred']!r} tok={ntoks[0]} pos={recs[-1]['cache_pos']}",
                flush=True,
            )
            del caches
            torch.cuda.empty_cache()
        return recs

    a = one_pass("a")
    b = one_pass("b")
    ok, reason = smoke_match(a, b)
    pos_ok = all(r["cache_pos"] > 0 for r in a)
    if not pos_ok:
        ok, reason = False, f"empty frozen cache pos={ [r['cache_pos'] for r in a] }"
    report = {
        "config": {"task": "math", "n": 2, "mode": "smoke", "seed": args.seed},
        "ok": bool(ok),
        "reason": reason,
        "pass_a": a,
        "pass_b": b,
        "recommend": "go" if ok else "stop",
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    write_gate(args, {"stage": "smoke", "smoke": report, "recommend": report["recommend"],
                      "reason": reason, "next": "MATH n=20 three roles" if ok else "fix harness"})
    print(f"[smoke] ok={ok} {reason}", flush=True)
    if not ok:
        raise SystemExit("ONE_ROLE_SMOKE_FAIL")
    print("ONE_ROLE_SMOKE_DONE", flush=True)
    return report


def _correct_by_idx(rows, method: str):
    rs = sorted((r for r in rows if r["method"] == method), key=lambda r: r["idx"])
    return [bool(r["correct"]) for r in rs], [int(r["idx"]) for r in rs]


def summarize(rows, args, pre_meta, items, specs):
    methods = []
    seen = set()
    for r in rows:
        if r["method"] not in seen:
            seen.add(r["method"])
            methods.append(r["method"])
    out: Dict[str, Any] = {
        "config": {
            "task": args.task, "n": len(items), "k": args.k,
            "generate_bs": args.generate_bs, "judger_budget": args.judger_budget,
            "cache": args.cache, "seed": args.seed, "indices": [it["idx"] for it in items],
            "arms": [s["name"] for s in specs],
        },
        "precompute": {
            k: pre_meta.get(k) for k in (
                "n_donors", "k", "scaffold", "donor_task", "n_pos", "mb", "model_name",
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
            "acc": float(np.mean([r["correct"] for r in rs])) if rs else None,
            "role": rs[0].get("role") if rs else None,
            "k": int(rs[0].get("k") or 0) if rs else 0,
        }
        for k in keys:
            if rs and k in rs[0]:
                blk[k] = mean_ci([r[k] for r in rs], seed=args.seed)
        out["arms"][m] = blk

    frozen_acc = (out["arms"].get("frozen") or {}).get("acc")
    pick = pick_best_role(out["arms"], frozen_acc=frozen_acc, k=args.k)
    out["best_role"] = pick
    paired = {}
    if "frozen" in by:
        f_idx = set(by["frozen"])
        for m, idxmap in by.items():
            if m == "frozen":
                continue
            common = sorted(f_idx & set(idxmap))
            f_ok = [bool(by["frozen"][i]["correct"]) for i in common]
            l_ok = [bool(idxmap[i]["correct"]) for i in common]
            stage = "aime6"
            if args.task.startswith("aime"):
                if len(common) >= 30:
                    stage = "aime30"
                elif len(common) >= 12:
                    stage = "aime12"
            else:
                stage = "math"
            if args.task.startswith("aime"):
                paired[m] = aime_subset_verdict(f_ok, l_ok, stage=stage)
            else:
                paired[m] = paired_flags(f_ok, l_ok)
    out["paired"] = paired
    return out


def print_table(report):
    print("\n=== ONE ROLE (mean / item) ===", flush=True)
    hdr = (
        f"{'arm':<24}{'acc':>8}{'e2e':>8}{'up':>8}{'J':>8}"
        f"{'tok':>8}{'pos':>8}{'fwd':>8}"
    )
    print(hdr, flush=True)
    for name, a in report.get("arms", {}).items():
        def m(key):
            blk = a.get(key) or {}
            return float(blk.get("mean") or 0.0)
        acc = a.get("acc")
        acc_s = f"{acc:8.3f}" if acc is not None else f"{'na':>8}"
        print(
            f"{name:<24}{acc_s}{m('e2e_s'):8.2f}{m('upstream_s'):8.2f}"
            f"{m('judger_s'):8.2f}{m('tokens'):8.1f}{m('cache_pos'):8.0f}"
            f"{m('upstream_forwards'):8.0f}",
            flush=True,
        )
    best = report.get("best_role") or {}
    if best.get("arm"):
        print(f"  best={best['arm']} ok={best.get('ok')} {best.get('reason')}", flush=True)


def gate_path(args) -> str:
    root = args.root_dir or os.path.dirname(args.out_dir.rstrip("/"))
    return os.path.join(root, "gate.json")


def checkin_path(args) -> str:
    root = args.root_dir or os.path.dirname(args.out_dir.rstrip("/"))
    return os.path.join(root, "CHECKIN.md")


def load_gate(args) -> Dict[str, Any]:
    path = gate_path(args)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"recommend": "running", "reason": "no stages yet"}


def write_gate(args, update: Dict[str, Any]):
    os.makedirs(args.root_dir, exist_ok=True)
    gate = load_gate(args)
    gate.update({k: v for k, v in update.items() if v is not None})
    # fold the latest eval report into the right slot
    report = update if "arms" in update else None
    if report and args.mode == "eval":
        if args.task == "math":
            frozen_acc = (report.get("arms") or {}).get("frozen", {}).get("acc")
            best = report.get("best_role") or {}
            gate["math"] = {
                "n": report.get("config", {}).get("n"),
                "frozen_acc": frozen_acc,
                "best_arm": best.get("arm"),
                "best_acc": best.get("acc"),
                "best_role": best.get("role"),
                "ok": best.get("ok"),
                "reason": best.get("reason"),
                "candidates": best.get("candidates"),
            }
            gate["recommend"] = "go" if best.get("ok") else "stop"
            gate["reason"] = best.get("reason") or ""
            gate["next"] = (
                f"AIME n=6 frozen + {best.get('arm')}" if best.get("ok") else "stop"
            )
            gate["best_arm"] = best.get("arm")
        elif args.task.startswith("aime"):
            n = int(report.get("config", {}).get("n") or 0)
            key = "aime6" if n <= 6 else ("aime12" if n <= 12 else "aime30")
            best_arm = gate.get("best_arm") or args.arms.split(",")[-1]
            paired = (report.get("paired") or {}).get(best_arm) or {}
            if not paired:
                # fall back to the first non-frozen paired block
                for k, v in (report.get("paired") or {}).items():
                    paired = v
                    best_arm = k
                    break
            gate[key] = paired
            gate["recommend"] = paired.get("recommend") or "unknown"
            gate["reason"] = paired.get("reason") or ""
            if gate["recommend"] in ("go", "borderline"):
                gate["next"] = {
                    "aime6": "AIME n=12 same role",
                    "aime12": "AIME n=30 same role",
                    "aime30": "Judger SEAL on this config (do not train)",
                }[key]
            else:
                gate["next"] = "stop — one live K=10 did not earn more GPU"
    with open(gate_path(args), "w") as f:
        json.dump(gate, f, indent=2)
    with open(checkin_path(args), "w") as f:
        f.write(render_checkin(gate))
    print(f"[gate] {gate_path(args)} recommend={gate.get('recommend')}", flush=True)
    print(open(checkin_path(args)).read(), flush=True)


def run_gate(args):
    gate = load_gate(args)
    os.makedirs(args.root_dir, exist_ok=True)
    with open(checkin_path(args), "w") as f:
        f.write(render_checkin(gate))
    print(open(checkin_path(args)).read(), flush=True)
    rec = gate.get("recommend") or "running"
    if rec == "stop":
        raise SystemExit("ONE_ROLE_GATE_STOP")
    print("ONE_ROLE_GATE_DONE", flush=True)
    return gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="eval", choices=["smoke", "eval", "gate"])
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--task", default="math",
                    choices=["gsm8k", "math", "aime2024", "aime2025"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--generate_bs", type=int, default=0)
    ap.add_argument("--judger_budget", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument(
        "--arms", default="",
        help="frozen,frozen_planner_k10,frozen_refiner_k10,frozen_critic_k10",
    )
    ap.add_argument("--indices", default="", help="Comma/range indices into the split.")
    ap.add_argument("--cache", default="")
    ap.add_argument("--seal_coef", type=float, default=0.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--root_dir", default="artifacts/one_role")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    if not args.out_dir:
        args.out_dir = os.path.join(args.root_dir, args.mode if args.mode != "eval" else f"{args.task}_n{args.n or default_n(args.task, False)}")
    if args.n <= 0:
        args.n = default_n(args.task, args.mode == "smoke")
    if args.generate_bs <= 0:
        args.generate_bs = default_bs(args.task)
    if args.judger_budget <= 0:
        args.judger_budget = 256 if args.mode == "smoke" else default_budget(args.task)
    if not args.cache:
        args.cache = cache_path(os.path.join(ROOT, "artifacts/math_ladder/math1k"))
    if not args.arms:
        args.arms = ",".join(expand_one_role_arms(None, k=args.k))

    set_seed(args.seed)
    os.makedirs(args.root_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "eval":
        run_eval(args)
    else:
        run_gate(args)


if __name__ == "__main__":
    main()
