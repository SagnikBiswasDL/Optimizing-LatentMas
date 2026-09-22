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
import io
import json
import os
import re
import shutil
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
UP_ROLES = ("planner", "critic", "refiner")
# Upstream compute a view actually still has to pay. In the growing tape a later
# role's write only exists because the earlier roles ran, so `c3` costs the same
# silent-agent time as `real`; only `isolated`/`none` can bank the saving.
PAID_ROLES = {
    "none": (),
    "c1": ("planner",),
    "c2": ("planner", "critic"),
    "c12": ("planner", "critic"),
}
# Arm-major order: the arms that decide the recommendation run first, so a
# partial run is still readable if we stop early.
VIEW_ARMS = (
    "real",
    "none",
    "c3",
    "c23",
    "c1",
    "c2",
    "c12",
    "evict_seg",
    "evict_uniform",
    "latents",
    "shuf",
)


_SEAL_ARM = re.compile(r"^(?P<base>.+)_seal(?P<coef>\d+(?:\.\d+)?)$")


def parse_arm(name: str):
    """`real_seal40` -> ("real", 40.0); plain arms -> (name, None).

    Encoding the coefficient in the arm name keeps each setting a distinct,
    restart-safe row and lets one process sweep coefficients without reloading
    the model — the steering hook is attached once and armed per decode.
    """
    m = _SEAL_ARM.match(name)
    if m:
        return m.group("base"), float(m.group("coef"))
    return name, None


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


def tape_root(args) -> str:
    """Tapes are ~130MB each and must not land on a network mount.

    /workspace on the pod is MooseFS, where torch.save's zip writer fails
    mid-file ("unexpected pos"). Keep the big tensors on local disk and leave
    only the small JSON artifacts in out_dir.
    """
    d = getattr(args, "tape_dir", "") or os.environ.get("TAPE_DIR") or ""
    return d if d else os.path.join(args.out_dir, "tapes")


def tape_path(args, idx: int) -> str:
    return os.path.join(tape_root(args), f"{int(idx):04d}.pt")


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


def agent_times_path(root: str) -> str:
    return os.path.join(root, "agent_times.jsonl")


def append_agent_times(root: str, idx: int, pass_name: str, times, peaks, spans) -> None:
    """Per-role silent-agent cost, kept tensor-free so `report` stays cheap."""
    os.makedirs(root, exist_ok=True)
    with open(agent_times_path(root), "a") as f:
        for sp in spans:
            role = sp["role"]
            f.write(json.dumps({
                "idx": int(idx),
                "pass": pass_name,
                "role": role,
                "seconds": float(times.get(role) or 0.0),
                "positions": int(sp["end"]) - int(sp["start"]),
                "peak_mb": float(peaks.get(role) or 0.0),
            }) + "\n")


def persist_small(args) -> None:
    """Copy the tensor-free artifacts to a location that outlives the container.

    Pod-local disk is wiped on stop, so a long sweep must check its results in
    somewhere durable as it goes rather than only at the end. Tapes are excluded
    on purpose: they are large, regenerable, and large writes are exactly what
    the network mount mishandles.
    """
    dest = getattr(args, "persist_dir", "") or os.environ.get("PERSIST_DIR") or ""
    if not dest:
        return
    try:
        os.makedirs(dest, exist_ok=True)
        for name in ("rows.jsonl", "agent_times.jsonl", "report.json",
                     "CHECKIN.md", "BUDGET.md", "budget.json"):
            src = os.path.join(args.out_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest, name))
        tsrc = os.path.join(args.out_dir, "texts")
        if os.path.isdir(tsrc):
            tdst = os.path.join(dest, "texts")
            os.makedirs(tdst, exist_ok=True)
            for fn in os.listdir(tsrc):
                shutil.copy2(os.path.join(tsrc, fn), os.path.join(tdst, fn))
    except Exception as exc:  # never let mirroring kill a sweep
        print(f"[persist] failed: {exc}", flush=True)


def load_agent_times(root: str) -> List[Dict]:
    path = agent_times_path(root)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


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
    """Serialize to memory, then lay the bytes down in one sequential write.

    torch.save straight to a path seeks while writing, which some network
    filesystems mishandle; a buffered write plus atomic rename avoids both the
    seek pattern and half-written tapes on crash.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    buf = io.BytesIO()
    torch.save(payload, buf)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(buf.getbuffer())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_tape(path: str) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def view_cache(name: str, tape: Dict, others: Sequence[Dict], args) -> Any:
    past = tape["past"]
    spans = tape["spans"]
    k = int(tape.get("k") or args.k)
    name, _coef = parse_arm(name)  # SEAL arms reuse their base arm's cache view
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
        path = tape_path(args, i)
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
        append_agent_times(args.out_dir, i, "growing", times, peaks, spans)
        print(
            f"[collect] idx={i} pos={payload['pos']} mb={payload['mb']:.1f} "
            f"up={times['upstream']:.2f}s spans={spans}",
            flush=True,
        )
        del past, cpu
        torch.cuda.empty_cache()


def list_tapes(args) -> List[Dict]:
    d = tape_root(args)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(d, name)
        # Truncated/zero-byte tapes are a real artifact of the network-mount
        # corruption; skip them loudly rather than dying on torch.load.
        if os.path.getsize(path) == 0:
            print(f"[tapes] skip empty {path}", flush=True)
            continue
        try:
            out.append(load_tape(path))
        except Exception as exc:
            print(f"[tapes] skip unreadable {path}: {exc}", flush=True)
    return out


def decode_group(wrapper, ns, args, items: Sequence[Dict], caches: Sequence[Any],
                 method: str, extras: Optional[Sequence[Optional[Dict]]] = None,
                 seal_on: Optional[bool] = None) -> List[Dict]:
    """Decode a group of items in one generate() call.

    generate() runs until every sequence in the batch finishes, so a group costs
    max(tokens) steps rather than sum(tokens). That is still a large throughput
    win because the per-step cost is dominated by streaming the weights, which
    the batch amortizes across items.

    The cost is that per-item wall clock is no longer observable: a short item
    sharing a batch with a runaway one appears to take as long as the runaway.
    Grouped rows therefore carry batch_s/batch_size and a null judger_s, and the
    latency tables skip them. Use --decode_bs 1 whenever latency is the question.
    """
    n = len(items)
    have = [c is not None for c in caches]
    if any(have) and not all(have):
        raise ValueError("cannot batch cached and cacheless items together")
    questions = [it["question"] for it in items]
    jids, jmask = judger_tensors(wrapper, questions, ns)
    dev_caches = [None if c is None else to_dev(deep_clone(c), wrapper.device) for c in caches]
    if seal_on is None:
        seal_on = bool(args.seal_on and "_seal" in method)
    reset_peak()
    sync()
    t0 = time.perf_counter()
    texts, ntoks, eoss = decode_batch_maybe_seal(
        wrapper, jids, jmask, dev_caches, args.judger_budget,
        temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k if float(args.temperature) > 0 else None,
        seal_on=bool(seal_on),
    )
    sync()
    t_batch = time.perf_counter() - t0
    del dev_caches
    tdir = os.path.join(args.out_dir, "texts")
    os.makedirs(tdir, exist_ok=True)
    rows: List[Dict] = []
    for b, it in enumerate(items):
        cache = caches[b]
        row = {
            "idx": int(it["idx"]),
            "method": method,
            "task": args.task,
            "correct": bool(graded(texts[b], it.get("gold") or "", args.task)),
            "pred": pred_short(texts[b]),
            "tokens": int(ntoks[b]),
            "eos": bool(eoss[b]),
            # Per-item timing is only meaningful when the item had the GPU alone.
            "judger_s": float(t_batch) if n == 1 else None,
            "batch_size": n,
            "batch_s": float(t_batch),
            "cache_pos": 0 if cache is None else int(num_positions(cache)),
            "cache_mb": 0.0 if cache is None else float(kv_mb(cache)),
            "critical": int(it["idx"]) in CRITICAL,
        }
        extra = (extras or [None] * n)[b]
        if extra:
            row["extra"] = extra
        # Full generation is kept so `--mode budget` can replay truncation offline
        # instead of re-decoding on the GPU.
        with open(os.path.join(tdir, f"{int(it['idx']):04d}_{method}.txt"), "w") as f:
            f.write(texts[b])
        rows.append(row)
    tag = f"{t_batch:.1f}s" if n == 1 else f"{t_batch:.1f}s/B{n}"
    for row in rows:
        print(
            f"[decode] idx={row['idx']} {method} acc={int(row['correct'])} "
            f"tok={row['tokens']} pos={row['cache_pos']} {tag} pred={row['pred']!r}",
            flush=True,
        )
    return rows


def decode_one(wrapper, ns, args, it, cache, method: str, extra: Optional[Dict],
               seal_on: Optional[bool] = None) -> Dict:
    return decode_group(wrapper, ns, args, [it], [cache], method, [extra], seal_on)[0]


def run_views(args) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for --mode views")
    tapes = list_tapes(args)
    if not tapes:
        raise SystemExit(f"no tapes in {tape_root(args)} — run --mode collect")
    ns = make_ns(args)
    wrapper = maybe_load_model(args, ns)
    arms = [a.strip() for a in args.view_arms.split(",") if a.strip()]
    needs_seal = any(parse_arm(a)[1] is not None for a in arms)
    if args.seal_vector:
        attach_judger_seal(wrapper, args.seal_vector, coef=args.seal_coef,
                           layer_index=args.seal_layer)
    elif needs_seal:
        raise SystemExit("SEAL arms requested but --seal_vector is empty")
    already = seen_keys(load_rows(args.out_dir))
    for name in arms:
        base, coef = parse_arm(name)
        if coef is not None:
            # one attached hook, retuned per arm — no reload between coefficients.
            # role_coefs shadows .coef inside _resolve, so update it too.
            wrapper.seal.coef = float(coef)
            rc = getattr(wrapper.seal, "role_coefs", None)
            if isinstance(rc, dict) and "judger" in rc:
                rc["judger"] = float(coef)
            print(f"[views] SEAL armed coef={coef} layer={args.seal_layer}", flush=True)
        method = name
        if float(args.temperature) > 0:
            method = f"{name}_t{str(args.temperature).replace('.', '')}"
        # A label that does not change the cache view, so the same arm can be
        # re-decoded under a different setting (batch size, sampler) and compared
        # against its own baseline instead of overwriting it.
        if args.method_tag:
            method = f"{method}__{args.method_tag}"
        pending = []
        for tape in tapes:
            key = (int(tape["idx"]), method)
            if key in already and not args.overwrite:
                print(f"[views] skip {key}", flush=True)
                continue
            pending.append(tape)
        bs = max(1, int(args.decode_bs))
        for start in range(0, len(pending), bs):
            chunk = pending[start : start + bs]
            items, caches, extras = [], [], []
            for tape in chunk:
                cache, extra = view_cache(name, tape, tapes, args)
                items.append({"idx": tape["idx"], "question": tape["question"],
                              "gold": tape["gold"]})
                caches.append(cache)
                extras.append(extra)
            rows = decode_group(wrapper, ns, args, items, caches, method, extras,
                                seal_on=coef is not None)
            for tape, row in zip(chunk, rows):
                row["seal_coef"] = coef
                times = tape.get("times") or {}
                paid = PAID_ROLES.get(base, UP_ROLES)
                row["upstream_s"] = float(sum(float(times.get(r) or 0.0) for r in paid))
                row["paid_roles"] = list(paid)
                append_row(args.out_dir, row)
                already.add((int(tape["idx"]), method))
            del caches
            torch.cuda.empty_cache()
        # refresh the check-in and mirror it off ephemeral disk after every arm
        try:
            write_report(args)
        except Exception as exc:  # report must never kill a long sweep
            print(f"[views] report failed: {exc}", flush=True)
        persist_small(args)


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
    tapes = list_tapes(args)
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
        row["paid_roles"] = list(UP_ROLES)
        append_row(args.out_dir, row)
        append_agent_times(args.out_dir, i, method, times, peaks, spans)
        tag = os.path.join(tape_root(args), "isolated", f"{int(i):04d}_{method}.pt")
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
        ups = [float(x.get("upstream_s") or 0.0) for x in rs]
        # Only rows that had the GPU to themselves carry interpretable per-item
        # latency; grouped rows are excluded rather than divided by batch size.
        timed = [x for x in rs if x.get("judger_s") is not None]
        js = [float(x["judger_s"]) for x in timed]
        e2e = [float(x["judger_s"]) + float(x.get("upstream_s") or 0.0) for x in timed]
        # batch_s/batch_size sums back to true wall clock regardless of grouping.
        wall = sum(float(x.get("batch_s") or x.get("judger_s") or 0.0)
                   / max(1, int(x.get("batch_size") or 1)) for x in rs)
        crit = [x for x in rs if x.get("critical")]
        n_eos = sum(1 for x in rs if x.get("eos"))
        out[m] = {
            "n": len(rs),
            "acc": float(np.mean(acc)) if acc else 0.0,
            "tokens": mean_ci(toks),
            "cache_pos": mean_ci(pos),
            "n_timed": len(timed),
            "max_batch": max(int(x.get("batch_size") or 1) for x in rs) if rs else 1,
            "judger_s": float(np.mean(js)) if js else 0.0,
            "upstream_s": float(np.mean(ups)) if ups else 0.0,
            "e2e_s": float(np.mean(e2e)) if e2e else 0.0,
            "wall_s": float(wall),
            "eos_rate": float(n_eos) / len(rs) if rs else 0.0,
            "tok_per_s": (float(np.sum(toks)) / float(np.sum(js))) if np.sum(js) > 0 else 0.0,
            "tok_per_s_wall": (float(np.sum(toks)) / wall) if wall > 0 else 0.0,
            "correct_idx": sorted(int(x["idx"]) for x in rs if x["correct"]),
            "critical_kept": sorted(int(x["idx"]) for x in crit if x["correct"]),
            "critical_lost": sorted(int(x["idx"]) for x in crit if not x["correct"]),
            "paid_roles": (rs[0].get("paid_roles") if rs else None),
        }
    return out


def _agent_table(root: str, arms: Dict[str, Dict]) -> tuple:
    """Per-agent wall clock: the four LatentMAS roles side by side."""
    recs = load_agent_times(root)
    if not recs:
        return {}, ["No per-agent timings yet — run `--mode collect`."]
    by: Dict[str, Dict[str, List[float]]] = {}
    for r in recs:
        if r.get("pass") != "growing":
            continue
        d = by.setdefault(r["role"], {"s": [], "pos": [], "mb": []})
        d["s"].append(float(r["seconds"]))
        d["pos"].append(float(r["positions"]))
        d["mb"].append(float(r["peak_mb"]))
    real = arms.get("real") or {}
    judger_s = float(real.get("judger_s") or 0.0)
    stats = {
        role: {
            "n": len(d["s"]),
            "seconds": float(np.mean(d["s"])),
            "positions": float(np.mean(d["pos"])),
            "peak_mb": float(np.mean(d["mb"])),
        }
        for role, d in by.items()
    }
    up_total = sum(v["seconds"] for v in stats.values())
    total = up_total + judger_s
    if judger_s > 0:
        stats["judger"] = {
            "n": int(real.get("n") or 0),
            "seconds": judger_s,
            "positions": float((real.get("tokens") or {}).get("mean") or 0.0),
            "peak_mb": 0.0,
        }
    lines = [
        "| Agent | n | seconds | % of e2e | KV positions written | peak MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for role in list(UP_ROLES) + ["judger"]:
        v = stats.get(role)
        if not v:
            continue
        pct = 100.0 * v["seconds"] / total if total > 0 else 0.0
        label = "tokens emitted" if role == "judger" else "KV positions"
        mb = f"{v['peak_mb']:.0f}" if v["peak_mb"] else "-"
        lines.append(
            f"| {role} | {v['n']} | {v['seconds']:.3f} | {pct:.2f}% | "
            f"{v['positions']:.0f} ({label}) | {mb} |"
        )
    lines.append("")
    lines.append(
        f"Silent agents together cost {up_total:.3f}s of the {total:.1f}s item. "
        "Anything that only removes silent-agent work is capped at that number — "
        "the Judger decode is the budget that matters."
    )
    return stats, lines


def _latency_notes(arms: Dict[str, Dict]) -> List[str]:
    """Where the AIME wall clock actually goes, and what each arm could bank."""
    real = arms.get("real")
    if not real:
        return ["No `real` arm yet — cannot attribute latency."]
    j, u = real["judger_s"], real["upstream_s"]
    tot = j + u or 1.0
    notes = []
    if real.get("n_timed"):
        notes += [
            f"Real end-to-end **{tot:.1f}s/item**: Judger decode {j:.1f}s "
            f"({100.0 * j / tot:.1f}%), silent agents {u:.2f}s ({100.0 * u / tot:.1f}%). "
            f"(n_timed={real['n_timed']} of {real['n']}, unbatched only.)",
            f"Real Judger emits {real['tokens']['mean']:.0f} tokens at "
            f"{real['tok_per_s']:.1f} tok/s; {100.0 * real['eos_rate']:.0f}% of items stop "
            f"on EOS (the rest burn the full budget).",
        ]
    else:
        notes.append(
            "No unbatched `real` rows — per-item latency is unavailable. "
            "Re-run the arm with `--decode_bs 1` if latency is the question."
        )
    notes += [
        "",
        "Per-item latency (unbatched rows only; `—` means the arm was run grouped):",
        "",
        "| Arm | acc | judger_s | upstream_s | e2e_s | vs Real | tokens | eos | n(timed) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m, b in sorted(arms.items(), key=lambda kv: kv[1]["e2e_s"]):
        if not b.get("n_timed"):
            notes.append(
                f"| {m} | {b['acc']:.3f} | — | {b['upstream_s']:.2f} | — | — | "
                f"{b['tokens']['mean']:.0f} | {100.0 * b['eos_rate']:.0f}% | "
                f"0/{b['n']} |"
            )
            continue
        d = b["e2e_s"] - tot
        notes.append(
            f"| {m} | {b['acc']:.3f} | {b['judger_s']:.1f} | {b['upstream_s']:.2f} | "
            f"{b['e2e_s']:.1f} | {d:+.1f}s ({100.0 * d / tot:+.1f}%) | "
            f"{b['tokens']['mean']:.0f} | {100.0 * b['eos_rate']:.0f}% | "
            f"{b['n_timed']}/{b['n']} |"
        )
    notes += [
        "",
        "GPU cost actually paid per arm (valid for grouped runs too — `batch_s/batch_size` "
        "sums back to true wall clock):",
        "",
        "| Arm | n | max batch | wall_s | tok/s (aggregate) |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, b in sorted(arms.items(), key=lambda kv: -kv[1]["wall_s"]):
        notes.append(
            f"| {m} | {b['n']} | {b['max_batch']} | {b['wall_s']:.0f} | "
            f"{b['tok_per_s_wall']:.1f} |"
        )
    notes.extend([
        "",
        "Upstream is charged honestly: in the growing tape `c3` still needs Planner and "
        "Critic to have run, so only `none`/`c1`/`c2`/`isolated` can bank silent-agent time.",
    ])
    return notes


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
    agent_stats, agent_lines = _agent_table(args.out_dir, arms)
    report = {
        "task": args.task,
        "critical": list(CRITICAL),
        "n_rows": len(rows),
        "arms": arms,
        "pairing": pairing,
        "per_agent": agent_stats,
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
        "## Where the latency is (per agent)",
        "",
    ]
    lines.extend(agent_lines)
    lines.extend([
        "",
        "## Latency per arm",
        "",
    ])
    lines.extend(_latency_notes(arms))
    lines.extend([
        "",
        "## Accuracy per arm",
        "",
        f"Critical items (Real-only vs Frozen in the locked ladder): {list(CRITICAL)}",
        "",
        "| Arm | n | acc | correct items | critical kept | lost vs Real | recovered vs none |",
        "|---|---:|---:|---|---|---|---|",
    ])
    for m, blk in sorted(arms.items(), key=lambda kv: -kv[1]["acc"]):
        p = pairing.get(m) or {}
        lines.append(
            f"| {m} | {blk['n']} | {blk['acc']:.3f} | {blk.get('correct_idx')} | "
            f"{blk.get('critical_kept')} | {p.get('lost_vs_real')} | "
            f"{p.get('recovered_vs_none')} |"
        )
    lines.extend(["", "Jiayi map: `real` = judger(c1+c2+c3). `c1`/`c2`/`c3` = private writes. "
                  "`isolated` = a1(c1')→… with empty/frozen c', judger still gets concat writes. "
                  "`evict_seg` = SnapKV budget per role span.", ""])
    with open(os.path.join(args.out_dir, "CHECKIN.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"recommend": rec, "reason": reason, "arms": list(arms)}, indent=2), flush=True)
    return report


BUDGETS = (512, 1024, 1536, 2048, 3072, 4096, 6144, 8192)


def run_compare(args) -> Dict:
    """Diff two arms' saved generations token-for-token. CPU only.

    The point is to certify grouped decoding before trusting a batched sweep.
    Greedy decoding *should* be batch-invariant, but left-padding the KV cache
    and the different GEMM shapes a batch takes can perturb logits enough to
    change a tie, and once two runs diverge they diverge for good. If this comes
    back less than fully identical, batched accuracy numbers are still valid on
    their own terms but are no longer strictly comparable to the n=1 baseline,
    and that has to be stated wherever they appear.
    """
    names = [a.strip() for a in (args.compare_arms or "").split(",") if a.strip()]
    if len(names) != 2:
        raise SystemExit("--mode compare needs --compare_arms A,B")
    a, b = names
    rows = load_rows(args.out_dir)
    tdir = os.path.join(args.out_dir, "texts")
    ra = {int(r["idx"]): r for r in rows if r["method"] == a}
    rb = {int(r["idx"]): r for r in rows if r["method"] == b}
    common = sorted(set(ra) & set(rb))
    if not common:
        raise SystemExit(f"no shared items between {a!r} and {b!r}")

    def _text(arm, idx):
        p = os.path.join(tdir, f"{idx:04d}_{arm}.txt")
        return open(p).read() if os.path.isfile(p) else None

    items, n_ident, n_tok, n_acc = [], 0, 0, 0
    for idx in common:
        ta, tb = _text(a, idx), _text(b, idx)
        ident = ta is not None and ta == tb
        div = None
        if ta is not None and tb is not None and not ident:
            lim = min(len(ta), len(tb))
            div = next((i for i in range(lim) if ta[i] != tb[i]), lim)
        same_tok = int(ra[idx]["tokens"]) == int(rb[idx]["tokens"])
        same_acc = bool(ra[idx]["correct"]) == bool(rb[idx]["correct"])
        n_ident += int(ident)
        n_tok += int(same_tok)
        n_acc += int(same_acc)
        items.append({
            "idx": idx, "identical": bool(ident), "diverge_char": div,
            "tokens_a": int(ra[idx]["tokens"]), "tokens_b": int(rb[idx]["tokens"]),
            "correct_a": bool(ra[idx]["correct"]), "correct_b": bool(rb[idx]["correct"]),
            "same_tokens": bool(same_tok), "same_correct": bool(same_acc),
        })
    n = len(common)
    acc_a = sum(bool(ra[i]["correct"]) for i in common) / n
    acc_b = sum(bool(rb[i]["correct"]) for i in common) / n
    out = {
        "arm_a": a, "arm_b": b, "n": n,
        "identical_text": n_ident, "same_tokens": n_tok, "same_correct": n_acc,
        "acc_a": acc_a, "acc_b": acc_b,
        "wall_a": _arm_table(rows).get(a, {}).get("wall_s"),
        "wall_b": _arm_table(rows).get(b, {}).get("wall_s"),
        "verdict": ("identical" if n_ident == n else
                    "same_decisions" if n_acc == n and n_tok == n else
                    "diverged"),
        "items": items,
    }
    with open(os.path.join(args.out_dir, "compare.json"), "w") as f:
        json.dump(out, f, indent=2)
    lines = [
        f"# Parity: `{a}` vs `{b}`", "",
        f"n={n} · identical text **{n_ident}/{n}** · same tokens {n_tok}/{n} · "
        f"same correct {n_acc}/{n}",
        f"acc {acc_a:.3f} -> {acc_b:.3f}",
    ]
    if out["wall_a"] and out["wall_b"]:
        lines.append(
            f"wall clock {out['wall_a']:.0f}s -> {out['wall_b']:.0f}s "
            f"(**{out['wall_a'] / out['wall_b']:.2f}x**)"
        )
    lines += [
        "", f"**verdict: {out['verdict']}**", "",
        "| idx | identical | tokens A | tokens B | correct A | correct B | diverge @char |",
        "|---:|---|---:|---:|---|---|---:|",
    ]
    for it in items:
        lines.append(
            f"| {it['idx']} | {'yes' if it['identical'] else 'NO'} | {it['tokens_a']} | "
            f"{it['tokens_b']} | {int(it['correct_a'])} | {int(it['correct_b'])} | "
            f"{'—' if it['diverge_char'] is None else it['diverge_char']} |"
        )
    if out["verdict"] != "identical":
        lines += [
            "",
            "Not bit-identical, so batched rows are self-consistent but not "
            "drop-in comparable to unbatched ones. Keep each comparison within a "
            "single batch size, and say so when quoting the numbers.",
        ]
    with open(os.path.join(args.out_dir, "COMPARE.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return out


def _unbatched_tps(rows: List[Dict], arm: Optional[str] = None) -> float:
    """Per-item decode speed, from rows that had the GPU to themselves.

    Grouped rows cannot price a single item's latency: their wall clock is set by
    the slowest sequence in the batch. Prefer the arm's own unbatched rows, then
    any unbatched row, so a grouped accuracy sweep can still be priced using
    speed measured elsewhere in the same run.
    """
    for pool in ([r for r in rows if arm and r["method"] == arm], rows):
        timed = [r for r in pool if r.get("judger_s") is not None]
        tot_s = sum(float(r["judger_s"]) for r in timed)
        if tot_s > 0:
            return sum(float(r["tokens"]) for r in timed) / tot_s
    return 0.0


def run_budget(args) -> Dict:
    """Replay a hard token cap over saved generations. Tokenizer only, no GPU.

    Answers the latency question directly: if the Judger were cut off at N
    tokens, what accuracy would we keep and what would we pay? A truncated
    generation with no extractable answer counts as wrong, so this is the
    conservative version of an early-exit policy.
    """
    from transformers import AutoTokenizer

    rows = load_rows(args.out_dir)
    tdir = os.path.join(args.out_dir, "texts")
    if not rows or not os.path.isdir(tdir):
        raise SystemExit("need rows.jsonl + texts/ — run --mode views first")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    arms = [a.strip() for a in (args.budget_arms or "real").split(",") if a.strip()]
    pool = load_pool(args.task)
    out: Dict[str, Any] = {}
    for arm in arms:
        rs = [r for r in rows if r["method"] == arm]
        if not rs:
            continue
        # measured decode speed for this arm, used to price each budget
        tps = _unbatched_tps(rows, arm)
        per_budget = []
        for b in BUDGETS:
            n_ok = 0
            spent = []
            for r in rs:
                p = os.path.join(tdir, f"{int(r['idx']):04d}_{arm}.txt")
                if not os.path.isfile(p):
                    continue
                with open(p) as f:
                    text = f.read()
                ids = tok(text, add_special_tokens=False)["input_ids"]
                used = min(len(ids), b)
                spent.append(used)
                cut = tok.decode(ids[:b], skip_special_tokens=True) if len(ids) > b else text
                gold = pool[int(r["idx"])].get("gold") or ""
                if graded(cut, gold, args.task):
                    n_ok += 1
            if not spent:
                continue
            mean_tok = float(np.mean(spent))
            per_budget.append({
                "budget": int(b),
                "acc": n_ok / len(spent),
                "mean_tokens": mean_tok,
                "est_judger_s": mean_tok / tps if tps > 0 else 0.0,
            })
        base = per_budget[-1] if per_budget else None
        if base:
            for e in per_budget:
                e["speedup_vs_full"] = (
                    base["est_judger_s"] / e["est_judger_s"] if e["est_judger_s"] > 0 else 0.0)
                e["acc_delta"] = e["acc"] - base["acc"]
        out[arm] = {"tok_per_s": tps, "n": len(rs), "curve": per_budget}
    path = os.path.join(args.out_dir, "budget.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    lines = [
        "# Judger token-budget curve", "",
        "Hard cap replayed over saved generations; no answer inside the cap = wrong, "
        "so this is the conservative bound on an early-exit policy (a real policy could "
        "force an answer at the cap and do better).", "",
        "Caveat: lengths come from re-tokenizing the saved text, which can differ by a "
        "few tokens from the generation-time count. `est_judger_s` prices each budget at "
        "the measured tok/s for that arm.", "",
    ]
    for arm, blk in out.items():
        lines.extend([
            f"## `{arm}` (n={blk['n']}, {blk['tok_per_s']:.1f} tok/s measured)", "",
            "| budget | acc | Δacc | mean tokens | est judger_s | speedup |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for e in blk["curve"]:
            lines.append(
                f"| {e['budget']} | {e['acc']:.3f} | {e.get('acc_delta', 0.0):+.3f} | "
                f"{e['mean_tokens']:.0f} | {e['est_judger_s']:.1f} | "
                f"{e.get('speedup_vs_full', 1.0):.2f}x |")
        lines.append("")
    with open(os.path.join(args.out_dir, "BUDGET.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return out


def _loop_onset(ids: Sequence[int], *, n: int = 8, window: int = 256,
                stride: int = 128, thresh: float = 0.8) -> Optional[int]:
    """First position where the tail has gone degenerate, or None.

    Slides a window and asks what fraction of its n-grams already appeared
    earlier in the generation. A run that is looping repeats almost everything;
    a run still making progress does not. Requires two consecutive windows over
    threshold so a repeated formula or restated equation does not trip it.
    """
    if len(ids) < window * 2:
        return None
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    hits = 0
    for start in range(window, len(grams) - window, stride):
        seen = set(grams[:start])
        win = grams[start:start + window]
        if not win:
            break
        rep = sum(1 for g in win if g in seen) / len(win)
        if rep >= thresh:
            hits += 1
            if hits >= 2:
                return int(start - stride)
        else:
            hits = 0
    return None


def run_loops(args) -> Dict:
    """Find degenerate tails in saved generations and price aborting there.

    This targets the actual latency lever: runs that never emit EOS burn the
    whole budget and are wrong anyway, so cutting them costs no accuracy. A
    hard token cap cannot do this job because the answer sits at the very end
    of a healthy generation.
    """
    from transformers import AutoTokenizer

    rows = load_rows(args.out_dir)
    tdir = os.path.join(args.out_dir, "texts")
    if not rows or not os.path.isdir(tdir):
        raise SystemExit("need rows.jsonl + texts/ — run --mode views first")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    arms = [a.strip() for a in (args.budget_arms or "real").split(",") if a.strip()]
    out: Dict[str, Any] = {}
    for arm in arms:
        rs = sorted([r for r in rows if r["method"] == arm], key=lambda r: int(r["idx"]))
        if not rs:
            continue
        tps = _unbatched_tps(rows, arm)
        items, saved_tok, lost_solves = [], 0.0, []
        for r in rs:
            p = os.path.join(tdir, f"{int(r['idx']):04d}_{arm}.txt")
            if not os.path.isfile(p):
                continue
            with open(p) as f:
                ids = tok(f.read(), add_special_tokens=False)["input_ids"]
            onset = _loop_onset(ids, n=args.loop_ngram, window=args.loop_window,
                               thresh=args.loop_thresh)
            n_tok = len(ids)
            save = max(0, n_tok - onset) if onset is not None else 0
            # aborting only costs accuracy if the item was correct AND the
            # answer lives after the abort point
            costs = bool(r["correct"]) and onset is not None
            if costs:
                lost_solves.append(int(r["idx"]))
            saved_tok += save
            items.append({
                "idx": int(r["idx"]), "correct": bool(r["correct"]),
                "eos": bool(r.get("eos")), "tokens": n_tok,
                "loop_onset": onset, "saved_tokens": save,
                "would_lose_solve": costs,
            })
        base = sum(i["tokens"] for i in items) or 1
        out[arm] = {
            "n": len(items), "tok_per_s": tps,
            "total_tokens": base,
            "saved_tokens": int(saved_tok),
            "saved_frac": saved_tok / base,
            "saved_seconds": saved_tok / tps if tps > 0 else 0.0,
            "would_lose_solves": sorted(lost_solves),
            "items": items,
        }
    with open(os.path.join(args.out_dir, "loops.json"), "w") as f:
        json.dump(out, f, indent=2)
    lines = [
        "# Degenerate-tail (loop) detection", "",
        f"Detector: {args.loop_ngram}-gram repetition over a {args.loop_window}-token "
        f"window, threshold {args.loop_thresh}, two consecutive windows required.", "",
        "Aborting at the onset costs accuracy only for rows marked "
        "`would_lose_solve` — those were correct, so the abort would have cut a "
        "healthy generation and the detector is too aggressive for them.", "",
    ]
    for arm, blk in out.items():
        lines.extend([
            f"## `{arm}`", "",
            f"Abort-at-onset saves **{blk['saved_tokens']} of {blk['total_tokens']} tokens "
            f"({100.0 * blk['saved_frac']:.1f}%)**, about "
            f"{blk['saved_seconds']:.0f}s at {blk['tok_per_s']:.1f} tok/s. "
            f"Solves lost: {blk['would_lose_solves'] or 'none'}.", "",
            "| item | tokens | EOS | correct | loop onset | tokens saved | would lose solve |",
            "|---:|---:|---|---|---:|---:|---|",
        ])
        for i in blk["items"]:
            lines.append(
                f"| {i['idx']} | {i['tokens']} | {'yes' if i['eos'] else 'no'} | "
                f"{'yes' if i['correct'] else 'no'} | "
                f"{i['loop_onset'] if i['loop_onset'] is not None else '—'} | "
                f"{i['saved_tokens']} | {'YES' if i['would_lose_solve'] else 'no'} |")
        lines.append("")
    with open(os.path.join(args.out_dir, "LOOPS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return out


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
                    choices=["smoke", "collect", "views", "isolated", "isolated_frozen",
                             "budget", "loops", "compare", "report"])
    ap.add_argument("--budget_arms", default="real")
    ap.add_argument("--loop_ngram", type=int, default=8)
    ap.add_argument("--loop_window", type=int, default=256)
    ap.add_argument("--loop_thresh", type=float, default=0.8)
    ap.add_argument("--out_dir", default="artifacts/aime_localize")
    ap.add_argument("--tape_dir", default="",
                    help="local-disk dir for the big KV tapes (default out_dir/tapes)")
    ap.add_argument("--persist_dir", default="",
                    help="durable dir (network volume) mirrored after each arm")
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
    ap.add_argument("--method_tag", default="",
                    help="Suffix appended to row method names, leaving the cache "
                         "view untouched. Lets one arm be re-decoded and compared.")
    ap.add_argument("--compare_arms", default="",
                    help="Two method names to diff in --mode compare.")
    ap.add_argument("--decode_bs", type=int, default=1,
                    help="Judger decodes per generate() call. >1 multiplies throughput "
                         "but forfeits per-item latency; keep 1 for latency runs.")
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
    elif args.mode == "budget":
        run_budget(args)
    elif args.mode == "loops":
        run_loops(args)
    elif args.mode == "compare":
        run_compare(args)
    else:
        write_report(args)


if __name__ == "__main__":
    main()
