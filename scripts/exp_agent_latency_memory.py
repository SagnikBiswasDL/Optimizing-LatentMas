#!/usr/bin/env python3
"""Per-agent latency (LatentMAS vs compressed) + batch-size memory sweep.

Jiayi asks (2026-08-23):
  1. latency breakdown of each agent, latentmas vs our method
  2. larger batch-size: contribution on the memory

Protocol
--------
Upstream Planner/Critic/Refiner are built ONCE per item (same as LatentMAS).
We clone the cache, then decode the Judger twice:
  latentmas  = full relay
  ours       = evict (sink + key-norm top-k)  [optional: obf]
So P/C/R times MUST match; the only legal deltas are t_compress and t_judger.

Latency uses batch_size=1 + CUDA synchronize so bars are honest.
Memory sweep clones one real cache to batch B (full vs compressed) and runs a
short Judger decode, increasing B until OOM. That is the "batch reveals the
relay" plot — not an accuracy claim.

Example (pod):
  source /workspace/env_native.sh
  python scripts/exp_agent_latency_memory.py --smoke
  python scripts/exp_agent_latency_memory.py --mode both --task gsm8k --n 50 \\
      --k 40 --relay_budget 16 --judger_budget 768 \\
      --out_dir artifacts/exp_latency_mem/gsm8k_b16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import load_gsm8k, load_math, load_aime2024, load_aime_pooled  # noqa: E402
try:
    from data import load_aime2025
except ImportError:
    load_aime2025 = None
try:
    from data import load_gpqa_diamond
except ImportError:
    load_gpqa_diamond = None
try:
    from data import load_humanevalplus
except ImportError:
    load_humanevalplus = None
from methods import default_agents  # noqa: E402
from models import ModelWrapper  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.relay_compress import RelayCompressor, from_legacy, kv_mb, num_positions, to_legacy  # noqa: E402
from utils import (  # noqa: E402
    auto_device,
    extract_boxed_answer,
    extract_gsm8k_answer,
    normalize_answer,
    normalize_math_answer,
    set_seed,
)


def load_task(task: str, split: str):
    if task == "math":
        return list(load_math(split=split))
    if task == "aime2024":
        return list(load_aime2024(split="train"))
    if task == "aime2025":
        if load_aime2025 is None:
            raise RuntimeError("load_aime2025 missing")
        return list(load_aime2025(split="train"))
    if task == "aime_pooled":
        return list(load_aime_pooled())
    if task == "gpqa":
        if load_gpqa_diamond is None:
            raise RuntimeError("load_gpqa_diamond is not in this checkout's data.py")
        return list(load_gpqa_diamond(split="test"))
    if task == "humanevalplus":
        if load_humanevalplus is None:
            raise RuntimeError("load_humanevalplus missing")
        return list(load_humanevalplus(split="test"))
    if task == "livecodebench":
        return list(_load_livecodebench())
    return list(load_gsm8k(split=split))


def _load_livecodebench():
    """Best-effort LiveCodeBench lite. No unit-test execution in this harness."""
    from datasets import load_dataset
    last_err = None
    ds = None
    for kwargs in (
        dict(path="livecodebench/code_generation_lite", split="test"),
        dict(path="livecodebench/code_generation_lite", name="release_v5", split="test"),
    ):
        try:
            ds = load_dataset(**kwargs)
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load LiveCodeBench: {last_err}")
    out = []
    for item in ds:
        q = (item.get("question") or item.get("problem") or item.get("prompt") or "").strip()
        if not q:
            continue
        out.append({"question": q, "gold": "", "solution": "", "source": "livecodebench"})
    if not out:
        raise RuntimeError("LiveCodeBench loaded empty")
    return out


def make_ns(args):
    use_asc = bool(getattr(args, "asc_vector", None))
    ptask = "humanevalplus" if args.task == "livecodebench" else args.task
    return SimpleNamespace(
        model_name=args.model_name, task=ptask, prompt="sequential", think=False,
        latent_only=False, sequential_info_only=False, agents=None, use_vllm=False,
        device=args.device, device2="cuda:1", max_new_tokens=args.judger_budget,
        text_mas_context_length=-1,
        temperature=float(getattr(args, "temperature", 0.0)),
        top_p=float(getattr(args, "top_p", 1.0)), seed=args.seed,
        seal=use_asc, seal_vector=getattr(args, "asc_vector", None),
        seal_coef=float(getattr(args, "asc_coef", 40.0)),
        seal_layer=int(getattr(args, "asc_layer", -1)),
        seal_apply_to="last", seal_agents="judger",
        kvsteer=False, ces=False, capture_acts=None,
        planner_steps=None, critic_steps=None, refiner_steps=None,
        latent_steps=0, latent_space_realign=False,
    )


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return 0.0


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def deep_clone(past):
    if past is None:
        return None
    return from_legacy([(k.clone(), v.clone()) for (k, v) in to_legacy(past)], like=past)


def to_cpu(past):
    if past is None:
        return None
    return from_legacy([(k.detach().cpu().contiguous(), v.detach().cpu().contiguous())
                        for (k, v) in to_legacy(past)], like=past)


def to_dev(past, device):
    if past is None:
        return None
    return from_legacy([(k.to(device), v.to(device)) for (k, v) in to_legacy(past)], like=past)


def _pad_caches_left(caches, device):
    legacies = [to_legacy(c) for c in caches]
    B = len(legacies)
    n_layers = len(legacies[0])
    lens = [lg[0][0].shape[-2] for lg in legacies]
    Pmax = max(lens)
    past_mask = torch.zeros(B, Pmax, dtype=torch.long, device=device)
    for b, ln in enumerate(lens):
        past_mask[b, Pmax - ln:] = 1
    layers = []
    for li in range(n_layers):
        Ks, Vs = [], []
        for b in range(B):
            k, v = legacies[b][li]
            k = k.to(device)
            v = v.to(device)
            ln = k.shape[-2]
            if ln < Pmax:
                pk = torch.zeros(1, k.shape[1], Pmax - ln, k.shape[3], dtype=k.dtype, device=device)
                pv = torch.zeros(1, v.shape[1], Pmax - ln, v.shape[3], dtype=v.dtype, device=device)
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            Ks.append(k)
            Vs.append(v)
        layers.append((torch.cat(Ks, 0), torch.cat(Vs, 0)))
    return from_legacy(layers), past_mask, Pmax


def decode_batch(wrapper, judger_ids, judger_mask, caches, budget, asc_on=False,
                 temperature=0.0, top_p=1.0):
    device = wrapper.device
    jids = judger_ids.to(device)
    jmask = judger_mask.to(device)
    if all(c is None for c in caches):
        past, full_mask, cache_position = None, jmask, None
    else:
        past, past_mask, Pmax = _pad_caches_left(caches, device)
        full_mask = torch.cat([past_mask, jmask], dim=1)
        cache_position = torch.arange(Pmax, Pmax + jids.shape[1], device=device)
    seal = getattr(wrapper, "seal", None)
    used_asc = False
    if asc_on and seal is not None:
        seal.set_active_role("judger")
        if seal.has_effect_for("judger"):
            seal.enable()
            used_asc = True
    sample = float(temperature) > 0
    gen_kwargs = dict(
        input_ids=jids, attention_mask=full_mask, past_key_values=past,
        max_new_tokens=int(budget), do_sample=sample,
        pad_token_id=wrapper.tokenizer.pad_token_id,
        return_dict_in_generate=True, output_scores=False,
    )
    if sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
    if cache_position is not None:
        gen_kwargs["cache_position"] = cache_position
    try:
        out = wrapper.model.generate(**gen_kwargs)
    finally:
        if used_asc:
            seal.disable()
    seqs = out.sequences
    gen_start = jids.shape[1]
    eos_id = wrapper.tokenizer.eos_token_id
    pad_id = wrapper.tokenizer.pad_token_id
    texts, ntoks, eoss = [], [], []
    for i in range(seqs.shape[0]):
        gen = seqs[i, gen_start:]
        texts.append(wrapper.tokenizer.decode(gen, skip_special_tokens=True).strip())
        cnt, eos = 0, False
        for tok in gen.tolist():
            if eos_id is not None and tok == eos_id:
                cnt += 1
                eos = True
                break
            if pad_id is not None and pad_id != eos_id and tok == pad_id:
                break
            cnt += 1
        ntoks.append(cnt)
        eoss.append(eos)
    return texts, ntoks, eoss


def graded(text, gold, task):
    if task in ("humanevalplus", "livecodebench"):
        return False
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


def build_upstream_timed(wrapper, question, k, ns, agents):
    """Return (past, {role: seconds}) with CUDA sync around each agent."""
    past = None
    times: Dict[str, float] = {}
    for agent in agents:
        messages = build_agent_message_sequential_latent_mas(
            role=agent.role, question=question, context="", method="latent_mas", args=ns)
        _, ids, mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
        sync()
        t0 = time.perf_counter()
        past = wrapper.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=int(k),
            past_key_values=past, role=agent.role)
        sync()
        times[agent.role] = time.perf_counter() - t0
    times["upstream"] = sum(times.values())
    return past, times


def mean_ci(xs, seed=0):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    boots = [xs[rng.integers(0, len(xs), len(xs))].mean() for _ in range(2000)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(xs.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(xs))}


# --------------------------------------------------------------------------- #
# latency
# --------------------------------------------------------------------------- #
def run_latency(wrapper, items, args, ns, up_agents, compressor, kind: str):
    rows = []
    t_wall = time.time()
    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass

    for i, it in enumerate(items):
        q, gold = it["question"], it["gold"]
        past, up_t = build_upstream_timed(wrapper, q, args.k, ns, up_agents)
        full_pos, full_mb = num_positions(past), kv_mb(past)

        sync()
        tc0 = time.perf_counter()
        compressed, st = compressor.compress(past)
        sync()
        t_compress = time.perf_counter() - tc0

        jmsgs = [build_agent_message_sequential_latent_mas(
            role="judger", question=q, context="", method="latent_mas", args=ns)]
        _, jids, jmask, _ = wrapper.prepare_chat_batch(jmsgs, add_generation_prompt=True)

        # LatentMAS = full relay. Clone so decode growth cannot poison `past`.
        sync()
        td0 = time.perf_counter()
        texts_a, ntoks_a, eoss_a = decode_batch(
            wrapper, jids, jmask, [deep_clone(past)], args.judger_budget,
            temperature=args.temperature, top_p=args.top_p)
        sync()
        t_judger_full = time.perf_counter() - td0

        sync()
        td1 = time.perf_counter()
        texts_c, ntoks_c, eoss_c = decode_batch(
            wrapper, jids, jmask, [deep_clone(compressed)], args.judger_budget, asc_on=False,
            temperature=args.temperature, top_p=args.top_p)
        sync()
        t_judger_ours = time.perf_counter() - td1

        t_judger_asc = None
        ntoks_d = eoss_d = texts_d = None
        if getattr(args, "asc_vector", None):
            sync()
            td2 = time.perf_counter()
            texts_d, ntoks_d, eoss_d = decode_batch(
                wrapper, jids, jmask, [deep_clone(compressed)], args.judger_budget, asc_on=True,
                temperature=args.temperature, top_p=args.top_p)
            sync()
            t_judger_asc = time.perf_counter() - td2

        row_base = {
            "idx": it.get("idx", i),
            "planner_s": up_t.get("planner", 0.0),
            "critic_s": up_t.get("critic", 0.0),
            "refiner_s": up_t.get("refiner", 0.0),
            "upstream_s": up_t["upstream"],
            "compress_s": t_compress,
            "full_pos": int(full_pos),
            "full_mb": float(full_mb),
            "ours_pos": int(st.positions_out),
            "ours_mb": float(st.mb_out),
            "relay_ratio": float(st.ratio),
        }
        rows.append({
            **row_base,
            "method": "latentmas",
            "judger_s": t_judger_full,
            "tokens": int(ntoks_a[0]),
            "eos": bool(eoss_a[0]),
            "correct": bool(graded(texts_a[0], gold, args.task)),
            "e2e_s": up_t["upstream"] + t_judger_full,
            "tok_per_s": (ntoks_a[0] / t_judger_full) if t_judger_full > 0 else 0.0,
        })
        rows.append({
            **row_base,
            "method": "ours",
            "kind": kind,
            "judger_s": t_judger_ours,
            "tokens": int(ntoks_c[0]),
            "eos": bool(eoss_c[0]),
            "correct": bool(graded(texts_c[0], gold, args.task)),
            "e2e_s": up_t["upstream"] + t_compress + t_judger_ours,
            "tok_per_s": (ntoks_c[0] / t_judger_ours) if t_judger_ours > 0 else 0.0,
        })
        if t_judger_asc is not None:
            rows.append({
                **row_base,
                "method": "ours_asc",
                "kind": kind,
                "judger_s": t_judger_asc,
                "tokens": int(ntoks_d[0]),
                "eos": bool(eoss_d[0]),
                "correct": bool(graded(texts_d[0], gold, args.task)),
                "e2e_s": up_t["upstream"] + t_compress + t_judger_asc,
                "tok_per_s": (ntoks_d[0] / t_judger_asc) if t_judger_asc > 0 else 0.0,
            })
        del past, compressed

        k10_note = ""
        if getattr(args, "with_k10", False):
            past10, up10 = build_upstream_timed(wrapper, q, 10, ns, up_agents)
            sync()
            td3 = time.perf_counter()
            texts_k, ntoks_k, eoss_k = decode_batch(
                wrapper, jids, jmask, [deep_clone(past10)], args.judger_budget,
                temperature=args.temperature, top_p=args.top_p)
            sync()
            t_judger_k10 = time.perf_counter() - td3
            rows.append({
                "idx": it.get("idx", i),
                "method": "k10",
                "planner_s": up10.get("planner", 0.0),
                "critic_s": up10.get("critic", 0.0),
                "refiner_s": up10.get("refiner", 0.0),
                "upstream_s": up10["upstream"],
                "compress_s": 0.0,
                "full_pos": int(num_positions(past10)),
                "full_mb": float(kv_mb(past10)),
                "ours_pos": int(num_positions(past10)),
                "ours_mb": float(kv_mb(past10)),
                "relay_ratio": 1.0,
                "judger_s": t_judger_k10,
                "tokens": int(ntoks_k[0]),
                "eos": bool(eoss_k[0]),
                "correct": bool(graded(texts_k[0], gold, args.task)),
                "e2e_s": up10["upstream"] + t_judger_k10,
                "tok_per_s": (ntoks_k[0] / t_judger_k10) if t_judger_k10 > 0 else 0.0,
            })
            k10_note = f" J_k10={t_judger_k10:.2f}s tok_k10={ntoks_k[0]} acc_k10={int(graded(texts_k[0], gold, args.task))}"
            del past10

        extra = ""
        if t_judger_asc is not None:
            extra = f" J_asc={t_judger_asc:.2f}s tok_asc={ntoks_d[0]}"
        print(
            f"[latency] {i+1}/{len(items)} "
            f"P={up_t.get('planner', 0):.2f}s C={up_t.get('critic', 0):.2f}s "
            f"R={up_t.get('refiner', 0):.2f}s "
            f"J_full={t_judger_full:.2f}s J_ours={t_judger_ours:.2f}s{extra}{k10_note} "
            f"tok {ntoks_a[0]}/{ntoks_c[0]} "
            f"acc {int(graded(texts_a[0], gold, args.task))}/{int(graded(texts_c[0], gold, args.task))} "
            f"elapsed={time.time()-t_wall:.0f}s",
            flush=True,
        )
    return rows


def summarize_latency(rows, seed):
    methods = ["latentmas", "ours", "ours_asc", "k10"]
    out = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        keys = ["planner_s", "critic_s", "refiner_s", "upstream_s", "compress_s",
                "judger_s", "e2e_s", "tokens", "full_mb", "ours_mb", "tok_per_s"]
        blk = {"n": len(rs), "acc": float(np.mean([r["correct"] for r in rs]))}
        for k in keys:
            if k in rs[0]:
                blk[k] = mean_ci([r[k] for r in rs], seed=seed)
        out[m] = blk
    by = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if rs:
            by[m] = {r["idx"]: r for r in rs}
    paired = {}

    def _pair(src, dst, label):
        if src not in by or dst not in by:
            return
        common = sorted(set(by[src]) & set(by[dst]))
        for k in ("judger_s", "e2e_s", "tokens", "correct"):
            diffs = [by[dst][i][k] - by[src][i][k] for i in common]
            paired[f"{label}_{k}"] = mean_ci(diffs, seed=seed)

    _pair("latentmas", "ours", "ours_minus_latentmas")
    _pair("latentmas", "ours_asc", "ours_asc_minus_latentmas")
    _pair("ours", "ours_asc", "asc_minus_ours")
    _pair("latentmas", "k10", "k10_minus_latentmas")
    _pair("k10", "ours", "ours_minus_k10")
    out["paired"] = paired
    out["n_paired"] = len(set(by.get("latentmas", {})) & set(by.get("ours", {})))
    return out


# --------------------------------------------------------------------------- #
# memory vs batch size
# --------------------------------------------------------------------------- #
def run_memory(wrapper, sample_item, args, ns, up_agents, compressor, kind: str):
    """Clone one real cache to batch B; decode a few tokens; climb until OOM."""
    print("[memory] building one donor upstream cache...", flush=True)
    past, up_t = build_upstream_timed(wrapper, sample_item["question"], args.k, ns, up_agents)
    compressed, st = compressor.compress(past)
    full_cpu = to_cpu(past)
    ours_cpu = to_cpu(compressed)
    del past, compressed
    torch.cuda.empty_cache()

    jmsgs = [build_agent_message_sequential_latent_mas(
        role="judger", question=sample_item["question"], context="",
        method="latent_mas", args=ns)]
    _, jids1, jmask1, _ = wrapper.prepare_chat_batch(jmsgs, add_generation_prompt=True)

    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass

    reset_peak()
    _ = torch.zeros(1, device=wrapper.device)
    sync()
    weights_mb = peak_mb()

    batches = [int(x) for x in args.batch_grid.split(",") if int(x) > 0]
    results = []
    decode_tok = int(args.memory_decode_tokens)
    for kind_name, cpu_cache, mb_each, pos in (
        ("latentmas", full_cpu, kv_mb(full_cpu), num_positions(full_cpu)),
        ("ours", ours_cpu, kv_mb(ours_cpu), num_positions(ours_cpu)),
    ):
        oom_at = None
        for B in batches:
            if oom_at is not None:
                results.append({
                    "method": kind_name, "batch_size": B, "ok": False, "oom": True,
                    "skipped": True, "relay_mb_each": mb_each, "relay_pos": pos,
                })
                continue
            torch.cuda.empty_cache()
            reset_peak()
            try:
                caches = [to_dev(deep_clone(cpu_cache), wrapper.device) for _ in range(B)]
                jids = jids1.repeat(B, 1)
                jmask = jmask1.repeat(B, 1)
                sync()
                t0 = time.perf_counter()
                decode_batch(wrapper, jids, jmask, caches, decode_tok)
                sync()
                dt = time.perf_counter() - t0
                pmb = peak_mb()
                rec = {
                    "method": kind_name, "batch_size": B, "ok": True, "oom": False,
                    "peak_mb": float(pmb), "weights_mb": float(weights_mb),
                    "relay_mb_each": float(mb_each), "relay_pos": int(pos),
                    "relay_mb_batch": float(mb_each * B),
                    "decode_s": float(dt), "decode_tokens": decode_tok,
                    "items_per_s": float(B / dt) if dt > 0 else 0.0,
                }
                results.append(rec)
                print(
                    f"[memory] {kind_name} B={B} peak={pmb/1024:.2f}GB "
                    f"relay_batch={mb_each*B:.1f}MB decode={dt:.2f}s "
                    f"({B/dt:.2f} it/s)",
                    flush=True,
                )
                del caches
            except torch.cuda.OutOfMemoryError:
                oom_at = B
                torch.cuda.empty_cache()
                results.append({
                    "method": kind_name, "batch_size": B, "ok": False, "oom": True,
                    "peak_mb": None, "relay_mb_each": float(mb_each),
                    "relay_pos": int(pos), "relay_mb_batch": float(mb_each * B),
                })
                print(f"[memory] {kind_name} B={B} OOM", flush=True)
            except Exception as e:
                torch.cuda.empty_cache()
                results.append({
                    "method": kind_name, "batch_size": B, "ok": False, "oom": False,
                    "error": repr(e), "relay_mb_each": float(mb_each),
                    "relay_pos": int(pos),
                })
                print(f"[memory] {kind_name} B={B} ERROR {e!r}", flush=True)
                break
    return {
        "upstream_s_donor": up_t,
        "compressor": compressor.summary(),
        "kind": kind,
        "weights_mb": weights_mb,
        "full_mb": kv_mb(full_cpu),
        "ours_mb": kv_mb(ours_cpu),
        "rows": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--task", default="gsm8k",
                    choices=["gsm8k", "math", "aime2024", "aime2025", "aime_pooled",
                             "gpqa", "humanevalplus", "livecodebench"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--n", type=int, default=0, help="0 -> 2 smoke / 50 full")
    ap.add_argument("--judger_budget", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Judger sampling temp; 0 = greedy. Paper uses 0.6")
    ap.add_argument("--top_p", type=float, default=1.0,
                    help="Judger top-p when temperature>0. Paper uses 0.95")
    ap.add_argument("--relay_budget", type=int, default=16)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--importance", default="key_norm",
                    choices=["key_norm", "value_norm", "recency"])
    ap.add_argument("--compress_mode", default="evict", choices=["evict", "obf"])
    ap.add_argument("--mode", default="both", choices=["latency", "memory", "both"])
    ap.add_argument("--batch_grid", default="1,2,4,8,16,24,32,48,64")
    ap.add_argument("--memory_decode_tokens", type=int, default=16,
                    help="short decode for the memory/OOM sweep (not accuracy)")
    ap.add_argument("--with_k10", action="store_true",
                    help="Also decode a K=10 full-relay arm (reviewer: K=10 vs K=40+top-k)")
    ap.add_argument("--asc_vector", default=None,
                    help="Judger-only SEAL vector; adds an ours_asc decode arm")
    ap.add_argument("--asc_coef", type=float, default=40.0)
    ap.add_argument("--asc_layer", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/exp_latency_mem/run")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(2)
    if args.n <= 0:
        args.n = 2 if args.smoke else 50
    if args.smoke:
        args.k = min(args.k, 4)
        args.judger_budget = min(args.judger_budget, 64)
        args.memory_decode_tokens = min(args.memory_decode_tokens, 8)
        args.batch_grid = "1,2,4"
        print(f"[smoke] k={args.k} n={args.n} judger_budget={args.judger_budget}", flush=True)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    items = load_task(args.task, args.split)[: args.n]
    for i, it in enumerate(items):
        it["idx"] = i
    ns = make_ns(args)
    print(f"[load] {args.model_name} task={args.task} n={len(items)} k={args.k} "
          f"temp={args.temperature} top_p={args.top_p}", flush=True)
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    if getattr(wrapper, "seal", None) is not None:
        wrapper.seal.disable()
        print(f"[asc] loaded {args.asc_vector} coef={args.asc_coef} (Judger only)", flush=True)
    up_agents = [a for a in default_agents() if a.role != "judger"]
    compressor = RelayCompressor(
        mode=args.compress_mode, budget=args.relay_budget,
        sink=args.sink, rank=args.rank, importance=args.importance,
    )
    print(f"[compress] {compressor.summary()}", flush=True)

    report: Dict = {
        "config": vars(args),
        "n": len(items),
        "compressor": compressor.summary(),
        "fork_note": "LatentMAS sequential HF path, K latent steps/agent. "
                     "Judger uses temperature/top_p from CLI (0/1 = greedy). "
                     "ours = same upstream + RelayCompressor before Judger.",
    }

    if args.mode in ("memory", "both"):
        report["memory"] = run_memory(
            wrapper, items[0], args, ns, up_agents, compressor, args.compress_mode)

    if args.mode in ("latency", "both"):
        rows = run_latency(wrapper, items, args, ns, up_agents, compressor, args.compress_mode)
        report["latency"] = summarize_latency(rows, args.seed)
        with open(os.path.join(args.out_dir, "latency_rows.json"), "w") as f:
            json.dump(rows, f)
        # print table
        lat = report["latency"]
        print("\n=== PER-AGENT LATENCY (mean) ===", flush=True)
        print(f"{'method':<12}{'P':>8}{'C':>8}{'R':>8}{'comp':>8}{'Judger':>8}{'e2e':>8}{'tok':>8}{'acc':>8}",
              flush=True)
        for m in ("latentmas", "ours", "ours_asc", "k10"):
            if m not in lat:
                continue
            a = lat[m]
            print(
                f"{m:<12}{a['planner_s']['mean']:8.2f}{a['critic_s']['mean']:8.2f}"
                f"{a['refiner_s']['mean']:8.2f}{a['compress_s']['mean']:8.3f}"
                f"{a['judger_s']['mean']:8.2f}{a['e2e_s']['mean']:8.2f}"
                f"{a['tokens']['mean']:8.1f}{a['acc']:8.3f}",
                flush=True,
            )
        print("\n=== PAIRED (ours - latentmas) ===", flush=True)
        for k, d in lat["paired"].items():
            print(f"  {k:<32} {d['mean']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] n={d['n']}",
                  flush=True)

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] {os.path.join(args.out_dir, 'report.json')}", flush=True)
    print("EXP_LATENCY_MEM_DONE", flush=True)


if __name__ == "__main__":
    main()
