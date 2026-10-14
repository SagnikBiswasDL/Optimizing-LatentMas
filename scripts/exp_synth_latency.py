#!/usr/bin/env python3
"""LatentMAS vs a type-level average KV cache at the paper's generate_bs.

  LatentMAS  = batched P+C+R + Judger
  ours       = frozen type-level cache + Judger (upstream = 0)
  none       = Judger, no prefix

Default scaffold is mean_aligned: right-align train-donor relays and average
over the donor axis (sequence kept). --scaffold gaussian is the i.i.d. draw
that worked on MedQA and was weak on GSM8K.

Precompute can be large (--stat_n, --filter_correct). It is not in per-query e2e.
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
    load_aime2024,
    load_aime2025,
    load_gsm8k,
    load_math,
    load_medqa,
    math_level_int,
)
from methods import default_agents  # noqa: E402
from models import ModelWrapper  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.cache_bank import (  # noqa: E402
    _slug,
    from_legacy,
    kv_mb,
    mean_aligned,
    mean_restore_latents,
    medoid_latent_index,
    num_positions,
    pool_stats,
    prototype_donor,
    sample_synth,
    to_legacy,
)
from utils import (  # noqa: E402
    auto_device,
    extract_boxed_answer,
    extract_gsm8k_answer,
    normalize_answer,
    normalize_math_answer,
    set_seed,
)


def load_task(task: str, split: str):
    if task == "medqa":
        return list(load_medqa(split=split))
    if task == "aime2024":
        return list(load_aime2024(split="train"))
    if task == "aime2025":
        return list(load_aime2025(split="train"))
    if task == "math":
        return list(load_math(split=split))
    return list(load_gsm8k(split=split))


def make_ns(args):
    return SimpleNamespace(
        model_name=args.model_name, task=args.task, prompt="sequential", think=False,
        latent_only=False, sequential_info_only=False, agents=None, use_vllm=False,
        device=args.device, device2="cuda:1", max_new_tokens=args.judger_budget,
        text_mas_context_length=-1,
        temperature=float(args.temperature),
        top_p=float(args.top_p), seed=args.seed,
        seal=False, kvsteer=False, ces=False, capture_acts=None,
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
    return from_legacy([(k.clone(), v.clone()) for (k, v) in to_legacy(past)])


def to_cpu(past):
    if past is None:
        return None
    return from_legacy([(k.detach().cpu().contiguous(), v.detach().cpu().contiguous())
                        for (k, v) in to_legacy(past)])


def to_dev(past, device):
    if past is None:
        return None
    return from_legacy([(k.to(device), v.to(device)) for (k, v) in to_legacy(past)])


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


def decode_batch(wrapper, judger_ids, judger_mask, caches, budget, temperature=0.0, top_p=1.0):
    device = wrapper.device
    jids = judger_ids.to(device)
    jmask = judger_mask.to(device)
    if all(c is None for c in caches):
        past, full_mask, cache_position = None, jmask, None
    else:
        past, past_mask, Pmax = _pad_caches_left(caches, device)
        full_mask = torch.cat([past_mask, jmask], dim=1)
        cache_position = torch.arange(Pmax, Pmax + jids.shape[1], device=device)
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
    out = wrapper.model.generate(**gen_kwargs)
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
    boxed = extract_boxed_answer(text) or extract_gsm8k_answer(text) or text
    if task == "math":
        pred = normalize_math_answer(boxed)
        g = normalize_math_answer(gold)
        return bool(pred) and bool(g) and pred == g
    pred = normalize_answer(boxed)
    g = normalize_answer(gold)
    if task == "medqa":
        if pred and g and (pred == g or pred[:1] == g[:1]):
            return True
        return False
    # Same as LatentMASMethod.run_batch: AIME golds are zero-padded ("025").
    if task in ("aime2024", "aime2025"):
        try:
            return int(pred) == int(g)
        except (TypeError, ValueError):
            return bool(pred) and bool(g) and pred == g
    return bool(pred) and bool(g) and pred == g


def type_question(task: str, subject: str = "") -> str:
    """Prefill text for Mean-Replay / medoid-replay: no specific test item."""
    task = (task or "").strip().lower()
    sub = (subject or "").replace("_", " ").strip()
    if task in ("gsm8k",) and not sub:
        return (
            "Solve a grade-school math word problem using elementary arithmetic. "
            "Show brief steps. Put the final numeric answer in \\boxed{}."
        )
    if sub:
        return (
            f"Solve a contest mathematics problem in {sub}. Reason carefully. "
            "Put the final answer in \\boxed{}."
        )
    if task in ("aime2024", "aime2025", "math"):
        return (
            "Solve a contest mathematics problem. Reason carefully. "
            "Put the final answer in \\boxed{}."
        )
    return (
        "Solve a multiple-choice question. Select A, B, C, or D. "
        "Put the letter in \\boxed{}."
    )


def subject_slug(item: Dict) -> str:
    return _slug(item.get("subject") or item.get("type") or "")


@torch.no_grad()
def last_token_embed(wrapper, texts: List[str], bs: int = 8) -> torch.Tensor:
    """L2-normalized last-token hidden states for subject assignment."""
    tok = wrapper.tokenizer
    chunks = []
    for start in range(0, len(texts), bs):
        batch = texts[start : start + bs]
        enc = tok(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(wrapper.device) for k, v in enc.items()}
        out = wrapper.model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]
        idx = enc["attention_mask"].sum(1) - 1
        b = torch.arange(h.size(0), device=h.device)
        vec = h[b, idx].float()
        vec = vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        chunks.append(vec.cpu())
        del out
        torch.cuda.empty_cache()
    return torch.cat(chunks, 0) if chunks else torch.zeros(0, 1)


def _squeeze_latents(emb: torch.Tensor) -> torch.Tensor:
    """[K, B, D] -> [K, D] for a single-item donor."""
    if emb.dim() == 3:
        return emb[:, 0, :].contiguous()
    return emb.contiguous()


def build_upstream_timed(
    wrapper, questions: List[str], k, ns, agents,
    collect_latents: bool = False,
    forced_by_role: Optional[Dict[str, torch.Tensor]] = None,
    out_latents: Optional[Dict[str, torch.Tensor]] = None,
):
    """Batched silent-agent pass. times[*] are wall-clock for the whole batch."""
    past = None
    times: Dict[str, float] = {}
    peaks: Dict[str, float] = {}
    for agent in agents:
        messages = [
            build_agent_message_sequential_latent_mas(
                role=agent.role, question=q, context="", method="latent_mas", args=ns)
            for q in questions
        ]
        _, ids, mask, _ = wrapper.prepare_chat_batch(messages, add_generation_prompt=True)
        forced = None
        if forced_by_role is not None:
            forced = forced_by_role[agent.role]
        reset_peak()
        sync()
        t0 = time.perf_counter()
        past = wrapper.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=int(k),
            past_key_values=past, role=agent.role,
            collect_latents=collect_latents, forced_latents=forced)
        sync()
        times[agent.role] = time.perf_counter() - t0
        peaks[agent.role] = peak_mb()
        if collect_latents and out_latents is not None:
            emb = getattr(wrapper, "last_latent_embeds", None)
            if emb is None:
                raise RuntimeError(f"collect_latents: no embeds for role={agent.role}")
            out_latents[agent.role] = _squeeze_latents(emb)
    times["upstream"] = sum(times[a.role] for a in agents)
    return past, times, peaks


def mean_ci(xs, seed=0):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    boots = [xs[rng.integers(0, len(xs), len(xs))].mean() for _ in range(2000)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(xs.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(xs))}


def score_frozen(wrapper, cache_cpu, items, args, ns, task: str):
    """Judger-only accuracy of one frozen cache on a held-out train slice."""
    if not items:
        return 0.0, 0.0
    bs = max(1, int(args.generate_bs))
    n_ok = n_tok = n = 0
    device = wrapper.device
    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass
    for start in range(0, len(items), bs):
        batch = items[start : start + bs]
        jmsgs = [
            build_agent_message_sequential_latent_mas(
                role="judger", question=it["question"], context="",
                method="latent_mas", args=ns)
            for it in batch
        ]
        _, jids, jmask, _ = wrapper.prepare_chat_batch(jmsgs, add_generation_prompt=True)
        caches = [to_dev(deep_clone(cache_cpu), device) for _ in batch]
        texts, ntoks, _ = decode_batch(
            wrapper, jids, jmask, caches,
            args.judger_budget, temperature=0.0, top_p=1.0)
        for text, nt, it in zip(texts, ntoks, batch):
            n += 1
            n_tok += int(nt)
            n_ok += int(graded(text, it["gold"], task))
        del caches
        torch.cuda.empty_cache()
    return n_ok / max(n, 1), n_tok / max(n, 1)


def past_from_legacy(lg, dtype):
    cpu = [(k.detach().cpu().contiguous(), v.detach().cpu().contiguous()) for (k, v) in lg]
    return from_legacy([(k.to(dtype=dtype), v.to(dtype=dtype)) for (k, v) in cpu])


def precompute_synth(wrapper, train_pool, args, ns, up_agents):
    """Offline: donor relays → one frozen type-level cache.

    Math lock: ``donor`` = one *real* train relay, reused for every test item.
    That is what cache-swap actually tested. Do not average KV tensors.
    ``heldout`` spends precompute to pick the donor whose frozen cache
    scores highest on a disjoint train slice.
    """
    kind = str(getattr(args, "scaffold", "donor"))
    want = int(args.stat_n)
    print(
        f"[precompute] pool={len(train_pool)} want={want} k={args.k} "
        f"scaffold={kind} select={getattr(args, 'donor_select', 'first')} "
        f"filter_correct={bool(args.filter_correct)}",
        flush=True,
    )
    t_all0 = time.perf_counter()
    legacies = []
    donor_latents: Dict[str, List] = {a.role: [] for a in up_agents}
    donor_rows = []
    n_tried = 0
    want_latents = kind == "replay"
    for i, it in enumerate(train_pool):
        if len(legacies) >= want:
            break
        n_tried += 1
        got_latents: Dict[str, torch.Tensor] = {}
        past, times, peaks = build_upstream_timed(
            wrapper, [it["question"]], args.k, ns, up_agents,
            collect_latents=want_latents,
            out_latents=got_latents if want_latents else None)
        rec = {
            "idx": i,
            "planner_s": times.get("planner", 0.0),
            "critic_s": times.get("critic", 0.0),
            "refiner_s": times.get("refiner", 0.0),
            "upstream_s": times["upstream"],
            "pos": int(num_positions(past)),
            "mb": float(kv_mb(past)),
            "kept": True,
            "train_correct": None,
            "peak_mb_planner": peaks.get("planner", 0.0),
            "peak_mb_critic": peaks.get("critic", 0.0),
            "peak_mb_refiner": peaks.get("refiner", 0.0),
        }
        keep = True
        if args.filter_correct:
            jmsg = build_agent_message_sequential_latent_mas(
                role="judger", question=it["question"], context="",
                method="latent_mas", args=ns)
            _, jids, jmask, _ = wrapper.prepare_chat_batch(
                [jmsg], add_generation_prompt=True)
            texts, _, _ = decode_batch(
                wrapper, jids, jmask, [deep_clone(past)],
                args.judger_budget, temperature=0.0, top_p=1.0)
            ok = graded(texts[0], it["gold"], args.donor_task or args.task)
            rec["train_correct"] = bool(ok)
            keep = bool(ok)
            rec["kept"] = keep
        donor_rows.append(rec)
        print(
            f"[precompute] donor {n_tried}/{len(train_pool)} kept={len(legacies)+int(keep)}/{want} "
            f"P={rec['planner_s']:.2f}s C={rec['critic_s']:.2f}s R={rec['refiner_s']:.2f}s "
            f"pos={rec['pos']} {rec['mb']:.1f}MB keep={keep}",
            flush=True,
        )
        if keep:
            legacies.append(to_legacy(to_cpu(past)))
            if want_latents:
                for role, emb in got_latents.items():
                    donor_latents[role].append(emb.cpu())
        del past
        torch.cuda.empty_cache()

    if not legacies:
        raise RuntimeError("precompute: no donor caches kept (filter_correct emptied the pool)")

    dtype = next(wrapper.model.parameters()).dtype
    proto_i, proto_d = None, None
    select_scores = []
    chosen = 0
    sync()
    ts0 = time.perf_counter()
    replay_norms = None
    type_q = None
    if kind == "gaussian":
        stats, L = pool_stats(legacies)
        synth = sample_synth(stats, L, dtype=dtype, device="cpu", seed=args.seed)
    elif kind == "replay":
        forced_by_role = {}
        replay_norms = {}
        for role, steps in donor_latents.items():
            if not steps:
                raise RuntimeError(f"replay: no latent embeddings for role={role}")
            mean_emb = mean_restore_latents(steps)
            forced_by_role[role] = mean_emb
            replay_norms[role] = {
                "k": int(mean_emb.shape[0]),
                "d": int(mean_emb.shape[1]),
                "mean_l2": float(mean_emb.norm(dim=-1).mean()),
                "n_donors": len(steps),
            }
            print(
                f"[replay] {role} mean-restore K={mean_emb.shape[0]} D={mean_emb.shape[1]} "
                f"l2={replay_norms[role]['mean_l2']:.3f} n={len(steps)}",
                flush=True,
            )
        type_q = type_question(args.donor_task or args.task)
        print(f"[replay] type prefill: {type_q[:80]}...", flush=True)
        past_r, _, _ = build_upstream_timed(
            wrapper, [type_q], args.k, ns, up_agents,
            forced_by_role=forced_by_role)
        synth = to_cpu(past_r)
        del past_r
        torch.cuda.empty_cache()
    elif kind == "mean_aligned":
        synth = mean_aligned(legacies, L=0, dtype=dtype, device="cpu")
    elif kind == "prototype":
        synth, proto_i, proto_d = prototype_donor(
            legacies, L=0, dtype=dtype, device="cpu")
        chosen = int(proto_i)
    else:
        # donor: one real cache, as-is. This is the swap-faithful object.
        sel = str(getattr(args, "donor_select", "heldout"))
        if sel == "random":
            chosen = int(np.random.default_rng(args.seed).integers(0, len(legacies)))
        elif sel == "medoid":
            _, proto_i, proto_d = prototype_donor(
                legacies, L=0, dtype=dtype, device="cpu")
            chosen = int(proto_i)
        elif sel == "heldout":
            select_n = int(getattr(args, "select_n", 0) or 0)
            select_items = train_pool[n_tried : n_tried + select_n]
            if not select_items:
                print("[precompute] heldout: no leftover train items, using first donor", flush=True)
                chosen = 0
            else:
                print(
                    f"[precompute] heldout scoring {len(legacies)} donors on "
                    f"{len(select_items)} train items",
                    flush=True,
                )
                best = None
                for i, lg in enumerate(legacies):
                    cache = past_from_legacy(lg, dtype)
                    acc, tok = score_frozen(
                        wrapper, cache, select_items, args, ns,
                        args.donor_task or args.task)
                    select_scores.append({"i": i, "acc": acc, "tokens": tok, "pos": int(lg[0][0].shape[-2])})
                    print(f"[select] donor {i} acc={acc:.3f} tok={tok:.1f}", flush=True)
                    key = (acc, -tok)
                    if best is None or key > best[0]:
                        best = (key, i)
                chosen = int(best[1])
        else:
            chosen = 0
        synth = past_from_legacy(legacies[chosen], dtype)
        proto_i = chosen
    sync()
    t_build = time.perf_counter() - ts0
    t_precompute = time.perf_counter() - t_all0
    blk = {
        "n_donors": len(legacies),
        "n_tried": n_tried,
        "k": int(args.k),
        "scaffold": kind,
        "donor_select": str(getattr(args, "donor_select", "")),
        "chosen_idx": int(chosen) if kind in ("donor", "prototype") else None,
        "filter_correct": bool(args.filter_correct),
        "synth_len": int(num_positions(synth)),
        "synth_mb": float(kv_mb(synth)),
        "prototype_idx": proto_i,
        "prototype_l2": proto_d,
        "select_scores": select_scores,
        "replay_type_question": type_q,
        "replay_norms": replay_norms,
        "donor_upstream_s": mean_ci([r["upstream_s"] for r in donor_rows if r.get("kept")], seed=args.seed),
        "t_stats_s": float(t_build),
        "t_sample_s": float(t_build),
        "t_precompute_s": float(t_precompute),
        "donors": donor_rows,
    }
    print(
        f"[precompute] done in {t_precompute:.1f}s  kind={kind} n={len(legacies)} "
        f"chosen={blk['chosen_idx']} pos={blk['synth_len']} {blk['synth_mb']:.1f}MB",
        flush=True,
    )
    return synth, blk


def precompute_medoid_bank(wrapper, train_pool, args, ns, up_agents):
    """AIME recipe: one replayed prefix per subject from a latent-space medoid.

    Donors are not averaged. The chosen tape is one real trajectory; the model
    writes KV on a type prompt with no test item.
    """
    want = int(args.stat_n)
    min_n = int(getattr(args, "min_subject_donors", 3) or 3)
    by = defaultdict(list)
    for it in train_pool:
        key = subject_slug(it)
        by[key if key != "unknown" else "math"].append(it)
    roles = [a.role for a in up_agents]
    print(
        f"[medoid] subjects={len(by)} want_per={want} min={min_n} k={args.k} "
        + ", ".join(f"{k}:{len(v)}" for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))),
        flush=True,
    )
    t_all0 = time.perf_counter()
    caches, centroids = {}, {}
    meta_subjects, donor_rows = [], []
    n_tried = 0
    for subj, pool in sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        take = pool[:want]
        if len(take) < min_n and len(by) > 1:
            print(f"[medoid] skip {subj}: {len(take)} donors < min {min_n}", flush=True)
            continue
        print(f"[medoid] subject={subj} using {len(take)}/{len(pool)}", flush=True)
        latents = {r: [] for r in roles}
        for j, it in enumerate(take):
            n_tried += 1
            got: Dict[str, torch.Tensor] = {}
            past, times, peaks = build_upstream_timed(
                wrapper, [it["question"]], args.k, ns, up_agents,
                collect_latents=True, out_latents=got)
            rec = {
                "subject": subj,
                "idx": j,
                "planner_s": times.get("planner", 0.0),
                "critic_s": times.get("critic", 0.0),
                "refiner_s": times.get("refiner", 0.0),
                "upstream_s": times["upstream"],
                "pos": int(num_positions(past)),
                "mb": float(kv_mb(past)),
                "kept": True,
                "train_correct": None,
                "peak_mb_planner": peaks.get("planner", 0.0),
                "peak_mb_critic": peaks.get("critic", 0.0),
                "peak_mb_refiner": peaks.get("refiner", 0.0),
            }
            donor_rows.append(rec)
            print(
                f"[precompute] {subj} donor {j+1}/{len(take)} "
                f"P={rec['planner_s']:.2f}s C={rec['critic_s']:.2f}s R={rec['refiner_s']:.2f}s "
                f"pos={rec['pos']}",
                flush=True,
            )
            for role in roles:
                latents[role].append(got[role].cpu())
            del past
            torch.cuda.empty_cache()
        qembs = last_token_embed(wrapper, [it["question"] for it in take])
        c = qembs.mean(0)
        centroids[subj] = (c / c.norm().clamp_min(1e-6)).cpu()
        mi, md = medoid_latent_index(latents, roles)
        forced = {role: latents[role][mi] for role in roles}
        type_q = type_question(args.donor_task or args.task, subject=subj)
        print(
            f"[medoid] {subj} pick={mi} dist={md:.3f} prefill={type_q[:70]}...",
            flush=True,
        )
        past_r, _, _ = build_upstream_timed(
            wrapper, [type_q], args.k, ns, up_agents, forced_by_role=forced)
        caches[subj] = to_cpu(past_r)
        meta_subjects.append({
            "subject": subj,
            "n_pool": len(pool),
            "n_donors": len(take),
            "medoid_idx": int(mi),
            "medoid_dist": float(md),
            "pos": int(num_positions(caches[subj])),
            "mb": float(kv_mb(caches[subj])),
            "type_question": type_q,
        })
        del past_r
        torch.cuda.empty_cache()
    if not caches:
        raise RuntimeError("medoid bank: no subjects with enough donors")
    default = max(meta_subjects, key=lambda s: s["n_donors"])["subject"]
    t_precompute = time.perf_counter() - t_all0
    blk = {
        "n_donors": sum(s["n_donors"] for s in meta_subjects),
        "n_tried": n_tried,
        "k": int(args.k),
        "scaffold": "medoid_replay",
        "donor_select": "medoid_latent",
        "chosen_idx": None,
        "filter_correct": False,
        "synth_len": int(np.mean([s["pos"] for s in meta_subjects])),
        "synth_mb": float(np.mean([s["mb"] for s in meta_subjects])),
        "prototype_idx": None,
        "prototype_l2": None,
        "select_scores": [],
        "replay_type_question": None,
        "replay_norms": None,
        "subjects": meta_subjects,
        "default_subject": default,
        "donor_upstream_s": mean_ci(
            [r["upstream_s"] for r in donor_rows if r.get("kept")], seed=args.seed),
        "t_stats_s": float(t_precompute),
        "t_sample_s": float(t_precompute),
        "t_precompute_s": float(t_precompute),
        "donors": donor_rows,
    }
    print(
        f"[precompute] medoid bank done in {t_precompute:.1f}s  "
        f"subjects={list(caches)} default={default}",
        flush=True,
    )
    return {"caches": caches, "centroids": centroids, "default": default}, blk


def assign_prefix_keys(wrapper, items, bank):
    """Map each test item to a bank key: labeled subject, else nearest centroid."""
    keys = list(bank["caches"])
    cents = torch.stack([bank["centroids"][k] for k in keys], 0)
    need = []
    for it in items:
        labeled = subject_slug(it)
        if labeled in bank["caches"]:
            it["_prefix_key"] = labeled
            it["_prefix_sim"] = 1.0
        else:
            need.append(it)
    if need:
        embs = last_token_embed(wrapper, [it["question"] for it in need])
        sim = embs @ cents.T
        for i, it in enumerate(need):
            j = int(sim[i].argmax().item())
            it["_prefix_key"] = keys[j]
            it["_prefix_sim"] = float(sim[i, j].item())
    counts = defaultdict(int)
    for it in items:
        counts[it["_prefix_key"]] += 1
    print(
        "[assign] " + ", ".join(f"{k}:{counts[k]}" for k in keys) +
        f"  unlabeled_nn={len(need)}",
        flush=True,
    )


def _split_past(past, B: int):
    """Slice a batched DynamicCache into B single-item caches (CPU)."""
    if past is None:
        return [None] * B
    leg = to_legacy(past)
    out = []
    for b in range(B):
        layers = [(k[b : b + 1].contiguous(), v[b : b + 1].contiguous()) for (k, v) in leg]
        out.append(from_legacy(layers))
    return out


def _prefix_for_item(it, synth_cpu, prefix_bank):
    if prefix_bank is None:
        return synth_cpu
    key = it.get("_prefix_key") or prefix_bank["default"]
    return prefix_bank["caches"].get(key) or prefix_bank["caches"][prefix_bank["default"]]


def _run_acc(rows, method):
    rs = [r for r in rows if r["method"] == method]
    if not rs:
        return "na"
    acc = float(np.mean([r["correct"] for r in rs]))
    return f"{acc:.3f}(n={len(rs)})"


def _pred_short(text):
    boxed = extract_boxed_answer(text) or extract_gsm8k_answer(text) or ""
    return str(boxed).replace("\n", " ")[:80]


def run_latency(wrapper, items, args, ns, up_agents, synth_cpu, prefix_bank=None,
                dump_path=None):
    """Same generate_bs as LatentMAS run.py. Per-item seconds = batch wall / B."""
    rows = []
    t_wall = time.time()
    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass
    bs = max(1, int(args.generate_bs))
    arms = set(getattr(args, "arms_list", ["latentmas", "ours", "none"]))
    want_real = "latentmas" in arms
    want_ours = "ours" in arms
    want_none = "none" in arms and bool(args.include_none)
    dump_f = open(dump_path, "w") if dump_path else None

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
        up_t, per_up, per_j, per_load, per_j_o = {}, 0.0, 0.0, 0.0, 0.0
        t_judger_full, t_judger_ours = 0.0, 0.0
        batch_rows = []

        if want_real:
            past, up_t, up_peak = build_upstream_timed(
                wrapper, questions, args.k, ns, up_agents)
            full_pos, full_mb = num_positions(past), kv_mb(past)
            per_p = up_t.get("planner", 0.0) / B
            per_c = up_t.get("critic", 0.0) / B
            per_r = up_t.get("refiner", 0.0) / B
            per_up = up_t["upstream"] / B
            reset_peak()
            sync()
            td0 = time.perf_counter()
            texts_a, ntoks_a, eoss_a = decode_batch(
                wrapper, jids, jmask, _split_past(deep_clone(past), B),
                args.judger_budget, temperature=args.temperature, top_p=args.top_p)
            sync()
            t_judger_full = time.perf_counter() - td0
            per_j = t_judger_full / B
            peak_j_full = peak_mb()
            del past
            torch.cuda.empty_cache()

            for b, it in enumerate(batch):
                batch_rows.append({
                    "idx": it.get("idx", start + b),
                    "method": "latentmas",
                    "batch_size": B,
                    "planner_s": per_p,
                    "critic_s": per_c,
                    "refiner_s": per_r,
                    "upstream_s": per_up,
                    "cache_load_s": 0.0,
                    "judger_s": per_j,
                    "e2e_s": per_up + per_j,
                    "planner_batch_s": up_t.get("planner", 0.0),
                    "critic_batch_s": up_t.get("critic", 0.0),
                    "refiner_batch_s": up_t.get("refiner", 0.0),
                    "judger_batch_s": t_judger_full,
                    "tokens": int(ntoks_a[b]),
                    "eos": bool(eoss_a[b]),
                    "correct": bool(graded(texts_a[b], golds[b], args.task)),
                    "pred": _pred_short(texts_a[b]),
                    "tok_per_s": (ntoks_a[b] / per_j) if per_j > 0 else 0.0,
                    "cache_pos": int(full_pos),
                    "cache_mb": float(full_mb) / B,
                    "peak_mb_judger": float(peak_j_full),
                    "peak_mb_planner": up_peak.get("planner", 0.0),
                    "peak_mb_critic": up_peak.get("critic", 0.0),
                    "peak_mb_refiner": up_peak.get("refiner", 0.0),
                })

        if want_ours:
            sync()
            tl0 = time.perf_counter()
            cpu_ours = [_prefix_for_item(it, synth_cpu, prefix_bank) for it in batch]
            caches_ours = [to_dev(deep_clone(c), wrapper.device) for c in cpu_ours]
            sync()
            t_load = time.perf_counter() - tl0
            reset_peak()
            sync()
            td1 = time.perf_counter()
            texts_c, ntoks_c, eoss_c = decode_batch(
                wrapper, jids, jmask, caches_ours, args.judger_budget,
                temperature=args.temperature, top_p=args.top_p)
            sync()
            t_judger_ours = time.perf_counter() - td1
            per_j_o = t_judger_ours / B
            per_load = t_load / B
            peak_j_ours = peak_mb()
            del caches_ours
            torch.cuda.empty_cache()

            for b, it in enumerate(batch):
                batch_rows.append({
                    "idx": it.get("idx", start + b),
                    "method": "ours",
                    "batch_size": B,
                    "planner_s": 0.0,
                    "critic_s": 0.0,
                    "refiner_s": 0.0,
                    "upstream_s": 0.0,
                    "cache_load_s": per_load,
                    "judger_s": per_j_o,
                    "e2e_s": per_load + per_j_o,
                    "planner_batch_s": 0.0,
                    "critic_batch_s": 0.0,
                    "refiner_batch_s": 0.0,
                    "judger_batch_s": t_judger_ours,
                    "tokens": int(ntoks_c[b]),
                    "eos": bool(eoss_c[b]),
                    "correct": bool(graded(texts_c[b], golds[b], args.task)),
                    "pred": _pred_short(texts_c[b]),
                    "tok_per_s": (ntoks_c[b] / per_j_o) if per_j_o > 0 else 0.0,
                    "cache_pos": int(num_positions(cpu_ours[b])),
                    "cache_mb": float(kv_mb(cpu_ours[b])),
                    "prefix_key": it.get("_prefix_key"),
                    "peak_mb_judger": float(peak_j_ours),
                    "peak_mb_planner": 0.0,
                    "peak_mb_critic": 0.0,
                    "peak_mb_refiner": 0.0,
                })

        if want_none:
            reset_peak()
            sync()
            td2 = time.perf_counter()
            texts_n, ntoks_n, eoss_n = decode_batch(
                wrapper, jids, jmask, [None] * B, args.judger_budget,
                temperature=args.temperature, top_p=args.top_p)
            sync()
            t_judger_none = time.perf_counter() - td2
            per_j_n = t_judger_none / B
            for b, it in enumerate(batch):
                batch_rows.append({
                    "idx": it.get("idx", start + b),
                    "method": "none",
                    "batch_size": B,
                    "planner_s": 0.0,
                    "critic_s": 0.0,
                    "refiner_s": 0.0,
                    "upstream_s": 0.0,
                    "cache_load_s": 0.0,
                    "judger_s": per_j_n,
                    "e2e_s": per_j_n,
                    "planner_batch_s": 0.0,
                    "critic_batch_s": 0.0,
                    "refiner_batch_s": 0.0,
                    "judger_batch_s": t_judger_none,
                    "tokens": int(ntoks_n[b]),
                    "eos": bool(eoss_n[b]),
                    "correct": bool(graded(texts_n[b], golds[b], args.task)),
                    "pred": _pred_short(texts_n[b]),
                    "tok_per_s": (ntoks_n[b] / per_j_n) if per_j_n > 0 else 0.0,
                    "cache_pos": 0,
                    "cache_mb": 0.0,
                    "peak_mb_judger": float(peak_mb()),
                    "peak_mb_planner": 0.0,
                    "peak_mb_critic": 0.0,
                    "peak_mb_refiner": 0.0,
                })

        rows.extend(batch_rows)
        if dump_f is not None:
            for rec in batch_rows:
                dump_f.write(json.dumps(rec) + "\n")
            dump_f.flush()
        print(
            f"[latency] items {start+1}-{start+B}/{len(items)} B={B} "
            f"P_batch={up_t.get('planner', 0):.2f}s C_batch={up_t.get('critic', 0):.2f}s "
            f"R_batch={up_t.get('refiner', 0):.2f}s "
            f"J_lat={t_judger_full:.2f}s J_ours={t_judger_ours:.2f}s "
            f"e2e/item {per_up+per_j:.2f}/{per_load+per_j_o:.2f} "
            f"acc L={_run_acc(rows, 'latentmas')} O={_run_acc(rows, 'ours')} "
            f"N={_run_acc(rows, 'none')} elapsed={time.time()-t_wall:.0f}s",
            flush=True,
        )
    if dump_f is not None:
        dump_f.close()
    return rows


def summarize_latency(rows, seed, precompute, n_test):
    methods = ["latentmas", "ours", "none"]
    out = {}
    keys = [
        "planner_s", "critic_s", "refiner_s", "upstream_s", "cache_load_s",
        "judger_s", "e2e_s", "tokens", "cache_mb", "tok_per_s", "peak_mb_judger",
    ]
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        blk = {"n": len(rs), "acc": float(np.mean([r["correct"] for r in rs]))}
        for k in keys:
            if k in rs[0]:
                blk[k] = mean_ci([r[k] for r in rs], seed=seed)
        out[m] = blk

    by = {m: {r["idx"]: r for r in rows if r["method"] == m} for m in methods}
    paired = {}
    common = sorted(set(by.get("latentmas", {})) & set(by.get("ours", {})))
    for k in ("judger_s", "e2e_s", "tokens", "correct", "upstream_s"):
        diffs = [by["ours"][i][k] - by["latentmas"][i][k] for i in common]
        paired[f"ours_minus_latentmas_{k}"] = mean_ci(diffs, seed=seed)
    out["paired"] = paired
    out["n_paired"] = len(common)
    amort = float(precompute.get("t_precompute_s", 0.0)) / max(n_test, 1)
    out["precompute_s"] = float(precompute.get("t_precompute_s", 0.0))
    out["precompute_amortized_per_query_s"] = amort
    if "ours" in out:
        e2e = out["ours"]["e2e_s"]["mean"]
        out["ours_e2e_plus_amortized_precompute_s"] = e2e + amort
    return out


def run_memory(wrapper, sample_item, args, ns, synth_cpu, real_cpu):
    """Clone one cache to batch B; short Judger decode; climb until OOM."""
    print("[memory] batch-size sweep (short decode, not accuracy)", flush=True)
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

    for kind_name, cpu_cache in (
        ("latentmas", real_cpu),
        ("ours", synth_cpu),
        ("none", None),
    ):
        mb_each = float(kv_mb(cpu_cache)) if cpu_cache is not None else 0.0
        pos = int(num_positions(cpu_cache)) if cpu_cache is not None else 0
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
                if cpu_cache is None:
                    caches = [None] * B
                else:
                    caches = [to_dev(deep_clone(cpu_cache), wrapper.device) for _ in range(B)]
                jids = jids1.repeat(B, 1)
                jmask = jmask1.repeat(B, 1)
                sync()
                t0 = time.perf_counter()
                decode_batch(wrapper, jids, jmask, caches, decode_tok)
                sync()
                dt = time.perf_counter() - t0
                pmb = peak_mb()
                results.append({
                    "method": kind_name, "batch_size": B, "ok": True, "oom": False,
                    "peak_mb": float(pmb), "weights_mb": float(weights_mb),
                    "relay_mb_each": mb_each, "relay_pos": pos,
                    "relay_mb_batch": float(mb_each * B),
                    "decode_s": float(dt), "decode_tokens": decode_tok,
                    "items_per_s": float(B / dt) if dt > 0 else 0.0,
                })
                print(
                    f"[memory] {kind_name} B={B} peak={pmb/1024:.2f}GB "
                    f"relay_batch={mb_each*B:.1f}MB decode={dt:.2f}s ({B/dt:.2f} it/s)",
                    flush=True,
                )
                del caches
            except torch.cuda.OutOfMemoryError:
                oom_at = B
                torch.cuda.empty_cache()
                results.append({
                    "method": kind_name, "batch_size": B, "ok": False, "oom": True,
                    "relay_mb_each": mb_each, "relay_pos": pos,
                    "relay_mb_batch": float(mb_each * B),
                })
                print(f"[memory] {kind_name} B={B} OOM", flush=True)
            except Exception as e:
                torch.cuda.empty_cache()
                results.append({
                    "method": kind_name, "batch_size": B, "ok": False, "oom": False,
                    "error": repr(e), "relay_mb_each": mb_each, "relay_pos": pos,
                })
                print(f"[memory] {kind_name} B={B} ERROR {e!r}", flush=True)
                break
    return {"weights_mb": weights_mb, "rows": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--task", default="medqa",
                    choices=["medqa", "gsm8k", "aime2024", "aime2025", "math"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--stat_split", default="train")
    ap.add_argument("--donor_task", default="",
                    help="Task to sample donors from (default: same as --task). "
                         "Use gsm8k/math train as the math-type bank for AIME.")
    ap.add_argument("--stat_n", type=int, default=8,
                    help="How many donor caches to collect (candidates for donor select, "
                         "or members of a mean/gaussian pool).")
    ap.add_argument("--stat_pool", type=int, default=0,
                    help="Max train items to scan for donors (0 -> stat_n, or 4*stat_n if --filter_correct).")
    ap.add_argument("--scaffold", default="donor",
                    choices=["donor", "gaussian", "mean_aligned", "prototype", "replay",
                             "medoid_replay"],
                    help="gaussian = MedQA operating point. "
                         "replay = GSM8K Mean-Replay (average latent embeds, model writes KV). "
                         "medoid_replay = AIME: per-subject latent medoid, then replay. "
                         "donor = one real train relay. mean_aligned = failed full-KV average.")
    ap.add_argument("--donor_level", type=int, default=0,
                    help="If >0 and donors are MATH, keep only this level (5 = contest-hard).")
    ap.add_argument("--min_subject_donors", type=int, default=3,
                    help="Skip a MATH subject with fewer than this many Level-filtered donors.")
    ap.add_argument("--donor_select", default="heldout",
                    choices=["first", "random", "medoid", "heldout"],
                    help="How to pick the one donor cache when --scaffold donor.")
    ap.add_argument("--select_n", type=int, default=32,
                    help="Held-out train items for donor_select=heldout.")
    ap.add_argument("--filter_correct", action="store_true",
                    help="Keep only train donors whose own Judger decode is correct.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=0, help="0 -> 2 smoke / 40 full")
    ap.add_argument("--generate_bs", type=int, default=20,
                    help="Same as run.py / LatentMAS. Fork replica used 25. Not a sweep.")
    ap.add_argument("--judger_budget", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--include_none", action="store_true", default=True)
    ap.add_argument("--no_none", action="store_true")
    ap.add_argument(
        "--arms", default="latentmas,ours,none",
        help="Comma-separated subset of latentmas,ours,none. "
             "Use latentmas,none to verify Real before spending GPU on Ours.")
    ap.add_argument("--mode", default="latency", choices=["latency", "memory", "both"])
    ap.add_argument("--batch_grid", default="1,2,4,8,16,32,64",
                    help="Unused unless --mode memory. Do not use this to claim a win.")
    ap.add_argument("--memory_decode_tokens", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/exp_synth_latency/run")
    args = ap.parse_args()
    if args.no_none:
        args.include_none = False
    allowed_arms = {"latentmas", "ours", "none"}
    args.arms_list = [a.strip() for a in str(args.arms).split(",") if a.strip()]
    bad = [a for a in args.arms_list if a not in allowed_arms]
    if bad or not args.arms_list:
        raise SystemExit(f"--arms must be a non-empty subset of {sorted(allowed_arms)}; got {args.arms!r}")
    if "none" not in args.arms_list:
        args.include_none = False
    want_ours = "ours" in args.arms_list

    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr)
        sys.exit(2)
    if args.n <= 0:
        args.n = 2 if args.smoke else 40
    if args.smoke:
        args.k = min(args.k, 4)
        args.stat_n = min(args.stat_n, 2)
        args.select_n = 0
        args.donor_select = "first"
        args.judger_budget = min(args.judger_budget, 64)
        args.generate_bs = min(args.generate_bs, args.n)
        print(f"[smoke] k={args.k} n={args.n} donors={args.stat_n} generate_bs={args.generate_bs}", flush=True)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    args.donor_task = (args.donor_task or args.task).strip()
    if args.scaffold == "medoid_replay" and args.stat_pool <= 0:
        args.stat_pool = 10 ** 9
    if args.stat_pool <= 0:
        extra = int(getattr(args, "select_n", 0) or 0)
        args.stat_pool = args.stat_n + extra
        if args.filter_correct:
            args.stat_pool = max(args.stat_pool, args.stat_n * 4 + extra)
    items = load_task(args.task, args.split)[: args.n]
    for i, it in enumerate(items):
        it["idx"] = i
    train_pool = []
    if want_ours:
        donor_split = args.stat_split
        if args.donor_task in ("aime2024", "aime2025"):
            donor_split = "train"
        train_pool = load_task(args.donor_task, donor_split)
        if int(getattr(args, "donor_level", 0) or 0) > 0:
            lv = int(args.donor_level)
            before = len(train_pool)
            train_pool = [it for it in train_pool if math_level_int(it) == lv]
            print(f"[load] donor_level={lv} kept {len(train_pool)}/{before}", flush=True)
        if args.stat_pool < 10 ** 9:
            train_pool = train_pool[: args.stat_pool]
    ns = make_ns(args)
    ns_donors = make_ns(args)
    ns_donors.task = args.donor_task
    print(
        f"[load] {args.model_name} task={args.task} n={len(items)} k={args.k} "
        f"temp={args.temperature} top_p={args.top_p} train_pool={len(train_pool)} "
        f"donor_task={args.donor_task} scaffold={args.scaffold} "
        f"select={args.donor_select} generate_bs={args.generate_bs} "
        f"arms={','.join(args.arms_list)}",
        flush=True,
    )
    wrapper = ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)
    up_agents = [a for a in default_agents() if a.role != "judger"]

    prefix_bank = None
    synth_cpu = None
    if not want_ours:
        precompute = {
            "n_donors": 0, "n_tried": 0, "k": int(args.k), "scaffold": "skipped",
            "donor_select": "", "chosen_idx": None, "filter_correct": False,
            "synth_len": 0, "synth_mb": 0.0, "prototype_idx": None, "prototype_l2": None,
            "select_scores": [], "replay_type_question": None, "replay_norms": None,
            "t_stats_s": 0.0, "t_sample_s": 0.0, "t_precompute_s": 0.0, "donors": [],
        }
        print("[precompute] skipped (ours not in --arms)", flush=True)
    elif args.scaffold == "medoid_replay":
        prefix_bank, precompute = precompute_medoid_bank(
            wrapper, train_pool, args, ns_donors, up_agents)
        synth_cpu = prefix_bank["caches"][prefix_bank["default"]]
        assign_prefix_keys(wrapper, items, prefix_bank)
    else:
        synth, precompute = precompute_synth(wrapper, train_pool, args, ns_donors, up_agents)
        synth_cpu = to_cpu(synth)
        del synth
    torch.cuda.empty_cache()

    report: Dict = {
        "config": vars(args),
        "n": len(items),
        "fork_note": (
            "Same generate_bs as LatentMAS run.py (default 20). "
            "LatentMAS = batched Planner/Critic/Refiner + Judger. "
            "ours = frozen type-level cache + Judger; silent agents not run on test. "
            "gaussian = MedQA operating point. "
            "replay = GSM8K Mean-Replay (average latent embeds, replay to write KV). "
            "medoid_replay = AIME subject-conditional medoid tape, model writes KV. "
            "Precompute is not in per-query e2e_s."
        ),
        "precompute": {
            k: v for k, v in precompute.items() if k != "donors"
        },
        "precompute_donors": precompute["donors"],
    }
    if prefix_bank is not None:
        report["prefix_assign"] = [
            {"idx": it.get("idx"), "key": it.get("_prefix_key"), "sim": it.get("_prefix_sim")}
            for it in items
        ]

    real_cpu = None
    if args.mode in ("memory", "both"):
        print("[memory] building one real LatentMAS cache for the batch sweep...", flush=True)
        past, _, _ = build_upstream_timed(
            wrapper, [items[0]["question"]], args.k, ns, up_agents)
        real_cpu = to_cpu(past)
        del past
        torch.cuda.empty_cache()
        report["memory"] = run_memory(wrapper, items[0], args, ns, synth_cpu, real_cpu)

    if args.mode in ("latency", "both"):
        dump_path = os.path.join(args.out_dir, "latency_rows.jsonl")
        rows = run_latency(
            wrapper, items, args, ns, up_agents, synth_cpu, prefix_bank=prefix_bank,
            dump_path=dump_path)
        report["latency"] = summarize_latency(rows, args.seed, precompute, len(items))
        with open(os.path.join(args.out_dir, "latency_rows.json"), "w") as f:
            json.dump(rows, f)
        lat = report["latency"]
        print(f"\n=== PER-AGENT LATENCY (mean sec/item at generate_bs={args.generate_bs}) ===", flush=True)
        hdr = f"{'method':<12}{'P':>8}{'C':>8}{'R':>8}{'up':>8}{'load':>8}{'Judger':>8}{'e2e':>8}{'tok':>8}{'acc':>8}"
        print(hdr, flush=True)
        for m in ("latentmas", "ours", "none"):
            if m not in lat:
                continue
            a = lat[m]
            print(
                f"{m:<12}{a['planner_s']['mean']:8.2f}{a['critic_s']['mean']:8.2f}"
                f"{a['refiner_s']['mean']:8.2f}{a['upstream_s']['mean']:8.2f}"
                f"{a['cache_load_s']['mean']:8.3f}{a['judger_s']['mean']:8.2f}"
                f"{a['e2e_s']['mean']:8.2f}{a['tokens']['mean']:8.1f}{a['acc']:8.3f}",
                flush=True,
            )
        print("\n=== PAIRED (ours - latentmas) ===", flush=True)
        for k, d in lat["paired"].items():
            print(f"  {k:<40} {d['mean']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}] n={d['n']}",
                  flush=True)
        print(
            f"\n[precompute] {lat['precompute_s']:.1f}s total  "
            f"amortized {lat['precompute_amortized_per_query_s']:.3f}s / test item  "
            f"ours e2e+amort {lat.get('ours_e2e_plus_amortized_precompute_s', 0):.2f}s",
            flush=True,
        )

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] {os.path.join(args.out_dir, 'report.json')}", flush=True)
    print("EXP_SYNTH_LATENCY_DONE", flush=True)


if __name__ == "__main__":
    main()
