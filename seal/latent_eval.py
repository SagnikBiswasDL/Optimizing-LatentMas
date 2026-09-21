"""Shared Judger / silent-agent eval helpers (CPU-safe pieces, CUDA when present)."""
from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch

from seal.cache_bank import from_legacy, kv_mb, num_positions, to_legacy
from utils import (
    extract_boxed_answer,
    extract_gsm8k_answer,
    normalize_answer,
    normalize_math_answer,
)


def make_ns(args, task: Optional[str] = None):
    return SimpleNamespace(
        model_name=args.model_name,
        task=task or args.task,
        prompt="sequential",
        think=False,
        latent_only=False,
        sequential_info_only=False,
        agents=None,
        use_vllm=False,
        device=args.device,
        device2="cuda:1",
        max_new_tokens=int(getattr(args, "judger_budget", 1024)),
        text_mas_context_length=-1,
        temperature=float(getattr(args, "temperature", 0.0)),
        top_p=float(getattr(args, "top_p", 1.0)),
        seed=int(getattr(args, "seed", 42)),
        seal=False,
        kvsteer=False,
        ces=False,
        capture_acts=None,
        planner_steps=None,
        critic_steps=None,
        refiner_steps=None,
        latent_steps=0,
        latent_space_realign=False,
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
    return from_legacy(
        [(k.detach().cpu().contiguous(), v.detach().cpu().contiguous())
         for (k, v) in to_legacy(past)]
    )


def to_dev(past, device):
    if past is None:
        return None
    return from_legacy([(k.to(device), v.to(device)) for (k, v) in to_legacy(past)])


def stack_past(caches):
    """Concatenate single-item caches on the batch axis."""
    if not caches:
        return None
    legs = [to_legacy(c) for c in caches]
    layers = []
    for li in range(len(legs[0])):
        k = torch.cat([lg[li][0] for lg in legs], 0)
        v = torch.cat([lg[li][1] for lg in legs], 0)
        layers.append((k.contiguous(), v.contiguous()))
    return from_legacy(layers)


def split_past(past, B: int):
    if past is None:
        return [None] * B
    leg = to_legacy(past)
    out = []
    for b in range(B):
        layers = [(k[b : b + 1].contiguous(), v[b : b + 1].contiguous()) for (k, v) in leg]
        out.append(from_legacy(layers))
    return out


def pad_caches_left(caches, device):
    legacies = [to_legacy(c) for c in caches]
    B = len(legacies)
    n_layers = len(legacies[0])
    lens = [lg[0][0].shape[-2] for lg in legacies]
    pmax = max(lens)
    past_mask = torch.zeros(B, pmax, dtype=torch.long, device=device)
    for b, ln in enumerate(lens):
        past_mask[b, pmax - ln :] = 1
    layers = []
    for li in range(n_layers):
        ks, vs = [], []
        for b in range(B):
            k, v = legacies[b][li]
            k = k.to(device)
            v = v.to(device)
            ln = k.shape[-2]
            if ln < pmax:
                pk = torch.zeros(1, k.shape[1], pmax - ln, k.shape[3], dtype=k.dtype, device=device)
                pv = torch.zeros(1, v.shape[1], pmax - ln, v.shape[3], dtype=v.dtype, device=device)
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            ks.append(k)
            vs.append(v)
        layers.append((torch.cat(ks, 0), torch.cat(vs, 0)))
    return from_legacy(layers), past_mask, pmax


def decode_batch(wrapper, judger_ids, judger_mask, caches, budget, temperature=0.0, top_p=1.0):
    device = wrapper.device
    jids = judger_ids.to(device)
    jmask = judger_mask.to(device)
    if all(c is None for c in caches):
        past, full_mask, cache_position = None, jmask, None
    else:
        past, past_mask, pmax = pad_caches_left(caches, device)
        full_mask = torch.cat([past_mask, jmask], dim=1)
        cache_position = torch.arange(pmax, pmax + jids.shape[1], device=device)
    sample = float(temperature) > 0
    gen_kwargs = dict(
        input_ids=jids,
        attention_mask=full_mask,
        past_key_values=past,
        max_new_tokens=int(budget),
        do_sample=sample,
        pad_token_id=wrapper.tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=False,
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


def attach_judger_seal(wrapper, vector_path: str, *, coef: float, layer_index: int = 28):
    """Register the GSM8K SEAL residual on the Judger only. Starts disabled."""
    from seal import SealSteerer

    if not os.path.isfile(vector_path):
        raise FileNotFoundError(vector_path)
    wrapper.seal = SealSteerer.from_artifact(
        vector_path, coef=float(coef), layer_index=int(layer_index),
    )
    wrapper.seal_active_roles = {"judger"}
    wrapper.seal.register(wrapper.model)
    wrapper.seal.disable()
    print(
        f"[SEAL] attached layer={wrapper.seal.layer_index} coef={wrapper.seal.coef} "
        f"vector={vector_path}",
        flush=True,
    )
    return wrapper.seal


def decode_batch_maybe_seal(
    wrapper, judger_ids, judger_mask, caches, budget,
    temperature=0.0, top_p=1.0, seal_on: bool = False,
):
    armed = False
    if seal_on:
        seal = getattr(wrapper, "seal", None)
        if seal is None:
            raise RuntimeError("decode_batch_maybe_seal(seal_on=True) needs attach_judger_seal")
        seal.set_active_role("judger")
        seal.enable()
        armed = True
    try:
        return decode_batch(
            wrapper, judger_ids, judger_mask, caches, budget,
            temperature=temperature, top_p=top_p,
        )
    finally:
        if armed:
            wrapper.seal.disable()


def graded(text, gold, task):
    boxed = extract_boxed_answer(text) or extract_gsm8k_answer(text) or text
    if task in ("math",):
        pred = normalize_math_answer(boxed)
        g = normalize_math_answer(gold)
        return bool(pred) and bool(g) and pred == g
    pred = normalize_answer(boxed)
    g = normalize_answer(gold)
    if task in ("aime2024", "aime2025"):
        try:
            return int(pred) == int(g)
        except (TypeError, ValueError):
            return bool(pred) and bool(g) and pred == g
    if task == "medqa":
        if pred and g and (pred == g or pred[:1] == g[:1]):
            return True
        return False
    return bool(pred) and bool(g) and pred == g


def pred_short(text):
    boxed = extract_boxed_answer(text) or extract_gsm8k_answer(text) or ""
    return str(boxed).replace("\n", " ")[:80]


def mean_ci(xs, seed=0):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    boots = [xs[rng.integers(0, len(xs), len(xs))].mean() for _ in range(2000)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(xs.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(xs))}


def squeeze_latents(emb: torch.Tensor) -> torch.Tensor:
    if emb.dim() == 3:
        return emb[:, 0, :].contiguous()
    return emb.contiguous()


@torch.no_grad()
def build_upstream_timed(
    wrapper,
    questions: List[str],
    k,
    ns,
    agents,
    *,
    collect_latents: bool = False,
    forced_by_role: Optional[Dict[str, torch.Tensor]] = None,
    out_latents: Optional[Dict[str, torch.Tensor]] = None,
    start_past=None,
    latent_steps: Optional[int] = None,
):
    """Silent-agent pass. ``start_past`` is the precomputed prefix (optional)."""
    from prompts import build_agent_message_sequential_latent_mas

    past = start_past
    k_use = int(k if latent_steps is None else latent_steps)
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
            ids,
            attention_mask=mask,
            latent_steps=int(k_use),
            past_key_values=past,
            role=agent.role,
            collect_latents=collect_latents,
            forced_latents=forced,
        )
        sync()
        times[agent.role] = time.perf_counter() - t0
        peaks[agent.role] = peak_mb()
        if collect_latents and out_latents is not None:
            emb = getattr(wrapper, "last_latent_embeds", None)
            if emb is None:
                raise RuntimeError(f"collect_latents: no embeds for role={agent.role}")
            out_latents[agent.role] = squeeze_latents(emb)
    times["upstream"] = sum(times[a.role] for a in agents)
    return past, times, peaks
