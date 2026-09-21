"""Slice / concat LatentMAS KV along the sequence axis, by silent-agent span.

Growing Real K=10 writes one cache that is Planner then Critic then Refiner.
These helpers cut that tape into ``c1``, ``c2``, ``c3`` so a later Judger decode
can see one role, a pair, latents-only, or a role-budgeted eviction — without
re-running the 33 silent forwards.

All ops are CPU-safe and work on B=1 caches (the AIME protocol).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from seal.cache_bank import from_legacy, num_positions, to_legacy
from seal.relay_compress import RelayCompressor, kv_mb


ROLES = ("planner", "critic", "refiner")


def slice_seq(past, start: int, end: int):
    """Keep positions ``[start, end)`` on the sequence axis. Empty if end<=start."""
    if past is None:
        return None
    start = int(max(0, start))
    end = int(end)
    if end <= start:
        return _empty_like(past)
    legacy = to_legacy(past)
    S = int(legacy[0][0].shape[-2])
    start = min(start, S)
    end = min(end, S)
    if end <= start:
        return _empty_like(past)
    layers = []
    for k, v in legacy:
        layers.append((k[..., start:end, :].contiguous(), v[..., start:end, :].contiguous()))
    return from_legacy(layers)


def _empty_like(past):
    legacy = to_legacy(past)
    layers = []
    for k, v in legacy:
        layers.append((k[..., :0, :].contiguous(), v[..., :0, :].contiguous()))
    return from_legacy(layers)


def concat_seq(parts: Sequence):
    """Concatenate caches along the sequence axis. Skips None / length-0."""
    usable = []
    for p in parts:
        if p is None:
            continue
        if num_positions(p) <= 0:
            continue
        usable.append(p)
    if not usable:
        return None
    if len(usable) == 1:
        lg = to_legacy(usable[0])
        return from_legacy([(k.contiguous(), v.contiguous()) for (k, v) in lg])
    n_layers = len(to_legacy(usable[0]))
    layers = []
    for li in range(n_layers):
        ks = [to_legacy(p)[li][0] for p in usable]
        vs = [to_legacy(p)[li][1] for p in usable]
        layers.append((torch.cat(ks, dim=-2).contiguous(), torch.cat(vs, dim=-2).contiguous()))
    return from_legacy(layers)


def span_map(spans: Sequence[Dict]) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for sp in spans:
        out[str(sp["role"])] = (int(sp["start"]), int(sp["end"]))
    return out


def take_roles(past, spans: Sequence[Dict], roles: Iterable[str]):
    """Concat the listed role spans in planner-critic-refiner order."""
    want = [r for r in ROLES if r in set(roles)]
    sm = span_map(spans)
    parts = []
    for r in want:
        if r not in sm:
            continue
        a, b = sm[r]
        parts.append(slice_seq(past, a, b))
    return concat_seq(parts)


def take_latents(past, spans: Sequence[Dict], k: int):
    """Keep the last ``k`` positions of each role (latent steps, drop prompt)."""
    k = int(k)
    parts = []
    for sp in spans:
        a, b = int(sp["start"]), int(sp["end"])
        keep = min(k, max(0, b - a))
        parts.append(slice_seq(past, b - keep, b))
    return concat_seq(parts)


def evict_uniform(past, budget: int, sink: int = 4, importance: str = "key_norm"):
    if past is None or num_positions(past) <= 0:
        return past, None
    comp = RelayCompressor(mode="evict", budget=int(budget), sink=int(sink),
                           importance=importance)
    out, st = comp.compress(past)
    return out, st.as_dict()


def evict_by_segment(
    past,
    spans: Sequence[Dict],
    *,
    budget_per_role: int,
    sink: int = 2,
    importance: str = "key_norm",
    roles: Optional[Sequence[str]] = None,
):
    """Independent SnapKV budget on each silent-agent span, then concat.

    This is the eviction that can *use* a localization result: if Planner is
    wasteful, drop it from ``roles`` or give it budget 0; if Refiner carries
    AIME, keep its full slice.
    """
    if past is None:
        return None, {"mode": "seg_evict", "parts": []}
    sm = span_map(spans)
    use = [r for r in ROLES if r in (set(roles) if roles is not None else set(ROLES))]
    parts = []
    info = []
    for r in use:
        if r not in sm:
            continue
        a, b = sm[r]
        sl = slice_seq(past, a, b)
        n = num_positions(sl)
        if n <= 0:
            continue
        if int(budget_per_role) <= 0:
            info.append({"role": r, "in": n, "out": 0})
            continue
        out, st = evict_uniform(sl, int(budget_per_role), sink=int(sink),
                                importance=importance)
        parts.append(out)
        rec = {"role": r, "in": n, "out": num_positions(out)}
        if st:
            rec["ratio"] = st.get("ratio")
        info.append(rec)
    merged = concat_seq(parts)
    return merged, {
        "mode": "seg_evict",
        "budget_per_role": int(budget_per_role),
        "sink": int(sink),
        "mb_out": float(kv_mb(merged)) if merged is not None else 0.0,
        "pos_out": int(num_positions(merged)) if merged is not None else 0,
        "parts": info,
    }
