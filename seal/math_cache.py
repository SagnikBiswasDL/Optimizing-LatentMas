"""Unified MATH Mean-Replay cache: one prefix from many train tapes.

Construction (the only averaging recipe for the math ladder):
  1. Run Planner/Critic/Refiner on N MATH-train items, keep the K latent
     embeddings each role actually feeds as the next input.
  2. Average those embeddings *by step*, restore median L2.
  3. Replay the mean tapes on a type prompt so the model writes KV.
  4. Freeze that KV. Never average finished KV tables.

The on-disk object is one cache used on every math eval item (GSM8K / MATH-500
/ AIME). Residual small-K and eviction are eval-time, not part of this file.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from seal.cache_bank import (
    from_legacy,
    kv_mb,
    mean_restore_latents,
    num_positions,
    to_legacy,
)


UNIFIED_MATH_PROMPT = (
    "Solve a mathematics problem. Reason carefully. "
    "Put the final answer in \\boxed{}."
)

ROLES = ("planner", "critic", "refiner")
_ARM_OURS = re.compile(r"^frozen(?:_k(\d+))?(?:_evict(\d+))?$")
_ARM_LIVE = re.compile(r"^k(\d+)$")


def parse_int_list(raw: str, default: Sequence[int] = (0,)) -> List[int]:
    if raw is None or str(raw).strip() == "":
        return [int(x) for x in default]
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or [int(x) for x in default]


def arm_name(k_ttc: int = 0, evict: int = 0) -> str:
    name = "frozen" if int(k_ttc) <= 0 else f"frozen_k{int(k_ttc)}"
    if int(evict) > 0:
        name += f"_evict{int(evict)}"
    return name


def parse_arm(name: str) -> Dict[str, Any]:
    """Map an arm label to {kind, k_ttc, evict}.

    kind is ``none`` / ``real`` / ``ours`` / ``live``.
    Ours is the precomputed prefix, optionally plus residual silent-agent
    steps and plain-H eviction. Live is Planner/Critic/Refiner at K from
    scratch (no frozen prefix), e.g. ``k2`` / ``k5``.
    """
    name = str(name).strip()
    if name in ("none", "real"):
        return {"kind": name, "k_ttc": 0, "evict": 0, "name": name}
    live = _ARM_LIVE.fullmatch(name)
    if live:
        k = int(live.group(1))
        if k <= 0:
            raise ValueError(f"live arm {name!r} needs K>=1")
        return {"kind": "live", "k_ttc": k, "evict": 0, "name": name}
    m = _ARM_OURS.fullmatch(name)
    if not m:
        raise ValueError(
            f"unknown arm {name!r}; expected none, real, frozen, "
            f"frozen_k2, k2, k5, frozen_evict64, frozen_k2_evict64, ..."
        )
    return {
        "kind": "ours",
        "k_ttc": int(m.group(1) or 0),
        "evict": int(m.group(2) or 0),
        "name": name,
    }


def expand_ladder_arms(
    k_ttc: Sequence[int],
    evict_budgets: Sequence[int],
    requested: Optional[Iterable[str]] = None,
) -> List[str]:
    """Cartesian product of residual-K × eviction, plus none and real."""
    names = ["none", "real"]
    ks = [int(x) for x in k_ttc] or [0]
    bs = [int(x) for x in evict_budgets] or [0]
    for k in ks:
        names.append(arm_name(k, 0))
        if k > 0:
            names.append(f"k{k}")
        for b in bs:
            if b > 0:
                names.append(arm_name(k, b))
    # unique, stable
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    if requested:
        want = {str(x).strip() for x in requested if str(x).strip()}
        missing = sorted(want - set(ordered))
        if missing:
            raise ValueError(f"--arms not in the expanded set {ordered}: {missing}")
        ordered = [n for n in ordered if n in want]
    if not ordered:
        raise ValueError("no arms left after filtering")
    return ordered


def stack_latents(steps: Sequence[torch.Tensor]) -> torch.Tensor:
    """List of [K, D] -> [N, K, D]."""
    if not steps:
        raise ValueError("stack_latents: empty")
    xs = []
    for t in steps:
        t = t.detach().float().cpu()
        if t.dim() == 3:
            t = t[:, 0, :]
        if t.dim() != 2:
            raise ValueError(f"stack_latents: expected [K,D], got {tuple(t.shape)}")
        xs.append(t)
    return torch.stack(xs, 0)


def donor_checkpoint_path(out_dir: str) -> str:
    return os.path.join(out_dir, "donors.pt")


def cache_path(out_dir: str) -> str:
    return os.path.join(out_dir, "cache.pt")


def empty_donor_state(k: int, model_name: str) -> Dict[str, Any]:
    return {
        "k": int(k),
        "model_name": model_name,
        "roles": list(ROLES),
        "latents": {r: [] for r in ROLES},
        "rows": [],
        "n_tried": 0,
        "next_idx": 0,
    }


def load_donor_state(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "latents" not in payload:
        raise ValueError(f"not a donor checkpoint: {path}")
    for r in ROLES:
        payload["latents"].setdefault(r, [])
    payload["rows"] = list(payload.get("rows") or [])
    payload["n_tried"] = int(payload.get("n_tried") or 0)
    payload["next_idx"] = int(payload.get("next_idx") or payload["n_tried"])
    return payload


def n_kept(state: Dict[str, Any]) -> int:
    lat = state.get("latents") or {}
    return int(min(len(lat.get(r) or []) for r in ROLES))


def append_donor(
    state: Dict[str, Any],
    *,
    latents: Dict[str, torch.Tensor],
    row: Dict[str, Any],
    next_idx: int,
) -> None:
    for r in ROLES:
        t = latents[r]
        if t.dim() == 3:
            t = t[:, 0, :]
        state["latents"][r].append(t.detach().float().cpu().contiguous())
    state["rows"].append(dict(row))
    state["n_tried"] = int(state.get("n_tried") or 0) + 1
    state["next_idx"] = int(next_idx)


def save_donor_state(state: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    packed = dict(state)
    packed["latents"] = {
        r: stack_latents(state["latents"][r]) if state["latents"][r] else torch.zeros(0, 1, 1)
        for r in ROLES
    }
    # keep list form in memory; disk stores stacked tensors
    torch.save(packed, path)


def load_donor_state_lists(path: str) -> Dict[str, Any]:
    """Load checkpoint and convert stacked latents back to a list per role."""
    state = load_donor_state(path)
    lat = {}
    for r in ROLES:
        t = state["latents"][r]
        if torch.is_tensor(t):
            if t.numel() == 0 or t.shape[0] == 0:
                lat[r] = []
            else:
                lat[r] = [t[i].contiguous() for i in range(t.shape[0])]
        else:
            lat[r] = list(t)
    state["latents"] = lat
    return state


def mean_tapes(state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out = {}
    for r in ROLES:
        steps = state["latents"][r]
        if not steps:
            raise RuntimeError(f"mean_tapes: no latents for role={r}")
        out[r] = mean_restore_latents(steps)
    return out


def legacy_from_past(past) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    lg = to_legacy(past)
    return [(k.detach().cpu().contiguous(), v.detach().cpu().contiguous()) for (k, v) in lg]


def save_unified_cache(
    path: str,
    *,
    past,
    meta: Dict[str, Any],
    mean_latents: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "meta": dict(meta),
        "legacy": legacy_from_past(past),
        "mean_latents": {
            r: t.detach().float().cpu().contiguous()
            for r, t in (mean_latents or {}).items()
        },
    }
    payload["meta"]["n_pos"] = num_positions(payload["legacy"])
    payload["meta"]["mb"] = float(kv_mb(payload["legacy"]))
    torch.save(payload, path)


def load_unified_cache(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "legacy" not in payload:
        raise ValueError(f"not a unified math cache: {path}")
    payload.setdefault("meta", {})
    payload.setdefault("mean_latents", {})
    return payload


def cache_as_past(payload: Dict[str, Any], *, device="cpu", dtype=None):
    lg = payload["legacy"]
    if dtype is not None:
        lg = [(k.to(dtype=dtype), v.to(dtype=dtype)) for (k, v) in lg]
    past = from_legacy(lg)
    if device != "cpu":
        past = from_legacy([(k.to(device), v.to(device)) for (k, v) in to_legacy(past)])
    return past


def gate_verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    """Jiayi: frozen good on eval? if not, run first 3 agents at small K."""
    arms = report.get("arms") or {}
    frozen = arms.get("frozen") or {}
    real = arms.get("real") or {}
    none = arms.get("none") or {}
    paired = report.get("paired") or {}
    d_acc = paired.get("frozen_minus_real_correct") or {}
    out = {
        "frozen_acc": frozen.get("acc"),
        "real_acc": real.get("acc"),
        "none_acc": none.get("acc"),
        "frozen_minus_real_acc": d_acc.get("mean"),
        "ci_lo": d_acc.get("ci_lo"),
        "ci_hi": d_acc.get("ci_hi"),
        "recommend": "unknown",
        "reason": "",
    }
    if frozen.get("acc") is None or real.get("acc") is None:
        out["recommend"] = "need_real_and_frozen"
        out["reason"] = "run --arms none,frozen,real"
        return out
    lo = d_acc.get("ci_lo")
    if lo is not None and lo >= -1e-9:
        out["recommend"] = "skip_ttc"
        out["reason"] = (
            "frozen matches Real (paired acc CI does not sit below 0); add eviction"
        )
        return out
    if none.get("acc") is not None and frozen["acc"] + 1e-9 < none["acc"]:
        out["recommend"] = "bug"
        out["reason"] = (
            "frozen < none: Mean-Replay collapsed (do not add a fourth recipe)"
        )
        return out
    out["recommend"] = "run_small_k"
    out["reason"] = (
        "frozen misses Real; run Planner/Critic/Refiner at K in {2,5} on top of the cache"
    )
    return out
