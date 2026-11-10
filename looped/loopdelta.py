"""Cross-loop KV deltas, MATH data and grading for the LoopDelta stage-0 checks.

Δ_t compares loop t with loop t-1 (t = 2..R, 1-based) on the same positions:

    Δ_t[layer] = ||X_t - X_{t-1}||^2 / ||X_{t-1}||^2,   X in {K, V}

summed over heads, head dims and the selected positions of each row. Normalising
by the previous loop's energy keeps layers with large activations from dominating
the layer mean. Loop t of a depth-R prefill holds exactly the keys a depth-t run
would hold (tests/test_looped_cache.py), so a trajectory read off one R_max
prefill is the trajectory every shallower run would see.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import torch

from utils import extract_boxed_answer, normalize_math_answer


@torch.no_grad()
def kv_deltas(cache, num_layers: int, depth: int, pos_mask: torch.Tensor) -> torch.Tensor:
    """``[B, depth-1, L, 2]`` normalised deltas (last axis: K, V) over ``pos_mask``.

    ``pos_mask`` is ``[B, S]`` over the cache's sequence axis; 1 marks positions
    that count. Rows with no selected position get NaN.
    """
    B, S = pos_mask.shape
    m = pos_mask.to(torch.float32).view(B, 1, S, 1)
    out = torch.full((B, max(depth - 1, 0), num_layers, 2), float("nan"))
    for t in range(1, depth):
        for layer in range(num_layers):
            for j, store in enumerate((cache.key_cache, cache.value_cache)):
                cur = store[t * num_layers + layer][:, :, -S:].float()
                prev = store[(t - 1) * num_layers + layer][:, :, -S:].float()
                num = ((cur - prev) ** 2 * m).sum(dim=(1, 2, 3))
                den = (prev ** 2 * m).sum(dim=(1, 2, 3)).clamp_min(1e-12)
                out[:, t - 1, layer, j] = (num / den).cpu()
    empty = (pos_mask.sum(-1) == 0).cpu()
    if not torch.isfinite(out[~empty]).all():
        raise FloatingPointError(
            "non-finite KV delta on a real position: fully padded query rows can turn NaN under "
            "some sdpa backends (seen on MPS) and leak into the next loop; rerun with --attn eager")
    out[empty] = float("nan")
    return out


def summarize(d: torch.Tensor) -> Dict[str, List[float]]:
    """Per-row summaries of a ``[depth-1, L, 2]`` delta block."""
    per_loop = d.nanmean(dim=(1, 2))                       # mean over layers and K/V
    return {
        "delta": per_loop.tolist(),                          # Δ_2..Δ_R
        "delta_k": d[..., 0].nanmean(-1).tolist(),
        "delta_v": d[..., 1].nanmean(-1).tolist(),
        "delta_auc": float(per_loop.nansum()),
        "delta_last": float(per_loop[-1]) if per_loop.numel() else float("nan"),
    }


# ------------------------------------------------------------------ MATH data

def load_math_train_stratified(per_level: int, seed: int = 0) -> List[Dict]:
    """``per_level`` items from each MATH level 1..5 of the train split, drawn by a
    seeded shuffle; ``idx`` is the position in the loaded train list."""
    from data import load_math

    rows = list(load_math("train"))
    by_level = defaultdict(list)
    for i, r in enumerate(rows):
        if 1 <= r["level_int"] <= 5:
            by_level[r["level_int"]].append(i)
    rng = random.Random(seed)
    out = []
    for lv in range(1, 6):
        ids = by_level[lv]
        rng.shuffle(ids)
        for i in ids[:per_level]:
            r = rows[i]
            out.append({"idx": i, "level": lv, "subject": r["subject"], "question": r["question"],
                        "gold": extract_boxed_answer(r["solution"]) or r["gold"]})
    return out


def fingerprint(items: Sequence[Dict]) -> str:
    blob = json.dumps([(it["idx"], it["question"]) for it in items]).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ------------------------------------------------------------------ grading

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
except Exception:  # math_verify is optional; string match is the fallback
    _mv_parse = _mv_verify = None


def extract_math_answer(text: str) -> Optional[str]:
    return extract_boxed_answer(text) if text else None


def grade_math(text: str, gold_raw: str) -> Dict:
    """``correct`` uses math_verify when installed, else normalised string match.
    Both verdicts are returned so the grader can be audited."""
    pred = extract_math_answer(text)
    str_ok = pred is not None and normalize_math_answer(pred) == normalize_math_answer(gold_raw)
    mv_ok = None
    if _mv_verify is not None and pred is not None:
        try:
            mv_ok = bool(_mv_verify(_mv_parse(f"${gold_raw}$"), _mv_parse(f"${pred}$")))
        except Exception:
            mv_ok = None
    return {"pred": pred, "correct": bool(mv_ok) if mv_ok is not None else str_ok,
            "str_ok": str_ok, "mv_ok": mv_ok}


def answer_key(pred: Optional[str]) -> Optional[str]:
    """Key used to group votes: normalised string of the boxed answer."""
    return normalize_math_answer(pred) if pred is not None else None
