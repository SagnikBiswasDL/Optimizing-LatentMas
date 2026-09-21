"""DeltaBridge: problem-conditioned residual over a frozen Mean-Replay cache.

The frozen 422-position MATH-1k cache is the task-level prior. The instance-
specific piece is a low-rank residual injected at one (or a few) designated
bridge token(s) immediately before Judger decode.

This module is the CPU-safe math + hook plumbing. The GPU driver is
``scripts/exp_delta_bridge.py``.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from seal.cache_bank import from_legacy, num_positions, to_legacy
from seal.hooks import _find_decoder_layers


DEFAULT_LAYERS_14B = (16, 24, 28, 32)
DEFAULT_RANKS = (4, 8, 16, 32)
ORACLE_ARMS = ("zero", "oracle_full", "oracle_r", "shuffled")


def parse_int_tuple(raw: str, default: Sequence[int]) -> Tuple[int, ...]:
    if raw is None or str(raw).strip() == "":
        return tuple(int(x) for x in default)
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return tuple(out) if out else tuple(int(x) for x in default)


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def bridge_token_id(tokenizer) -> int:
    """Single-token suffix; do not resize embeddings."""
    for s in ("\n", ":", " "):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("tokenizer has no single-token bridge candidate")
    return int(eos)


class LayerRecorder:
    """Last-token residual of selected decoder layers (latest forward)."""

    def __init__(self, model, layers: Sequence[int]):
        self.layers = _find_decoder_layers(model)
        self.want = [int(i) for i in layers]
        n = len(self.layers)
        for i in self.want:
            if not (0 <= i < n):
                raise IndexError(f"layer {i} out of range (model has {n})")
        self.buf: Dict[int, torch.Tensor] = {}
        self.handles = []

    def _mk(self, i: int):
        def hook(_m, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            self.buf[i] = hs[:, -1, :].detach().float().cpu()
        return hook

    def register(self):
        if self.handles:
            return
        for i in self.want:
            self.handles.append(self.layers[i].register_forward_hook(self._mk(i)))

    def snapshot(self) -> Dict[int, torch.Tensor]:
        out = {}
        for i in self.want:
            t = self.buf.get(i)
            if t is None:
                raise RuntimeError(f"LayerRecorder: layer {i} was not written")
            out[i] = t.contiguous()
        return out

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.buf = {}


class ResidualInjector:
    """Add a per-layer vector to the last token of the current forward only."""

    def __init__(self, deltas: Optional[Dict[int, torch.Tensor]] = None):
        self.deltas: Dict[int, torch.Tensor] = {
            int(k): v.detach() for k, v in (deltas or {}).items() if v is not None
        }
        self.handles = []
        self._enabled = False
        self._layers = None
        self.once = True
        self._fired = set()

    def set_deltas(self, deltas: Optional[Dict[int, torch.Tensor]]):
        self.deltas = {
            int(k): v.detach() for k, v in (deltas or {}).items() if v is not None
        }

    def _mk(self, i: int):
        def hook(_m, _inp, out):
            if not self._enabled:
                return out
            vec = self.deltas.get(i)
            if vec is None:
                return out
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.clone()
            delta = vec.to(dtype=hs.dtype, device=hs.device)
            if delta.dim() == 1:
                delta = delta.view(1, 1, -1)
            elif delta.dim() == 2:
                delta = delta.view(delta.shape[0], 1, -1)
            hs[:, -1:, :] = hs[:, -1:, :] + delta
            self._fired.add(i)
            if self.once and self.deltas and self._fired >= set(self.deltas):
                self._enabled = False
            if isinstance(out, tuple):
                return (hs,) + tuple(out[1:])
            return hs
        return hook

    def register(self, model) -> None:
        if self.handles:
            return
        layers = _find_decoder_layers(model)
        self._layers = layers
        want = sorted(self.deltas)
        if not want:
            return
        registered = set()
        for i in want:
            if i in registered:
                continue
            if not (0 <= i < len(layers)):
                raise IndexError(f"inject layer {i} out of range (model has {len(layers)})")
            self.handles.append(layers[i].register_forward_hook(self._mk(i)))
            registered.add(i)

    def ensure_layers(self, model, layers: Sequence[int]) -> None:
        """Register hooks for these layers even if deltas are empty (will no-op)."""
        if self.handles:
            return
        all_layers = _find_decoder_layers(model)
        for i in layers:
            i = int(i)
            if not (0 <= i < len(all_layers)):
                raise IndexError(f"inject layer {i} out of range")
            self.handles.append(all_layers[i].register_forward_hook(self._mk(i)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def enable(self) -> None:
        self._enabled = True
        self._fired = set()

    def disable(self) -> None:
        self._enabled = False
        self._fired = set()


def fit_pca(x: torch.Tensor, rank: int) -> Dict[str, Any]:
    """PCA on [N, D] residuals. Returns orthonormal columns U [D, r], mean [D]."""
    if x.dim() != 2:
        raise ValueError(f"fit_pca expected [N,D], got {tuple(x.shape)}")
    n, d = x.shape
    r = int(max(1, min(rank, n, d)))
    x = x.detach().float()
    mean = x.mean(dim=0)
    xc = x - mean
    # SVD on centered data: X = U S Vh, principal axes = Vh.T = V
    try:
        _, s, vh = torch.linalg.svd(xc, full_matrices=False)
    except RuntimeError:
        # tiny N or degenerate
        s = torch.zeros(min(n, d))
        vh = torch.zeros(min(n, d), d)
        vh.fill_diagonal_(1.0)
    u = vh[:r].T.contiguous()  # [D, r]
    tot = float((s.pow(2).sum()).clamp_min(1e-12))
    explained = float((s[:r].pow(2).sum() / tot).item()) if tot > 0 else 0.0
    return {
        "mean": mean.contiguous(),
        "U": u,
        "singular": s[:r].contiguous(),
        "rank": r,
        "explained": explained,
        "n": int(n),
        "d": int(d),
        "energy": float(x.norm(dim=-1).mean().item()),
    }


def project_residual(d: torch.Tensor, bank: Dict[str, Any]) -> torch.Tensor:
    """Reconstruct d in the PCA subspace: mu + U U^T (d - mu)."""
    u = bank["U"].to(dtype=d.dtype, device=d.device)
    mean = bank["mean"].to(dtype=d.dtype, device=d.device)
    centered = d - mean
    alpha = u.T @ centered
    return mean + (u @ alpha)


def coefficients(d: torch.Tensor, bank: Dict[str, Any]) -> torch.Tensor:
    u = bank["U"].to(dtype=d.dtype, device=d.device)
    mean = bank["mean"].to(dtype=d.dtype, device=d.device)
    return u.T @ (d - mean)


def reconstruct_from_alpha(alpha: torch.Tensor, bank: Dict[str, Any]) -> torch.Tensor:
    u = bank["U"].to(dtype=alpha.dtype, device=alpha.device)
    mean = bank["mean"].to(dtype=alpha.dtype, device=alpha.device)
    return mean + (u @ alpha)


def pick_layer_rank(
    banks: Dict[int, Dict[int, Dict[str, Any]]],
    *,
    prefer_layer: int = 28,
    target_rank: int = 16,
    min_explained: float = 0.25,
) -> Dict[str, Any]:
    """Choose one layer / rank from fitted PCA banks.

    ``banks[layer][rank]`` is a fit_pca dict (possibly nested per bridge slot;
    pass already-selected slot 0).
    """
    layers = sorted(int(x) for x in banks)
    if not layers:
        raise ValueError("pick_layer_rank: empty banks")
    ranks_avail = sorted({int(r) for li in layers for r in banks[li]})
    rank = int(target_rank)
    if rank not in ranks_avail:
        rank = min(ranks_avail, key=lambda r: abs(r - target_rank))

    scored = []
    for li in layers:
        blk = banks[li][rank]
        exp = float(blk.get("explained") or 0.0)
        energy = float(blk.get("energy") or 0.0)
        scored.append((li, exp, energy))

    eligible = [s for s in scored if s[1] >= min_explained]
    pool = eligible or scored
    # Prefer energy; break ties toward prefer_layer
    pool.sort(key=lambda t: (t[2], -abs(t[0] - prefer_layer)), reverse=True)
    layer = int(pool[0][0])
    if prefer_layer in {s[0] for s in pool}:
        pref = [s for s in pool if s[0] == prefer_layer][0]
        best = pool[0]
        if pref[2] >= 0.8 * best[2]:
            layer = prefer_layer
    reason = (
        f"layer={layer} rank={rank} explained={banks[layer][rank]['explained']:.3f} "
        f"energy={banks[layer][rank]['energy']:.4f} "
        f"(prefer {prefer_layer} if close)"
    )
    return {"layer": layer, "rank": rank, "reason": reason}


def oracle_verdict(
    *,
    zero_correct: Sequence[bool],
    oracle_correct: Sequence[bool],
    shuffled_correct: Optional[Sequence[bool]] = None,
    full_correct: Optional[Sequence[bool]] = None,
    min_recover: int = 2,
) -> Dict[str, Any]:
    """Go/no-go from paired per-item flags. Oracle is the capacity test."""
    n = len(zero_correct)
    if len(oracle_correct) != n:
        raise ValueError("oracle_verdict: length mismatch")
    z = [bool(x) for x in zero_correct]
    o = [bool(x) for x in oracle_correct]
    recovered = [i for i in range(n) if o[i] and not z[i]]
    lost = [i for i in range(n) if z[i] and not o[i]]
    n_rec = len(recovered)
    shuf_rec = []
    if shuffled_correct is not None and len(shuffled_correct) == n:
        s = [bool(x) for x in shuffled_correct]
        shuf_rec = [i for i in range(n) if s[i] and not z[i]]
    full_rec = []
    if full_correct is not None and len(full_correct) == n:
        f = [bool(x) for x in full_correct]
        full_rec = [i for i in range(n) if f[i] and not z[i]]

    acc_z = sum(z) / max(n, 1)
    acc_o = sum(o) / max(n, 1)
    acc_s = (sum(shuffled_correct) / n) if shuffled_correct is not None and len(shuffled_correct) == n else None
    acc_f = (sum(full_correct) / n) if full_correct is not None and len(full_correct) == n else None

    shuffled_matches = (
        shuffled_correct is not None
        and len(shuf_rec) >= n_rec
        and n_rec >= 1
        and len(set(shuf_rec) & set(recovered)) >= max(1, n_rec - 1)
    )

    if n_rec >= min_recover and not shuffled_matches:
        recommend = "go"
        reason = (
            f"oracle recovered {n_rec} zero-misses {recovered} and lost {lost}; "
            "shuffled does not match. Train the coefficient predictor."
        )
    elif acc_f is not None and acc_f > acc_o + 1e-9 and (len(full_rec) >= min_recover) and n_rec < min_recover:
        recommend = "raise_rank_or_tokens"
        reason = (
            f"full residual recovers {full_rec} but rank-r does not ({recovered}). "
            "Raise rank or inject more layers before training."
        )
    elif n_rec < min_recover and (acc_f is None or len(full_rec) < min_recover):
        recommend = "retry_4token"
        reason = (
            f"oracle recovered {n_rec} {recovered} (need {min_recover}). "
            "A single residual likely cannot replace K=10; try four bridge tokens."
        )
    elif shuffled_matches:
        recommend = "stop_not_instance_specific"
        reason = (
            f"oracle recovered {recovered} but shuffled recovered {shuf_rec}. "
            "Gain is not instance-specific."
        )
    else:
        recommend = "stop"
        reason = f"oracle recovered {n_rec} {recovered}, lost {lost}."

    return {
        "n": n,
        "acc_zero": acc_z,
        "acc_oracle": acc_o,
        "acc_shuffled": acc_s,
        "acc_full": acc_f,
        "recovered_idx": recovered,
        "lost_idx": lost,
        "shuffled_recovered_idx": shuf_rec,
        "full_recovered_idx": full_rec,
        "n_recovered": n_rec,
        "recommend": recommend,
        "reason": reason,
    }


class CoefPredictor(nn.Module):
    """Tiny MLP: frozen-prefill features q [L, D] -> alpha [L, R] (or one layer)."""

    def __init__(self, d: int, n_layers: int, rank: int, hidden: int = 256):
        super().__init__()
        self.d = int(d)
        self.n_layers = int(n_layers)
        self.rank = int(rank)
        self.net = nn.Sequential(
            nn.Linear(self.d * self.n_layers, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_layers * self.rank),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: [B, L, D] or [L, D]
        if q.dim() == 2:
            q = q.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        b, ell, d = q.shape
        if ell != self.n_layers or d != self.d:
            raise ValueError(f"q shape {tuple(q.shape)} != [B,{self.n_layers},{self.d}]")
        alpha = self.net(q.reshape(b, ell * d)).view(b, ell, self.rank)
        return alpha[0] if squeeze else alpha


def stack_layer_dict(rows: Sequence[Dict[int, torch.Tensor]], layers: Sequence[int]) -> torch.Tensor:
    """List of {layer: [1,D] or [D]} -> [N, L, D]."""
    mats = []
    for row in rows:
        vecs = []
        for li in layers:
            t = row[int(li)]
            if t.dim() == 2:
                t = t[0]
            vecs.append(t.float().reshape(-1))
        mats.append(torch.stack(vecs, 0))
    return torch.stack(mats, 0)


def continue_forward(
    wrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past,
    *,
    deltas: Optional[Dict[int, torch.Tensor]] = None,
    record_layers: Optional[Sequence[int]] = None,
    injector: Optional[ResidualInjector] = None,
) -> Tuple[Any, Optional[Dict[int, torch.Tensor]]]:
    """One model forward that continues ``past`` (Judger prefill or 1-token bridge)."""
    from seal.latent_eval import pad_caches_left

    device = wrapper.device
    ids = input_ids.to(device)
    mask = attention_mask.to(device)
    rec = None
    own_injector = False
    if record_layers:
        rec = LayerRecorder(wrapper.model, record_layers)
        rec.register()
    if deltas:
        if injector is None:
            injector = ResidualInjector(deltas)
            injector.register(wrapper.model)
            own_injector = True
        else:
            injector.set_deltas(deltas)
            if not injector.handles:
                injector.register(wrapper.model)
        injector.enable()
    elif injector is not None:
        injector.disable()

    try:
        if past is None:
            out = wrapper.model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=True,
                return_dict=True,
            )
        else:
            past_b, past_mask, pmax = pad_caches_left([past], device)
            full_mask = torch.cat([past_mask, mask], dim=1)
            cache_position = torch.arange(
                pmax, pmax + ids.shape[1], dtype=torch.long, device=device,
            )
            out = wrapper.model(
                input_ids=ids,
                attention_mask=full_mask,
                past_key_values=past_b,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
        hid = rec.snapshot() if rec is not None else None
        return out.past_key_values, hid
    finally:
        if rec is not None:
            rec.remove()
        if injector is not None:
            injector.disable()
        if own_injector and injector is not None:
            injector.remove()


def bridge_ids(wrapper, n_bridge: int = 1) -> torch.Tensor:
    tok = bridge_token_id(wrapper.tokenizer)
    return torch.full((1, int(n_bridge)), tok, dtype=torch.long, device=wrapper.device)


def forward_bridge(
    wrapper,
    past,
    *,
    n_bridge: int = 1,
    deltas_by_slot: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
    record_layers: Optional[Sequence[int]] = None,
) -> Tuple[Any, List[Dict[int, torch.Tensor]]]:
    """Process ``n_bridge`` designated tokens, optionally injecting per slot."""
    tok = bridge_token_id(wrapper.tokenizer)
    hiddens: List[Dict[int, torch.Tensor]] = []
    cur = past
    for s in range(int(n_bridge)):
        ids = torch.tensor([[tok]], dtype=torch.long, device=wrapper.device)
        mask = torch.ones_like(ids)
        deltas = None
        if deltas_by_slot is not None and s < len(deltas_by_slot):
            deltas = deltas_by_slot[s]
        cur, hid = continue_forward(
            wrapper, ids, mask, cur, deltas=deltas, record_layers=record_layers,
        )
        if hid is not None:
            hiddens.append(hid)
    return cur, hiddens


def deltas_from_residual(
    residual_by_layer: Dict[int, torch.Tensor],
    *,
    layers: Sequence[int],
    bank: Optional[Dict[int, Dict[str, Any]]] = None,
    mode: str = "full",
) -> Dict[int, torch.Tensor]:
    """Build inject dict. ``bank`` is {layer: fit_pca} for mode=oracle_r."""
    out = {}
    for li in layers:
        li = int(li)
        d = residual_by_layer[li]
        if d.dim() == 2:
            d = d[0]
        d = d.float().reshape(-1)
        if mode in ("zero",):
            continue
        if mode in ("full", "oracle_full"):
            out[li] = d
        elif mode in ("r", "oracle_r", "projected"):
            if bank is None or li not in bank:
                raise KeyError(f"no PCA bank for layer {li}")
            out[li] = project_residual(d, bank[li])
        else:
            raise ValueError(f"unknown residual mode {mode}")
    return out


def derange(n: int, rng: torch.Generator) -> List[int]:
    if n <= 1:
        return list(range(n))
    perm = torch.randperm(n, generator=rng).tolist()
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    if n == 2 and perm[0] == 0:
        perm = [1, 0]
    return perm
