"""Task-banked KV prefixes for LatentMAS.

The method: run Planner/Critic/Refiner once on a few *donor* items per problem
type, save those relays, and at test time skip the silent agents. The Judger
reads a precomputed prefix keyed by task (or MATH subject), not by the test
item. That is the claim: the latent channel is a type-level decoding scaffold,
not an instance-specific message.

No eviction. The stored object is a full-length, on-manifold (or
stat-matched synthetic) cache.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

try:
    from transformers.cache_utils import Cache, DynamicCache
except Exception:  # pragma: no cover
    Cache = None
    DynamicCache = None


def to_legacy(past) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    if past is None:
        return None
    if Cache is not None and isinstance(past, Cache):
        return [tuple(layer) for layer in past.to_legacy_cache()]
    return [tuple(layer) for layer in past]


def from_legacy(layers, like=None):
    tup = tuple((k, v) for (k, v) in layers)
    if like is not None and Cache is not None and isinstance(like, Cache):
        return like.__class__.from_legacy_cache(tup)
    if DynamicCache is not None:
        return DynamicCache.from_legacy_cache(tup)
    return tup


def num_positions(past) -> int:
    legacy = to_legacy(past) if not isinstance(past, list) else past
    if not legacy:
        return 0
    return int(legacy[0][0].shape[-2])


def kv_mb(past) -> float:
    legacy = to_legacy(past) if not isinstance(past, list) else past
    if not legacy:
        return 0.0
    n = 0
    for k, v in legacy:
        n += k.numel() * k.element_size()
        n += v.numel() * v.element_size()
    return n / (1024.0 * 1024.0)


def clone_legacy(legacy):
    return [(k.detach().contiguous().clone(), v.detach().contiguous().clone())
            for (k, v) in legacy]


def to_cpu_legacy(past):
    legacy = to_legacy(past)
    if legacy is None:
        return None
    return [(k.detach().to("cpu").contiguous(), v.detach().to("cpu").contiguous())
            for (k, v) in legacy]


def to_device(past, device, dtype=None):
    legacy = to_legacy(past) if not isinstance(past, list) else past
    if legacy is None:
        return None
    out = []
    for k, v in legacy:
        kk = k.to(device)
        vv = v.to(device)
        if dtype is not None:
            kk = kk.to(dtype)
            vv = vv.to(dtype)
        out.append((kk.contiguous(), vv.contiguous()))
    return from_legacy(out)


def clone_past(past):
    """Fresh cache object. generate() mutates past_key_values in place."""
    if past is None:
        return None
    legacy = to_legacy(past) if not isinstance(past, list) else past
    return from_legacy(clone_legacy(legacy))


def _slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unknown"


def problem_type(item: Dict, task: str, granularity: str = "task") -> str:
    """Lookup key for the bank. ``task`` or ``task:subject``."""
    task = (task or "unknown").strip().lower()
    if granularity == "subtype":
        sub = item.get("subject") or item.get("type") or item.get("problem_type") or ""
        sub = str(sub).strip()
        if sub:
            return f"{task}:{_slug(sub)}"
    return task


def parent_task(key: str) -> str:
    return key.split(":", 1)[0] if ":" in key else key


def pool_stats(legacies: Sequence[List[Tuple[torch.Tensor, torch.Tensor]]]):
    """Per-(layer, K/V, head, channel) mean/std over positions, pooled across donors."""
    if not legacies:
        raise ValueError("pool_stats: empty donor list")
    acc = None
    lengths = []
    for legacy in legacies:
        lengths.append(int(legacy[0][0].shape[-2]))
        if acc is None:
            acc = [[{"s": torch.zeros(1, t.shape[1], 1, t.shape[3]),
                     "ss": torch.zeros(1, t.shape[1], 1, t.shape[3]),
                     "cnt": 0}
                    for t in (K, V)] for (K, V) in legacy]
        for li, (K, V) in enumerate(legacy):
            for ti, t in enumerate((K, V)):
                tf = t.float().cpu()
                acc[li][ti]["s"] += tf.sum(dim=2, keepdim=True)
                acc[li][ti]["ss"] += (tf * tf).sum(dim=2, keepdim=True)
                acc[li][ti]["cnt"] += int(tf.shape[2])
    stats = []
    for li in range(len(acc)):
        pair = []
        for ti in range(2):
            cnt = max(acc[li][ti]["cnt"], 1)
            mean = acc[li][ti]["s"] / cnt
            var = (acc[li][ti]["ss"] / cnt - mean * mean).clamp_min(0)
            pair.append((mean, var.sqrt()))
        stats.append(pair)
    L = int(sorted(lengths)[len(lengths) // 2])
    return stats, L


def sample_synth(stats, L: int, *, dtype, device, seed: int):
    """I.i.d. Gaussian positions from pooled (head, channel) moments.

    Destroys sequence structure: every position is an independent draw from the
    same marginal. Fine as a MedQA-style *mode* prefix; weak as a math
    trajectory. Prefer ``mean_aligned`` for math.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    legacy = []
    for pair in stats:
        tens = []
        for mean, std in pair:
            H, D = mean.shape[1], mean.shape[3]
            noise = torch.randn(1, H, int(L), D, generator=g)
            tens.append((noise * std + mean).to(dtype=dtype))
        legacy.append((tens[0], tens[1]))
    past = from_legacy(legacy)
    return to_device(past, device, dtype=dtype)


def _right_align(t: torch.Tensor, L: int) -> torch.Tensor:
    """Keep the last L positions (latent-step tail); left-pad if shorter."""
    ln = int(t.shape[2])
    if ln == L:
        return t
    if ln > L:
        return t[:, :, ln - L :, :]
    pad = torch.zeros(t.shape[0], t.shape[1], L - ln, t.shape[3], dtype=t.dtype, device=t.device)
    return torch.cat([pad, t], dim=2)


def mean_aligned(legacies: Sequence[List[Tuple[torch.Tensor, torch.Tensor]]],
                 L: int = 0, *, dtype=None, device="cpu"):
    """Position-wise mean of donor caches, right-aligned to length L.

    This is the literal average cache: donor axis only, sequence kept.
    Latent steps sit at the *end* of each relay, so we align on the tail.
    """
    if not legacies:
        raise ValueError("mean_aligned: empty donor list")
    lengths = [int(lg[0][0].shape[-2]) for lg in legacies]
    if L is None or int(L) <= 0:
        L = int(sorted(lengths)[len(lengths) // 2])
    L = int(L)
    n_layers = len(legacies[0])
    out = []
    dt = dtype if dtype is not None else legacies[0][0][0].dtype
    for li in range(n_layers):
        ks, vs = [], []
        for lg in legacies:
            k, v = lg[li]
            ks.append(_right_align(k.float().cpu(), L))
            vs.append(_right_align(v.float().cpu(), L))
        k_mean = torch.stack(ks, 0).mean(0).to(dtype=dt)
        v_mean = torch.stack(vs, 0).mean(0).to(dtype=dt)
        out.append((k_mean.contiguous(), v_mean.contiguous()))
    return to_device(from_legacy(out), device, dtype=dt)


def prototype_donor(legacies: Sequence[List[Tuple[torch.Tensor, torch.Tensor]]],
                     L: int = 0, *, dtype=None, device="cpu"):
    """Real donor cache closest (L2) to the position-wise mean.

    On-manifold: one actual Planner/Critic/Refiner trajectory, still
    question-independent at test time.
    """
    mean = mean_aligned(legacies, L=L, dtype=torch.float32, device="cpu")
    mean_lg = to_legacy(mean)
    L = int(mean_lg[0][0].shape[-2])
    best_i, best_d = 0, None
    for i, lg in enumerate(legacies):
        dist = 0.0
        for li, (k, v) in enumerate(lg):
            mk, mv = mean_lg[li]
            kk = _right_align(k.float().cpu(), L)
            vv = _right_align(v.float().cpu(), L)
            dist += float((kk - mk).pow(2).mean() + (vv - mv).pow(2).mean())
        if best_d is None or dist < best_d:
            best_i, best_d = i, dist
    lg = [(_right_align(k.cpu(), L), _right_align(v.cpu(), L)) for (k, v) in legacies[best_i]]
    dt = dtype if dtype is not None else legacies[0][0][0].dtype
    return to_device(from_legacy(lg), device, dtype=dt), int(best_i), float(best_d or 0.0)


def mean_restore_latents(donor_steps: Sequence[torch.Tensor]) -> torch.Tensor:
    """Average donor latent embeddings per step, restore median donor L2.

    Each tensor is ``[K, D]`` (batch squeezed). Averaging in embedding space is
    the math construction: the model later *writes* KV from these means.
    """
    if not donor_steps:
        raise ValueError("mean_restore_latents: empty donor list")
    x = torch.stack([t.detach().float().cpu() for t in donor_steps], 0)
    if x.dim() == 4:
        x = x.squeeze(2)
    if x.dim() != 3:
        raise ValueError(f"mean_restore_latents: expected [N,K,D], got {tuple(x.shape)}")
    mu = x.mean(0)
    med = x.norm(dim=-1).median(dim=0).values
    scale = med / mu.norm(dim=-1).clamp_min(1e-6)
    return (mu * scale.unsqueeze(-1)).contiguous()


def medoid_latent_index(donor_latents: Dict[str, Sequence[torch.Tensor]],
                        roles: Sequence[str]) -> Tuple[int, float]:
    """Donor whose concatenated latent tape is closest to the mean tape.

    AIME construction: do not average contest thoughts. Return the index of
    one real on-manifold trajectory (the medoid).
    """
    n = None
    for role in roles:
        steps = donor_latents.get(role) or []
        if not steps:
            raise ValueError(f"medoid_latent_index: no latents for role={role}")
        if n is None:
            n = len(steps)
        elif len(steps) != n:
            raise ValueError("medoid_latent_index: donor counts differ across roles")
    vecs = []
    for i in range(int(n)):
        chunks = []
        for role in roles:
            t = donor_latents[role][i].detach().float().cpu().reshape(-1)
            chunks.append(t)
        vecs.append(torch.cat(chunks, 0))
    x = torch.stack(vecs, 0)
    mu = x.mean(0)
    dist = (x - mu).pow(2).sum(-1)
    i = int(dist.argmin().item())
    return i, float(dist[i].sqrt().item())


class CacheBank:
    """One reusable KV prefix per problem-type key, plus optional synth stats."""

    def __init__(self, meta: Optional[Dict] = None, entries: Optional[Dict[str, Dict]] = None):
        self.meta: Dict[str, Any] = dict(meta or {})
        self.entries: Dict[str, Dict[str, Any]] = dict(entries or {})

    def keys(self) -> List[str]:
        return sorted(self.entries)

    def resolve(self, key: str) -> str:
        if key in self.entries:
            return key
        parent = parent_task(key)
        if parent in self.entries:
            return parent
        raise KeyError(f"no bank for {key!r}; have {self.keys()}")

    def n_pos(self, key: str) -> int:
        e = self.entries[self.resolve(key)]
        return int(e.get("n_pos") or num_positions(e["legacy"]))

    def donor_cache(self, key: str, device, dtype=None):
        e = self.entries[self.resolve(key)]
        return to_device(e["legacy"], device, dtype=dtype)

    def synth_cache(self, key: str, device, dtype, seed: int, length: int = 0):
        e = self.entries[self.resolve(key)]
        stats = e.get("stats")
        if stats is None:
            raise KeyError(f"bank {key!r} has no pooled stats (build with n_donors>=1)")
        L = int(length) if length and length > 0 else int(e.get("synth_len") or e.get("n_pos") or 0)
        if L <= 0:
            L = num_positions(e["legacy"])
        return sample_synth(stats, L, dtype=dtype, device=device, seed=seed)

    def add(self, key: str, *, legacy, donors: List[Dict], stats=None, synth_len: int = 0,
            extra: Optional[Dict] = None):
        layers = to_legacy(legacy) if not isinstance(legacy, list) else legacy
        cpu = [(k.detach().to("cpu").contiguous(), v.detach().to("cpu").contiguous())
               for (k, v) in layers]
        entry = {
            "key": key,
            "legacy": cpu,
            "n_pos": num_positions(cpu),
            "mb": kv_mb(cpu),
            "n_donors": len(donors),
            "donors": donors,
            "stats": stats,
            "synth_len": int(synth_len) or num_positions(cpu),
        }
        if extra:
            entry.update(extra)
        self.entries[key] = entry

    def save(self, path: str) -> None:
        payload = {"meta": self.meta, "entries": self.entries}
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "CacheBank":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "entries" not in payload:
            raise ValueError(f"not a CacheBank file: {path}")
        return cls(meta=payload.get("meta") or {}, entries=payload["entries"])
