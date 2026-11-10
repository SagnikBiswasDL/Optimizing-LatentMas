"""Paired accuracy comparisons with uncertainty."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from discordant counts b (A only) and c (B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a: Sequence[bool], b: Sequence[bool], *, iters: int = 10000, seed: int = 0) -> Dict:
    """Accuracy difference ``a - b`` on the same items: bootstrap CI and McNemar."""
    if len(a) != len(b):
        raise ValueError("paired arms must cover the same items")
    n = len(a)
    d = [int(x) - int(y) for x, y in zip(a, b)]
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return {
        "n": n,
        "acc_a": round(sum(a) / n, 4),
        "acc_b": round(sum(b) / n, 4),
        "diff": round(sum(d) / n, 4),
        "ci95": [round(boots[int(0.025 * iters)], 4), round(boots[int(0.975 * iters) - 1], 4)],
        "only_a": only_a,
        "only_b": only_b,
        "mcnemar_p": round(mcnemar_exact(only_a, only_b), 4),
    }


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")
