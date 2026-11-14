"""Query-aware loop selection: labels, decision rule, controls and statistics.

Notation (docs/looped_serving/LOOP_ROUTING.md):

    C[q, j]     1 if the vote of the four sampled answers at depth DEPTHS[j] is correct
    cost[q, j]  loop-tokens spent on question q at that depth (prompt and all answers)
    P[q, j]     a selector's estimate of C[q, j]

A policy is one depth index per question, fixed before the answer is written.
Everything here is numpy only, so the tests run on the repo's Python 3.9 venv.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEPTHS = (1, 2, 3, 4)


# ------------------------------------------------------------------ labels

def vote_key(samples: Sequence[Tuple[int, Optional[str]]]) -> Optional[str]:
    """Majority answer key over ``(sample_index, key)`` pairs.

    ``key`` is None for an unparseable or cut-off answer, which casts no vote.
    Ties go to the key whose first vote has the lowest sample index, the rule
    of ``es.LOWEST_CANDIDATE`` in the existing grader.
    """
    counts: Dict[str, int] = {}
    first: Dict[str, int] = {}
    for idx, key in samples:
        if key is None:
            continue
        counts[key] = counts.get(key, 0) + 1
        first[key] = min(first.get(key, idx), idx)
    if not counts:
        return None
    return min(counts, key=lambda k: (-counts[k], first[k]))


def vote_correct(samples: Sequence[Tuple[int, Optional[str], bool]]) -> bool:
    """``samples`` are ``(sample_index, key, correct)``; the vote is correct when
    the winning key's answers are graded correct. No vote counts as wrong."""
    win = vote_key([(i, k) for i, k, _ in samples])
    if win is None:
        return False
    return any(ok for _, k, ok in samples if k == win)


# ------------------------------------------------------------------ decision rule

def policy_from_probs(P: np.ndarray, tau: float) -> np.ndarray:
    """Smallest depth whose estimate is within ``tau`` of the best depth's."""
    best = P.max(axis=1, keepdims=True)
    ok = P >= best - tau
    return ok.argmax(axis=1)


def policy_from_scores(score: np.ndarray, cuts: Sequence[float]) -> np.ndarray:
    """Depth index = number of cut points the score exceeds (higher score, more loops)."""
    return np.searchsorted(np.sort(np.asarray(cuts)), score, side="left").clip(0, len(DEPTHS) - 1)


def evaluate(policy: np.ndarray, C: np.ndarray, cost: np.ndarray) -> Tuple[float, float]:
    rows = np.arange(len(policy))
    return float(C[rows, policy].mean()), float(cost[rows, policy].mean())


def depth_counts(policy: np.ndarray) -> List[int]:
    return [int((policy == j).sum()) for j in range(len(DEPTHS))]


# ------------------------------------------------------------------ controls

def matched_random(policy: np.ndarray, C: np.ndarray, cost: np.ndarray, n_perm: int = 1000,
                   seed: int = 0) -> Dict[str, float]:
    """The same depth counts assigned to random questions.

    ``p_value`` is the share of permutations whose accuracy is at least the
    policy's: a one-sided permutation test of "knowing the question helps".
    """
    rng = np.random.default_rng(seed)
    acc, cst = evaluate(policy, C, cost)
    accs = np.empty(n_perm)
    costs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(policy)
        accs[i], costs[i] = evaluate(perm, C, cost)
    return {"acc": float(accs.mean()), "cost": float(costs.mean()),
            "acc_lo": float(np.quantile(accs, 0.025)), "acc_hi": float(np.quantile(accs, 0.975)),
            "p_value": float((np.sum(accs >= acc - 1e-12) + 1) / (n_perm + 1))}


def mixture_frontier(acc_d: Sequence[float], cost_d: Sequence[float]) -> List[Tuple[float, float]]:
    """Upper concave hull of the fixed-depth points, as (cost, acc) sorted by cost.

    A question-independent policy sends each question to depth d with
    probability w_d; its expected accuracy and cost are linear in w, so the best
    such policy at a given cost lies on this hull.
    """
    pts = sorted(zip(cost_d, acc_d))
    hull: List[Tuple[float, float]] = []
    for c, a in pts:
        if hull and a <= hull[-1][1]:
            continue                                     # dearer and no better: never on the frontier
        while len(hull) >= 2:
            (c1, a1), (c2, a2) = hull[-2], hull[-1]
            if (a2 - a1) * (c - c1) <= (a - a1) * (c2 - c1):
                hull.pop()
            else:
                break
        hull.append((c, a))
    return hull


def best_mixture_acc(frontier: List[Tuple[float, float]], budget: float) -> float:
    """Best question-independent accuracy at mean cost <= budget."""
    if budget <= frontier[0][0]:
        return frontier[0][1]
    for (c1, a1), (c2, a2) in zip(frontier, frontier[1:]):
        if budget <= c2:
            return a1 + (a2 - a1) * (budget - c1) / (c2 - c1)
    return frontier[-1][1]


def oracle_policy(C: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Cheapest depth with a correct vote; the cheapest depth when none is correct."""
    big = cost.max() + 1.0
    masked = np.where(C > 0, cost, big + cost)
    return masked.argmin(axis=1)


# ------------------------------------------------------------------ statistics

def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Two-sided exact McNemar on paired 0/1 outcomes; ``only_a`` / ``only_b`` counts."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p": 1.0}
    k = min(n10, n01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"only_a": n10, "only_b": n01, "p": float(min(1.0, 2 * tail))}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean of a - b with a 95% percentile interval, resampling questions."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cohen_kappa(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, bool)
    y = np.asarray(y, bool)
    po = float((x == y).mean())
    px, py = float(x.mean()), float(y.mean())
    pe = px * py + (1 - px) * (1 - py)
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def kappa_interval(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, bool)
    y = np.asarray(y, bool)
    ks = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), size=len(x))
        k = cohen_kappa(x[i], y[i])
        if not math.isnan(k):
            ks.append(k)
    return cohen_kappa(x, y), float(np.quantile(ks, 0.025)), float(np.quantile(ks, 0.975))


# ------------------------------------------------------------------ pilot gate

def needs_next(C: np.ndarray, j: int) -> np.ndarray:
    """1 where the vote is wrong at DEPTHS[j] and right at DEPTHS[j + 1]."""
    return (C[:, j] == 0) & (C[:, j + 1] == 1)


def transfer_oracle(C_from: np.ndarray, cost_from: np.ndarray) -> np.ndarray:
    """Cheapest depth whose vote was correct in one run; the most accurate
    fixed depth of that run when none was. Scored on a replicate run, this is
    the best any router could learn from one run's labels."""
    best_fixed = int(C_from.mean(axis=0).argmax())
    pol = oracle_policy(C_from, cost_from)
    pol[C_from.max(axis=1) == 0] = best_fixed
    return pol


def transfer_margin(C_from, cost_from, C_to, cost_to) -> float:
    """Accuracy on the replicate run of the transferred oracle, minus the best
    question-independent mixture of that run at the same mean cost."""
    pol = transfer_oracle(C_from, cost_from)
    acc, cst = evaluate(pol, C_to, cost_to)
    fr = mixture_frontier(C_to.mean(axis=0), cost_to.mean(axis=0))
    return acc - best_mixture_acc(fr, cst)


def repeatability_gate(CA: np.ndarray, CB: np.ndarray, costA: np.ndarray, costB: np.ndarray,
                       steps: Iterable[int] = (0, 1), n_boot: int = 2000, seed: int = 0) -> Dict[str, object]:
    """The pilot gate of LOOP_ROUTING.md.

    Check 1: for each step j -> j+1, Cohen's kappa between the "needs the next
    loop" indicators of runs A and B, with a bootstrap interval that must
    exclude 0.
    Check 2: the oracle routing read off one run, scored on the other, beats
    the best question-independent mixture at the same cost. Averaged over
    both directions; the bootstrap resamples questions and must stay above 0.
    """
    out: Dict[str, object] = {"steps": {}}
    passed = True
    for j in steps:
        k, lo, hi = kappa_interval(needs_next(CA, j), needs_next(CB, j), n_boot, seed)
        ok = lo > 0
        passed &= ok
        out["steps"][f"{DEPTHS[j]}->{DEPTHS[j + 1]}"] = {
            "needs_next_A": int(needs_next(CA, j).sum()), "needs_next_B": int(needs_next(CB, j).sum()),
            "both": int((needs_next(CA, j) & needs_next(CB, j)).sum()),
            "kappa": k, "kappa_lo": lo, "kappa_hi": hi, "pass": bool(ok)}

    def margin(i):
        return 0.5 * (transfer_margin(CA[i], costA[i], CB[i], costB[i])
                      + transfer_margin(CB[i], costB[i], CA[i], costA[i]))

    rng = np.random.default_rng(seed)
    n = len(CA)
    point = margin(np.arange(n))
    boots = np.array([margin(rng.integers(0, n, size=n)) for _ in range(n_boot)])
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    polA = transfer_oracle(CA, costA)
    accB, costB_mean = evaluate(polA, CB, costB)
    out["transfer"] = {"margin": float(point), "margin_lo": lo, "margin_hi": hi, "pass": bool(lo > 0),
                       "A_to_B_acc": accB, "A_to_B_cost": costB_mean, "A_to_B_depths": depth_counts(polA)}
    passed &= lo > 0
    out["pass"] = bool(passed)
    return out
