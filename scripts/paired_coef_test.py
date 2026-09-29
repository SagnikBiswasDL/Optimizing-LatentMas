"""Pre-registered paired test: does a steering coefficient reduce Judger tokens?

Written and committed BEFORE the 30-item cohort was collected. The point of
fixing it in advance is that the 2026-09-29 small sweep showed token count swings
by >150% of baseline between adjacent coefficients, so with enough freedom in the
analysis almost any conclusion can be extracted after the fact.

DESIGN
------
Paired by item: every item contributes one unsteered run and one steered run over
the same upstream tape, both greedy on the same decode path. Decoding is
deterministic, so each pair is an exact measurement of that item's response; the
variation being averaged out is across *items*, not across repeats.

The statistic is the log token ratio, log(steered / unsteered), because token
counts are multiplicative and right-skewed. Its median is reported back as a
percentage change.

CENSORING
---------
A run that hits the cap did not finish, so its token count is a lower bound and
its correctness is unknown-at-best. Censored pairs are therefore EXCLUDED from
the token test and reported separately as a count, because censoring is itself a
harm: an arm that saves tokens on the items it finishes while running more items
into the cap is worse, not better.

PRE-REGISTERED DECISION RULE
----------------------------
The steered arm is promoted only if ALL of:
  1. median paired saving >= MIN_SAVING (default 10%)
  2. two-sided paired permutation p < ALPHA (default 0.05) on log ratios
  3. accuracy not worse: steered_correct >= base_correct over the whole cohort
  4. censoring not worse: steered_censored <= base_censored
Otherwise it is rejected, and which clause failed is printed.

Usage:
  python scripts/paired_coef_test.py --coef 5
  python scripts/paired_coef_test.py --coef 5 --rows artifacts/aime_localize/rows.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


def load_pairs(rows_path: str, tag: str, coef: str, base_arm: str = "real",
               ) -> Dict[int, Dict[str, dict]]:
    """Group rows by item index into {'base': row, 'steer': row}."""
    base_method = f"{base_arm}__{tag}"
    steer_method = f"{base_arm}_seal{coef}__{tag}"
    out: Dict[int, Dict[str, dict]] = {}
    with open(rows_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            m = str(r.get("method", ""))
            which = "base" if m == base_method else ("steer" if m == steer_method else None)
            if which is None:
                continue
            out.setdefault(int(r["idx"]), {})[which] = r
    return {i: v for i, v in out.items() if "base" in v and "steer" in v}


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p at p0=0.5 (sign test)."""
    if n == 0:
        return 1.0
    def pmf(j: int) -> float:
        return math.comb(n, j) * 0.5 ** n
    obs = pmf(k)
    # Sum every outcome no more likely than the observed one.
    return min(1.0, sum(pmf(j) for j in range(n + 1) if pmf(j) <= obs + 1e-15))


def perm_two_sided(diffs: Sequence[float], n_mc: int = 200_000,
                   seed: int = 0) -> Tuple[float, bool]:
    """Two-sided paired permutation p on the mean, by sign flipping.

    Returns (p, exact). Enumerates all 2^n sign assignments when n is small
    enough, otherwise samples. The null is that each pair's difference was
    equally likely to come out with either sign.
    """
    n = len(diffs)
    if n == 0:
        return 1.0, True
    obs = abs(sum(diffs) / n)
    if n <= 20:
        hits = 0
        total = 0
        for signs in itertools.product((1, -1), repeat=n):
            total += 1
            if abs(sum(s * d for s, d in zip(signs, diffs)) / n) >= obs - 1e-15:
                hits += 1
        return hits / total, True
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_mc):
        if abs(sum(d if rng.random() < 0.5 else -d for d in diffs) / n) >= obs - 1e-15:
            hits += 1
    # Add-one smoothing so a p of exactly 0 is not reported from finite sampling.
    return (hits + 1) / (n_mc + 1), False


def mcnemar_two_sided(b: int, c: int) -> float:
    """Exact McNemar on discordant counts: b fixed-by-steering, c broken."""
    n = b + c
    return binom_two_sided(b, n) if n else 1.0


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="artifacts/aime_localize/rows.jsonl")
    ap.add_argument("--tag", default="b16k")
    ap.add_argument("--coef", default="5")
    ap.add_argument("--base_arm", default="real")
    ap.add_argument("--min_saving", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pairs = load_pairs(args.rows, args.tag, args.coef, args.base_arm)
    if not pairs:
        raise SystemExit(f"no complete pairs for coef {args.coef} at tag {args.tag}")

    base_cens = sum(1 for v in pairs.values() if not v["base"].get("eos"))
    steer_cens = sum(1 for v in pairs.values() if not v["steer"].get("eos"))
    base_ok = sum(1 for v in pairs.values() if v["base"].get("correct"))
    steer_ok = sum(1 for v in pairs.values() if v["steer"].get("correct"))

    # Primary set: both arms finished and both got it right, so the token counts
    # describe two complete correct solutions to the same problem.
    usable: List[Tuple[int, int, int, float]] = []
    for i, v in sorted(pairs.items()):
        b, s = v["base"], v["steer"]
        if not (b.get("eos") and s.get("eos")):
            continue
        if not (b.get("correct") and s.get("correct")):
            continue
        usable.append((i, int(b["tokens"]), int(s["tokens"]),
                       math.log(int(s["tokens"]) / int(b["tokens"]))))

    print(f"cohort: {len(pairs)} paired items at coef {args.coef} (tag {args.tag})")
    print(f"  censored (hit cap): baseline {base_cens}, steered {steer_cens}")
    print(f"  correct:            baseline {base_ok}, steered {steer_ok}")
    print(f"  usable for the token test (both finished, both correct): {len(usable)}")

    if usable:
        print("\n  item   baseline   steered    change")
        for i, bt, st, lr in usable:
            print(f"  {i:>4d} {bt:>10,} {st:>9,} {100 * (math.exp(lr) - 1):>+8.1f}%")

    logs = [lr for _, _, _, lr in usable]
    med_pct = 100 * (math.exp(median(logs)) - 1) if logs else float("nan")
    n_short = sum(1 for lr in logs if lr < 0)
    sign_p = binom_two_sided(n_short, len(logs))
    perm_p, exact = perm_two_sided(logs)

    # Discordant accuracy pairs: steering fixed it (b) or broke it (c).
    b_fix = sum(1 for v in pairs.values()
                if v["steer"].get("correct") and not v["base"].get("correct"))
    c_break = sum(1 for v in pairs.values()
                  if v["base"].get("correct") and not v["steer"].get("correct"))
    acc_p = mcnemar_two_sided(b_fix, c_break)

    print(f"\n  median change: {med_pct:+.1f}%  (n={len(logs)})")
    print(f"  shorter on {n_short}/{len(logs)} items; sign test p={sign_p:.4f}")
    print(f"  paired permutation p={perm_p:.4f} ({'exact' if exact else 'monte carlo'})")
    print(f"  accuracy: steering fixed {b_fix}, broke {c_break}, McNemar p={acc_p:.4f}")

    saving = -med_pct / 100.0
    clauses = {
        f"median saving >= {100 * args.min_saving:.0f}%": saving >= args.min_saving,
        f"permutation p < {args.alpha}": perm_p < args.alpha,
        "accuracy not worse": steer_ok >= base_ok,
        "censoring not worse": steer_cens <= base_cens,
    }
    print("\n  pre-registered decision rule:")
    for name, passed in clauses.items():
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
    verdict = "PROMOTE" if all(clauses.values()) else "REJECT"
    print(f"  => {verdict} coef {args.coef}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "coef": args.coef, "tag": args.tag, "n_pairs": len(pairs),
                "n_usable": len(usable), "median_pct": med_pct,
                "n_shorter": n_short, "sign_p": sign_p, "perm_p": perm_p,
                "perm_exact": exact, "base_censored": base_cens,
                "steer_censored": steer_cens, "base_correct": base_ok,
                "steer_correct": steer_ok, "acc_fixed": b_fix,
                "acc_broke": c_break, "mcnemar_p": acc_p,
                "clauses": {k: bool(v) for k, v in clauses.items()},
                "verdict": verdict,
                "items": [{"idx": i, "base_tokens": bt, "steer_tokens": st,
                           "pct": 100 * (math.exp(lr) - 1)}
                          for i, bt, st, lr in usable],
            }, fh, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
