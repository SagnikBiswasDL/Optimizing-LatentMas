#!/usr/bin/env python
"""What the latent channel actually buys on AIME24: tokens, not accuracy.

The accuracy ablation (scripts/channel_ablation.py) came back null: deleting the
entire channel costs one item out of thirty, CI [-9.9, +3.2] pp. That is the
metric the paper reports, and at n=30 it cannot resolve the effect it claims.

The Judger's *token count* is a different matter. It is a per-item continuous
measurement rather than a 30-way binomial, so the same 30 problems support a far
better powered test, and greedy decoding makes it exactly reproducible. This
script runs that test.

THE CHAIN
---------
  none -> shuf   attach *some* cache of similar size, from a DIFFERENT problem
  shuf -> real   make that cache the RIGHT problem's
  none -> real   both, i.e. the published configuration

If most of the saving appears in the first step, the channel's benefit is not
communication: a cache from an unrelated problem supplies the same number of
positions to attend over, so any improvement it produces cannot be information
about the problem being solved. If most appears in the second step, the channel
carries problem-specific content and localization is worth pursuing.

CENSORING, AND WHY THE PRIMARY TEST DROPS NOTHING
-------------------------------------------------
Runs that hit the token cap are censored: their true length is unknown, only
bounded below. The tempting fix -- keep the pairs where both arms terminated --
conditions on the outcome, which is the exact error §3.7 of the results doc
records us making before (items picked because a coefficient broke them).

The sign of a paired difference, though, is *identified under censoring*: if A
stops at 5000 and B is still going at the 16384 cap, B is longer, full stop. So
the primary test is an exact sign test over all items, discarding only pairs
where BOTH arms hit the cap and the order is genuinely unknown. Magnitudes need
completed runs and are reported as conditional estimates, with the
correctness-filtered version alongside to show the filter changes nothing.

WALL CLOCK, AND ONE CONFOUND WE CANNOT REMOVE
---------------------------------------------
Token counts understate the case for the channel, because the upstream roles
emit only k latent steps and no text, so they cost ~1s against a ~2min Judger.
End-to-end time is therefore reported too.

But `shuf` is length-matched to `real` only *on average*, not per item: item i
borrows item i+1's cache, which is a different length. That does not touch the
token comparison (tokens generated do not depend on how long the prefix is) but
it does contaminate `shuf`'s *time*, since prefill and per-step attention scale
with prefix length. This script prints the per-item mismatch so the confound is
visible in the output, and refuses to quote a time verdict for `shuf`.

Usage:
  python scripts/channel_token_cost.py --rows artifacts/aime_localize/rows.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple


def load(rows_path: str, tag: str) -> Dict[str, Dict[int, dict]]:
    by: Dict[str, Dict[int, dict]] = collections.defaultdict(dict)
    with open(rows_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            m = str(r.get("method", ""))
            if tag and not m.endswith(f"__{tag}"):
                continue
            arm = m[: -(len(tag) + 2)] if tag else m
            if "_seal" in arm:  # steering is a different experiment
                continue
            by[arm][int(r["idx"])] = r
    return dict(by)


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def sign_test(a: Dict[int, dict], b: Dict[int, dict]) -> Dict:
    """Exact sign test on tokens, valid under censoring, dropping no item.

    A pair is undetermined only when both arms hit the cap, since then neither
    true length is known. Every other pair has a known sign.
    """
    shared = sorted(set(a) & set(b))
    wins = losses = undet = 0
    for i in shared:
        both_capped = (not a[i].get("eos")) and (not b[i].get("eos"))
        if both_capped:
            undet += 1
            continue
        ta, tb = int(a[i]["tokens"]), int(b[i]["tokens"])
        if tb < ta:
            wins += 1
        elif tb > ta:
            losses += 1
        else:
            undet += 1
    return {"n_items": len(shared), "shorter": wins, "longer": losses,
            "undetermined": undet, "p": round(exact_sign_p(wins, losses), 5)}


def magnitude(a: Dict[int, dict], b: Dict[int, dict],
              require_correct: bool = False) -> Optional[Dict]:
    """Paired log token ratio on completed runs. Conditional by necessity."""
    use = [i for i in sorted(set(a) & set(b))
           if a[i].get("eos") and b[i].get("eos")
           and (not require_correct
                or (a[i].get("correct") and b[i].get("correct")))]
    if len(use) < 3:
        return None
    lr = [math.log(int(b[i]["tokens"]) / int(a[i]["tokens"])) for i in use]
    n = len(lr)
    mean = sum(lr) / n
    sd = (sum((x - mean) ** 2 for x in lr) / (n - 1)) ** 0.5
    se = sd / n ** 0.5
    sl = sorted(lr)
    med = sl[n // 2] if n % 2 else (sl[n // 2 - 1] + sl[n // 2]) / 2
    pct = lambda x: round(100 * (math.exp(x) - 1), 1)  # noqa: E731
    return {"n": n, "median_pct": pct(med), "mean_pct": pct(mean),
            "ci_pct": [pct(mean - 1.96 * se), pct(mean + 1.96 * se)]}


def total_s(r: dict) -> float:
    return float(r.get("upstream_s") or 0.0) + float(r.get("judger_s") or 0.0)


def time_magnitude(a: Dict[int, dict], b: Dict[int, dict]) -> Optional[Dict]:
    """End-to-end paired time ratio, upstream roles included.

    Restricted to completed runs for the same reason as tokens: a censored run's
    time is the time to reach the cap, not the time to answer.
    """
    use = [i for i in sorted(set(a) & set(b))
           if a[i].get("eos") and b[i].get("eos")]
    if len(use) < 3:
        return None
    lr = [math.log(total_s(b[i]) / total_s(a[i])) for i in use
          if total_s(a[i]) > 0 and total_s(b[i]) > 0]
    if len(lr) < 3:
        return None
    n = len(lr)
    mean = sum(lr) / n
    sd = (sum((x - mean) ** 2 for x in lr) / (n - 1)) ** 0.5
    se = sd / n ** 0.5
    sl = sorted(lr)
    med = sl[n // 2] if n % 2 else (sl[n // 2 - 1] + sl[n // 2]) / 2
    faster = sum(1 for x in lr if x < 0)
    pct = lambda x: round(100 * (math.exp(x) - 1), 1)  # noqa: E731
    return {"n": n, "median_pct": pct(med), "mean_pct": pct(mean),
            "ci_pct": [pct(mean - 1.96 * se), pct(mean + 1.96 * se)],
            "faster": faster, "p": round(exact_sign_p(faster, n - faster), 5)}


def length_mismatch(a: Dict[int, dict], b: Dict[int, dict]) -> Optional[Dict]:
    """Per-item cache-size gap, to expose a control matched only on average."""
    shared = [i for i in sorted(set(a) & set(b))]
    d = [abs(float(b[i].get("cache_mb") or 0.0) - float(a[i].get("cache_mb") or 0.0))
         for i in shared]
    if not d:
        return None
    mean_a = sum(float(a[i].get("cache_mb") or 0.0) for i in shared) / len(shared)
    mean_b = sum(float(b[i].get("cache_mb") or 0.0) for i in shared) / len(shared)
    return {"mean_mb_a": round(mean_a, 1), "mean_mb_b": round(mean_b, 1),
            "per_item_abs_gap_mean_mb": round(sum(d) / len(d), 1),
            "per_item_abs_gap_max_mb": round(max(d), 1)}


# `shuf` borrows another item's cache, so its prefix length is matched only in
# the mean. Token counts are unaffected; times are not.
TIME_CONFOUNDED = {"shuf"}


def step(by: Dict[str, Dict[int, dict]], a: str, b: str, label: str) -> Optional[Dict]:
    if a not in by or b not in by:
        print(f"{label}: needs both `{a}` and `{b}`; skipped")
        return None
    A, B = by[a], by[b]
    s = sign_test(A, B)
    m_all = magnitude(A, B, require_correct=False)
    m_ok = magnitude(A, B, require_correct=True)
    t = time_magnitude(A, B)
    lm = length_mismatch(A, B)

    print(f"\n{label}")
    print(f"  tokens, PRIMARY (exact sign test, all {s['n_items']} items, "
          f"nothing dropped on outcome)")
    print(f"      {s['shorter']} shorter / {s['longer']} longer / "
          f"{s['undetermined']} undetermined      p = {s['p']:.4f}")
    if m_all:
        print(f"  tokens, magnitude (both terminated, n={m_all['n']}): "
              f"median {m_all['median_pct']:+.1f}%  "
              f"CI {m_all['ci_pct']} %")
    if m_ok:
        print(f"  tokens, magnitude (also both correct, n={m_ok['n']}): "
              f"median {m_ok['median_pct']:+.1f}%  "
              f"CI {m_ok['ci_pct']} %")
    if t:
        note = ""
        if a in TIME_CONFOUNDED or b in TIME_CONFOUNDED:
            note = "   [CONFOUNDED: prefix length matched only on average]"
        print(f"  end-to-end time incl. upstream (n={t['n']}): "
              f"median {t['median_pct']:+.1f}%  CI {t['ci_pct']} %  "
              f"{t['faster']}/{t['n']} faster  p={t['p']:.4f}{note}")
    # Only meaningful when both arms carry a cache; against `none` the gap is
    # just the cache itself and says nothing about matching.
    if lm and min(lm["mean_mb_a"], lm["mean_mb_b"]) > 0 \
            and lm["per_item_abs_gap_mean_mb"] > 1.0:
        print(f"  cache size: {a} {lm['mean_mb_a']} MB vs {b} {lm['mean_mb_b']} MB "
              f"on average, but differing by {lm['per_item_abs_gap_mean_mb']} MB "
              f"per item (max {lm['per_item_abs_gap_max_mb']})")
    return {"from": a, "to": b, "sign_test": s, "tokens_all": m_all,
            "tokens_correct": m_ok, "time": t, "length_mismatch": lm,
            "time_confounded": bool({a, b} & TIME_CONFOUNDED)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="artifacts/aime_localize/rows.jsonl")
    ap.add_argument("--tag", default="b16k")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    by = load(args.rows, args.tag)
    print(f"arms available at tag `{args.tag}`: {sorted(by)}")
    for a in sorted(by):
        rs = list(by[a].values())
        cap = sum(1 for r in rs if not r.get("eos"))
        toks = sorted(int(r["tokens"]) for r in rs)
        print(f"  {a:<8} n={len(rs):<3} median {toks[len(toks) // 2]:>6} tokens, "
              f"{sum(1 for r in rs if r.get('correct'))}/{len(rs)} correct, "
              f"{cap} censored")

    print("\n" + "=" * 74)
    print("DOES THE LATENT CHANNEL SHORTEN THE ANSWER, AND IS IT THE CONTENT?")
    steps = [s for s in (
        step(by, "none", "shuf", "none -> shuf   (attach ANY cache, wrong problem)"),
        step(by, "shuf", "real", "shuf -> real   (make it the RIGHT problem's cache)"),
        step(by, "none", "real", "none -> real   (the published configuration)"),
    ) if s]

    # `single` isolates the prompt text, which also changes length.
    if "single" in by:
        steps += [s for s in (
            step(by, "single", "none",
                 "single -> none (prompt text only, no cache either side)"),
            step(by, "single", "real",
                 "single -> real (the paper's own comparison)"),
        ) if s]

    got = {(s["from"], s["to"]): s for s in steps}
    any_c, right_c = got.get(("none", "shuf")), got.get(("shuf", "real"))
    print("\n" + "=" * 74)
    print("READING")
    if not (any_c and right_c):
        print("  needs none, shuf and real to apportion the saving.")
    else:
        a_sig = any_c["sign_test"]["p"] < 0.05
        r_sig = right_c["sign_test"]["p"] < 0.05
        a_med = (any_c["tokens_all"] or {}).get("median_pct")
        r_med = (right_c["tokens_all"] or {}).get("median_pct")
        print(f"  attaching any cache at all:      {a_med:+.1f}% tokens, "
              f"p={any_c['sign_test']['p']:.4f}")
        print(f"  upgrading it to the right cache: {r_med:+.1f}% tokens, "
              f"p={right_c['sign_test']['p']:.4f}")
        if a_sig and not r_sig:
            print("  => The saving is real but is NOT the channel's content. A cache")
            print("     from an unrelated problem reproduces most of it, so what the")
            print("     Judger gains is positions to attend over, not communication.")
            print("     Caveat: a donor cache is still mathematical reasoning. To call")
            print("     it content-free needs a cache with no structure at all.")
        elif r_sig and not a_sig:
            print("  => The saving IS the channel's content: an unrelated cache does")
            print("     not reproduce it. Localization and compression are worth it.")
        elif a_sig and r_sig:
            print("  => Both steps contribute; report the split, do not round it to")
            print("     one story.")
        else:
            print("  => Neither step is resolved at this n. Quote the intervals.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"tag": args.tag, "steps": steps}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
