"""What does the latent channel between LatentMAS agents actually carry?

Pre-registered before the arms were run.

LatentMAS has the Planner, Critic and Refiner write a shared KV cache that only
the Judger reads. The claim is that passing latents beats passing text. Nobody has
asked what is *in* that channel, which is the algorithmic question: the Judger
currently receives all three roles' KV unconditionally (~120 MB/problem here), and
if most of it is inert then the channel is mostly overhead.

THE ARMS THAT DECIDE IT
-----------------------
  real   all three roles. The published configuration.
  none   no cache at all; the Judger solves from the question alone. The floor.
  shuf   *another problem's* cache, same size, same structure. The placebo.
  c1/c2/c3/c12/c23   single roles and pairs, for localization.

`shuf` is the arm that makes this an experiment rather than an ablation. A cache
from a different problem supplies exactly as many positions to attend over, so if
it performs like `real`, the benefit is not problem-specific information -- it is a
length or attention-sink effect, and that is a finding about latent communication
rather than about this implementation.

PRE-REGISTERED READING
----------------------
Everything is paired per item against `real`, tested with exact McNemar on the
discordant pairs, and reported with a confidence interval on the paired
difference. In order:

  1. real vs none. If indistinguishable, the latent channel buys no measurable
     accuracy on this task at this n, and nothing below matters.
  2. real vs shuf, given that real > none:
       shuf ~ none  -> the channel carries problem-specific information. Proceed to
                       localization, and KV savings are a real optimization.
       shuf ~ real  -> the channel's benefit is NOT its content. The strong claim,
                       and the one that would need the most defending.
       in between   -> partial; report as such and do not round to either story.
  3. Localization, only if (2) says content matters: the smallest role subset whose
     interval overlaps `real`, priced in KV bytes.

POWER, STATED UP FRONT
----------------------
With 30 paired items this design detects *drops* far better than it establishes
*equivalence*. A "shuf ~ real" conclusion is therefore reported as an interval, not
as a null, and the interval will be wide. Read the bound, not the point estimate.

Usage:
  python scripts/channel_ablation.py --arms none,shuf,c3
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

KV_KB_PER_POSITION = 160.0  # measured from the pipeline's cache_mb field


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    def pmf(j: int) -> float:
        return math.comb(n, j) * 0.5 ** n
    obs = pmf(k)
    return min(1.0, sum(pmf(j) for j in range(n + 1) if pmf(j) <= obs + 1e-15))


def wilson_diff_ci(b: int, c: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """CI for the paired difference in proportions (b-c)/n, Agresti-Min style.

    b = items the arm solves and `real` does not, c = the reverse. Only discordant
    pairs carry information about the difference.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    d = (b - c) / n
    # Variance of the paired difference, with a small-sample floor so a zero
    # discordant count does not report a zero-width interval.
    var = (b + c - (b - c) ** 2 / n) / (n * n)
    se = math.sqrt(max(var, 1.0 / (n * n)))
    return (max(-1.0, d - z * se), min(1.0, d + z * se))


def load(rows_path: str, tag: str) -> Dict[str, Dict[int, dict]]:
    by_arm: Dict[str, Dict[int, dict]] = {}
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
            # SEAL arms are a different experiment; keep the channel views only.
            if "_seal" in arm:
                continue
            by_arm.setdefault(arm, {})[int(r["idx"])] = r
    return by_arm


def kv_mb(rows: Sequence[dict]) -> Optional[float]:
    vals = [float(r["cache_mb"]) for r in rows if r.get("cache_mb")]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def compare(base: Dict[int, dict], arm: Dict[int, dict], name: str) -> Dict:
    shared = sorted(set(base) & set(arm))
    b = sum(1 for i in shared
            if arm[i].get("correct") and not base[i].get("correct"))
    c = sum(1 for i in shared
            if base[i].get("correct") and not arm[i].get("correct"))
    n = len(shared)
    base_ok = sum(1 for i in shared if base[i].get("correct"))
    arm_ok = sum(1 for i in shared if arm[i].get("correct"))
    lo, hi = wilson_diff_ci(b, c, n)
    toks = [int(arm[i]["tokens"]) for i in shared if arm[i].get("eos")]
    return {
        "arm": name, "n_paired": n,
        "real_correct": base_ok, "arm_correct": arm_ok,
        "gained": b, "lost": c,
        "delta_items": arm_ok - base_ok,
        "delta_pct_ci": [round(100 * lo, 1), round(100 * hi, 1)],
        "mcnemar_p": round(binom_two_sided(b, b + c), 4),
        "censored": sum(1 for i in shared if not arm[i].get("eos")),
        "mean_tokens_finished": round(sum(toks) / len(toks), 1) if toks else None,
        "kv_mb": round(kv_mb([arm[i] for i in shared]), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="artifacts/aime_localize/rows.jsonl")
    ap.add_argument("--tag", default="b16k")
    ap.add_argument("--base", default="real")
    ap.add_argument("--arms", default="none,shuf,c1,c2,c3,c12,c23")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    by_arm = load(args.rows, args.tag)
    if args.base not in by_arm:
        raise SystemExit(f"baseline arm {args.base} not found at tag {args.tag}; "
                         f"have {sorted(by_arm)}")
    base = by_arm[args.base]
    base_kv = kv_mb(list(base.values()))
    print(f"baseline `{args.base}`: {sum(1 for r in base.values() if r.get('correct'))}"
          f"/{len(base)} correct, {base_kv:.1f} MB KV passed to the Judger")

    results: List[Dict] = []
    print(f"\n{'arm':<6} {'n':>3} {'correct':>9} {'delta':>6} {'95% CI (pp)':>16} "
          f"{'McNemar':>8} {'KV MB':>7} {'KV %':>6} {'cens':>5}")
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if name not in by_arm:
            print(f"{name:<6}   -- not run --")
            continue
        r = compare(base, by_arm[name], name)
        results.append(r)
        pct = 100 * r["kv_mb"] / base_kv if base_kv else 0.0
        print(f"{name:<6} {r['n_paired']:>3} {r['arm_correct']:>4}/{r['n_paired']:<4} "
              f"{r['delta_items']:>+6} "
              f"{str(r['delta_pct_ci']):>16} {r['mcnemar_p']:>8.4f} "
              f"{r['kv_mb']:>7.1f} {pct:>5.0f}% {r['censored']:>5}")

    print("\n(delta is in items out of n; CI is on the paired accuracy difference "
          "in percentage points)")

    got = {r["arm"]: r for r in results}
    print("\n" + "=" * 70)
    print("PRE-REGISTERED READING")
    none_r, shuf_r = got.get("none"), got.get("shuf")
    if not none_r:
        print("  `none` not run — step 1 undecided, so nothing below is interpretable.")
    else:
        sig = none_r["mcnemar_p"] < 0.05
        print(f"  1. real vs none: {none_r['delta_items']:+d} items, "
              f"CI {none_r['delta_pct_ci']} pp, p={none_r['mcnemar_p']:.4f}")
        if not sig:
            print("     => The latent channel does not measurably change accuracy "
                  "at this n. Read the CI for what size of effect is still "
                  "possible; this is the result that most needs more items.")
        else:
            print("     => The latent channel does change accuracy. Step 2 applies.")
        if shuf_r:
            print(f"  2. real vs shuf: {shuf_r['delta_items']:+d} items, "
                  f"CI {shuf_r['delta_pct_ci']} pp, p={shuf_r['mcnemar_p']:.4f}")
            if sig:
                near_none = abs(shuf_r["delta_items"] - none_r["delta_items"]) <= 1
                near_real = shuf_r["mcnemar_p"] >= 0.05
                if near_none and not near_real:
                    print("     => shuf behaves like none: the channel carries "
                          "problem-specific information. Localization and KV "
                          "savings are worth pursuing.")
                elif near_real and not near_none:
                    print("     => shuf behaves like real: the benefit is NOT the "
                          "channel's content. Treat as the strong claim and "
                          "defend it (more items, and a length-matched control).")
                else:
                    print("     => Partial: shuf sits between none and real. "
                          "Report the interval; do not round to either story.")
            else:
                print("     (step 1 undecided, so this is descriptive only)")
    loc = [got[a] for a in ("c1", "c2", "c3", "c12", "c23") if a in got]
    if loc:
        ok = [r for r in loc if r["mcnemar_p"] >= 0.05]
        print(f"  3. localization: {len(loc)} subsets run; "
              f"{len(ok)} indistinguishable from real")
        if ok:
            best = min(ok, key=lambda r: r["kv_mb"])
            print(f"     cheapest such subset: {best['arm']} at {best['kv_mb']:.1f} MB "
                  f"({100 * best['kv_mb'] / base_kv:.0f}% of full), "
                  f"{best['delta_items']:+d} items, CI {best['delta_pct_ci']} pp")
            print("     (an interval overlapping zero is not proof of equivalence; "
                  "the bound is what to quote)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"base": args.base, "tag": args.tag,
                       "base_correct": sum(1 for r in base.values()
                                           if r.get("correct")),
                       "base_n": len(base), "base_kv_mb": base_kv,
                       "arms": results}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
