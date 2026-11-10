#!/usr/bin/env python3
"""Analysis for LoopDelta stage 0 (numpy only).

Stage 0a (prefill signal vs difficulty):
  - Spearman of each signal with MATH level, raw and partial on prompt length
    and subject (rank residualisation); level-1 vs level-5 AUROC
  - 5-fold CV: does adding the signal to length + subject improve prediction of
    level? (Spearman of out-of-fold predictions with level)

Stage 0b (needed depth and vote weighting), if --stage0b is given:
  - accuracy and loop-tokens per depth: greedy, mean sample, majority vote,
    Δ-weighted vote, log-prob-weighted vote
  - min sufficient depth per question vs the planner signal (joined from 0a)
  - does a solver's late-loop Δ pick its correct samples? pooled AUROC and the
    within-question pairwise AUROC (the quantity vote weighting relies on),
    against mean token log-prob
  - adaptive-depth policies fitted on even idx and evaluated on odd idx:
    planner signal, prompt length, random, oracle; vs fixed depth

The direction "larger Δ = harder = deeper" is the proposal's and is fixed in
advance; the reversed direction is reported separately, as a diagnostic only.

    python scripts/loopdelta_analyze.py --stage0a artifacts/loopdelta/stage0a_r4 \\
        --stage0b artifacts/loopdelta/stage0b --out artifacts/loopdelta/analysis
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped.harness import read_jsonl, write_json  # noqa: E402

PLANNER_SIGNALS = {
    "delta_auc": lambda r: r["delta_auc"],
    "delta_first": lambda r: r["delta"][0],
    "delta_last": lambda r: r["delta_last"],
    "delta_auc_lastpos": lambda r: r["delta_auc_lastpos"],
    "h_delta_sum": lambda r: float(np.nansum(r["h_delta"])),
    "kl_consec_sum": lambda r: float(np.nansum(r["kl_consec"])),
    "entropy_last_loop": lambda r: r["entropy"][-1],
    "gate_expected_exit": lambda r: r["gate_expected_exit"],
    "prompt_tokens": lambda r: float(r["prompt_tokens"]),
}


# ------------------------------------------------------------------ statistics

def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def residualize(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def partial_spearman(a, b, covars):
    """Spearman of a and b after regressing the ranks of both on ``covars``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & np.all(np.isfinite(covars), axis=1)
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ok.sum() < 3 or ra.std() == 0 or rb.std() == 0:
        return float("nan")
    C = covars[ok]
    return float(np.corrcoef(residualize(ra, C), residualize(rb, C))[0, 1])


def auroc(score, label):
    """P(score of a positive > score of a negative), ties count half."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s) + 1.0                                # 1-based ranks, ties averaged
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_ci(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else None


def covariates(rows):
    subjects = sorted({r["subject"] for r in rows})
    length = np.array([rankdata([r["prompt_tokens"] for r in rows])]).T
    onehot = np.array([[1.0 if r["subject"] == s else 0.0 for s in subjects[1:]] for r in rows])
    return np.column_stack([length, onehot]) if onehot.size else length


def cv_level_prediction(rows, signal_vals, k=5, seed=0):
    """Out-of-fold Spearman(level, prediction) for length+subject with and without the signal."""
    y = np.array([r["level"] for r in rows], float)
    base = covariates(rows)
    sig = rankdata(signal_vals)[:, None]
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    preds = {"length_subject": np.zeros(len(rows)), "length_subject_signal": np.zeros(len(rows))}
    for f in folds:
        test = np.array(f)
        train = np.array(sorted(set(idx) - set(f)))
        for name, X in (("length_subject", base), ("length_subject_signal", np.column_stack([base, sig]))):
            X1 = np.column_stack([np.ones(len(rows)), X])
            beta, *_ = np.linalg.lstsq(X1[train], y[train], rcond=None)
            preds[name][test] = X1[test] @ beta
    return {k_: round(spearman(v, y), 4) for k_, v in preds.items()}


# ------------------------------------------------------------------ stage 0a

def analyze_0a(rows):
    covs = covariates(rows)
    levels = np.array([r["level"] for r in rows])
    out = {"n": len(rows), "per_level_n": dict(Counter(int(v) for v in levels)), "signals": {}}
    for name, fn in PLANNER_SIGNALS.items():
        v = np.array([fn(r) for r in rows], float)
        n = len(rows)
        rec = {
            "spearman_level": round(spearman(v, levels), 4),
            "spearman_level_ci95": bootstrap_ci(lambda i: spearman(v[i], levels[i]), n),
            "mean_by_level": {int(lv): round(float(np.nanmean(v[levels == lv])), 6) for lv in range(1, 6)},
            "auroc_l5_vs_l1": round(auroc(v[(levels == 1) | (levels == 5)], levels[(levels == 1) | (levels == 5)] == 5), 4),
        }
        if name != "prompt_tokens":
            rec["partial_spearman_level_given_length_subject"] = round(partial_spearman(v, levels, covs), 4)
            rec["partial_ci95"] = bootstrap_ci(lambda i: partial_spearman(v[i], levels[i], covs[i]), n, B=1000)
            rec["cv_level_spearman"] = cv_level_prediction(rows, v)
        out["signals"][name] = rec
    if rows and len(rows[0]["delta"]):
        traj = np.array([r["delta"] for r in rows], float)
        out["mean_delta_trajectory_by_level"] = {
            int(lv): [round(float(x), 6) for x in np.nanmean(traj[levels == lv], 0)] for lv in range(1, 6)}
    return out


# ------------------------------------------------------------------ stage 0b

def vote(samples, weight=None):
    tally = defaultdict(float)
    for s in samples:
        if s["key"] is None:
            continue
        tally[s["key"]] += 1.0 if weight is None else weight(s)
    if not tally:
        return None
    best = max(tally.values())
    for s in samples:                                   # deterministic tie-break: first sample's answer
        if s["key"] is not None and tally[s["key"]] == best:
            return s
    return None


def load_0b(path):
    rows = []
    for f in sorted(os.listdir(path)):
        if f.startswith("scored_r") and f.endswith(".jsonl"):
            rows += read_jsonl(os.path.join(path, f))
    by = defaultdict(lambda: defaultdict(list))         # idx -> depth -> rows
    for r in rows:
        by[r["idx"]][r["depth"]].append(r)
    return rows, by


def per_question(by, depths, c_delta):
    """Per question and depth: greedy/vote correctness and loop-token cost."""
    out = {}
    for idx, dd in by.items():
        rec = {}
        for r in depths:
            rs = dd.get(r, [])
            greedy = [x for x in rs if x["sample"] == -1]
            samp = [x for x in rs if x["sample"] >= 0]
            if not greedy or not samp:
                continue
            g = greedy[0]
            v_un = vote(samp)
            v_d = vote(samp, (lambda s: 1.0 / (s.get("delta_last", float("nan")) + c_delta)
                              if math.isfinite(s.get("delta_last", float("nan"))) else 1.0)) if r >= 2 else v_un
            v_lp = vote(samp, lambda s: math.exp(s.get("mean_logprob", -10.0)))
            rec[r] = {
                "level": g["level"], "prompt_tokens": g["prompt_tokens"],
                "greedy_ok": g["correct"], "greedy_cost": g["loop_tokens"],
                "sample_acc": float(np.mean([x["correct"] for x in samp])),
                "vote_ok": bool(v_un and v_un["correct"]),
                "vote_delta_ok": bool(v_d and v_d["correct"]),
                "vote_logprob_ok": bool(v_lp and v_lp["correct"]),
                "vote_cost": float(sum(x["loop_tokens"] for x in samp)),
                "slowest_gen": max(x["gen_tokens"] for x in samp),
                "greedy_gen": g["gen_tokens"],
            }
        if rec:
            out[idx] = rec
    return out


def analyze_0b(rows, by, depths, planner):
    dl = [r["delta_last"] for r in rows if r["sample"] >= 0 and math.isfinite(r.get("delta_last", float("nan")))]
    c_delta = float(np.median(dl)) if dl else 1.0
    pq = per_question(by, depths, c_delta)
    out = {"n_questions": len(pq), "c_delta": c_delta, "by_depth": {}, "solver_delta": {}}

    for r in depths:
        recs = [v[r] for v in pq.values() if r in v]
        if not recs:
            continue
        lv = np.array([x["level"] for x in recs])
        out["by_depth"][r] = {
            k: round(float(np.mean([x[k] for x in recs])), 4)
            for k in ("greedy_ok", "sample_acc", "vote_ok", "vote_delta_ok", "vote_logprob_ok", "greedy_cost", "vote_cost")}
        out["by_depth"][r]["greedy_ok_by_level"] = {
            int(l): round(float(np.mean([x["greedy_ok"] for x, ll in zip(recs, lv) if ll == l])), 3) for l in range(1, 6)}
        out["by_depth"][r]["straggler_ratio_mean"] = round(
            float(np.mean([x["slowest_gen"] for x in recs]) / max(1e-9, np.mean([x["greedy_gen"] for x in recs]))), 3)

        samp = [x for x in rows if x["depth"] == r and x["sample"] >= 0]
        y = np.array([x["correct"] for x in samp])
        res = {"n_samples": len(samp), "acc": round(float(y.mean()), 4) if len(y) else None}
        for name, key, sign in (("neg_delta_last", "delta_last", -1), ("mean_logprob", "mean_logprob", 1),
                                ("neg_mean_entropy", "mean_entropy", -1)):
            s = np.array([sign * x.get(key, float("nan")) for x in samp], float)
            res[f"auroc_{name}"] = round(auroc(s, y), 4) if np.isfinite(s).any() else None
            pair_hits = pair_n = 0.0
            for idx, dd in by.items():
                qs = [x for x in dd.get(r, []) if x["sample"] >= 0]
                good = [sign * x.get(key, float("nan")) for x in qs if x["correct"]]
                bad = [sign * x.get(key, float("nan")) for x in qs if not x["correct"]]
                for a, b in itertools.product(good, bad):
                    if math.isfinite(a) and math.isfinite(b):
                        pair_n += 1
                        pair_hits += 1.0 if a > b else (0.5 if a == b else 0.0)
            res[f"within_question_auroc_{name}"] = round(pair_hits / pair_n, 4) if pair_n else None
            res["within_question_pairs"] = int(pair_n)
        out["solver_delta"][r] = res

    # min sufficient depth (by vote) and its relation to the planner signal
    full = max(depths)
    if planner:
        mind, feats = [], defaultdict(list)
        for idx, v in pq.items():
            if idx not in planner:
                continue
            md = next((r for r in sorted(depths) if r in v and v[r]["vote_ok"]), full + 1)
            mind.append(md)
            for name, fn in PLANNER_SIGNALS.items():
                feats[name].append(fn(planner[idx]))
        mind = np.array(mind, float)
        cov = rankdata(feats["prompt_tokens"])[:, None]
        out["min_sufficient_depth"] = {
            "histogram": dict(Counter(int(x) for x in mind)),
            "spearman": {n: round(spearman(np.array(v, float), mind), 4) for n, v in feats.items()},
            "partial_given_length": {n: round(partial_spearman(np.array(v, float), mind, cov), 4)
                                     for n, v in feats.items() if n != "prompt_tokens"},
        }
        out["policies"] = policies(pq, planner, depths)
    return out


def policies(pq, planner, depths, r_max=4):
    """Fit depth thresholds on even idx, evaluate on odd idx. Costs include the
    planner's prefill (r_max loops over the prompt) for every adaptive policy."""
    choices = [r for r in sorted(depths) if r >= 2] or sorted(depths)
    ids = [i for i in pq if i in planner and all(r in pq[i] for r in choices)]
    train = [i for i in ids if i % 2 == 0]
    test = [i for i in ids if i % 2 == 1]

    def evaluate(assign, mode):
        acc = np.mean([pq[i][assign[i]][f"{mode}_ok"] for i in test])
        cost = np.mean([pq[i][assign[i]][f"{mode}_cost"] for i in test])
        return float(acc), float(cost)

    out = {"choices": choices, "n_test": len(test), "fixed": {}, "adaptive": {}}
    for mode in ("greedy", "vote"):
        out["fixed"][mode] = {r: [round(x, 4) for x in evaluate({i: r for i in test}, mode)] for r in choices}
        grid = [p for p in itertools.product(*(np.arange(0, 1.01, 0.1) for _ in choices)) if abs(sum(p) - 1) < 1e-6]
        res = {}
        signals = {"delta_auc": True, "delta_last": True, "prompt_tokens": True,
                   "delta_auc_reversed": False}
        for name, harder_is_larger in signals.items():
            fn = PLANNER_SIGNALS[name.replace("_reversed", "")]
            sgn = 1.0 if harder_is_larger else -1.0
            tr = sgn * np.array([fn(planner[i]) for i in train], float)
            pts = []
            for p in grid:
                cuts = np.quantile(tr, np.cumsum(p)[:-1])
                assign = {i: choices[int(np.searchsorted(cuts, sgn * fn(planner[i]), side="right"))] for i in test}
                a, c_ = evaluate(assign, mode)
                planner_cost = r_max * np.mean([pq[i][choices[0]]["prompt_tokens"] for i in test])
                pts.append({"mix": [round(x, 1) for x in p], "acc": round(a, 4), "cost": round(c_ + planner_cost, 1)})
            res[name] = pareto(pts)
        rng = random.Random(0)
        pts = []
        for p in grid:
            accs, costs = [], []
            for _ in range(50):
                assign = {i: rng.choices(choices, weights=p)[0] for i in test}
                a, c_ = evaluate(assign, mode)
                accs.append(a)
                costs.append(c_)
            pts.append({"mix": [round(x, 1) for x in p], "acc": round(float(np.mean(accs)), 4),
                        "cost": round(float(np.mean(costs)), 1)})
        res["random"] = pareto(pts)
        oracle = {i: next((r for r in choices if pq[i][r][f"{mode}_ok"]), choices[-1]) for i in test}
        a, c_ = evaluate(oracle, mode)
        res["oracle"] = [{"acc": round(a, 4), "cost": round(c_, 1)}]
        out["adaptive"][mode] = res
    return out


def pareto(pts):
    pts = sorted(pts, key=lambda p: (p["cost"], -p["acc"]))
    front, best = [], -1.0
    for p in pts:
        if p["acc"] > best:
            front.append(p)
            best = p["acc"]
    return front


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0a", required=True)
    ap.add_argument("--stage0b", default=None)
    ap.add_argument("--out", default="artifacts/loopdelta/analysis")
    args = ap.parse_args()
    rows_a = read_jsonl(os.path.join(args.stage0a, "rows.jsonl"))
    summary = {"stage0a": analyze_0a(rows_a)}
    if args.stage0b:
        rows_b, by = load_0b(args.stage0b)
        depths = sorted({r["depth"] for r in rows_b})
        planner = {r["idx"]: r for r in rows_a}
        summary["stage0b"] = analyze_0b(rows_b, by, depths, planner)
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "summary.json"), summary)
    s = summary["stage0a"]["signals"]
    print(f"{'signal':24s} {'rho(level)':>11s} {'partial':>8s} {'AUROC 5v1':>10s}  cv(len+subj -> +signal)")
    for name, rec in s.items():
        cv = rec.get("cv_level_spearman", {})
        print(f"{name:24s} {rec['spearman_level']:11.3f} {rec.get('partial_spearman_level_given_length_subject', float('nan')):8.3f} "
              f"{rec['auroc_l5_vs_l1']:10.3f}  {cv.get('length_subject', '')} -> {cv.get('length_subject_signal', '')}")
    print(f"wrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
