#!/usr/bin/env python3
"""Compare loop selectors on questions answered at every depth.

Each selector maps a question to a depth before its answer is written. Learned
selectors give out-of-fold estimates P[q, d] of the 4-answer vote being correct
(5 folds stratified by MATH level, one regularised logistic head per depth); the
tau rule in ``looped.loop_select`` turns P into a depth. Threshold selectors (Δ,
Ouro's gate) are swept directly. Every policy is scored on accuracy and mean
loop-tokens and compared with fixed depths, the best question-independent
mixture at the same cost, and matched-random permutations of its own depths.

Sources:
  stage0b   LoopDelta stage 0b (300 questions, depths 1-4, samples 0-3; the greedy
            answer is excluded) with stage 0a prefill features
  practice  the 400 practice questions at depths 2-4 (forced short answers)
  pilot     the router pilot runs (labels per seed) with loop_probe_features.py rows

Needs Python >= 3.10 with numpy, scikit-learn and (stage0b only) datasets:
    uv run --no-project --python 3.12 --with numpy --with scikit-learn --with datasets \
        python scripts/eval_loop_selectors.py --source stage0b --out artifacts/looped_serving/loop_routing/stage_a_stage0b
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from looped import loop_select as ls  # noqa: E402

LD = "artifacts/loopdelta_run/artifacts/loopdelta"
TAUS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0]


# ------------------------------------------------------------------ data

class Data:
    def __init__(self, qids, C, cost, depths, text, level, subject, feats):
        self.qids, self.C, self.cost, self.depths = qids, C, cost, depths
        self.text, self.level, self.subject, self.feats = text, level, subject, feats


def load_stage0b(args) -> Data:
    by = defaultdict(dict)
    for d in (1, 2, 3, 4):
        rows = defaultdict(list)
        for line in open(f"{LD}/stage0b/scored_r{d}.jsonl"):
            r = json.loads(line)
            if r["sample"] < 0:
                continue                                       # the greedy answer is not part of the vote
            rows[r["idx"]].append(r)
        for idx, rs in rows.items():
            assert len(rs) == 4, (d, idx, len(rs))
            by[idx][d] = (ls.vote_correct([(r["sample"], r["key"], bool(r["correct"])) for r in rs]),
                          sum(r["loop_tokens"] for r in rs), rs[0]["level"], rs[0]["subject"])
    pre = {}
    for line in open(f"{LD}/stage0a_r4/rows.jsonl"):
        r = json.loads(line)
        pre[r["idx"]] = r
    from data import load_math
    train = [r["question"] for r in load_math("train")]       # stage 0's idx is a position in this list
    idxs = sorted(i for i in by if len(by[i]) == 4 and i in pre)
    for i in idxs:
        assert len(train[i]) == pre[i]["question_chars"], f"question text order drifted at idx {i}"
    C = np.array([[by[i][d][0] for d in (1, 2, 3, 4)] for i in idxs], float)
    cost = np.array([[by[i][d][1] for d in (1, 2, 3, 4)] for i in idxs], float)
    text = [train[i] for i in idxs]
    feats = [{"prefill": pre[i], "prompt_tokens": pre[i]["prompt_tokens"],
              "question_chars": pre[i]["question_chars"]} for i in idxs]
    return Data([str(i) for i in idxs], C, cost, (1, 2, 3, 4), text,
                [by[i][1][2] for i in idxs], [by[i][1][3] for i in idxs], feats)


def run_labels(run_dir: str):
    """{qid: (vote_correct, loop_tokens)} for one graded looped_profile arm."""
    graded = defaultdict(list)
    for line in open(os.path.join(run_dir, "graded.jsonl")):
        r = json.loads(line)
        s = int(re.search(r":s(\d+)", r["candidate_id"]).group(1))
        key = r["answer_key"] if r.get("parse_status") == "ok" else None
        graded[r["question_id"]].append((s, key, bool(r["correct"])))
    cost = defaultdict(float)
    for line in open(os.path.join(run_dir, "candidates.jsonl")):
        r = json.loads(line)
        cost[r["question_id"]] += r["logical_loop_tokens"]
    return {q: (ls.vote_correct(v), cost[q]) for q, v in graded.items()}


def split_meta(split: str):
    spec = json.load(open("configs/looped_serving/splits.json"))
    return {it["question_id"]: it for it in spec["splits"][split]["items"]}


def load_practice(args) -> Data:
    base = "artifacts/looped_serving/pod_out/modes_calibration"
    labs = {d: run_labels(f"{base}/d{d}_w4_direct_cap2048") for d in (2, 3, 4)}
    meta = split_meta("math_train_calibration")
    items = {json.loads(l)["question_id"]: json.loads(l)["text"]
             for l in open("artifacts/looped_serving/splits/math_train_calibration.items.jsonl")}
    qids = sorted(set.intersection(*[set(v) for v in labs.values()]))
    C = np.array([[labs[d][q][0] for d in (2, 3, 4)] for q in qids], float)
    cost = np.array([[labs[d][q][1] for d in (2, 3, 4)] for q in qids], float)
    return Data(qids, C, cost, (2, 3, 4), [items[q] for q in qids], [meta[q]["level"] for q in qids],
                [meta[q]["subject"] for q in qids], [{"question_chars": len(items[q])} for q in qids])


def load_pilot(args, seed: int) -> Data:
    base = args.pilot_dir
    labs = {}
    for d in (1, 2, 3, 4):
        hits = glob.glob(f"{base}/seed{seed}/d{d}_w4*cap8192*")
        if len(hits) != 1:
            raise SystemExit(f"expected one depth-{d} arm under {base}/seed{seed}, found {hits}")
        labs[d] = run_labels(hits[0])
    meta = split_meta("math_train_calibration")
    items = {json.loads(l)["question_id"]: json.loads(l)["text"]
             for l in open("artifacts/looped_serving/splits/math_train_calibration.items.jsonl")}
    feats = {}
    fpath = f"{base}/features/features.jsonl"
    if os.path.exists(fpath):
        feats = {json.loads(l)["question_id"]: json.loads(l) for l in open(fpath)}
    qids = sorted(set.intersection(*[set(v) for v in labs.values()]))
    C = np.array([[labs[d][q][0] for d in (1, 2, 3, 4)] for q in qids], float)
    cost = np.array([[labs[d][q][1] for d in (1, 2, 3, 4)] for q in qids], float)
    fl = [feats.get(q, {"question_chars": len(items[q])}) for q in qids]
    return Data(qids, C, cost, (1, 2, 3, 4), [items[q] for q in qids], [meta[q]["level"] for q in qids],
                [meta[q]["subject"] for q in qids], fl)


# ------------------------------------------------------------------ features

def _log(x):
    return [math.log(max(float(v), 1e-8)) for v in x]


def scalar_block(f: dict) -> list:
    return (_log(f["delta"]) + _log(f.get("delta_k", [])) + _log(f.get("delta_v", []))
            + list(f["entropy"]) + list(f["top1_agree_last"]) + _log(f["kl_consec"])
            + list(f.get("gate_exit_pdf", f.get("gate_lambda", []))))


def profile_block(f: dict) -> list:
    return _log(np.ravel(f["delta_layers_k"])) + _log(np.ravel(f["delta_layers_v"]))


def length_block(f: dict) -> list:
    out = [math.log(f["question_chars"])]
    if "prompt_tokens" in f:
        out.append(math.log(f["prompt_tokens"]))
    return out


def feature_sets(D: Data, probe_k: str):
    subjects = sorted(set(D.subject))
    meta = np.array([[float(l == k) for k in range(1, 6)] + [float(s == x) for x in subjects]
                     for l, s in zip(D.level, D.subject)])
    length = np.array([length_block(f) for f in D.feats])
    sets = {"length": length, "level+subject": meta}
    if all("prefill" in f for f in D.feats):
        pre = np.array([scalar_block(f["prefill"]) + [f["prefill"]["gate_expected_exit"]] for f in D.feats])
        sets["prefill"] = np.hstack([pre, length])
        sets["prefill_profile"] = np.hstack([np.array([profile_block(f["prefill"]) for f in D.feats]), pre, length])
    if all("probe" in f for f in D.feats):
        pb = np.array([scalar_block(f["probe"][probe_k])
                       + [float(f["probe_first_is_think"]), math.log(f["probe_tokens"]), float(f["probe_hit_eos"])]
                       for f in D.feats])
        sets["probe"] = np.hstack([pb, length])
        sets["probe_profile"] = np.hstack([np.array([profile_block(f["probe"][probe_k]) for f in D.feats]), pb, length])
        sets["prefill+probe"] = np.hstack([sets["prefill"], pb])
    return sets


# ------------------------------------------------------------------ learned heads

def oof_probs(X, C, level, kind: str, seed: int = 0, text=None) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n, nd = C.shape
    P = np.zeros((n, nd))
    folds = StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), level)
    for tr, te in folds:
        for j in range(nd):
            y = C[tr, j]
            if y.min() == y.max():
                P[te, j] = y[0]
                continue
            head = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=5, scoring="neg_log_loss",
                                        max_iter=5000)
            if kind == "text":
                model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), head)
                model.fit([text[i] for i in tr], y)
                P[te, j] = model.predict_proba([text[i] for i in te])[:, 1]
                continue
            steps = [StandardScaler()]
            if kind == "profile":
                steps.append(PCA(n_components=min(10, len(tr) - 1), random_state=seed))
            model = make_pipeline(*steps, head)
            model.fit(X[tr], y)
            P[te, j] = model.predict_proba(X[te])[:, 1]
    return P


# ------------------------------------------------------------------ scoring

def score_policy(pol, D: Data, fr, base_j: int, n_perm: int):
    acc, cost = ls.evaluate(pol, D.C, D.cost)
    mr = ls.matched_random(pol, D.C, D.cost, n_perm=n_perm)
    rows = np.arange(len(pol))
    mc = ls.mcnemar_exact(D.C[rows, pol], D.C[:, base_j])
    return {"acc": acc, "cost": cost, "depths": ls.depth_counts(pol),
            "mixture_acc_same_cost": ls.best_mixture_acc(fr, cost),
            "margin_vs_mixture": acc - ls.best_mixture_acc(fr, cost),
            "random_acc": mr["acc"], "random_p": mr["p_value"],
            "vs_base": mc}


def at_budget(points, budget):
    """Best point whose mean cost is within the budget (development read: the
    operating point is picked on the same out-of-fold predictions)."""
    ok = [p for p in points if p["cost"] <= budget * 1.0001]
    return max(ok, key=lambda p: (p["acc"], -p["cost"])) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["stage0b", "practice", "pilot"], required=True)
    ap.add_argument("--pilot_dir", default="artifacts/looped_serving/loop_routing/pilot")
    ap.add_argument("--seed_run", type=int, default=0, help="pilot: which run's labels to fit and score")
    ap.add_argument("--probe_k", default="32")
    ap.add_argument("--base_depth", type=int, default=3)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    D = {"stage0b": load_stage0b, "practice": load_practice}.get(args.source, None)
    D = D(args) if D else load_pilot(args, args.seed_run)
    acc_d, cost_d = D.C.mean(0), D.cost.mean(0)
    fr = ls.mixture_frontier(acc_d, cost_d)
    base_j = D.depths.index(args.base_depth)
    budget = cost_d[base_j]
    oracle = ls.oracle_policy(D.C, D.cost)
    report = {"source": args.source, "n": len(D.qids), "depths": list(D.depths),
              "fixed": {str(d): {"acc": float(a), "cost": float(c)} for d, a, c in zip(D.depths, acc_d, cost_d)},
              "frontier": fr, "base_depth": args.base_depth, "budget": float(budget),
              "oracle": {"acc": ls.evaluate(oracle, D.C, D.cost)[0], "cost": ls.evaluate(oracle, D.C, D.cost)[1],
                         "depths": ls.depth_counts(oracle)},
              "selectors": {}}

    sets = feature_sets(D, args.probe_k)
    learned = {name: ("profile" if "profile" in name else "dense", X) for name, X in sets.items()}
    learned["text"] = ("text", None)
    for name, (kind, X) in learned.items():
        P = oof_probs(X, D.C, D.level, kind, text=D.text)
        pts = []
        for tau in TAUS:
            pol = ls.policy_from_probs(P, tau)
            p = score_policy(pol, D, fr, base_j, n_perm=200)
            p["tau"] = tau
            pts.append(p)
        best = at_budget(pts, budget)
        if best is not None:
            best = {**score_policy(ls.policy_from_probs(P, best["tau"]), D, fr, base_j, args.n_perm), "tau": best["tau"]}
        report["selectors"][name] = {"curve": pts, "at_budget": best}
        print(f"{name:16s} " + (f"acc={best['acc']:.3f} cost={best['cost']:.0f} mix={best['mixture_acc_same_cost']:.3f} "
                                f"rand={best['random_acc']:.3f} p_rand={best['random_p']:.3f} depths={best['depths']}"
                                if best else "no point within budget"), flush=True)

    thresholds = {}
    if "prefill" in sets:
        dl = np.array([f["prefill"]["delta"] for f in D.feats])
        thresholds["prefill_delta_threshold"] = dl
        ge = np.array([f["prefill"]["gate_exit_pdf"] for f in D.feats])
        thresholds["gate_threshold"] = ge
    if "probe" in sets:
        thresholds["probe_delta_threshold"] = np.array([f["probe"][args.probe_k]["delta"] for f in D.feats])
    nd = len(D.depths)
    for name, M in thresholds.items():
        pts = []
        if name == "gate_threshold":
            cum = np.cumsum(M, axis=1)
            grid = np.linspace(0.05, 0.95, 19)
            for q in grid:
                depth = (cum >= q).argmax(axis=1) + 1                       # 1-based loop count
                pol = np.clip(np.searchsorted(D.depths, depth), 0, nd - 1)
                p = score_policy(pol, D, fr, base_j, n_perm=200)
                p["param"] = float(q)
                pts.append(p)
        else:
            grid = np.quantile(M, np.linspace(0.02, 0.98, 25))
            for eps in grid:
                below = M < eps                                             # Δ_t for t = 2..4
                first = np.where(below.any(1), below.argmax(1) + 1, 4)      # stop at loop t-1 = index + 1
                pol = np.clip(np.searchsorted(D.depths, first), 0, nd - 1)
                p = score_policy(pol, D, fr, base_j, n_perm=200)
                p["param"] = float(eps)
                pts.append(p)
        best = at_budget(pts, budget)
        report["selectors"][name] = {"curve": pts, "at_budget": best}
        print(f"{name:16s} " + (f"acc={best['acc']:.3f} cost={best['cost']:.0f} mix={best['mixture_acc_same_cost']:.3f} "
                                f"rand={best['random_acc']:.3f} p_rand={best['random_p']:.3f} depths={best['depths']}"
                                if best else "no point within budget"), flush=True)

    with open(os.path.join(args.out, "selectors.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    lines = [f"# Loop selectors, source `{args.source}` ({report['n']} questions)", "",
             "Fixed depths: " + ", ".join(f"{d} loops {100*v['acc']:.1f}% at {v['cost']:.0f} loop-tokens"
                                          for d, v in report["fixed"].items()),
             f"Oracle (cheapest correct depth): {100*report['oracle']['acc']:.1f}% at {report['oracle']['cost']:.0f}, "
             f"depths {report['oracle']['depths']}", "",
             f"Operating point: best accuracy at mean cost <= fixed {args.base_depth} loops ({budget:.0f}); "
             "picked on the same out-of-fold predictions, so this is a development read.", "",
             "| selector | accuracy | loop-tokens | best mixture, same cost | matched random | p (random) | "
             f"vs {args.base_depth} loops (only sel / only fixed, p) | depths |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in report["selectors"].items():
        b = s["at_budget"]
        if not b:
            lines.append(f"| {name} | - | - | - | - | - | - | - |")
            continue
        v = b["vs_base"]
        lines.append(f"| {name} | {100*b['acc']:.1f}% | {b['cost']:.0f} | {100*b['mixture_acc_same_cost']:.1f}% | "
                     f"{100*b['random_acc']:.1f}% | {b['random_p']:.3f} | {v['only_a']} / {v['only_b']}, {v['p']:.3g} | "
                     f"{b['depths']} |")
    open(os.path.join(args.out, "selectors.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
