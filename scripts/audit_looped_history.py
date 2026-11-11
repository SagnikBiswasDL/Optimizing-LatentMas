#!/usr/bin/env python3
"""CPU replay audit of the saved looped-model candidates (no model is run).

Reads the Stage-0b candidate pool (300 MATH-Train questions x depths 1-4 x
1 greedy + 4 sampled answers) and the GSM8K pilot outputs, validates the grid,
and recomputes the tables the looped-serving plan relies on:

  * greedy and fixed depth x width accuracy with logical loop-token cost
  * two-then-two agreement stopping, with a same-depth or deeper fallback
  * the 3+3+4+4 mixed-depth portfolio
  * strict vote-margin cancellation under a synchronized length proxy
  * agreement, cap, and prompt-reuse statistics
  * solver-delta vs log-probability within-question AUROC
  * GSM8K pilot headline accuracies

Every replay is done twice: with the vote key stored in the artifacts
(``legacy-v1``, to reproduce historical numbers) and with ``structural-v2``
(``looped/answer_identity.py``), so the effect of the vote-identity repair is
visible. Correctness labels are the ones stored in the artifacts.

What these numbers are not: fresh evaluations, latency measurements, or
estimates over generation seeds. Costs are logical loop-tokens. Intervals are
question bootstraps over this one saved pool. Adaptive rows average over every
choice of initial samples from the four saved ones.

    python scripts/audit_looped_history.py \\
        --root artifacts/loopdelta_run/artifacts/loopdelta/stage0b \\
        --out artifacts/looped_serving/history_audit
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped.answer_identity import LEGACY, STRUCTURAL, identify, legacy_key  # noqa: E402
from looped.ensemble_state import (FIRST_LISTED, Vote, completion_rounds, select,  # noqa: E402
                                   strict_certificate)
from looped.experiment_manifest import file_sha256  # noqa: E402

PAIRS = list(itertools.combinations(range(4), 2))


# ------------------------------------------------------------------ loading

def read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pool(root: str) -> Tuple[Dict[int, Dict[int, Dict[int, Dict]]], Dict[str, str]]:
    """idx -> depth -> sample -> row, plus the sha256 of every file read."""
    by: Dict[int, Dict[int, Dict[int, Dict]]] = defaultdict(lambda: defaultdict(dict))
    hashes = {}
    for name in sorted(os.listdir(root)):
        if not (name.startswith("scored_r") and name.endswith(".jsonl")):
            continue
        path = os.path.join(root, name)
        hashes[name] = file_sha256(path)
        for r in read_jsonl(path):
            if r["sample"] in by[r["idx"]][r["depth"]]:
                raise ValueError(f"duplicate row idx={r['idx']} depth={r['depth']} sample={r['sample']}")
            by[r["idx"]][r["depth"]][r["sample"]] = r
    for extra in ("env.json", "timing.json"):
        if os.path.exists(os.path.join(root, extra)):
            hashes[extra] = file_sha256(os.path.join(root, extra))
    return by, hashes


def validate_grid(by, depths: Sequence[int], n_samples: int, max_new: Optional[int]) -> Dict:
    """Fail loudly on anything that would make the replay silently wrong."""
    problems = []
    want = set(range(-1, n_samples))
    fallback_graded = 0
    for idx, dd in by.items():
        prompts = set()
        for d in depths:
            if set(dd.get(d, {})) != want:
                problems.append(f"idx {idx} depth {d}: samples {sorted(dd.get(d, {}))}")
                continue
            for r in dd[d].values():
                prompts.add(r["prompt_tokens"])
                # the last sampled token is never fed back, so it costs nothing
                if r["loop_tokens"] != d * (r["prompt_tokens"] + r["gen_tokens"] - 1):
                    problems.append(f"idx {idx} depth {d} sample {r['sample']}: loop_tokens mismatch")
                if r["key"] != legacy_key(r["pred"]):
                    problems.append(f"idx {idx} depth {d} sample {r['sample']}: stored key is not legacy-v1")
                if max_new is not None and not r["hit_eos"] and r["gen_tokens"] != max_new:
                    problems.append(f"idx {idx} depth {d} sample {r['sample']}: stopped without EOS below the cap")
                if r.get("mv_ok") is None and r["pred"] is not None:
                    fallback_graded += 1
        if len(prompts) > 1:
            problems.append(f"idx {idx}: prompt length differs across candidates {sorted(prompts)}")
    if problems:
        raise ValueError(f"{len(problems)} grid problems, first: {problems[:5]}")
    levels = Counter(next(iter(dd[depths[0]].values()))["level"] for dd in by.values())
    return {"questions": len(by), "depths": list(depths), "samples_per_depth": n_samples, "greedy_per_depth": 1,
            "questions_per_level": {str(k): v for k, v in sorted(levels.items())},
            "rows_graded_by_string_fallback": fallback_graded}


# ------------------------------------------------------------------ replay primitives

class Pool:
    """One question's candidates under one vote-identity version."""

    def __init__(self, rows: Dict[int, Dict[int, Dict]], version: str) -> None:
        self.rows = rows
        self.version = version

    def key(self, d: int, s: int) -> Optional[str]:
        r = self.rows[d][s]
        return r["key"] if self.version == LEGACY else identify(r["pred"], STRUCTURAL).key

    def votes(self, members: Sequence[Tuple[int, int]]) -> List[Vote]:
        return [Vote((d, s), self.key(d, s)) for d, s in members]

    def pick(self, members: Sequence[Tuple[int, int]]) -> Tuple[bool, float]:
        """(selected answer is correct, logical loop-tokens of all members).
        Historical rule: unweighted vote, first listed answer wins a tie."""
        win = select(self.votes(members), FIRST_LISTED)
        cost = float(sum(self.rows[d][s]["loop_tokens"] for d, s in members))
        return (bool(win and self.rows[win.candidate[0]][win.candidate[1]]["correct"]), cost)

    def agree(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        ka, kb = self.key(*a), self.key(*b)
        return ka is not None and ka == kb


Policy = Callable[[Pool], Tuple[float, float]]            # -> (P(correct), mean cost) for one question


def greedy(d: int) -> Policy:
    return lambda p: (float(p.rows[d][-1]["correct"]), float(p.rows[d][-1]["loop_tokens"]))


def fixed(d: int, width: int) -> Policy:
    """Vote over ``width`` samples at depth ``d``, averaged over which samples."""
    def run(p: Pool):
        out = [p.pick([(d, s) for s in sub]) for sub in itertools.combinations(range(4), width)]
        return float(np.mean([o[0] for o in out])), float(np.mean([o[1] for o in out]))
    return run


def portfolio(depths: Sequence[int]) -> Policy:
    """Two samples at each listed depth, averaged over which samples."""
    def run(p: Pool):
        out = [p.pick([(d, s) for d, pair in zip(depths, combo) for s in pair])
               for combo in itertools.product(PAIRS, repeat=len(depths))]
        return float(np.mean([o[0] for o in out])), float(np.mean([o[1] for o in out]))
    return run


def two_then_two(d0: int, d_fallback: int) -> Policy:
    """Start two depth-``d0`` samples. If they agree, return that answer. Otherwise
    acquire two more at ``d_fallback`` and vote over all four acquired."""
    def run(p: Pool):
        accs, costs = [], []
        for pair in PAIRS:
            first = [(d0, s) for s in pair]
            if p.agree(*first):
                a, c = p.pick(first)
            elif d_fallback == d0:
                a, c = p.pick(first + [(d0, s) for s in range(4) if s not in pair])
            else:
                sub = [p.pick(first + [(d_fallback, s) for s in more]) for more in PAIRS]
                a, c = float(np.mean([o[0] for o in sub])), float(np.mean([o[1] for o in sub]))
            accs.append(a)
            costs.append(c)
        return float(np.mean(accs)), float(np.mean(costs))
    return run


def evaluate(pools: Sequence[Pool], policy: Policy) -> Tuple[np.ndarray, np.ndarray]:
    out = [policy(p) for p in pools]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def boot_ci(values: np.ndarray, iters: int, seed: int) -> List[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(iters, n))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def row(acc: np.ndarray, cost: np.ndarray, iters: int, seed: int) -> Dict:
    return {"accuracy": float(acc.mean()), "accuracy_ci95_question_bootstrap": boot_ci(acc, iters, seed),
            "mean_loop_tokens": float(cost.mean())}


# ------------------------------------------------------------------ diagnostics

def agreement(pools: Sequence[Pool], d: int) -> Dict:
    agree = wrong = total = 0
    for p in pools:
        for a, b in PAIRS:
            total += 1
            if p.agree((d, a), (d, b)):
                agree += 1
                wrong += not p.rows[d][a]["correct"]
    return {"pairs": total, "agreeing_pairs": agree, "agreement_rate": agree / total,
            "wrong_among_agreeing": wrong / agree if agree else None,
            "note": "pairs from one question are dependent; agreement is evidence, not a certificate"}


def key_conflicts(pools: Sequence[Pool], depths: Sequence[int]) -> Dict:
    """Same vote key, different stored correctness, within one question."""
    conflicts, examples = 0, []
    for p in pools:
        seen: Dict[str, set] = defaultdict(set)
        preds: Dict[str, set] = defaultdict(set)
        for d in depths:
            for s in range(-1, 4):
                k = p.key(d, s)
                if k is not None:
                    seen[k].add(bool(p.rows[d][s]["correct"]))
                    preds[k].add(p.rows[d][s]["pred"])
        for k, verdicts in seen.items():
            if len(verdicts) > 1:
                conflicts += 1
                if len(examples) < 5:
                    examples.append({"key": k, "preds": sorted(preds[k])[:4]})
    return {"question_keys_with_mixed_correctness": conflicts, "examples": examples}


def strict_cancellation(pools: Sequence[Pool], d: int) -> Dict:
    """All four depth-``d`` samples start together and emit one token per round.
    Cancel the rest as soon as the completed votes fix the winner."""
    full_rounds, stop_rounds, full_tokens, kept_tokens = [], [], 0, 0
    cancelled_questions = winner_changes = 0
    for p in pools:
        lengths = [p.rows[d][s]["gen_tokens"] for s in range(4)]
        full = p.votes([(d, s) for s in range(4)])
        stop, done = max(lengths), []
        for group in completion_rounds(lengths):
            done += [full[i] for i in group]
            if strict_certificate(done, 4 - len(done)):
                stop = lengths[group[0]]
                break
        if len(done) < 4:
            cancelled_questions += 1
            a, b = select(done, FIRST_LISTED), select(full, FIRST_LISTED)
            winner_changes += a.key != b.key
        full_rounds.append(max(lengths))
        stop_rounds.append(stop)
        full_tokens += sum(lengths)
        kept_tokens += sum(min(n, stop) for n in lengths)
    return {
        "questions_with_a_cancellation": cancelled_questions,
        "winner_changes": winner_changes,
        "completion_rounds_saved_length_proxy": 1 - float(np.mean(stop_rounds)) / float(np.mean(full_rounds)),
        "decode_tokens_saved": 1 - kept_tokens / full_tokens,
        "note": "synchronized equal-token-time proxy from output lengths; not a measured latency",
    }


def cap_and_reuse(by, depths: Sequence[int], max_new: Optional[int]) -> Dict:
    out = {}
    for d in depths:
        samp = [dd[d][s] for dd in by.values() for s in range(4)]
        prompt = [dd[d][0]["prompt_tokens"] for dd in by.values()]
        total = sum(r["loop_tokens"] for r in samp)
        out[str(d)] = {
            "sampled_traces_at_cap": float(np.mean([not r["hit_eos"] for r in samp])),
            "greedy_traces_at_cap": float(np.mean([not dd[d][-1]["hit_eos"] for dd in by.values()])),
            "mean_prompt_tokens": float(np.mean(prompt)),
            "mean_generated_tokens_sampled": float(np.mean([r["gen_tokens"] for r in samp])),
            "sampled_traces_without_a_boxed_answer": float(np.mean([r["pred"] is None for r in samp])),
            "capped_sampled_traces_without_a_boxed_answer": float(
                np.mean([r["pred"] is None for r in samp if not r["hit_eos"]] or [float("nan")])),
            # one prompt computed, three reused, out of four identical prompts
            "sc4_loop_tokens_saved_by_ideal_prompt_reuse": 3 * d * sum(prompt) / total,
            "slowest_of_4_over_greedy_length_ratio_proxy": float(
                np.mean([max(dd[d][s]["gen_tokens"] for s in range(4)) for dd in by.values()])
                / np.mean([dd[d][-1]["gen_tokens"] for dd in by.values()])),
        }
    out["max_new_tokens"] = max_new
    out["note"] = ("prompt reuse is logical loop-token work, not a wall-time bound; the length ratio is the "
                   "quantity the old analysis called straggler_ratio_mean, an output-length proxy")
    return out


def within_question_auroc(by, d: int, field: str, sign: float) -> Dict:
    hits = pairs = 0.0
    questions = 0
    for dd in by.values():
        rows = [dd[d][s] for s in range(4)]
        good = [sign * r[field] for r in rows if r["correct"] and math.isfinite(r.get(field, float("nan")))]
        bad = [sign * r[field] for r in rows if not r["correct"] and math.isfinite(r.get(field, float("nan")))]
        questions += bool(good and bad)
        for a, b in itertools.product(good, bad):
            pairs += 1
            hits += 1.0 if a > b else 0.5 if a == b else 0.0
    return {"auroc": hits / pairs if pairs else None, "correct_incorrect_pairs": int(pairs),
            "questions_contributing": questions}


def identity_comparison(by, depths: Sequence[int]) -> Dict:
    """Where the two vote keys disagree about which sampled answers are the same."""
    merged_by_legacy_only, merged_by_structural_only, statuses = [], [], Counter()
    votes_dropped = []                 # legacy counted a vote that structural calls a parse failure
    for idx, dd in by.items():
        for d in depths:
            rows = [dd[d][s] for s in range(4)]
            ids = [identify(r["pred"], STRUCTURAL) for r in rows]
            for r, i in zip(rows, ids):
                statuses[i.status] += 1
                if r["key"] is not None and i.key is None:
                    votes_dropped.append({"idx": idx, "depth": d, "sample": r["sample"], "pred": r["pred"],
                                          "status": i.status, "hit_eos": r["hit_eos"], "correct": r["correct"]})
            for a, b in PAIRS:
                la = rows[a]["key"] is not None and rows[a]["key"] == rows[b]["key"]
                sa = ids[a].key is not None and ids[a].key == ids[b].key
                if la != sa and rows[a]["pred"] != rows[b]["pred"]:
                    rec = {"idx": idx, "depth": d, "preds": [rows[a]["pred"], rows[b]["pred"]],
                           "correct": [rows[a]["correct"], rows[b]["correct"]]}
                    (merged_by_legacy_only if la else merged_by_structural_only).append(rec)
    # Two answers both graded correct are the same answer, so a pair of them with
    # different keys is a split vote: a lower bound on what a key fails to merge.
    split = {LEGACY: [], STRUCTURAL: []}
    correct_pairs = 0
    for idx, dd in by.items():
        for d in depths:
            rows = [dd[d][s] for s in range(4)]
            for a, b in PAIRS:
                if not (rows[a]["correct"] and rows[b]["correct"]):
                    continue
                correct_pairs += 1
                if rows[a]["key"] != rows[b]["key"]:
                    split[LEGACY].append({"idx": idx, "depth": d, "preds": [rows[a]["pred"], rows[b]["pred"]]})
                if identify(rows[a]["pred"], STRUCTURAL).key != identify(rows[b]["pred"], STRUCTURAL).key:
                    split[STRUCTURAL].append({"idx": idx, "depth": d, "preds": [rows[a]["pred"], rows[b]["pred"]]})
    return {
        "both_correct_sampled_pairs": correct_pairs,
        "both_correct_pairs_split_by_legacy": len(split[LEGACY]),
        "both_correct_pairs_split_by_structural": len(split[STRUCTURAL]),
        "questions_with_a_split_correct_pair": {v: len({e["idx"] for e in split[v]}) for v in split},
        "examples_split_by_structural": split[STRUCTURAL][:10],
        "examples_split_by_legacy_only": [e for e in split[LEGACY] if e not in split[STRUCTURAL]][:10],
        "sampled_parse_status_structural": dict(statuses),
        "sampled_votes_counted_by_legacy_but_not_structural": votes_dropped,
        "pairs_merged_by_legacy_only": len(merged_by_legacy_only),
        "pairs_merged_by_structural_only": len(merged_by_structural_only),
        "examples_legacy_only": merged_by_legacy_only[:8],
        "examples_structural_only": merged_by_structural_only[:8],
    }


# ------------------------------------------------------------------ other saved evidence

def timing_summary(root: str, n_questions: int, n_samples: int) -> Optional[Dict]:
    path = os.path.join(root, "timing.json")
    if not os.path.exists(path):
        return None
    t = json.load(open(path))
    hours = sum(t[f"r{d}"]["wall_sampled_s"] for d in (2, 3, 4)) / 3600
    scale = (600 * 8) / (n_questions * n_samples)
    return {"sampled_pass_gpu_hours_depths_2_3_4": hours,
            "proportional_estimate_600_questions_x_8_samples": hours * scale,
            "note": "old reference path on the recorded GPU, excluding rescoring and greedy runs; "
                    "a proportional guess, to be replaced by the smoke-run measurement"}


def gsm8k_pilots(repo: str) -> Dict:
    out = {}
    for name, rel in (("thinking_valid", "artifacts/looped/mas_gate_r4"),
                      ("plain_valid", "artifacts/looped_base2/mas_gate")):
        path = os.path.join(repo, rel, "per_question.jsonl")
        if not os.path.exists(path):
            out[name] = None
            continue
        rows = read_jsonl(path)
        rec = {"n": len(rows), "sha256": file_sha256(path)}
        for arm in ("single", "selfrev", "mas"):
            rec[arm] = float(np.mean([r[arm]["ok"] for r in rows]))
        for m in sorted({s["m"] for r in rows for s in r["sc"]}):
            rec[f"sc@{m}"] = float(np.mean([next(s["ok"] for s in r["sc"] if s["m"] == m) for r in rows]))
        out[name] = rec
    out["note"] = "artifacts/looped_base/ is the invalid pre-mask-fix pilot and is deliberately not read"
    return out


# ------------------------------------------------------------------ main

def audit(root: str, repo: str, iters: int, seed: int) -> Dict:
    by, hashes = load_pool(root)
    env = json.load(open(os.path.join(root, "env.json"))) if os.path.exists(os.path.join(root, "env.json")) else {}
    depths = sorted({d for dd in by.values() for d in dd})
    max_new = env.get("args", {}).get("max_new")
    out: Dict = {
        "source": {"root": root, "sha256": hashes, "model_id": env.get("model_id"),
                   "model_revision": env.get("model_revision"), "device_name": env.get("device_name"),
                   "decode": {k: env.get("args", {}).get(k) for k in ("temperature", "top_p", "max_new")},
                   "subset_fingerprint": env.get("subset")},
        "bootstrap": {"iters": iters, "seed": seed, "unit": "question"},
        "grid": validate_grid(by, depths, 4, max_new),
        "cap_and_reuse": cap_and_reuse(by, depths, max_new),
        "identity_comparison": identity_comparison(by, depths),
        "timing": timing_summary(root, len(by), 4),
        "gsm8k_pilots": gsm8k_pilots(repo),
        "solver_signal_within_question": {
            str(d): {"neg_delta_last": within_question_auroc(by, d, "delta_last", -1.0),
                     "mean_logprob": within_question_auroc(by, d, "mean_logprob", 1.0)}
            for d in depths if d >= 2},
        "replay": {},
    }
    ids = sorted(by)
    for version in (LEGACY, STRUCTURAL):
        pools = [Pool(by[i], version) for i in ids]
        policies: Dict[str, Policy] = {}
        for d in depths:
            policies[f"greedy_d{d}"] = greedy(d)
            for w in (1, 2, 4):
                policies[f"fixed_d{d}_w{w}"] = fixed(d, w)
        if {3, 4} <= set(depths):
            policies["two_then_two_d3_fallback_d3"] = two_then_two(3, 3)
            policies["two_then_two_d3_fallback_d4"] = two_then_two(3, 4)
            policies["two_then_two_d4_fallback_d4"] = two_then_two(4, 4)
            policies["portfolio_3_3_4_4"] = portfolio([3, 4])
        if 2 in depths:
            policies["two_then_two_d2_fallback_d2"] = two_then_two(2, 2)
            policies["two_then_two_d2_fallback_d3"] = two_then_two(2, 3)
        res = {name: evaluate(pools, pol) for name, pol in policies.items()}
        table = {name: row(a, c, iters, seed) for name, (a, c) in res.items()}
        rec: Dict = {"policies": table, "tie_rule": FIRST_LISTED,
                     "agreement": {str(d): agreement(pools, d) for d in depths},
                     "strict_cancellation_sc4": {str(d): strict_cancellation(pools, d) for d in depths},
                     "key_conflicts": key_conflicts(pools, depths)}
        if {3, 4} <= set(depths):
            a3, a4 = res["fixed_d3_w4"][0], res["fixed_d4_w4"][0]
            half = 1.96 * float((a3 - a4).std(ddof=1)) / math.sqrt(len(a3))
            rec["paired_sc4_d3_minus_d4"] = {
                "difference": float((a3 - a4).mean()), "ci95_question_bootstrap": boot_ci(a3 - a4, iters, seed),
                "ci95_normal_approximation": [float((a3 - a4).mean()) - half, float((a3 - a4).mean()) + half],
                "bootstrap_note": "percentile endpoints fall on a 1/n grid (1/3 point here) and move by one "
                                  "step between bootstrap seeds",
                "d3_only": int(((a3 == 1) & (a4 == 0)).sum()), "d4_only": int(((a4 == 1) & (a3 == 0)).sum()),
                "note": "an interval containing zero does not establish equivalence"}
            ref = table["fixed_d3_w4"]["mean_loop_tokens"]
            rec["work_relative_to_fixed_d3_w4"] = {
                name: 1 - table[name]["mean_loop_tokens"] / ref
                for name in ("two_then_two_d3_fallback_d3", "two_then_two_d3_fallback_d4")}
            rec["fixed_d3_w4_work_saved_vs_fixed_d4_w4"] = 1 - ref / table["fixed_d4_w4"]["mean_loop_tokens"]
        out["replay"][version] = rec
    old, new = (out["replay"][v]["policies"] for v in (LEGACY, STRUCTURAL))
    out["identity_comparison"]["policy_accuracy_change_structural_minus_legacy"] = {
        name: new[name]["accuracy"] - old[name]["accuracy"]
        for name in old if abs(new[name]["accuracy"] - old[name]["accuracy"]) > 1e-12}
    return out


def print_table(out: Dict) -> None:
    for version, rec in out["replay"].items():
        print(f"\n== vote identity: {version}")
        print(f"{'policy':34s} {'accuracy':>9s} {'95% CI (questions)':>20s} {'loop-tokens':>12s}")
        for name, r in rec["policies"].items():
            lo, hi = r["accuracy_ci95_question_bootstrap"]
            print(f"{name:34s} {100 * r['accuracy']:8.2f}% [{100 * lo:6.2f}, {100 * hi:6.2f}] {r['mean_loop_tokens']:12.2f}")
        if "paired_sc4_d3_minus_d4" in rec:
            p = rec["paired_sc4_d3_minus_d4"]
            print(f"SC@4 depth 3 minus depth 4: {100 * p['difference']:+.2f} pts "
                  f"[{100 * p['ci95_question_bootstrap'][0]:+.2f}, {100 * p['ci95_question_bootstrap'][1]:+.2f}], "
                  f"depth-3 only {p['d3_only']}, depth-4 only {p['d4_only']}")
        ag, sc = rec["agreement"].get("3"), rec["strict_cancellation_sc4"].get("3")
        if ag:
            print(f"depth-3 pair agreement {100 * ag['agreement_rate']:.2f}%, wrong among agreeing "
                  f"{100 * ag['wrong_among_agreeing']:.2f}%")
            print(f"strict cancellation at depth 3: rounds saved (length proxy) "
                  f"{100 * sc['completion_rounds_saved_length_proxy']:.2f}%, decode tokens saved "
                  f"{100 * sc['decode_tokens_saved']:.2f}%, winner changes {sc['winner_changes']}")


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(repo, "artifacts/loopdelta_run/artifacts/loopdelta/stage0b"))
    ap.add_argument("--out", default=os.path.join(repo, "artifacts/looped_serving/history_audit"))
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = audit(args.root, repo, args.iters, args.seed)
    out["source"]["root"] = os.path.relpath(args.root, repo)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "audit.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print_table(out)
    print(f"\nwrote {os.path.relpath(path, repo)}")


if __name__ == "__main__":
    main()
