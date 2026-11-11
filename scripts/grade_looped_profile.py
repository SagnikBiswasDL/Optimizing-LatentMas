#!/usr/bin/env python
"""Grade the arms written by ``scripts/looped_profile.py`` and build the
latency-versus-quality table.

Gold answers are joined here and nowhere earlier. Votes are grouped by the
``structural-v2`` answer key with the ``lowest_candidate`` tie rule; correctness
comes from the pinned grader (``math-verify``), never from a string fallback.

Latency and throughput columns are measurements of the arm as it ran. The
``two-then-two`` rows are replays over an arm's first four samples: their
accuracy and work are exact for that policy, but they have no latency because
the policy was not executed live.

    python scripts/grade_looped_profile.py --runs artifacts/looped_serving/profile_smoke \
        --gold artifacts/looped_serving/splits/math_train_smoke.gold.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped import answer_identity as ai  # noqa: E402
from looped import ensemble_state as es  # noqa: E402
from looped import experiment_manifest as em  # noqa: E402


# ``<think>`` in the pinned Ouro tokenizer. New candidate rows carry a ``think``
# field; rows written before modes were recorded are classified by this id.
THINK_TOKEN_ID = 3


def opened_think(row) -> bool:
    if row.get("think") is not None:
        return bool(row["think"])
    return bool(row["generated_token_ids"]) and row["generated_token_ids"][0] == THINK_TOKEN_ID


def mode_stats(rows) -> dict:
    """Share, cap rate, accuracy and length of reasoning and direct answers (cancelled ones excluded)."""
    out = {}
    done = [r for r in rows if r["finish_reason"] in ("eos", "length_cap")]
    for name, sel in (("think", [r for r in done if opened_think(r)]), ("direct", [r for r in done if not opened_think(r)])):
        out[name] = {
            "share": len(sel) / len(done) if done else None,
            "cap_rate": float(np.mean([r["finish_reason"] == "length_cap" for r in sel])) if sel else None,
            "accuracy": float(np.mean([r["correct"] for r in sel])) if sel else None,
            "accuracy_when_finished": (float(np.mean([r["correct"] for r in sel if r["finish_reason"] == "eos"]))
                                       if any(r["finish_reason"] == "eos" for r in sel) else None),
            "mean_gen_tokens": float(np.mean([r["gen_tokens"] for r in sel])) if sel else None,
            "share_of_tokens": (sum(r["gen_tokens"] for r in sel) / max(1, sum(r["gen_tokens"] for r in done))),
        }
    return out


# One KV block of one depth plane: 16 tokens x 24 layers x 16 heads x 128 dims x (K, V) x 2 bytes (bf16).
BLOCK_BYTES = 16 * 24 * 16 * 128 * 2 * 2


def boot_ci(values, iters: int, seed: int):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(iters, len(v)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def grade_rows(rows, gold, grader: str):
    for r in rows:
        g = ai.grade(r["pred"], gold[r["question_id"]]["gold"], grader)
        r["correct"], r["grader_error"] = g["correct"], g["error"]
    return rows


def vote(cands):
    """Selected candidate row (or None) by majority over structural keys."""
    votes = [es.Vote(c["sample"], c["answer_key"]) for c in cands]
    win = es.select(votes, es.LOWEST_CANDIDATE)
    return None if win is None else next(c for c in cands if c["sample"] == win.candidate)


def two_then_two(cands):
    """(selected row or None, candidates used) for samples s0..s3."""
    by = {c["sample"]: c for c in cands}
    a, b = by[0], by[1]
    if a["answer_key"] is not None and a["answer_key"] == b["answer_key"]:
        return a, [a, b]
    used = [by[s] for s in range(4)]
    return vote(used), used


def arm_table(name, rows, questions, summary, args):
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["question_id"]].append(r)
    qids = [q["question_id"] for q in questions]
    correct = [bool((sel := vote(by_q[q])) and sel["correct"]) for q in qids]
    arm = summary["arm"]
    out = {
        "arm": name, "depth": arm["depth"], "width": arm["width"], "cap": arm["cap"], "run_id": summary["run_id"],
        "policy": ("greedy" if arm["width"] == 1 else f"SC@{arm['width']}")
                  + ("" if arm.get("mode", "natural") == "natural" else f", {arm['mode']}"),
        "mode": arm.get("mode", "natural"), "modes": mode_stats(rows),
        "questions": len(qids), "accuracy": float(np.mean(correct)),
        "accuracy_ci95": boot_ci(correct, args.iters, args.seed),
        "candidate_accuracy": float(np.mean([r["correct"] for r in rows])),
        "answered_rate": float(np.mean([r["answer_key"] is not None for r in rows])),
        "cap_rate": float(np.mean([r["finish_reason"] == "length_cap" for r in rows])),
        "error_candidates": sum(r["finish_reason"] == "error" for r in rows),
        "mean_gen_tokens": summary["mean_gen_tokens"],
        "logical_loop_tokens_per_question": summary["logical_loop_tokens"] / len(qids),
        "charged_loop_tokens_per_question": summary["charged_loop_tokens"] / len(qids),
        "latency_s": summary["question_latency_s"], "first_token_s": summary["first_token_s"],
        "tokens_per_s": summary["tokens_per_s"], "questions_per_min": summary["questions_per_min"],
        "wall_s": summary["wall_s"], "slots": summary["slots"],
        "peak_kv_gb": None,
    }
    derived = []
    if arm["width"] >= 4:
        acc, work, used_n = [], [], []
        for q in qids:
            sel, used = two_then_two(by_q[q])
            acc.append(bool(sel and sel["correct"]))
            work.append(sum(c["logical_loop_tokens"] for c in used))
            used_n.append(len(used))
        sc4 = [bool((sel := vote([c for c in by_q[q] if c["sample"] < 4])) and sel["correct"]) for q in qids]
        sc4_work = [sum(c["logical_loop_tokens"] for c in by_q[q] if c["sample"] < 4) for q in qids]
        derived.append({
            "arm": name, "depth": arm["depth"], "policy": "two-then-two (replay, first 4 samples)",
            "accuracy": float(np.mean(acc)), "accuracy_ci95": boot_ci(acc, args.iters, args.seed),
            "logical_loop_tokens_per_question": float(np.mean(work)), "stopped_after_two": float(np.mean(np.array(used_n) == 2)),
            "paired_diff_vs_sc4": float(np.mean(np.array(acc, float) - np.array(sc4, float))),
            "work_saved_vs_sc4": 1 - float(np.sum(work)) / float(np.sum(sc4_work)),
            "sc4_accuracy_same_samples": float(np.mean(sc4)),
        })
    return out, derived, dict(zip(qids, correct))


def policy_table(name, rows, questions, summary, args):
    """A live policy run: the answer is the candidate the policy selected."""
    by_id = {r["candidate_id"]: r for r in rows}
    correct = [bool(q["selected_candidate_id"] and by_id[q["selected_candidate_id"]]["correct"]) for q in questions]
    pol = summary["policy"]
    finished = [r for r in rows if r["finish_reason"] != "cancelled"]
    depth = pol.get("depth", pol.get("direct_depth"))
    width = pol.get("width", pol.get("direct_width", 4))
    label = {"fixed_width": "SC@{width} (live driver)", "two_then_two": "two-then-two (live)",
             "certificate_cancel": "SC@{width} + certificate cancel (live)",
             "quorum_cancel": "{width} launched, stop at {quorum} agreeing (live)",
             "fast_then_think": "{direct_width} direct (quorum {quorum}), then {think_width} think at depth "
                                "{think_depth} (live)"}.get(pol["name"], pol["name"]).format(**pol)
    if pol.get("mode", "natural") != "natural":
        label += f", {pol['mode']}"
    if pol.get("think_trumps"):
        label += ", finished think answer wins"
    if pol.get("fallback_depth", depth) != depth:
        label += f", fallback depth {pol['fallback_depth']}"
    return {
        "arm": name, "depth": depth, "width": width, "cap": summary["cap"], "run_id": summary["run_id"],
        "policy": label, "mode": pol.get("mode", "natural"), "modes": mode_stats(rows),
        "decision_reasons": summary.get("decision_reasons"),
        "questions": len(questions), "accuracy": float(np.mean(correct)),
        "accuracy_ci95": boot_ci(correct, args.iters, args.seed),
        "candidate_accuracy": float(np.mean([r["correct"] for r in finished])),
        "answered_rate": float(np.mean([r["answer_key"] is not None for r in finished])),
        "cap_rate": float(np.mean([r["finish_reason"] == "length_cap" for r in rows])),
        "error_candidates": sum(r["finish_reason"] == "error" for r in rows),
        "mean_gen_tokens": summary["mean_gen_tokens"],
        "candidates_per_question": summary["candidates_per_question"],
        "cancelled_candidates": summary["cancelled_candidates"],
        "logical_loop_tokens_per_question": summary["logical_loop_tokens"] / len(questions),
        "charged_loop_tokens_per_question": summary["charged_loop_tokens"] / len(questions),
        "latency_s": summary["question_latency_s"], "first_token_s": summary["first_token_s"],
        "tokens_per_s": summary["tokens_per_s"], "questions_per_min": summary["questions_per_min"],
        "wall_s": summary["wall_s"], "slots": summary["slots"], "peak_kv_gb": None,
    }, {q["question_id"]: c for q, c in zip(questions, correct)}


def extension_table(name, rows, summary, source_rows, source_questions):
    """Effect of a longer cap on the candidates that hit the old cap."""
    src = {r["candidate_id"]: r for r in source_rows}
    merged = dict(src)
    merged.update({r["candidate_id"]: r for r in rows})
    by_q_old, by_q_new = defaultdict(list), defaultdict(list)
    for cid, r in src.items():
        by_q_old[r["question_id"]].append(r)
        by_q_new[r["question_id"]].append(merged[cid])
    qids = [q["question_id"] for q in source_questions]
    old = [bool((s := vote(by_q_old[q])) and s["correct"]) for q in qids]
    new = [bool((s := vote(by_q_new[q])) and s["correct"]) for q in qids]
    return {
        "arm": name, "from_cap": summary["arm"]["from_cap"], "cap": summary["arm"]["cap"],
        "depth": summary["arm"]["depth"], "width": summary["arm"]["width"],
        "capped_candidates_rerun": len(rows), "prefix_reproduced": summary.get("prefix_reproduced"),
        "now_finish_with_eos": sum(r["finish_reason"] == "eos" for r in rows),
        "still_capped": sum(r["finish_reason"] == "length_cap" for r in rows),
        "answered_before": sum(src[r["candidate_id"]]["answer_key"] is not None for r in rows),
        "answered_after": sum(r["answer_key"] is not None for r in rows),
        "correct_before": sum(src[r["candidate_id"]]["correct"] for r in rows),
        "correct_after": sum(r["correct"] for r in rows),
        "extra_gen_tokens": sum(r["gen_tokens"] - src[r["candidate_id"]]["gen_tokens"] for r in rows),
        "policy_accuracy_before": float(np.mean(old)), "policy_accuracy_after": float(np.mean(new)),
        "questions_changed": int(np.sum(np.array(old) != np.array(new))),
    }


def fmt(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def markdown(table, derived, ext) -> str:
    lines = ["| depth | policy | cap | accuracy % (95% CI) | candidates/q | think % | answered % | capped % | "
             "loop-tokens/q (charged) | latency mean s | p50 s | p95 s | tokens/s | questions/min | peak KV GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(table, key=lambda r: (r["depth"], r["width"], r["policy"])):
        lines.append(
            f"| {r['depth']} | {r['policy']} | {r['cap']} | {100 * r['accuracy']:.1f} ({100 * r['accuracy_ci95'][0]:.1f}, "
            f"{100 * r['accuracy_ci95'][1]:.1f}) | {r.get('candidates_per_question', r['width']):.2f} | "
            f"{fmt(None if r['modes']['think']['share'] is None else 100 * r['modes']['think']['share'])} | "
            f"{100 * r['answered_rate']:.1f} | {100 * r['cap_rate']:.1f} | "
            f"{r['charged_loop_tokens_per_question']:.0f} | {fmt(r['latency_s'].get('mean'))} | "
            f"{fmt(r['latency_s'].get('p50'))} | {fmt(r['latency_s'].get('p95'))} | {r['tokens_per_s']:.0f} | "
            f"{r['questions_per_min']:.1f} | {fmt(r['peak_kv_gb'])} |")
    if derived:
        lines += ["", "| depth | policy | from arm | accuracy % (95% CI) | logical loop-tokens/q | stopped after two % | "
                      "work saved vs SC@4 % | paired diff vs SC@4 (pts) |", "|---|---|---|---|---|---|---|---|"]
        for r in sorted(derived, key=lambda r: (r["depth"], r["arm"])):
            lines.append(
                f"| {r['depth']} | {r['policy']} | {r['arm']} | {100 * r['accuracy']:.1f} ({100 * r['accuracy_ci95'][0]:.1f}, "
                f"{100 * r['accuracy_ci95'][1]:.1f}) | {r['logical_loop_tokens_per_question']:.0f} | "
                f"{100 * r['stopped_after_two']:.1f} | {100 * r['work_saved_vs_sc4']:.1f} | {100 * r['paired_diff_vs_sc4']:+.1f} |")
    if ext:
        lines += ["", "| arm | cap | capped candidates rerun | old tokens reproduced | finish with EOS | answered before -> after | "
                      "correct before -> after | policy accuracy before -> after % |", "|---|---|---|---|---|---|---|---|"]
        for r in ext:
            lines.append(
                f"| {r['arm']} | {r['from_cap']} -> {r['cap']} | {r['capped_candidates_rerun']} | {r['prefix_reproduced']} | "
                f"{r['now_finish_with_eos']} | {r['answered_before']} -> {r['answered_after']} | "
                f"{r['correct_before']} -> {r['correct_after']} | {100 * r['policy_accuracy_before']:.1f} -> "
                f"{100 * r['policy_accuracy_after']:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--grader", default="math-verify")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gold = {r["question_id"]: r for r in read_jsonl(args.gold)}
    info = ai.grader_info(args.grader)
    arms = {}
    for name in sorted(os.listdir(args.runs)):
        d = os.path.join(args.runs, name)
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        rows = grade_rows(read_jsonl(os.path.join(d, "candidates.jsonl")), gold, args.grader)
        with open(os.path.join(d, "graded.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("run_id", "question_id", "candidate_id", "pred", "answer_key",
                                                      "parse_status", "finish_reason", "correct", "grader_error")},
                                   ensure_ascii=False) + "\n")
        arms[name] = {"rows": rows, "questions": read_jsonl(os.path.join(d, "questions.jsonl")),
                      "summary": json.load(open(os.path.join(d, "summary.json"))),
                      "manifest": json.load(open(os.path.join(d, em.MANIFEST_FILE)))}
    table, derived, ext, per_q = [], [], [], {}
    by_hash = {a["manifest"]["config_hash"]: n for n, a in arms.items()}
    for name, a in arms.items():
        if "policy" in a["summary"]:
            row, pq = policy_table(name, a["rows"], a["questions"], a["summary"], args)
            row["peak_kv_gb"] = a["summary"]["peak_used_blocks"] * BLOCK_BYTES / 2 ** 30
            table.append(row)
            per_q[name] = pq
        elif a["summary"]["arm"]["kind"] == "fixed":
            row, der, pq = arm_table(name, a["rows"], a["questions"], a["summary"], args)
            row["peak_kv_gb"] = a["summary"]["peak_used_blocks"] * BLOCK_BYTES / 2 ** 30
            table.append(row)
            derived += der
            per_q[name] = pq
        else:
            src = arms[by_hash[a["summary"]["arm"]["from_run"]]]
            ext.append(extension_table(name, a["rows"], a["summary"], src["rows"], src["questions"]))
    out = {"grader": info, "identity_version": ai.STRUCTURAL, "tie_rule": es.LOWEST_CANDIDATE,
           "gold_sha256": em.file_sha256(args.gold), "bootstrap": {"iters": args.iters, "seed": args.seed},
           "arms": table, "replayed_policies": derived, "cap_extensions": ext, "per_question_correct": per_q}
    with open(os.path.join(args.runs, "table.json"), "w") as f:
        json.dump(out, f, indent=1)
    md = markdown(table, derived, ext)
    with open(os.path.join(args.runs, "table.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
