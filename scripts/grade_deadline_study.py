#!/usr/bin/env python
"""Score deadline-study runs and apply the predeclared decision rules.

Gold is read here and nowhere in the driver. ``--role select`` freezes the
mixture from the calibration runs. ``--role gate`` scores the validation
repetitions against that frozen mixture. A smoke run is ignored by both.

    python scripts/grade_deadline_study.py --role select \
        --runs artifacts/looped_serving/deadline_study/calibration \
        --gold artifacts/looped_serving/splits/math_train_calibration.gold.jsonl \
        --splits configs/looped_serving/splits.json

    python scripts/grade_deadline_study.py --role gate \
        --runs artifacts/looped_serving/deadline_study/validation \
        --gold artifacts/looped_serving/splits/math_train_validation.gold.jsonl \
        --splits configs/looped_serving/splits.json \
        --selection artifacts/looped_serving/deadline_study/calibration/selection.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped import answer_identity as ai  # noqa: E402
from looped import deadline as D  # noqa: E402
from looped import experiment_manifest as em  # noqa: E402


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score_run(path: str, gold: dict) -> dict:
    info = json.load(open(os.path.join(path, "run.json")))
    if info.get("study") != D.STUDY_NAME:
        raise SystemExit(f"{path} is not a run of {D.STUDY_NAME}")
    questions = read_jsonl(os.path.join(path, "questions.jsonl"))
    if [q["question_id"] for q in questions] != info["question_ids"]:
        raise SystemExit(f"{path}: questions.jsonl is not the recorded arrival list")
    preds = {}
    cand_path = os.path.join(path, "candidates.jsonl")
    if os.path.exists(cand_path):
        for row in read_jsonl(cand_path):
            preds[row["candidate_id"]] = row.get("pred")
    correct = {}
    for q in questions:
        cid = q.get("selected_candidate_id")
        if not q["deadline_met"] or not cid:
            correct[q["question_id"]] = False
            continue
        graded = ai.grade(preds.get(cid), gold[q["question_id"]]["gold"])
        correct[q["question_id"]] = bool(graded["correct"])
    scored = D.primary_fraction(questions, correct)
    scored.update(arm=info["arm"], candidate=info.get("candidate"), seed=info["seed"],
                  split=info["split"], smoke=bool(info.get("smoke")),
                  long_questions=sum(bool(q["launched_long"]) for q in questions),
                  question_ids=info["question_ids"])
    return scored


def discover(runs_dir: str, gold: dict) -> list:
    found = []
    for name in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, name)
        if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "run.json")):
            continue
        found.append(score_run(path, gold))
    return [r for r in found if not r["smoke"]]


def load_gold(path: str) -> dict:
    return {r["question_id"]: r for r in read_jsonl(path)}


def select(runs, splits) -> dict:
    mixture = [r for r in runs if r["arm"] == "mixture" and r["seed"] == D.CALIBRATION_SEED
               and r["split"] == D.CALIBRATION_SPLIT]
    expected = [e["question_id"] for e in splits["splits"][D.CALIBRATION_SPLIT]["items"]]
    for r in mixture:
        if r["question_ids"] != expected:
            raise SystemExit(f"{r['candidate']} is not the calibration split in its recorded order")
        banned = D.locked_overlap(r["question_ids"], splits)
        if banned:
            raise SystemExit(f"calibration run overlaps a locked split: {sorted(banned)}")
    chosen = D.choose_mixture(mixture)
    return {"study": D.STUDY_NAME, "split": D.CALIBRATION_SPLIT, "chosen": chosen,
            "runs": [{k: r[k] for k in ("candidate", "hits", "n", "fraction", "long_questions")}
                     for r in mixture]}


def gate(runs, selection, splits) -> dict:
    if selection.get("study") != D.STUDY_NAME or selection.get("chosen") not in D.MIXTURE_CANDIDATES:
        raise SystemExit("the selection file is not a frozen mixture for this study")
    expected = [e["question_id"] for e in splits["splits"][D.VALIDATION_SPLIT]["items"]]
    chosen = selection["chosen"]
    reps = []
    for seed in D.REPETITIONS:
        rep = {}
        for arm in (*D.STATIC_ARMS, D.ADAPTIVE_ARM):
            rows = [r for r in runs if r["arm"] == arm and r["seed"] == seed and r["split"] == D.VALIDATION_SPLIT]
            if arm == "mixture":
                rows = [r for r in rows if r["candidate"] == chosen]
            if len(rows) != 1:
                raise SystemExit(f"seed {seed} arm {arm}: expected one validation run, found {len(rows)}")
            row = rows[0]
            if row["question_ids"] != expected:
                raise SystemExit(f"{arm} seed {seed} is not the validation split in its recorded order")
            banned = D.locked_overlap(row["question_ids"], splits)
            if banned:
                raise SystemExit(f"validation run overlaps a locked split: {sorted(banned)}")
            rep[arm] = row["fraction"]
        reps.append(rep)
    return {"study": D.STUDY_NAME, "chosen_mixture": chosen, "repetitions": reps, **D.development_gate(reps)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=("select", "gate"))
    ap.add_argument("--runs", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--splits", default="configs/looped_serving/splits.json")
    ap.add_argument("--selection", default="")
    args = ap.parse_args()
    splits = em.load_splits(args.splits)
    gold = load_gold(args.gold)
    runs = discover(args.runs, gold)
    if args.role == "select":
        result = select(runs, splits)
        dest = os.path.join(args.runs, "selection.json")
    else:
        if not args.selection:
            raise SystemExit("--role gate needs --selection")
        result = gate(runs, json.load(open(args.selection)), splits)
        dest = os.path.join(args.runs, "gate.json")
    with open(dest, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: result[k] for k in result if k != "runs"}))


if __name__ == "__main__":
    main()
