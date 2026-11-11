#!/usr/bin/env python
"""Export one frozen split as two files: serving inputs and (separately) gold.

The generator only ever reads ``<split>.items.jsonl`` (question id and text).
Gold answers, level and subject go to ``<split>.gold.jsonl`` and are joined at
grading time, so no serving-side record can carry a label. Every exported row is checked
against the stable question id in ``configs/looped_serving/splits.json``; a
loader that silently fell back to another dataset fails here, not on the GPU box.

    python scripts/export_looped_split.py --split math_train_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped import experiment_manifest as em  # noqa: E402

MATH_TRAIN = "EleutherAI/hendrycks_math"


def export(split: str, splits_path: str) -> tuple:
    from data import load_math
    from utils import extract_boxed_answer

    spec = json.load(open(splits_path))
    if split not in spec["splits"] or not split.startswith("math_train_"):
        raise SystemExit(f"{split}: not a MATH-Train split in {splits_path}")
    revision = spec["datasets"]["math_train"]["revision"]
    train = list(load_math("train"))
    items, gold = [], []
    for e in spec["splits"][split]["items"]:
        r = train[e["loader_index"]]
        qid = em.question_id(MATH_TRAIN, revision, "train", r["question"])
        if qid != e["question_id"] or em.text_hash(r["question"]) != e["text_hash"]:
            raise SystemExit(f"loader row {e['loader_index']} does not match frozen id {e['question_id']}")
        items.append({"question_id": qid, "text": r["question"]})
        gold.append({"question_id": qid, "gold": extract_boxed_answer(r["solution"]) or r["gold"],
                     "level": e["level"], "subject": e["subject"]})
    return spec["split_hash"], items, gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--splits", default="configs/looped_serving/splits.json")
    ap.add_argument("--out", default="artifacts/looped_serving/splits")
    args = ap.parse_args()
    split_hash, items, gold = export(args.split, args.splits)
    for row in items:
        em.assert_no_gold(row)
    os.makedirs(args.out, exist_ok=True)
    for name, rows in (("items", items), ("gold", gold)):
        path = os.path.join(args.out, f"{args.split}.{name}.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(rows)} rows sha256={em.file_sha256(path)[:16]}")
    print(f"split_hash={split_hash}")


if __name__ == "__main__":
    main()
