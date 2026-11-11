#!/usr/bin/env python
"""Write the ids of the questions on which a saved arm's votes do not reach a quorum.

This is the population a "reasoning on disagreement" fallback would act on. It
uses only the candidates' vote keys, never gold.

    python scripts/select_looped_questions.py --arm <arm dir> --quorum 3 --out ids.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped import answer_identity as ai  # noqa: E402
from looped import ensemble_state as es  # noqa: E402


def without_quorum(rows, quorum: int, width: int) -> list:
    """Question ids whose first ``width`` samples have no answer shared by ``quorum`` of them, or tie for the lead."""
    by_q = defaultdict(list)
    for r in rows:
        if 0 <= r["sample"] < width:
            by_q[r["question_id"]].append(es.Vote(r["sample"], ai.answer_from_text(r["text"])[1].key))
    out = []
    for q, votes in by_q.items():
        c1, c2 = es.lead(votes)
        if c1 < quorum or c1 == c2:
            out.append(q)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--quorum", type=int, default=3)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(os.path.join(args.arm, "candidates.jsonl"))]
    ids = without_quorum(rows, args.quorum, args.width)
    with open(args.out, "w") as f:
        f.write("".join(q + "\n" for q in ids))
    print(f"{len(ids)} of {len({r['question_id'] for r in rows})} questions have no quorum of {args.quorum} among {args.width}")


if __name__ == "__main__":
    main()
