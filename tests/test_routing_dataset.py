"""Failure modes for the router-pilot loader and the historical vote."""
import json
import os

import pytest

from looped.routing_dataset import (
    PilotDataError,
    join_arm,
    load_outcomes,
    load_pilot,
    vote_from_candidates,
)


def _row(qid, depth, sample, tokens, parse="ok", finish="eos", key="7", pred="7"):
    return {
        "question_id": qid,
        "candidate_id": f"{qid}:d{depth}:s{sample}",
        "depth": depth,
        "sample": sample,
        "mode": "natural",
        "parse_status": parse,
        "finish_reason": finish,
        "answer_key": key,
        "pred": pred,
        "logical_loop_tokens": tokens,
    }


def _grade(row, correct):
    return {
        "question_id": row["question_id"],
        "candidate_id": row["candidate_id"],
        "pred": row["pred"],
        "answer_key": row["answer_key"],
        "parse_status": row["parse_status"],
        "finish_reason": row["finish_reason"],
        "correct": correct,
        "grader_error": None,
    }


def _four(qid, depth, finishes, parses, keys, corrects, tokens):
    rows = []
    for sample in range(4):
        key = keys[sample]
        rows.append(_row(qid, depth, sample, tokens[sample], parse=parses[sample],
                         finish=finishes[sample], key=key, pred=key))
        rows[-1]["correct"] = bool(corrects[sample])
    return rows


def test_historical_vote_keeps_a_completed_capped_answer_and_strict_drops_it():
    rows = _four(
        "q", 1,
        ["length_cap", "eos", "eos", "eos"],
        ["ok", "no_boxed", "no_boxed", "no_boxed"],
        ["1", None, None, None],
        [True, False, False, False],
        [10, 5, 5, 5],
    )
    assert vote_from_candidates(rows, strict_cap=False) is True
    assert vote_from_candidates(rows, strict_cap=True) is False
    assert sum(row["logical_loop_tokens"] for row in rows) == 25


def test_tie_goes_to_lowest_sample_and_an_empty_vote_is_wrong():
    rows = _four(
        "q", 2,
        ["eos"] * 4,
        ["ok"] * 4,
        ["A", "B", "A", "B"],
        [False, True, False, True],
        [1, 1, 1, 1],
    )
    assert vote_from_candidates(rows, strict_cap=False) is False
    empty = _four(
        "q", 1,
        ["eos"] * 4,
        ["no_boxed"] * 4,
        [None] * 4,
        [False] * 4,
        [3, 4, 5, 6],
    )
    assert vote_from_candidates(empty, strict_cap=False) is False


def test_cost_includes_wrong_unparseable_and_capped_candidates():
    rows = _four(
        "q", 1,
        ["eos", "length_cap", "eos", "length_cap"],
        ["ok", "no_boxed", "empty", "unterminated_boxed"],
        ["9", None, None, None],
        [False, False, False, False],
        [2, 4, 8, 16],
    )
    grades = [_grade(row, row_correct) for row, row_correct in zip(rows, [False, False, False, False])]
    joined = join_arm(rows, grades, question_ids=["q"], depth=1, run_seed=0, arm="t")
    assert sum(item["logical_loop_tokens"] for item in joined["q"]) == 30
    assert vote_from_candidates(joined["q"], strict_cap=False) is False


def test_duplicate_candidate_missing_grade_and_unequal_questions_fail():
    row = _row("q", 1, 0, 1)
    with pytest.raises(PilotDataError, match="duplicate"):
        join_arm([row, row], [_grade(row, False)], question_ids=["q"], depth=1, run_seed=0, arm="t")
    with pytest.raises(PilotDataError, match="missing grade"):
        join_arm([row], [], question_ids=["q"], depth=1, run_seed=0, arm="t")
    other = _row("other", 1, 0, 1)
    with pytest.raises(PilotDataError, match="unexpected"):
        join_arm([other], [_grade(other, False)], question_ids=["q"], depth=1, run_seed=0, arm="t")
    with pytest.raises(PilotDataError, match="missing questions"):
        join_arm([row], [_grade(row, True)], question_ids=["q", "absent"], depth=1, run_seed=0, arm="t")


def test_missing_arm_fails(tmp_path):
    with pytest.raises(PilotDataError, match="missing arm"):
        load_outcomes(str(tmp_path), ["q"], depths=(1,), run_seeds=(0,))


def test_unexpected_finish_reason_is_not_silently_counted():
    rows = _four("q", 1, ["stop"] + ["eos"] * 3, ["ok"] * 4, ["1"] * 4, [True] * 4, [1, 1, 1, 1])
    with pytest.raises(PilotDataError, match="finish_reason"):
        vote_from_candidates(rows, strict_cap=False)


def test_saved_pilot_reproduces_published_counts():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pilot = os.path.join(root, "artifacts/looped_serving/loop_routing/pilot")
    if not os.path.isdir(pilot):
        pytest.skip("router pilot artifacts are not in this checkout")
    data = load_pilot(root, reproduce=True)
    assert data.n == 200
    assert len(set(data.group_ids)) == 200
    assert data.audit["reproduction"]["primary_correct"]["0"] == (57, 152, 175, 179)
    assert json.dumps(data.audit)
