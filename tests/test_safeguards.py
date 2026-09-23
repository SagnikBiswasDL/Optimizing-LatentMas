"""Tests for the measurement safeguards, not for the model.

Each test here corresponds to a way a sweep silently produced a wrong number:
grouped decoding that failed parity but kept going, a tape reused across
datasets, and an arm compared against a baseline over a different item set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import exp_aime_localize as E  # noqa: E402


def cfg(**kw):
    d = dict(
        tape_dir="", tape_dir_exact=False, out_dir="/out", task="aime2024",
        model_name="Qwen/Qwen3-14B", k=10, allow_diverge=False,
        baseline_arm="real", exclude_arms="", judger_budget=8192,
    )
    d.update(kw)
    return argparse.Namespace(**d)


# ---------------------------------------------------------------- tape identity

def test_tape_root_isolates_by_task_model_and_k(tmp_path):
    base = cfg(tape_dir=str(tmp_path))
    paths = {
        E.tape_path(base, 0),
        E.tape_path(cfg(tape_dir=str(tmp_path), task="aime2025"), 0),
        E.tape_path(cfg(tape_dir=str(tmp_path), k=5), 0),
        E.tape_path(cfg(tape_dir=str(tmp_path), model_name="Qwen/Qwen3-8B"), 0),
    }
    assert len(paths) == 4, "each config must get its own tape directory"


def test_tape_dir_exact_opts_out_of_the_subdir(tmp_path):
    p = E.tape_path(cfg(tape_dir=str(tmp_path), tape_dir_exact=True), 0)
    assert p == os.path.join(str(tmp_path), "0000.pt")


@pytest.mark.parametrize("override,field", [
    ({"task": "aime2025"}, "task"),
    ({"k": 5}, "k"),
    ({"model_name": "Qwen/Qwen3-8B"}, "model"),
])
def test_identity_mismatch_is_detected(override, field):
    args = cfg()
    tape = {"idx": 0, "task": "aime2024", "k": 10, "question": "Q",
            "identity": E.tape_identity(args, 0, "Q")}
    bad = E.check_tape_identity(tape, cfg(**override), "Q")
    assert any(field in b for b in bad), f"{field} mismatch not reported: {bad}"


def test_identity_accepts_a_matching_tape():
    args = cfg()
    tape = {"idx": 0, "task": "aime2024", "k": 10, "question": "Q",
            "identity": E.tape_identity(args, 0, "Q")}
    assert E.check_tape_identity(tape, args, "Q") == []


def test_question_mismatch_is_detected_even_when_task_and_k_match():
    """The failure that motivated this: right index, right task, wrong problem."""
    args = cfg()
    tape = {"idx": 0, "task": "aime2024", "k": 10, "question": "ORIGINAL",
            "identity": E.tape_identity(args, 0, "ORIGINAL")}
    bad = E.check_tape_identity(tape, args, "A DIFFERENT PROBLEM")
    assert any("question" in b for b in bad)


def test_pre_identity_tape_is_unverifiable_not_assumed_good():
    bad = E.check_tape_identity({"idx": 0, "question": "Q"}, cfg(), "Q")
    assert bad, "a tape with no provenance must not silently pass"


def _write_tape(path, idx, task="aime2024", k=10, model="Qwen/Qwen3-14B"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"idx": idx, "task": task, "k": k, "question": f"Q{idx}",
                "identity": {"idx": idx, "task": task, "k": k, "model": model,
                             "question_sha": "0" * 16}}, path)


def test_list_tapes_rejects_a_foreign_tape(tmp_path):
    args = cfg(tape_dir=str(tmp_path))
    root = E.tape_root(args)
    _write_tape(os.path.join(root, "0000.pt"), 0)
    _write_tape(os.path.join(root, "0001.pt"), 1, task="aime2025")
    with pytest.raises(SystemExit) as ei:
        E.list_tapes(args)
    assert "aime2025" in str(ei.value)


def test_list_tapes_filters_by_index_before_loading(tmp_path, monkeypatch):
    args = cfg(tape_dir=str(tmp_path))
    root = E.tape_root(args)
    for i in (0, 1, 2):
        _write_tape(os.path.join(root, f"{i:04d}.pt"), i)
    loaded = []
    real_load = E.load_tape
    monkeypatch.setattr(E, "load_tape", lambda p: (loaded.append(p), real_load(p))[1])
    got = E.list_tapes(args, [1])
    assert [t["idx"] for t in got] == [1]
    assert len(loaded) == 1, "scoping must skip the ~130MB load for unwanted tapes"


# ------------------------------------------------------------------ parity gate

def _fixture(tmp_path, rows, texts):
    out = tmp_path / "out"
    (out / "texts").mkdir(parents=True)
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for (idx, arm), body in texts.items():
        (out / "texts" / f"{idx:04d}_{arm}.txt").write_text(body)
    return str(out)


def _row(arm, idx, tokens, correct, eos, judger_s=100.0):
    return {"method": arm, "idx": idx, "tokens": tokens, "correct": correct,
            "eos": eos, "judger_s": judger_s, "task": "aime2024",
            "pred": "1", "cache_pos": 100, "cache_mb": 1.0}


def test_compare_exits_nonzero_when_parity_fails(tmp_path):
    out = _fixture(
        tmp_path,
        [_row("a", 0, 10, True, True), _row("b", 0, 11, False, True)],
        {(0, "a"): "hello", (0, "b"): "world"},
    )
    args = cfg(out_dir=out, compare_arms="a,b")
    with pytest.raises(SystemExit) as ei:
        E.run_compare(args)
    assert "PARITY FAILED" in str(ei.value)


def test_allow_diverge_downgrades_the_gate_to_a_record(tmp_path):
    out = _fixture(
        tmp_path,
        [_row("a", 0, 10, True, True), _row("b", 0, 11, False, True)],
        {(0, "a"): "hello", (0, "b"): "world"},
    )
    args = cfg(out_dir=out, compare_arms="a,b", allow_diverge=True)
    res = E.run_compare(args)
    assert res["verdict"] == "diverged"


def test_compare_passes_when_texts_match(tmp_path):
    out = _fixture(
        tmp_path,
        [_row("a", 0, 10, True, True), _row("b", 0, 10, True, True)],
        {(0, "a"): "same", (0, "b"): "same"},
    )
    res = E.run_compare(cfg(out_dir=out, compare_arms="a,b"))
    assert res["verdict"] == "identical"


# --------------------------------------------------------------- latency report

def _latency_fixture(tmp_path):
    rows = [
        # baseline: solves 0 and 1 cleanly, censored on 2
        _row("real", 0, 1000, True, True),
        _row("real", 1, 2000, True, True),
        _row("real", 2, 8192, False, False),
        # arm wins on item 0, but loses item 1 to the cap
        _row("cand", 0, 500, True, True),
        _row("cand", 1, 8192, False, False),
        _row("cand", 2, 8192, False, False),
    ]
    return _fixture(tmp_path, rows, {})


def test_latency_pairs_on_baseline_completed_correct_items(tmp_path):
    out = _latency_fixture(tmp_path)
    res = E.run_latency(cfg(out_dir=out))
    p = res["paired_vs_baseline"]["cand"]
    # item 2 is excluded: the baseline never completed it, so it anchors nothing
    assert p["anchor_items"] == [0, 1]
    assert p["baseline_tokens"] == 3000
    assert p["arm_tokens"] == 8692


def test_latency_marks_a_censored_aggregate_as_a_lower_bound(tmp_path):
    out = _latency_fixture(tmp_path)
    res = E.run_latency(cfg(out_dir=out))
    p = res["paired_vs_baseline"]["cand"]
    assert p["arm_tokens_is_lower_bound"] is True
    assert p["censored_items"] == [1]
    assert p["ratio"] > 1.0, "spending more tokens than baseline must read as slower"


def test_latency_separates_the_win_on_kept_solves_from_the_losses(tmp_path):
    out = _latency_fixture(tmp_path)
    res = E.run_latency(cfg(out_dir=out))
    p = res["paired_vs_baseline"]["cand"]
    assert p["solves_kept"] == [0]
    assert p["solves_lost"] == [1]
    # 1000 -> 500 on the one item it kept: a real 2x, but not the whole story
    assert p["speedup_on_kept"] == pytest.approx(2.0)


def test_latency_reports_missing_items_as_missing(tmp_path):
    """An arm that never ran an item must not be credited with a wrong answer."""
    rows = [
        _row("real", 0, 1000, True, True),
        _row("real", 1, 2000, True, True),
        _row("thin", 0, 800, True, True),   # item 1 never run
    ]
    out = _fixture(tmp_path, rows, {})
    res = E.run_latency(cfg(out_dir=out))
    assert res["arms"]["thin"]["missing"] == [1]
    assert res["arms"]["thin"]["n"] == 1, "n must count observations, not the universe"
    # and the pairing must say which anchor items it could not cover
    assert res["paired_vs_baseline"]["thin"]["missing_from_anchor"] == [1]


def test_latency_excludes_named_arms(tmp_path):
    rows = [_row("real", 0, 1000, True, True), _row("bad__bs16", 0, 900, True, True)]
    out = _fixture(tmp_path, rows, {})
    res = E.run_latency(cfg(out_dir=out, exclude_arms="bad__bs16"))
    assert "bad__bs16" not in res["arms"]


def test_latency_mean_tokens_uses_completed_runs_only(tmp_path):
    out = _latency_fixture(tmp_path)
    res = E.run_latency(cfg(out_dir=out))
    # real completed items 0 and 1 only -> mean over {1000, 2000}, not the 8192
    assert res["arms"]["real"]["mean_tokens_completed"] == pytest.approx(1500.0)
    assert res["arms"]["real"]["n_censored"] == 1
