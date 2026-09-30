"""Tests for the measurement safeguards, not for the model.

Each test here corresponds to a way a sweep silently produced a wrong number:
grouped decoding that failed parity but kept going, a tape reused across
datasets, and an arm compared against a baseline over a different item set.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import exp_aime_localize as E  # noqa: E402


def cfg(**kw):
    d = dict(
        tape_dir="", tape_dir_exact=False, out_dir="/out", task="aime2024",
        model_name="Qwen/Qwen3-14B", k=10, allow_diverge=False,
        baseline_arm="real", exclude_arms="", judger_budget=8192, decode_path="",
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


def _row(arm, idx, tokens, correct, eos, judger_s=100.0, path="sdpa+dynamic"):
    return {"method": arm, "idx": idx, "tokens": tokens, "correct": correct,
            "eos": eos, "judger_s": judger_s, "task": "aime2024",
            "pred": "1", "cache_pos": 100, "cache_mb": 1.0, "decode_path": path}


# --------------------------------------------------------- decode provenance

def test_decode_path_names_the_kernels():
    class M:
        class config:
            _attn_implementation = "flash_attention_2"

    w = argparse.Namespace(model=M(), use_static_cache=True, compiled=True)
    assert E.decode_path(w) == "flash_attention_2+static+compile"
    w2 = argparse.Namespace(model=M(), use_static_cache=False, compiled=False)
    assert E.decode_path(w2) == "flash_attention_2+dynamic"


def test_mixing_decode_paths_is_refused():
    """Different kernels change reduction order, and bf16 greedy is not invariant."""
    rows = [_row("real", 0, 100, True, True, path="sdpa+dynamic"),
            _row("cand", 0, 90, True, True, path="sdpa+static+compile")]
    with pytest.raises(SystemExit) as ei:
        E.assert_one_decode_path(rows, "test")
    assert "mixes decode paths" in str(ei.value)


def test_one_decode_path_is_accepted():
    rows = [_row("real", 0, 100, True, True), _row("cand", 0, 90, True, True)]
    assert E.assert_one_decode_path(rows, "test") == "sdpa+dynamic"


def test_decode_path_filter_scopes_an_analysis():
    rows = [_row("real", 0, 100, True, True, path="sdpa+dynamic"),
            _row("real", 1, 90, True, True, path="sdpa+static+compile")]
    got = E.filter_decode_path(rows, cfg(decode_path="sdpa+static+compile"))
    assert [r["idx"] for r in got] == [1]


def test_latency_refuses_mixed_paths(tmp_path):
    rows = [_row("real", 0, 1000, True, True, path="sdpa+dynamic"),
            _row("fast", 0, 900, True, True, path="sdpa+static+compile")]
    out = _fixture(tmp_path, rows, {})
    with pytest.raises(SystemExit) as ei:
        E.run_latency(cfg(out_dir=out, decode_path=""))
    assert "mixes decode paths" in str(ei.value)


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


# --------------------------------------------------------- promotion decision

CAP = 16384


def _promo_cfg(out, cands, **kw):
    d = dict(out_dir=out, candidate_arms=cands, promote_cap=CAP, promote_tag="b16k",
             promote_indices="0,1,4,10", min_saving=0.10, baseline_arm="real")
    d.update(kw)
    return cfg(**d)


def _promo_rows(cand_rows):
    """Baseline: solves 0,4,10 cheaply; item 1 censored even at the raised cap."""
    return [
        _row("real", 0, 2600, True, True),
        _row("real", 4, 7653, True, True),
        _row("real", 10, 5848, True, True),
        _row("real__b16k", 1, CAP, False, False),
    ] + cand_rows


def test_promote_refuses_to_score_against_a_censored_baseline(tmp_path):
    """Item 1 censored at the OLD cap is not a usable baseline number."""
    rows = [_row("real", 0, 2600, True, True), _row("real", 1, 8192, False, False)]
    out = _fixture(tmp_path, rows, {})
    with pytest.raises(SystemExit) as ei:
        E.run_promote(_promo_cfg(out, "real_seal20", promote_indices="0,1"))
    assert "UNUSABLE" in str(ei.value)


def test_promote_reuses_a_terminated_row_from_a_smaller_cap(tmp_path):
    """A run that emitted EOS below the old cap is cap-independent, so reusable."""
    out = _fixture(tmp_path, _promo_rows([
        _row("real_seal20__b16k", 0, 1700, True, True),
        _row("real_seal20__b16k", 4, 5200, True, True),
        _row("real_seal20__b16k", 10, 4100, True, True),
        _row("real_seal20__b16k", 1, CAP, False, False),
    ]), {})
    res = E.run_promote(_promo_cfg(out, "real_seal20"))
    # baseline items 0,4,10 have no b16k row, so they are reused from the 8192 run;
    # item 1 does, because it was censored there and had to be re-decoded.
    assert res["baseline_total"] == 2600 + 7653 + 5848 + CAP
    cand = res["candidates"]["real_seal20"]["per_item"]
    assert cand["0"]["source"] == "measured"


def test_promote_accepts_a_candidate_that_keeps_solves_and_cuts_enough(tmp_path):
    out = _fixture(tmp_path, _promo_rows([
        _row("real_seal20__b16k", 0, 1700, True, True),
        _row("real_seal20__b16k", 4, 5200, True, True),
        _row("real_seal20__b16k", 10, 4100, True, True),
        _row("real_seal20__b16k", 1, CAP, False, False),
    ]), {})
    res = E.run_promote(_promo_cfg(out, "real_seal20"))
    d = res["candidates"]["real_seal20"]
    assert d["keeps_all_solves"] and d["meets_token_rule"] and d["promote"]
    assert res["promoted"] == ["real_seal20"]


def test_promote_rejects_a_candidate_that_loses_a_solve_however_cheap(tmp_path):
    """Token cuts must not be purchasable by failing to finish."""
    out = _fixture(tmp_path, _promo_rows([
        _row("real_seal80__b16k", 0, 100, True, True),
        _row("real_seal80__b16k", 4, 100, True, True),
        _row("real_seal80__b16k", 10, 100, True, False),   # cheap but WRONG
        _row("real_seal80__b16k", 1, 100, True, False),
    ]), {})
    res = E.run_promote(_promo_cfg(out, "real_seal80"))
    d = res["candidates"]["real_seal80"]
    assert d["solves_lost"] == [10]
    assert d["meets_token_rule"] is True, "it is cheap, but that must not be enough"
    assert d["promote"] is False


def test_promote_rejects_an_insufficient_saving(tmp_path):
    out = _fixture(tmp_path, _promo_rows([
        _row("real_seal80__b16k", 0, 2500, True, True),
        _row("real_seal80__b16k", 4, 7300, True, True),
        _row("real_seal80__b16k", 10, 5600, True, True),
        _row("real_seal80__b16k", 1, CAP, False, False),
    ]), {})
    d = E.run_promote(_promo_cfg(out, "real_seal80"))["candidates"]["real_seal80"]
    assert d["keeps_all_solves"] and not d["meets_token_rule"] and not d["promote"]


def test_promote_separates_termination_failure_from_verbosity(tmp_path):
    """The case that matters: efficient on everything it finishes, loses one item.

    Losing item 10 books (CAP - 5848) extra tokens, which alone exceeds the 10%
    bar, so the headline saving goes negative. The counterfactual must still show
    the efficiency gain, or the report would read as 'steering is verbose' when the
    truth is 'steering broke EOS'.
    """
    out = _fixture(tmp_path, _promo_rows([
        _row("real_seal60__b16k", 0, 1500, True, True),
        _row("real_seal60__b16k", 4, 4800, True, True),
        _row("real_seal60__b16k", 10, CAP, False, False),
        _row("real_seal60__b16k", 1, CAP, False, False),
    ]), {})
    d = E.run_promote(_promo_cfg(out, "real_seal60"))["candidates"]["real_seal60"]
    assert d["saving"] < 0, "headline saving is dominated by the lost solve"
    assert d["cost_per_lost_solve"]["10"] == CAP - 5848
    assert d["counterfactual_saving"] > d["saving"]
    assert d["efficiency_gain_absent_termination_failure"] is True


def test_promote_marks_saving_as_an_upper_bound_when_censored(tmp_path):
    out = _fixture(tmp_path, _promo_rows([
        _row("real_seal20__b16k", 0, 1700, True, True),
        _row("real_seal20__b16k", 4, 5200, True, True),
        _row("real_seal20__b16k", 10, 4100, True, True),
        _row("real_seal20__b16k", 1, CAP, False, False),
    ]), {})
    d = E.run_promote(_promo_cfg(out, "real_seal20"))["candidates"]["real_seal20"]
    assert d["total_is_upper_bound_on_saving"] is True


# ------------------------------------------------------------ extension check

def test_prefix_check_passes_when_the_longer_run_retraces(tmp_path):
    out = _fixture(
        tmp_path,
        [_row("real", 0, 10, False, False), _row("real__b16k", 0, 20, True, True)],
        {(0, "real"): "abcdef", (0, "real__b16k"): "abcdefGHIJ"},
    )
    res = E.run_prefix(cfg(out_dir=out, compare_arms="real,real__b16k"))
    assert res["verdict"] == "reproduced"


def test_prefix_check_fails_and_exits_when_the_extension_diverges(tmp_path):
    out = _fixture(
        tmp_path,
        [_row("real", 0, 10, False, False), _row("real__b16k", 0, 20, True, True)],
        {(0, "real"): "abcdef", (0, "real__b16k"): "abcXYZ"},
    )
    with pytest.raises(SystemExit) as ei:
        E.run_prefix(cfg(out_dir=out, compare_arms="real,real__b16k"))
    assert "EXTENSION CHECK FAILED" in str(ei.value)


def test_latency_mean_tokens_uses_completed_runs_only(tmp_path):
    out = _latency_fixture(tmp_path)
    res = E.run_latency(cfg(out_dir=out))
    # real completed items 0 and 1 only -> mean over {1000, 2000}, not the 8192
    assert res["arms"]["real"]["mean_tokens_completed"] == pytest.approx(1500.0)
    assert res["arms"]["real"]["n_censored"] == 1


def test_parse_arm_reads_a_positive_coefficient():
    assert E.parse_arm("real_seal40") == ("real", 40.0)
    assert E.parse_arm("real_seal7.5") == ("real", 7.5)


def test_parse_arm_reads_a_negative_coefficient():
    """The sign is the open question: positive coefficients lengthen output on
    AIME, so the opposite sign has to be expressible to be tested."""
    assert E.parse_arm("real_seal-40") == ("real", -40.0)
    assert E.parse_arm("frozen_seal-7.5") == ("frozen", -7.5)


def test_parse_arm_leaves_unsteered_arms_alone():
    for name in ("real", "none", "frozen", "c23", "real__b16k"):
        assert E.parse_arm(name) == (name, None)


def test_parse_arm_does_not_mistake_a_trailing_dash_for_a_coefficient():
    # A bare sign is not a number; this must not parse as coef 0 or crash.
    assert E.parse_arm("real_seal-") == ("real_seal-", None)
    assert E.parse_arm("real_seal") == ("real_seal", None)


# --- brevity sampling: the pooling must not encode length -------------------

def test_window_recorder_drops_the_prefill_entry():
    """The first forward is the prompt; its last position is not a generated
    token, and mixing it in would put question-dependent state into a direction
    meant to describe generation."""
    rec = E.WindowRecorder(layer_index=0, window=8)
    rec.buffer = [torch.full((1, 4), 99.0)]          # prefill
    rec.buffer += [torch.full((1, 4), 1.0), torch.full((1, 4), 3.0)]
    pooled = rec.pooled()
    assert pooled is not None
    assert torch.allclose(pooled, torch.full((4,), 2.0))  # not 34.33


def test_window_recorder_needs_at_least_one_generated_token():
    rec = E.WindowRecorder(layer_index=0, window=8)
    rec.buffer = [torch.zeros((1, 4))]
    assert rec.pooled() is None


def test_window_recorder_stops_at_the_window():
    """Pooling over a bounded window is what makes the representation
    length-independent; an unbounded buffer would reintroduce the confound."""
    rec = E.WindowRecorder(layer_index=0, window=3)
    for _ in range(50):
        rec._hook(None, None, torch.zeros((1, 2, 4)))
    assert len(rec.buffer) == 4  # prefill + window


def test_window_pooling_is_invariant_to_run_length():
    short, long = E.WindowRecorder(0, 2), E.WindowRecorder(0, 2)
    for rec, extra in ((short, 0), (long, 40)):
        rec.buffer = [torch.full((1, 3), 7.0)]
        rec.buffer += [torch.full((1, 3), 1.0), torch.full((1, 3), 2.0)]
        rec.buffer += [torch.full((1, 3), 9.0)] * extra  # beyond the window
        rec.buffer = rec.buffer[: rec.window + 1]
    assert torch.allclose(short.pooled(), long.pooled())


def test_variance_report_counts_items_usable_for_a_within_problem_contrast(capsys):
    meta = [
        # idx 0: two correct at different lengths -> usable
        {"idx": 0, "sample": 0, "tokens": 3000, "correct": True, "eos": True},
        {"idx": 0, "sample": 1, "tokens": 9000, "correct": True, "eos": True},
        # idx 1: one correct only -> not usable for a within-problem contrast
        {"idx": 1, "sample": 0, "tokens": 5000, "correct": True, "eos": True},
        {"idx": 1, "sample": 1, "tokens": 8000, "correct": False, "eos": True},
    ]
    res = E.report_sample_variance(meta)
    assert res[0]["n_correct"] == 2
    assert res[0]["spread"] == pytest.approx(3.0)
    assert res[1]["n_correct"] == 1 and res[1]["spread"] is None
    assert "**1**" in capsys.readouterr().out  # exactly one usable item


def test_variance_report_says_so_when_no_item_has_two_correct_samples(capsys):
    meta = [{"idx": 0, "sample": 0, "tokens": 100, "correct": False, "eos": True},
            {"idx": 0, "sample": 1, "tokens": 200, "correct": True, "eos": True}]
    E.report_sample_variance(meta)
    assert "cannot be estimated" in capsys.readouterr().out


def test_samples_mode_refuses_greedy_decoding_before_touching_a_gpu():
    """At temperature 0 every sample is the same run, so the variance this mode
    exists to measure is zero by construction. Must fail on a laptop."""
    with pytest.raises(SystemExit) as ei:
        E.run_samples(cfg(temperature=0.0, k_samples=4))
    assert "temperature" in str(ei.value)


def test_samples_mode_refuses_a_single_sample():
    with pytest.raises(SystemExit) as ei:
        E.run_samples(cfg(temperature=0.6, k_samples=1))
    assert "k_samples" in str(ei.value)


# ------------------------------------------------- pre-registered paired test
# The small sweep of 2026-09-29 showed token counts swinging >150% of baseline
# between adjacent coefficients, so the paired analysis has to be trustworthy
# before it is pointed at 6 GPU-hours of data.

import paired_coef_test as P  # noqa: E402


def _pair_row(idx, method, tokens, correct=True, eos=True):
    return {"idx": idx, "method": method, "tokens": tokens,
            "correct": correct, "eos": eos}


def _pair_rows_file(tmp_path, rows):
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def _run(monkeypatch, path, coef="5"):
    """Invoke the CLI without leaking sys.argv into other tests."""
    monkeypatch.setattr(sys, "argv", ["paired_coef_test", "--rows", path,
                                      "--coef", coef])
    P.main()


def test_load_pairs_only_keeps_items_with_both_arms(tmp_path):
    path = _pair_rows_file(tmp_path, [
        _pair_row(0, "real__b16k", 100), _pair_row(0, "real_seal5__b16k", 90),
        _pair_row(1, "real__b16k", 100),                      # no steered partner
        _pair_row(2, "real_seal5__b16k", 90),                 # no baseline partner
    ])
    pairs = P.load_pairs(path, "b16k", "5")
    assert sorted(pairs) == [0]


def test_load_pairs_does_not_confuse_coefficients_or_tags(tmp_path):
    path = _pair_rows_file(tmp_path, [
        _pair_row(0, "real__b16k", 100),
        _pair_row(0, "real_seal50__b16k", 90),   # coef 50, not 5
        _pair_row(1, "real__b16k", 100),
        _pair_row(1, "real_seal5__fast", 90),    # different tag
        _pair_row(2, "real__b16k", 100),
        _pair_row(2, "real_seal5__b16k", 80),
    ])
    pairs = P.load_pairs(path, "b16k", "5")
    assert sorted(pairs) == [2]


def test_negative_coefficient_pairs_resolve(tmp_path):
    path = _pair_rows_file(tmp_path, [
        _pair_row(0, "real__b16k", 100), _pair_row(0, "real_seal-5__b16k", 90),
    ])
    assert sorted(P.load_pairs(path, "b16k", "-5")) == [0]


def test_binom_sign_test_matches_known_values():
    assert P.binom_two_sided(0, 0) == 1.0
    assert P.binom_two_sided(5, 10) == pytest.approx(1.0)
    # 10/10 one way: 2 * 0.5**10
    assert P.binom_two_sided(10, 10) == pytest.approx(2 * 0.5 ** 10)
    assert P.binom_two_sided(9, 10) == pytest.approx(2 * 11 * 0.5 ** 10)


def test_permutation_test_is_exact_and_symmetric_for_small_n():
    p_a, exact_a = P.perm_two_sided([-0.3] * 6)
    p_b, exact_b = P.perm_two_sided([0.3] * 6)
    assert exact_a and exact_b
    # A unanimous effect of either sign is the most extreme of 2**6 assignments,
    # and the test must not care which direction it points.
    assert p_a == pytest.approx(p_b)
    assert p_a == pytest.approx(2 / 64)


def test_permutation_test_finds_no_effect_in_symmetric_noise():
    p, _ = P.perm_two_sided([0.5, -0.5, 0.4, -0.4, 0.3, -0.3])
    assert p > 0.9


def test_permutation_test_switches_to_monte_carlo_and_never_returns_zero():
    p, exact = P.perm_two_sided([-0.5] * 25, n_mc=2000, seed=1)
    assert not exact
    assert p > 0.0


def test_censored_pairs_are_excluded_from_the_token_test(tmp_path, capsys, monkeypatch):
    # The steered arm "saves" tokens on item 0 but ran item 1 into the cap. The
    # capped pair must not be counted as a 60% saving.
    path = _pair_rows_file(tmp_path, [
        _pair_row(0, "real__b16k", 1000), _pair_row(0, "real_seal5__b16k", 800),
        _pair_row(1, "real__b16k", 1000), _pair_row(1, "real_seal5__b16k", 400, eos=False),
    ])
    _run(monkeypatch, path)
    out = capsys.readouterr().out
    assert "usable for the token test (both finished, both correct): 1" in out
    assert "steered 1" in out  # censoring reported


def test_incorrect_pairs_are_excluded_from_the_token_test(tmp_path, capsys, monkeypatch):
    path = _pair_rows_file(tmp_path, [
        _pair_row(0, "real__b16k", 1000), _pair_row(0, "real_seal5__b16k", 800),
        _pair_row(1, "real__b16k", 1000), _pair_row(1, "real_seal5__b16k", 100, correct=False),
    ])
    _run(monkeypatch, path)
    out = capsys.readouterr().out
    assert "both finished, both correct): 1" in out


def test_decision_rule_rejects_a_saving_that_costs_accuracy(tmp_path, capsys, monkeypatch):
    # Every item 30% shorter, but steering breaks two solves: must REJECT.
    rows = []
    for i in range(8):
        rows += [_pair_row(i, "real__b16k", 1000),
                 _pair_row(i, "real_seal5__b16k", 700, correct=i >= 2)]
    _run(monkeypatch, _pair_rows_file(tmp_path, rows))
    out = capsys.readouterr().out
    assert "[FAIL] accuracy not worse" in out
    assert "=> REJECT" in out


def test_decision_rule_rejects_a_saving_that_costs_termination(tmp_path, capsys, monkeypatch):
    rows = []
    for i in range(8):
        rows += [_pair_row(i, "real__b16k", 1000),
                 _pair_row(i, "real_seal5__b16k", 700, eos=i >= 2)]
    _run(monkeypatch, _pair_rows_file(tmp_path, rows))
    out = capsys.readouterr().out
    assert "[FAIL] censoring not worse" in out
    assert "=> REJECT" in out


def test_decision_rule_rejects_a_saving_too_small_to_matter(tmp_path, capsys, monkeypatch):
    rows = []
    for i in range(12):
        rows += [_pair_row(i, "real__b16k", 1000), _pair_row(i, "real_seal5__b16k", 980)]
    _run(monkeypatch, _pair_rows_file(tmp_path, rows))
    out = capsys.readouterr().out
    # Perfectly consistent, so significant, but only 2% -- not worth shipping.
    assert "[FAIL] median saving >= 10%" in out
    assert "=> REJECT" in out


def test_decision_rule_rejects_a_large_saving_on_too_few_items(tmp_path, capsys, monkeypatch):
    # Two items, both 30% shorter. Consistent, but 2 items cannot clear p<0.05:
    # the smallest possible two-sided p at n=2 is 0.5.
    rows = []
    for i in range(2):
        rows += [_pair_row(i, "real__b16k", 1000), _pair_row(i, "real_seal5__b16k", 700)]
    _run(monkeypatch, _pair_rows_file(tmp_path, rows))
    out = capsys.readouterr().out
    assert "[FAIL] permutation p < 0.05" in out
    assert "=> REJECT" in out


def test_decision_rule_promotes_a_real_consistent_saving(tmp_path, capsys, monkeypatch):
    rows = []
    for i in range(20):
        rows += [_pair_row(i, "real__b16k", 1000 + 10 * i),
                 _pair_row(i, "real_seal5__b16k", int((1000 + 10 * i) * 0.75))]
    _run(monkeypatch, _pair_rows_file(tmp_path, rows))
    out = capsys.readouterr().out
    assert "=> PROMOTE" in out
    assert "-25.0%" in out


def test_a_noisy_cohort_with_one_big_winner_is_not_promoted(tmp_path, capsys, monkeypatch):
    # This is the shape of the 2026-09-29 data: one item much shorter, the rest
    # scattered. The test must not promote on the strength of a single item.
    deltas = [0.66, 1.2, 0.95, 1.4, 1.05, 0.9, 1.3, 1.1, 0.98, 1.15]
    rows = []
    for i, d in enumerate(deltas):
        rows += [_pair_row(i, "real__b16k", 1000),
                 _pair_row(i, "real_seal5__b16k", int(1000 * d))]
    _run(monkeypatch, _pair_rows_file(tmp_path, rows))
    out = capsys.readouterr().out
    assert "=> REJECT" in out


def test_item_major_rejects_batching_before_loading_a_model():
    # The guard must fire without CUDA and without touching the weights: a
    # misconfigured 6-hour run should fail in a second.
    args = cfg(arm_order="item", decode_bs=4, view_arms="real", view_indices="",
               temperature=0.0, method_tag="", overwrite=False)
    with pytest.raises(SystemExit) as e:
        E.run_views(args)
    assert "decode_bs 1" in str(e.value)


def test_arm_major_is_unaffected_by_the_new_guard():
    # Batching is legitimate in the default order; the failure must then be the
    # ordinary missing-CUDA one, not the ordering guard.
    args = cfg(arm_order="arm", decode_bs=4, view_arms="real", view_indices="",
               temperature=0.0, method_tag="", overwrite=False)
    with pytest.raises(SystemExit) as e:
        E.run_views(args)
    assert "decode_bs" not in str(e.value)


def test_single_arm_gets_the_papers_baseline_prompt_not_the_judger_prompt():
    # The whole point of the `single` arm is that its prompt never mentions
    # latents. If it silently inherited the Judger prompt the arm would measure
    # nothing and look like a successful replication.
    assert E.PROMPT_KIND.get("single") == "single"
    assert E.PROMPT_KIND.get("real", "judger") == "judger"
    assert E.PROMPT_KIND.get("none", "judger") == "judger"


def test_single_and_none_differ_only_by_the_latent_sentences():
    # `none` vs `real` is the latent channel with the prompt fixed; `single` vs
    # `none` is the prompt text with the cache fixed (both absent). That reading
    # only holds if the two prompts differ nowhere else that matters.
    ns = SimpleNamespace(model_name="Qwen/Qwen3-14B", task="aime2024")
    judger = E.build_agent_message_sequential_latent_mas(
        role="judger", question="Q?", context="", method="latent_mas", args=ns)
    sns = copy.copy(ns)
    sns.method = "baseline"
    single = E.build_agent_messages_single_agent(question="Q?", args=sns)
    assert judger[0]["content"] == single[0]["content"]
    assert "latent information" in judger[1]["content"]
    assert "latent" not in single[1]["content"].lower()
    for shared in ("Target Question: Q?", "\\boxed{YOUR_FINAL_ANSWER}",
                   "reason step by step"):
        assert shared in judger[1]["content"]
        assert shared in single[1]["content"]


def test_single_arm_pays_for_no_upstream_role():
    # A single agent runs no Planner/Critic/Refiner, so charging it upstream
    # time would understate exactly the cost the paper is trading away.
    assert E.PAID_ROLES["single"] == ()
    assert "single" in E.VIEW_ARMS


def test_single_arm_is_decoded_without_a_cache():
    cache, extra = E.view_cache("single", {"idx": 0}, [], cfg())
    assert cache is None
    assert extra == {"prompt_kind": "single"}


def test_prompt_tensors_rejects_an_unknown_prompt_kind():
    with pytest.raises(ValueError):
        E.prompt_tensors(None, ["Q?"], SimpleNamespace(), "not_a_prompt")


def test_method_name_matches_the_labels_the_analysis_pairs_on():
    # paired_coef_test keys off these exact strings, so a change here silently
    # unpairs a cohort.
    assert E.method_name(cfg(temperature=0.0, method_tag="b16k"), "real") == "real__b16k"
    assert (E.method_name(cfg(temperature=0.0, method_tag="b16k"), "real_seal5")
            == "real_seal5__b16k")
    assert E.method_name(cfg(temperature=0.0, method_tag=""), "real") == "real"
    assert (E.method_name(cfg(temperature=0.6, method_tag="b16k"), "real")
            == "real_t06__b16k")


def test_analysis_pairs_the_labels_method_name_produces(tmp_path):
    # Close the loop: generate labels from the runner and confirm the analysis
    # groups them into one pair.
    a = cfg(temperature=0.0, method_tag="b16k")
    path = _pair_rows_file(tmp_path, [
        _pair_row(7, E.method_name(a, "real"), 1000),
        _pair_row(7, E.method_name(a, "real_seal5"), 800),
    ])
    pairs = P.load_pairs(path, "b16k", "5")
    assert sorted(pairs) == [7]
    assert pairs[7]["steer"]["tokens"] == 800


# ------------------------------------------------- latent channel ablation
# The question is what the inter-agent KV channel carries, so the analysis has to
# separate "the arm lost solves" from "the arm was never run on those items".

import channel_ablation as C  # noqa: E402


def _ch_row(idx, method, correct, tokens=1000, eos=True, cache_mb=100.0):
    return {"idx": idx, "method": method, "correct": correct, "tokens": tokens,
            "eos": eos, "cache_mb": cache_mb}


def test_channel_load_strips_tag_and_ignores_seal_arms(tmp_path):
    path = _pair_rows_file(tmp_path, [
        _ch_row(0, "real__b16k", True),
        _ch_row(0, "none__b16k", False),
        _ch_row(0, "real_seal5__b16k", True),   # different experiment
        _ch_row(0, "real__fast", True),         # different tag
    ])
    by = C.load(path, "b16k")
    assert sorted(by) == ["none", "real"]


def test_channel_compare_counts_gained_and_lost_separately(tmp_path):
    base = {0: _ch_row(0, "real", True), 1: _ch_row(1, "real", True),
            2: _ch_row(2, "real", False)}
    arm = {0: _ch_row(0, "none", True), 1: _ch_row(1, "none", False),
           2: _ch_row(2, "none", True)}
    r = C.compare(base, arm, "none")
    assert (r["gained"], r["lost"]) == (1, 1)
    assert r["delta_items"] == 0
    assert r["n_paired"] == 3


def test_channel_compare_pairs_only_on_shared_items():
    # The arm ran on one item only; the other two must not count as failures.
    base = {i: _ch_row(i, "real", True) for i in range(3)}
    arm = {0: _ch_row(0, "none", True)}
    r = C.compare(base, arm, "none")
    assert r["n_paired"] == 1
    assert r["lost"] == 0


def test_channel_a_total_collapse_is_significant():
    base = {i: _ch_row(i, "real", True) for i in range(12)}
    arm = {i: _ch_row(i, "none", False) for i in range(12)}
    r = C.compare(base, arm, "none")
    assert r["delta_items"] == -12
    assert r["mcnemar_p"] < 0.01
    assert r["delta_pct_ci"][1] < 0  # interval excludes zero


def test_channel_identical_arms_give_a_wide_not_zero_interval():
    # Equivalence is the weak direction of this design: with no discordant pairs
    # the interval must still be reported wide, never as a point at zero.
    base = {i: _ch_row(i, "real", i % 3 != 0) for i in range(30)}
    arm = {i: _ch_row(i, "shuf", i % 3 != 0) for i in range(30)}
    r = C.compare(base, arm, "shuf")
    assert r["delta_items"] == 0
    assert r["mcnemar_p"] == 1.0
    lo, hi = r["delta_pct_ci"]
    assert lo < 0 < hi and (hi - lo) > 0


def test_channel_kv_mb_reports_the_cost_of_the_arm():
    arm = {i: _ch_row(i, "c3", True, cache_mb=40.0) for i in range(4)}
    base = {i: _ch_row(i, "real", True, cache_mb=120.0) for i in range(4)}
    assert C.compare(base, arm, "c3")["kv_mb"] == 40.0
    assert C.kv_mb(list(base.values())) == 120.0


def test_channel_none_arm_costs_zero_kv():
    # The `none` arm writes no cache, so cache_mb is absent rather than zero.
    arm = {0: {"idx": 0, "method": "none", "correct": False, "tokens": 5,
               "eos": True}}
    assert C.kv_mb(list(arm.values())) == 0.0


def test_channel_censored_runs_are_counted_not_scored_as_short():
    base = {0: _ch_row(0, "real", True, tokens=1000)}
    arm = {0: _ch_row(0, "shuf", False, tokens=16384, eos=False)}
    r = C.compare(base, arm, "shuf")
    assert r["censored"] == 1
    assert r["mean_tokens_finished"] is None
