"""Vote identity: collisions the legacy key creates, equivalences the structural
key must keep, and explicit parse failures."""
import glob
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from looped.answer_identity import (  # noqa: E402
    EMPTY, LEGACY, MALFORMED, NO_ANSWER, OK, STRUCTURAL, GraderUnavailable, answer_key,
    extract_final_answer, grade, grader_info, identify, legacy_key,
)

STAGE0B = os.path.join(ROOT, "artifacts/loopdelta_run/artifacts/loopdelta/stage0b")


def k(s):
    return answer_key(s, STRUCTURAL)


# ---------------------------------------------------------------- collisions

DIFFERENT = [
    ("\\frac{1}{23}", "\\frac{12}{3}"),          # the case named in the plan
    ("\\frac{1}{23}", "\\frac123"),              # TeX reads \frac123 as (1/2) then 3
    ("\\frac{a}{bc}", "\\frac{ab}{c}"),
    ("x^{12}3", "x^{1}23"),
    ("x^{12}", "x^12"),                          # TeX reads x^12 as x^1 then 2
    ("\\sqrt{2}3", "\\sqrt{23}"),
    ("\\sqrt[3]{2}", "\\sqrt{32}"),
    ("\\binom{1}{23}", "\\binom{12}{3}"),
    ("a_{1}2", "a_{12}"),
    ("\\pi r", "\\pir"),
    ("1 2", "12"),
    ("0.333", "\\frac{1}{3}"),                   # close, not equal
    ("1,000", "1000"),                           # could be the list 1, 000: left apart
    ("(1,2)", "[1,2]"),
    ("x", "\\frac{x}{2}"),
    ("\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}", "\\begin{pmatrix} 1 & 2 \\end{pmatrix}"),
]


@pytest.mark.parametrize("a,b", DIFFERENT)
def test_structural_keeps_distinct_answers_apart(a, b):
    assert k(a) is not None and k(b) is not None
    assert k(a) != k(b)


def test_legacy_collides_where_structural_does_not():
    # Documents the defect: brace removal merges two different fractions.
    assert legacy_key("\\frac{1}{23}") == legacy_key("\\frac{12}{3}") == "\\frac123"
    assert k("\\frac{1}{23}") == "Q(1/23)" and k("\\frac{12}{3}") == "Q(4/1)"


# ---------------------------------------------------------------- equivalences

SAME = [
    ["1/2", "0.5", ".5", "0.50", "+0.5", "\\frac{1}{2}", "\\frac12", "\\dfrac{1}{2}", "\\tfrac12",
     "\\frac{2}{4}", "$\\frac{1}{2}$", "\\boxed{\\frac{1}{2}}", "{\\frac{1}{2}}", "\\frac{1}{2}."],
    ["-\\frac{1}{2}", "\\frac{-1}{2}", "-0.5", "-1/2", "\u22120.5"],
    ["5", "5.0", "\\boxed{5}", "5.", " 5 ", "05", "{5}"],
    ["10,\\!000", "10000", "10{,}000"],
    ["50\\%", "50"],
    ["90^\\circ", "90^{\\circ}", "90"],
    ["x^2", "x^{2}", "x ^ 2"],
    ["\\sqrt2", "\\sqrt{2}"],
    ["\\frac{a}{b}", "\\dfrac{a}{b}", "\\frac ab"],
    ["\\left(1,2\\right)", "(1, 2)", "(1,2)"],
    ["10\\text{ cm}", "10 cm", "10\\,\\text{cm}"],
    ["\\text{Yes}", "yes", "Yes", "\\text{yes}"],
    ["2\\pi r", "2 \\pi r"],
]


@pytest.mark.parametrize("group", SAME)
def test_structural_groups_equivalent_forms(group):
    keys = {k(x) for x in group}
    assert None not in keys
    assert len(keys) == 1, {x: k(x) for x in group}


def test_macros_that_look_like_left_right_survive():
    assert "\\rightarrow" in k("a\\rightarrow b")
    assert "\\leftarrow" in k("a\\leftarrow b")
    assert k("a\\rightarrow b") != k("a\\leftarrow b")


def test_matrix_row_breaks_survive_cleaning():
    key = k("\\begin{pmatrix} \\frac{1}{2} \\\\ 2 \\end{pmatrix}")
    assert key == "S(\\begin{pmatrix}\\frac{1}{2}\\\\2\\end{pmatrix})"
    assert key == k("\\begin{pmatrix}\\frac12\\\\ 2\\end{pmatrix}")


def test_zero_denominator_is_not_a_number():
    assert k("\\frac{1}{0}").startswith("S(") and k("1/0").startswith("S(")


def test_case_is_kept_in_math_but_not_in_words():
    assert k("\\Delta") != k("\\delta")
    assert k("Xy+1") != k("xy+1")
    assert k("East") == k("east")


# ---------------------------------------------------------------- parse failures

@pytest.mark.parametrize("bad", ["\\frac{1}{2", "\\frac{1}", "\\frac", "1}", "x^", "\\sqrt[3", "\\text x",
                                 "\\boxed{\\frac{1}{2"])
def test_malformed_answers_cast_no_vote(bad):
    ident = identify(bad, STRUCTURAL)
    assert ident.key is None and ident.status == MALFORMED


def test_missing_and_empty_are_explicit():
    assert identify(None).status == NO_ANSWER and identify(None).key is None
    for empty in ["", "  ", "$$", "{}", "."]:
        assert identify(empty).status == EMPTY, empty
        assert identify(empty).key is None
    assert identify("7").status == OK
    assert identify(None, LEGACY).status == NO_ANSWER
    assert identify("$$", LEGACY).status == EMPTY


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError):
        identify("1", "v3")


# ---------------------------------------------------------------- extraction

def test_extraction_reports_unterminated_boxes():
    assert extract_final_answer("so \\boxed{1} then \\boxed{\\frac{3}{4}}.") == ("\\frac{3}{4}", "ok")
    assert extract_final_answer("no box here") == (None, "no_boxed")
    assert extract_final_answer(None) == (None, "no_boxed")
    assert extract_final_answer("cut off \\boxed{\\frac{3}{4") == ("\\frac{3}{4", "unterminated_boxed")


# ---------------------------------------------------------------- legacy parity

def _saved_preds():
    preds = set()
    for path in sorted(glob.glob(os.path.join(STAGE0B, "scored_r*.jsonl"))):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                preds.add(r["pred"])
    return preds


def test_legacy_key_matches_utils_and_saved_keys():
    pytest.importorskip("torch")                  # utils imports torch
    from utils import extract_boxed_answer, normalize_math_answer

    samples = [x for g in SAME for x in g] + [x for pair in DIFFERENT for x in pair] + [None, "", "\\boxed{2"]
    for s in samples:
        assert legacy_key(s) == normalize_math_answer(s), s
    for text in ["a \\boxed{1} b \\boxed{2}", "\\boxed {x}", "\\boxed{\\frac{1}{2", "none", "\\boxed"]:
        assert extract_final_answer(text)[0] == extract_boxed_answer(text), text
    if not os.path.isdir(STAGE0B):
        pytest.skip("Stage-0b artifacts not present")
    preds = _saved_preds()
    assert len(preds) > 500
    for p in preds:
        assert legacy_key(p) == normalize_math_answer(p), p
    # and the keys stored in the artifacts are exactly the legacy keys
    with open(os.path.join(STAGE0B, "scored_r3.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            assert r["key"] == legacy_key(r["pred"])


def test_structural_key_never_raises_on_saved_predictions():
    if not os.path.isdir(STAGE0B):
        pytest.skip("Stage-0b artifacts not present")
    statuses = {}
    for p in _saved_preds():
        ident = identify(p, STRUCTURAL)
        statuses[ident.status] = statuses.get(ident.status, 0) + 1
        assert (ident.key is None) == (ident.status != OK)
    assert statuses.get(OK, 0) > 500


# ---------------------------------------------------------------- grader pin

def test_string_grader_is_available_and_labelled():
    out = grade("\\frac{1}{2}", "\\dfrac{1}{2}", grader="string-legacy-v1")
    assert out["correct"] is True and out["grader"] == "string-legacy-v1"
    assert grade(None, "1", grader="string-legacy-v1")["correct"] is False


def test_missing_pinned_grader_fails_loudly():
    try:
        import math_verify  # noqa: F401
    except Exception:
        with pytest.raises(GraderUnavailable):
            grader_info("math-verify")
        with pytest.raises(GraderUnavailable):
            grade("1", "1")
    else:
        assert grader_info("math-verify")["grader"] == "math-verify"
        assert grade("\\frac{1}{2}", "0.5")["correct"] is True
