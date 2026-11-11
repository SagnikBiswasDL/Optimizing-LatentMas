"""Replay policies on a pool small enough to check by hand, and a regression
test that the saved Stage-0b pool still gives the numbers the plan quotes."""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import audit_looped_history as A  # noqa: E402
from looped.answer_identity import LEGACY, STRUCTURAL, legacy_key  # noqa: E402

STAGE0B = os.path.join(ROOT, "artifacts/loopdelta_run/artifacts/loopdelta/stage0b")


def cand(depth, sample, pred, correct, gen, prompt=10, hit_eos=True):
    return {"depth": depth, "sample": sample, "pred": pred, "key": legacy_key(pred), "correct": correct,
            "prompt_tokens": prompt, "gen_tokens": gen, "hit_eos": hit_eos,
            "loop_tokens": depth * (prompt + gen - 1), "level": 1}


def question(d3, d4=None):
    """d3/d4: four (pred, correct, gen) sampled candidates; greedy copies the first."""
    rows = {}
    for depth, spec in ((3, d3), (4, d4 or d3)):
        rows[depth] = {s: cand(depth, s, *spec[s]) for s in range(4)}
        rows[depth][-1] = cand(depth, -1, *spec[0])
    return rows


def cost(rows, members):
    return sum(rows[d][s]["loop_tokens"] for d, s in members)


def test_two_then_two_same_depth_by_hand():
    # answers a a b b; "a" is right. Tie among four goes to the first listed.
    rows = question([("1", True, 5), ("1", True, 6), ("2", False, 7), ("2", False, 8)])
    p = A.Pool(rows, LEGACY)
    acc, c = A.two_then_two(3, 3)(p)
    pair01, pair23, all4 = cost(rows, [(3, 0), (3, 1)]), cost(rows, [(3, 2), (3, 3)]), cost(rows, [(3, s) for s in range(4)])
    # (0,1) agree on a: right. (2,3) agree on b: wrong. The four mixed pairs fall back to all
    # four, tie 2-2, and the first listed (always an "a" sample) wins: right.
    assert acc == pytest.approx(5 / 6)
    assert c == pytest.approx((pair01 + pair23 + 4 * all4) / 6)


def test_fixed_width_and_greedy_by_hand():
    rows = question([("1", True, 5), ("1", True, 6), ("2", False, 7), ("2", False, 8)])
    p = A.Pool(rows, LEGACY)
    assert A.fixed(3, 4)(p) == (1.0, float(cost(rows, [(3, s) for s in range(4)])))
    acc, c = A.fixed(3, 2)(p)
    assert acc == pytest.approx(5 / 6)                 # only the pair (2,3) picks "2"
    assert c == pytest.approx(sum(cost(rows, [(3, a), (3, b)]) for a, b in A.PAIRS) / 6)
    assert A.fixed(3, 1)(p)[0] == pytest.approx(0.5)
    assert A.greedy(3)(p) == (1.0, float(rows[3][-1]["loop_tokens"]))


def test_deeper_fallback_and_portfolio_by_hand():
    # depth 3: a b c d (never agree, "a" right); depth 4: all "a".
    rows = question([("1", True, 5), ("2", False, 5), ("3", False, 5), ("4", False, 5)],
                    [("1", True, 9), ("1", True, 9), ("1", True, 9), ("1", True, 9)])
    p = A.Pool(rows, LEGACY)
    acc, c = A.two_then_two(3, 4)(p)
    assert acc == 1.0                                   # two depth-4 "a" votes always win
    assert c == pytest.approx(2 * 3 * 14 + 2 * 4 * 18)  # every branch acquires 2 + 2
    assert A.portfolio([3, 4])(p) == (acc, c)           # same members, no early stop to save work
    # same-depth fallback: a 1-1-1-1 tie goes to the first listed, which is the lower-numbered
    # sample of the initial pair. Right only for the pairs (0,1), (0,2), (0,3).
    assert A.two_then_two(3, 3)(p)[0] == pytest.approx(3 / 6)
    moved = A.Pool(question([("2", False, 5), ("1", True, 5), ("3", False, 5), ("4", False, 5)]), LEGACY)
    assert A.two_then_two(3, 3)(moved)[0] == pytest.approx(2 / 6)     # now only (1,2) and (1,3)


def test_missing_answers_never_agree_and_never_vote():
    rows = question([(None, False, 9), (None, False, 9), ("1", True, 5), ("2", False, 5)])
    p = A.Pool(rows, LEGACY)
    assert not p.agree((3, 0), (3, 1))                  # two missing answers are not agreement
    assert p.pick([(3, 0), (3, 1)])[0] is False
    assert p.pick([(3, 0), (3, 2)])[0] is True          # the only valid vote wins


def test_structural_identity_changes_a_replay_only_through_parse_failures():
    rows = question([("\\dfrac{", False, 9), ("7", True, 5), ("7", True, 5), ("7", True, 5)])
    assert A.Pool(rows, LEGACY).pick([(3, 0), (3, 1)])[0] is False      # truncated answer wins the tie
    assert A.Pool(rows, STRUCTURAL).pick([(3, 0), (3, 1)])[0] is True   # it casts no vote


def test_strict_cancellation_by_hand():
    rows = question([("1", True, 5), ("1", True, 6), ("1", True, 7), ("2", False, 8)])
    out = A.strict_cancellation([A.Pool(rows, LEGACY)], 3)
    # after rounds 5 and 6 the lead is 1 then 2 with 3 then 2 unfinished: not safe.
    # after round 7 it is 3-0 with one unfinished: stop at 7 of 8 rounds.
    assert out["questions_with_a_cancellation"] == 1 and out["winner_changes"] == 0
    assert out["completion_rounds_saved_length_proxy"] == pytest.approx(1 / 8)
    assert out["decode_tokens_saved"] == pytest.approx(1 / 26)          # 5+6+7+7 of 5+6+7+8


def test_grid_validation_rejects_bad_pools():
    good = {0: question([("1", True, 5)] * 4)}
    A.validate_grid(good, [3, 4], 4, None)
    missing = {0: question([("1", True, 5)] * 4)}
    del missing[0][4][2]
    with pytest.raises(ValueError):
        A.validate_grid(missing, [3, 4], 4, None)
    wrong_cost = {0: question([("1", True, 5)] * 4)}
    wrong_cost[0][3][1]["loop_tokens"] += 1
    with pytest.raises(ValueError):
        A.validate_grid(wrong_cost, [3, 4], 4, None)
    wrong_key = {0: question([("\\frac{1}{2}", True, 5)] * 4)}
    wrong_key[0][3][0]["key"] = "0.5"
    with pytest.raises(ValueError):
        A.validate_grid(wrong_key, [3, 4], 4, None)


@pytest.fixture(scope="module")
def saved():
    if not os.path.isdir(STAGE0B):
        pytest.skip("Stage-0b artifacts not present")
    return A.audit(STAGE0B, ROOT, iters=200, seed=0)


def test_saved_pool_reproduces_the_plan_table(saved):
    pol = saved["replay"][LEGACY]["policies"]
    expect = {                                           # accuracy %, mean loop-tokens (plan section 4)
        "greedy_d3": (66.33, 3524.93), "greedy_d4": (68.33, 5139.88),
        "fixed_d3_w4": (77.33, 13677.85), "fixed_d4_w4": (77.00, 19860.12),
        "two_then_two_d3_fallback_d3": (77.61, 10667.79), "two_then_two_d3_fallback_d4": (77.06, 12118.63),
        "portfolio_3_3_4_4": (77.06, 16768.99),
    }
    for name, (acc, work) in expect.items():
        assert 100 * pol[name]["accuracy"] == pytest.approx(acc, abs=0.005), name
        assert pol[name]["mean_loop_tokens"] == pytest.approx(work, abs=0.006), name
    rec = saved["replay"][LEGACY]
    assert rec["work_relative_to_fixed_d3_w4"]["two_then_two_d3_fallback_d3"] == pytest.approx(0.2201, abs=5e-5)
    assert rec["fixed_d3_w4_work_saved_vs_fixed_d4_w4"] == pytest.approx(0.3113, abs=5e-5)
    paired = rec["paired_sc4_d3_minus_d4"]
    assert (paired["d3_only"], paired["d4_only"]) == (15, 14)
    assert paired["difference"] == pytest.approx(1 / 300)


def test_saved_pool_reproduces_the_plan_diagnostics(saved):
    rec = saved["replay"][LEGACY]
    assert rec["agreement"]["3"]["agreement_rate"] == pytest.approx(0.6183, abs=5e-5)
    assert rec["agreement"]["3"]["wrong_among_agreeing"] == pytest.approx(0.0225, abs=5e-5)
    sc = rec["strict_cancellation_sc4"]["3"]
    assert sc["winner_changes"] == 0
    assert sc["completion_rounds_saved_length_proxy"] == pytest.approx(0.1281, abs=5e-5)
    assert sc["decode_tokens_saved"] == pytest.approx(0.0425, abs=5e-5)
    stats = saved["cap_and_reuse"]
    assert stats["3"]["sampled_traces_at_cap"] == pytest.approx(295 / 1200)      # 24.6%
    assert stats["4"]["sampled_traces_at_cap"] == pytest.approx(339 / 1200)      # 28.3%
    assert stats["3"]["mean_prompt_tokens"] == pytest.approx(111.68, abs=0.005)
    assert stats["3"]["sc4_loop_tokens_saved_by_ideal_prompt_reuse"] == pytest.approx(0.0735, abs=5e-5)
    sig = saved["solver_signal_within_question"]
    assert sig["3"]["neg_delta_last"]["auroc"] == pytest.approx(0.518, abs=5e-4)
    assert sig["3"]["mean_logprob"]["auroc"] == pytest.approx(0.659, abs=5e-4)
    assert sig["4"]["neg_delta_last"]["auroc"] == pytest.approx(0.591, abs=5e-4)
    assert sig["4"]["mean_logprob"]["auroc"] == pytest.approx(0.839, abs=5e-4)
    assert sig["3"]["mean_logprob"]["correct_incorrect_pairs"] == 276      # pairs, from 82 questions
    assert sig["4"]["mean_logprob"]["correct_incorrect_pairs"] == 254      # pairs, from 80 questions
    assert (sig["3"]["mean_logprob"]["questions_contributing"], sig["4"]["mean_logprob"]["questions_contributing"]) == (82, 80)
    assert saved["timing"]["sampled_pass_gpu_hours_depths_2_3_4"] == pytest.approx(4.66, abs=0.005)
    assert rec["key_conflicts"]["question_keys_with_mixed_correctness"] == 0


def test_saved_gsm8k_pilots_reproduce(saved):
    g = saved["gsm8k_pilots"]
    if g["thinking_valid"] is None or g["plain_valid"] is None:
        pytest.skip("GSM8K pilot artifacts not present")
    assert (g["thinking_valid"]["mas"], g["thinking_valid"]["single"]) == (0.835, 0.895)
    assert (g["plain_valid"]["mas"], g["plain_valid"]["single"]) == (0.72, 0.845)
    assert (g["plain_valid"]["sc@4"], g["plain_valid"]["selfrev"]) == (0.89, 0.88)


def test_saved_pool_identity_findings(saved):
    ic = saved["identity_comparison"]
    assert ic["pairs_merged_by_legacy_only"] == 0 and ic["pairs_merged_by_structural_only"] == 0
    dropped = ic["sampled_votes_counted_by_legacy_but_not_structural"]
    assert [d["pred"] for d in dropped] == ["\\dfrac{", "\\dfrac{"]          # truncated at the cap
    assert not any(d["hit_eos"] or d["correct"] for d in dropped)
    assert (ic["both_correct_sampled_pairs"], ic["both_correct_pairs_split_by_structural"]) == (3323, 4)
    assert max(abs(v) for v in ic["policy_accuracy_change_structural_minus_legacy"].values()) < 0.002
    stats = saved["cap_and_reuse"]
    for d in ("2", "3", "4"):                             # capped traces almost never carry an answer
        assert stats[d]["capped_sampled_traces_without_a_boxed_answer"] > 0.9
