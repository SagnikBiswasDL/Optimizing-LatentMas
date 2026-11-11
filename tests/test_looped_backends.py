"""Backend contract, boundary-shift bookkeeping and profile/grade helpers.

Pure Python: the engine is replaced by a scripted fake, so these run without a
GPU and without vllm-rlt. The real engine is covered by
``scripts/looped_fidelity_gate.py`` on a GPU box.
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from serving.looped import backend as B  # noqa: E402
from serving.looped.rlt_backend import RltBackend  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ contract

def test_request_validation():
    with pytest.raises(ValueError):
        B.GenerationRequest("", (1,), 2, 4)
    with pytest.raises(ValueError):
        B.GenerationRequest("a", (), 2, 4)
    with pytest.raises(ValueError):
        B.GenerationRequest("a", (1,), 0, 4)
    with pytest.raises(ValueError):
        B.GenerationRequest("a", (1,), 2, 0)


def test_logical_loop_tokens_matches_saved_runs():
    # Stage-0 rows satisfy loop_tokens == depth * (prompt_tokens + gen_tokens - 1).
    assert B.logical_loop_tokens(3, 100, 50) == 3 * 149
    assert B.logical_loop_tokens(4, 10, 1) == 40
    assert B.logical_loop_tokens(2, 10, 0) == 20      # cancelled before any token


def test_strip_stop():
    assert B.strip_stop([5, 6, 2, 7, 0], {0, 2}) == [5, 6, 7]


# ------------------------------------------------------------------ fake engine

class FakeParams:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeEngine:
    """Emits one scripted token per live request per step."""

    def __init__(self, scripts, full_depth=4, stop=(0,)):
        self.scripts, self.full_depth, self.stop = scripts, full_depth, set(stop)
        self.live, self.aborted, self.added = {}, [], {}

    def add_request(self, rid, prompt, params):
        self.added[rid] = (list(prompt), params)
        self.live[rid] = {"tokens": [], "depths": [], "params": params}

    def step(self):
        outs = []
        for rid in list(self.live):
            st = self.live[rid]
            tok = self.scripts[rid][len(st["tokens"])]
            st["tokens"].append(tok)
            st["depths"].append(self.full_depth if len(st["tokens"]) == 1 else st["params"].max_loops)
            reason = "stop" if tok in self.stop else ("length" if len(st["tokens"]) >= st["params"].max_tokens else None)
            outs.append(SimpleNamespace(request_id=rid, token_ids=list(st["tokens"]), exit_depths=list(st["depths"]),
                                        finished=reason is not None, finish_reason=reason))
            if reason:
                del self.live[rid]
        return outs

    def abort_request(self, rid):
        st = self.live.pop(rid)
        self.aborted.append(rid)
        return SimpleNamespace(request_id=rid, token_ids=list(st["tokens"]), exit_depths=list(st["depths"]),
                               finished=True, finish_reason="abort")


def make_backend(scripts, mode):
    be = RltBackend.__new__(RltBackend)
    be._SamplingParams, be.prompt_mode, be.full_depth = FakeParams, mode, 4
    ticks = iter(range(10_000))
    be.clock = lambda: float(next(ticks))
    be.engine = FakeEngine(scripts)
    be.stop_ids, be._live = (0,), {}
    return be


def test_stock_mode_submits_whole_prompt_and_charges_full_depth_prefill():
    be = make_backend({"r": [11, 12, 0]}, "stock")
    done = B.run_all(be, [B.GenerationRequest("r", (7, 8, 9), 2, 10)])["r"]
    prompt, params = be.engine.added["r"]
    assert prompt == [7, 8, 9] and params.max_tokens == 10
    assert (params.min_loops, params.max_loops, params.exit_threshold) == (2, 2, 1.0)
    assert done.token_ids == [11, 12, 0] and done.finish_reason == B.EOS
    assert done.logical_loop_tokens == 2 * (3 + 3 - 1)
    assert done.charged_loop_tokens == 4 * 3 + 2 * 2          # full-depth prompt, two fed-back tokens
    assert done.first_token_time is not None and done.first_token_time <= done.finish_time


def test_boundary_shift_withholds_last_prompt_token_and_hides_it():
    be = make_backend({"r": [9, 11, 12, 0]}, "boundary_shift")
    done = B.run_all(be, [B.GenerationRequest("r", (7, 8, 9), 2, 10)])["r"]
    prompt, params = be.engine.added["r"]
    assert prompt == [7, 8] and params.max_tokens == 11       # one extra for the withheld token
    assert done.token_ids == [11, 12, 0] and done.finish_reason == B.EOS
    assert done.prompt_tokens == 3
    assert done.logical_loop_tokens == 2 * (3 + 3 - 1)
    # prefill of two tokens at full depth, then three depth-2 steps (withheld token, 11, 12)
    assert done.charged_loop_tokens == 4 * 2 + 2 * 3
    assert done.charged_loop_tokens - done.logical_loop_tokens == (4 - 2) * (3 - 1)


def test_boundary_shift_first_token_time_ignores_the_withheld_token():
    be = make_backend({"r": [9, 11, 12, 13]}, "boundary_shift")
    be.submit(B.GenerationRequest("r", (7, 8, 9), 3, 3))
    assert be.step() == [] and be._live["r"]["first"] is None   # only the withheld token so far
    assert be.partial("r") == []
    be.step()
    assert be._live["r"]["first"] is not None and be.partial("r") == [11]
    be.step()
    done = be.step()[0]
    assert done.finish_reason == B.LENGTH_CAP and done.token_ids == [11, 12, 13]


def test_boundary_violation_is_an_error_not_an_answer():
    be = make_backend({"r": [5, 11, 0]}, "boundary_shift")
    done = B.run_all(be, [B.GenerationRequest("r", (7, 8, 9), 2, 10)])["r"]
    assert done.finish_reason == B.ERROR
    assert done.detail["boundary_violation"] == {"expected": 9, "got": 5}
    assert be.engine.aborted == ["r"] and be.active() == 0


def test_cancel_returns_tokens_so_far_and_frees_the_slot():
    be = make_backend({"a": [9, 1, 2, 3, 4, 0], "b": [9, 5, 6, 0]}, "boundary_shift")
    be.submit(B.GenerationRequest("a", (7, 9), 3, 10))
    be.submit(B.GenerationRequest("b", (7, 9), 3, 10))
    be.step(), be.step(), be.step()
    c = be.cancel("a")
    assert c.finish_reason == B.CANCELLED and c.token_ids == [1, 2]
    assert be.active() == 1
    rest = {}
    while be.active():
        for x in be.step():
            rest[x.request_id] = x
    assert rest["b"].token_ids == [5, 6, 0]


def test_duplicate_and_too_deep_requests_are_rejected():
    be = make_backend({"r": [9, 0]}, "boundary_shift")
    be.submit(B.GenerationRequest("r", (7, 9), 2, 4))
    with pytest.raises(ValueError):
        be.submit(B.GenerationRequest("r", (7, 9), 2, 4))
    with pytest.raises(ValueError):
        be.submit(B.GenerationRequest("s", (7, 9), 5, 4))
    with pytest.raises(ValueError):
        be.submit(B.GenerationRequest("t", (7,), 2, 4))      # nothing left to prefill


# ------------------------------------------------------------------ grading helpers

@pytest.fixture(scope="module")
def grade():
    return load_script("grade_looped_profile")


def cand(sample, key, correct, work=10, finish="eos", q="q", gen=5):
    return {"question_id": q, "candidate_id": f"{q}:d3:s{sample}", "sample": sample, "answer_key": key,
            "correct": correct, "logical_loop_tokens": work, "finish_reason": finish, "gen_tokens": gen}


def test_vote_uses_majority_then_lowest_candidate(grade):
    rows = [cand(0, "A", False), cand(1, "B", True), cand(2, "B", True), cand(3, None, False)]
    assert grade.vote(rows)["sample"] == 1
    tie = [cand(0, "A", False), cand(1, "B", True)]
    assert grade.vote(tie)["sample"] == 0
    assert grade.vote([cand(0, None, False)]) is None


def test_two_then_two_stops_on_agreement_and_charges_only_what_ran(grade):
    agree = [cand(0, "A", True), cand(1, "A", True), cand(2, "B", False), cand(3, "B", False)]
    sel, used = grade.two_then_two(agree)
    assert sel["sample"] == 0 and len(used) == 2
    split = [cand(0, "A", False), cand(1, "B", True), cand(2, "B", True), cand(3, "C", False)]
    sel, used = grade.two_then_two(split)
    assert sel["sample"] == 1 and len(used) == 4
    unanswered = [cand(0, None, False), cand(1, None, False), cand(2, "B", True), cand(3, None, False)]
    sel, used = grade.two_then_two(unanswered)                # two missing answers are not agreement
    assert sel["sample"] == 2 and len(used) == 4


def test_extension_table_counts_rescued_candidates(grade):
    old = [cand(0, None, False, finish="length_cap", gen=8), cand(1, "A", False), cand(2, None, False, finish="length_cap", gen=8)]
    new = [cand(0, "B", True, gen=12), cand(2, None, False, finish="length_cap", gen=16)]
    summary = {"arm": {"from_cap": 8, "cap": 16, "depth": 3, "width": 3}, "prefix_reproduced": 2}
    t = grade.extension_table("x", new, summary, old, [{"question_id": "q"}])
    assert (t["capped_candidates_rerun"], t["now_finish_with_eos"], t["still_capped"]) == (2, 1, 1)
    assert (t["answered_before"], t["answered_after"]) == (0, 1)
    assert (t["correct_before"], t["correct_after"]) == (0, 1)
    assert t["extra_gen_tokens"] == 4 + 8
    # old vote: only "A" (wrong). new vote: A and B tie, lowest candidate (s0, correct) wins.
    assert (t["policy_accuracy_before"], t["policy_accuracy_after"], t["questions_changed"]) == (0.0, 1.0, 1)


# ------------------------------------------------------------------ profile helpers

@pytest.fixture(scope="module")
def profile():
    pytest.importorskip("torch")
    return load_script("looped_profile")


def test_candidates_are_stable_across_widths(profile):
    args = SimpleNamespace(run_seed=0, temperature=0.6, top_p=0.95)
    four = profile.candidates_for("q_x", 3, 4, args)
    eight = profile.candidates_for("q_x", 3, 8, args)
    assert eight[:4] == four                                   # a wider arm extends, never reshuffles
    assert [c[1] for c in eight] == list(range(8))
    assert len({c[4] for c in eight}) == 8                     # one seed per candidate
    greedy = profile.candidates_for("q_x", 3, 1, args)
    assert greedy == [("q_x:d3:s-1", -1, 0.0, 1.0, greedy[0][4])]
    other_depth = profile.candidates_for("q_x", 4, 4, args)
    assert {c[4] for c in other_depth}.isdisjoint({c[4] for c in four})
    reseeded = profile.candidates_for("q_x", 3, 4, SimpleNamespace(run_seed=1, temperature=0.6, top_p=0.95))
    assert {c[4] for c in reseeded}.isdisjoint({c[4] for c in four})


def test_percentiles(profile):
    assert profile.pct([], .5) is None
    assert profile.pct([3.0], .95) == 3.0
    assert profile.pct([1.0, 2.0, 3.0, 4.0], .5) == 2.5
    assert profile.pct([4.0, 1.0, 3.0, 2.0], 1.0) == 4.0
    d = profile.dist([1.0, 2.0, 3.0])
    assert d["mean"] == 2.0 and d["max"] == 3.0 and profile.dist([]) == {}


def test_arm_specs_and_mode_candidate_ids(profile):
    assert profile.parse_arm("3x4", 2048) == {"depth": 3, "width": 4, "cap": 2048, "kind": "fixed", "mode": "natural"}
    arm = profile.parse_arm("4x2:think@4096", 2048)
    assert (arm["depth"], arm["width"], arm["mode"], arm["cap"]) == (4, 2, "think", 4096)
    assert profile.arm_name(profile.parse_arm("3x4", 2048)) == "d3_w4_cap2048"          # unchanged for natural arms
    assert profile.arm_name(arm) == "d4_w2_think_cap4096"
    with pytest.raises(ValueError):
        profile.parse_arm("3x4:sideways", 2048)
    args = SimpleNamespace(run_seed=0, temperature=0.6, top_p=0.95)
    natural = profile.candidates_for("q_x", 3, 4, args)
    direct = profile.candidates_for("q_x", 3, 4, args, "direct")
    assert [c[0] for c in direct] == [f"q_x:d3:s{s}:direct" for s in range(4)]
    assert {c[4] for c in direct}.isdisjoint({c[4] for c in natural})


def test_forced_modes_need_a_prompt_state_at_the_request_depth(profile):
    be = SimpleNamespace(semantics={"prompt_mode": "boundary_shift", "full_depth": 4})
    profile.check_mode_semantics(be, 3, "natural")
    with pytest.raises(SystemExit):
        profile.check_mode_semantics(be, 4, "think")
    be = SimpleNamespace(semantics={"prompt_mode": "stock", "full_depth": 4})
    profile.check_mode_semantics(be, 4, "direct")
    with pytest.raises(SystemExit):
        profile.check_mode_semantics(be, 3, "direct")
    profile.check_mode_semantics(be, 3, "direct", allow_full_depth_prefix_state=True)


def test_forced_prefix_is_reported_as_the_start_of_the_answer(profile):
    comp = B.Completion("q:d3:s0:think", 3, 11, [50, 51, 0], "eos", 1.0, 1.5, 2.0, 3 * (11 + 3 - 1), 40)
    row = profile.record("run", "q", ("q:d3:s0:think", 0, 0.6, 0.95, 7), comp, "text \\boxed{4}", 0.0, "think", (3,), 3)
    assert row["generated_token_ids"] == [3, 50, 51, 0] and row["gen_tokens"] == 4
    assert row["prompt_tokens"] == 10 and row["prefix_tokens"] == 1 and row["mode"] == "think" and row["think"] is True
    assert row["logical_loop_tokens"] == 3 * (row["prompt_tokens"] + row["gen_tokens"] - 1)
    natural = profile.record("run", "q", ("q:d3:s0", 0, 0.6, 0.95, 7), comp, "text", 0.0, think_token=3)
    assert natural["think"] is False and natural["prefix_tokens"] == 0 and natural["prompt_tokens"] == 11


def test_modes_prefixes():
    from looped import modes as M

    class Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, t):
            return {"<think>": 3}.get(t, 0)

        def encode(self, text, add_special_tokens=False):
            return {"To": [2068]}[text]

    assert M.prefix_ids(Tok(), "natural") == () and M.prefix_ids(Tok(), "think") == (3,)
    assert M.prefix_ids(Tok(), "direct") == (2068,)
    assert M.observed_think([3, 9], 3) is True and M.observed_think([2068], 3) is False and M.observed_think([], 3) is None
    with pytest.raises(ValueError):
        M.prefix_ids(Tok(), "sideways")
