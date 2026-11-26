"""The deadline study: the trace, the four arms, and the open-loop score."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from looped import deadline as D  # noqa: E402
from looped import experiment_manifest as em  # noqa: E402
from serving.looped import backend as B  # noqa: E402
from serving.looped.deadline_driver import run_deadline  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class Scripted(B.Backend):
    """``script[request_id] = (steps, answer key)``. Each step is one second."""

    def __init__(self, script, clock):
        self.script, self.clock, self.live, self.submitted, self.cancelled = script, clock, {}, [], []

    def submit(self, r):
        self.live[r.request_id] = [r, 0]
        self.submitted.append(r.request_id)

    def active(self):
        return len(self.live)

    def step(self):
        self.clock.advance(1.0)
        out = []
        for rid in list(self.live):
            self.live[rid][1] += 1
            r, n = self.live[rid]
            if n >= self.script[rid][0]:
                del self.live[rid]
                out.append(B.Completion(rid, r.depth, len(r.prompt_ids), [7] * n, B.EOS,
                                        0.0, 0.0, self.clock(), 0, 0))
        return out

    def cancel(self, rid):
        r, n = self.live.pop(rid)
        self.cancelled.append(rid)
        return B.Completion(rid, r.depth, len(r.prompt_ids), [7] * n, B.CANCELLED, 0.0, 0.0, self.clock(), 0, 0)


def rid(qid, mode, sample):
    return f"{qid}:{mode}:s{sample}"


def serve(arm, script, questions, slots, deadline_s=D.DEADLINE_S, on_outcome=None):
    clock = Clock()
    be = Scripted(script, clock)

    def make_request(qid, prompt, launch):
        return B.GenerationRequest(rid(qid, launch.mode, launch.sample), tuple(prompt), launch.depth, launch.cap or 8)

    out = run_deadline(
        be, arm, questions, slots=slots, deadline_s=deadline_s, clock=clock, sleeper=clock.advance,
        make_request=make_request, keyer=lambda c: script[c.request_id][1],
        on_outcome=on_outcome or (lambda q: None),
    )
    return be, out


def q(i, arrival, prompt=(1, 2)):
    return (f"q{i}", prompt, arrival)


# ------------------------------------------------------------------ protocol

def test_protocol_file_matches_the_predeclared_study():
    path = os.path.join(ROOT, "configs", "looped_serving", "deadline_study.json")
    assert D.check_study_file(path)["name"] == D.STUDY_NAME
    assert D.LONG_SLACK_S == D.DEADLINE_S / 2


def test_trace_alternates_dense_and_sparse_blocks():
    # period 2, rates 4 then 12 per minute: gaps 15s, 15s, 5s.
    assert D.arrival_trace(4, (4, 12), period=2) == [0.0, 15.0, 20.0, 25.0]
    study = D.arrival_trace(40)
    assert study[0] == 0.0
    assert study[1] - study[0] == pytest.approx(15.0)          # 4 per minute
    assert study[20] - study[19] == pytest.approx(5.0)         # first question of the fast block
    with pytest.raises(ValueError):
        D.arrival_trace(5, period=2)


def test_fraction_assignment_ignores_the_run_seed_and_the_endpoints():
    assert D.assigns_long("q_a", 0.0) is False
    assert D.assigns_long("q_a", 1.0) is True
    assert D.assigns_long("q_a", 0.5) == D.assigns_long("q_a", 0.5)
    longs = sum(D.assigns_long(f"q{i}", 0.5) for i in range(2000))
    assert 900 < longs < 1100


def test_mixture_selection_is_the_predeclared_tie_break():
    ids = ["q0", "q1"]
    base = {"n": 2, "question_ids": ids, "hits": 1, "long_questions": 1}
    runs = [{**base, "candidate": name} for name in D.MIXTURE_CANDIDATES]
    runs[1]["hits"] = 2                                     # p50
    assert D.choose_mixture(runs) == "p50"
    tied = [{**base, "candidate": name, "hits": 1, "long_questions": 5 if name != "p25" else 1}
            for name in D.MIXTURE_CANDIDATES]
    assert D.choose_mixture(tied) == "p25"                  # same hits, fewer long answers
    with pytest.raises(ValueError):
        D.choose_mixture(runs[:3])


def test_gate_wants_two_points_on_average_and_a_gain_every_repetition():
    good = [{"always_direct": 0.70, "always_long": 0.60, "mixture": 0.72, "queue_aware": 0.75}] * 3
    assert D.development_gate(good)["passed"]
    short = [{"always_direct": 0.70, "always_long": 0.60, "mixture": 0.72, "queue_aware": 0.73}] * 3
    assert not D.development_gate(short)["passed"]          # +1 point, under the gate
    mixed = [dict(good[0]), dict(good[0]),
             {"always_direct": 0.80, "always_long": 0.70, "mixture": 0.78, "queue_aware": 0.79}]
    assert not D.development_gate(mixed)["passed"]          # one repetition is not ahead
    zero = [{"always_direct": 0.70, "always_long": 0.60, "mixture": 0.72, "queue_aware": 0.72}] * 3
    assert not D.development_gate(zero)["passed"]


def test_score_keeps_expired_questions_in_the_denominator():
    records = [
        {"question_id": "a", "deadline_met": True},
        {"question_id": "b", "deadline_met": False},
        {"question_id": "c", "deadline_met": True},
    ]
    scored = D.primary_fraction(records, {"a": True, "c": False})
    assert scored == {"n": 3, "hits": 1, "in_time": 2, "fraction": 1 / 3}


def test_locked_splits_are_refused():
    splits = {"splits": {name: {"items": [{"question_id": f"{name}_q"}]} for name in
                         ("math_train_calibration", *D.LOCKED_SPLITS)}}
    assert D.locked_overlap(["math_train_calibration_q"], splits) == {}
    hit = D.locked_overlap(["gsm8k_test_q"], splits)
    assert list(hit) == ["gsm8k_test"]


# ------------------------------------------------------------------ driver

def direct_script(n, steps, key="A"):
    script = {}
    for i in range(n):
        for s in range(4):
            script[rid(f"q{i}", "direct", s)] = (steps, key)
    return script


def test_driver_reports_an_outcome_when_the_question_closes():
    seen = []
    _, out = serve(D.AlwaysDirect(), direct_script(2, 1), [q(0, 0.0), q(1, 0.0)], slots=8, deadline_s=30,
                   on_outcome=lambda q: seen.append(q.outcome["question_id"]))
    assert seen == ["q0", "q1"]
    assert [row["question_id"] for row in out["questions"]] == ["q0", "q1"]


def test_driver_waits_for_an_arrival_and_answers_inside_the_deadline():
    be, out = serve(D.AlwaysDirect(), direct_script(1, 2), [q(0, 10.0)], slots=4, deadline_s=30)
    row = out["questions"][0]
    assert be.submitted and row["deadline_met"] and row["selected_answer"] == "A"
    assert row["scheduled_arrival"] == 10.0 and row["completion_time"] == 12.0
    assert row["admitted"] and row["launched_long"] is False


def test_driver_expires_a_question_that_is_still_running_at_the_deadline():
    be, out = serve(D.AlwaysDirect(), direct_script(1, 10), [q(0, 0.0)], slots=4, deadline_s=3)
    row = out["questions"][0]
    assert row["deadline_met"] is False and row["selected_answer"] is None
    assert row["decision_reason"] == "deadline" and be.cancelled
    assert out["stats"]["expired"] == 1


def test_driver_expires_a_question_that_never_gets_a_slot():
    # q0 holds the only four slots past q1's deadline.
    script = direct_script(2, 10)
    be, out = serve(D.AlwaysDirect(), script, [q(0, 0.0), q(1, 0.0)], slots=4, deadline_s=3)
    assert out["questions"][0]["admitted"] and out["questions"][0]["deadline_met"] is False
    assert out["questions"][1]["admitted"] is False and out["questions"][1]["deadline_met"] is False
    assert all(not s.startswith("q1:") for s in be.submitted)
    assert D.primary_fraction(out["questions"], {"q0": True, "q1": True})["fraction"] == 0.0


def test_queue_aware_uses_the_long_answer_only_when_the_server_is_idle():
    script = {}
    for mode in ("direct", "natural"):
        for s in range(4):
            script[rid("q0", mode, s)] = (5, "A")
            script[rid("q1", mode, s)] = (2, "B")
    _, out = serve(D.QueueAware(), script, [q(0, 0.0), q(1, 0.0)], slots=8, deadline_s=240)
    assert out["questions"][0]["choice"] == "long" and out["questions"][0]["launched_long"]
    assert out["questions"][1]["choice"] == "direct" and out["questions"][1]["launched_long"] is False
    assert out["questions"][0]["selected_answer"] == "A"
    assert out["questions"][1]["selected_answer"] == "B"


def test_a_finish_in_the_step_that_crosses_the_deadline_does_not_count():
    _, out = serve(D.AlwaysDirect(), direct_script(1, 2), [q(0, 0.0)], slots=4, deadline_s=1)
    assert out["questions"][0]["deadline_met"] is False
    assert out["questions"][0]["selected_answer"] is None


def test_disagree_arm_stops_on_a_direct_quorum_and_otherwise_launches_long():
    agree = {rid("q0", "direct", s): (1, "A") for s in range(3)}
    agree[rid("q0", "direct", 3)] = (50, "A")
    _, out = serve(D.DisagreeMixture(), agree, [q(0, 0.0)], slots=4, deadline_s=30)
    assert out["questions"][0]["decision_reason"] == "direct_quorum"
    assert out["questions"][0]["launched_long"] is False
    assert out["stats"]["cancelled"] == 1                       # the fourth direct sample

    script = {rid("q0", "direct", s): (1, "ABCD"[s]) for s in range(4)}
    script.update({rid("q0", "natural", D.LONG_BASE + s): (2, "Z") for s in range(4)})
    _, out = serve(D.DisagreeMixture(), script, [q(0, 0.0)], slots=8, deadline_s=30)
    row = out["questions"][0]
    assert row["launched_long"] and row["decision_reason"] == "long_vote"
    assert row["selected_answer"] == "Z" and row["deadline_met"]


def test_a_queued_long_launch_is_dropped_when_the_deadline_passes():
    # Three direct samples finish at t=1 and disagree; the fourth is still
    # running, so the long launch is queued behind it. The deadline is 1.5s.
    script = {rid("q0", "direct", s): (1, "ABC"[s]) for s in range(3)}
    script[rid("q0", "direct", 3)] = (50, "D")
    for s in range(4):
        script[rid("q0", "natural", D.LONG_BASE + s)] = (5, "Z")
    be, out = serve(D.DisagreeMixture(), script, [q(0, 0.0)], slots=4, deadline_s=1.5)
    assert out["questions"][0]["deadline_met"] is False
    started = [s for s in be.submitted if ":natural:" in s]
    assert len(started) < 4                                 # at least one long request never ran
    assert out["stats"]["dropped_queued"] >= 1


def test_outcomes_carry_no_gold():
    _, out = serve(D.AlwaysLong(), {rid("q0", "natural", s): (1, "A") for s in range(4)},
                   [q(0, 0.0)], slots=4, deadline_s=10)
    em.assert_no_gold(out["questions"][0])


def test_fraction_mixture_is_a_property_of_the_question_id():
    script = {}
    for i in range(4):
        for mode in ("direct", "natural"):
            for s in range(4):
                script[rid(f"q{i}", mode, s)] = (1, "A")
    arm = D.FractionMixture("p50")
    _, out = serve(arm, script, [q(i, 0.0) for i in range(4)], slots=16, deadline_s=10)
    choices = [row["choice"] for row in out["questions"]]
    assert choices == ["long" if D.assigns_long(f"q{i}", 0.5) else "direct" for i in range(4)]
