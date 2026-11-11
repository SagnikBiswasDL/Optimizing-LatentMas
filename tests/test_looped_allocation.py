"""Allocation policies and the closed-loop driver, on a scripted backend."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from looped import allocation as al  # noqa: E402
from looped import experiment_manifest as em  # noqa: E402
from serving.looped import backend as B  # noqa: E402
from serving.looped.driver import run_closed_loop  # noqa: E402


def view(sample, key=None, status=al.DONE, depth=3):
    return al.CandidateView(sample, depth, status, key, "eos" if status == al.DONE else None, 10)


# ------------------------------------------------------------------ policies

def test_fixed_width_launches_greedy_or_samples_and_waits_for_all():
    assert al.FixedWidth(3, 1).initial() == [al.Launch(3, al.GREEDY)]
    p = al.FixedWidth(2, 4)
    assert p.initial() == [al.Launch(2, s) for s in range(4)]
    assert not p.decide([view(0, "A"), view(1, status=al.RUNNING)]).done
    d = p.decide([view(0, "A"), view(1, "B")])
    assert d.done and not d.launch and not d.cancel


def test_two_then_two_stops_only_on_a_real_agreement():
    p = al.TwoThenTwo(3)
    assert p.initial() == [al.Launch(3, 0), al.Launch(3, 1)]
    assert not p.decide([view(0, "A"), view(1, status=al.RUNNING)]).done
    d = p.decide([view(0, "A"), view(1, "A")])
    assert d.done and d.reason == "first_two_agree"
    d = p.decide([view(0, "A"), view(1, "B")])
    assert not d.done and d.launch == [al.Launch(3, 2), al.Launch(3, 3)]
    d = p.decide([view(0, None), view(1, None)])            # two missing answers are not agreement
    assert not d.done and len(d.launch) == 2
    four = [view(0, "A"), view(1, "B"), view(2, "B"), view(3, None)]
    assert p.decide(four).done
    assert al.selected(four).sample == 1


def test_two_then_two_can_fall_back_deeper():
    p = al.TwoThenTwo(3, 4)
    assert p.decide([view(0, "A"), view(1, "B")]).launch == [al.Launch(4, 2), al.Launch(4, 3)]
    assert p.max_candidates == 4


def test_certificate_cancels_exactly_when_the_lead_is_out_of_reach():
    p = al.CertificateCancel(3, 4)
    two = [view(0, "A"), view(1, "A"), view(2, status=al.RUNNING), view(3, status=al.RUNNING)]
    assert not p.decide(two).done                           # 2 - 0 is not > 2
    three = [view(0, "A"), view(1, "A"), view(2, "A"), view(3, status=al.RUNNING)]
    d = p.decide(three)
    assert d.done and d.cancel == [3] and d.reason == "strict_certificate"
    split = [view(0, "A"), view(1, "A"), view(2, "B"), view(3, status=al.RUNNING)]
    assert not p.decide(split).done                         # 2 - 1 is not > 1
    unanswered = [view(0, "A"), view(1, None), view(2, None), view(3, status=al.RUNNING)]
    assert not p.decide(unanswered).done                    # 1 - 0 is not > 1
    done = [view(0, "A"), view(1, "B"), view(2, "B"), view(3, "C")]
    assert p.decide(done).done and not p.decide(done).cancel


def test_certificate_never_changes_the_full_pool_winner():
    import itertools
    p = al.CertificateCancel(3, 4)
    for keys in itertools.product(["A", "B", None], repeat=4):
        full = al.selected([view(s, k) for s, k in enumerate(keys)])
        for order in itertools.permutations(range(4)):
            finished = []
            for n, s in enumerate(order):
                finished.append(s)
                cands = [view(i, keys[i]) if i in finished else view(i, status=al.RUNNING) for i in range(4)]
                d = p.decide(cands)
                if d.done:
                    sel = al.selected([c for c in cands if c.status == al.DONE])
                    assert (sel.answer_key if sel else None) == (full.answer_key if full else None)
                    break


def test_parse_policy():
    assert isinstance(al.parse_policy("fixed:3x4"), al.FixedWidth)
    p = al.parse_policy("two_then_two:3->4")
    assert (p.depth, p.fallback_depth) == (3, 4)
    assert al.parse_policy("two_then_two:2").fallback_depth == 2
    assert al.parse_policy("cert:3x8").width == 8
    with pytest.raises(ValueError):
        al.parse_policy("magic:1")


# ------------------------------------------------------------------ driver

class ScriptedBackend(B.Backend):
    """Request ``rid`` finishes after ``script[rid][0]`` steps with answer ``script[rid][1]``."""

    def __init__(self, script):
        self.script, self.live, self.t, self.max_live, self.cancelled = script, {}, 0, 0, []

    def submit(self, r):
        self.live[r.request_id] = [r, 0]
        self.max_live = max(self.max_live, len(self.live))

    def active(self):
        return len(self.live)

    def _completion(self, r, steps, reason):
        return B.Completion(r.request_id, r.depth, len(r.prompt_ids), [7] * steps, reason, 0.0, 0.0, float(self.t), 0, 0)

    def step(self):
        self.t += 1
        out = []
        for rid in list(self.live):
            self.live[rid][1] += 1
            r, n = self.live[rid]
            if n >= self.script[rid][0]:
                del self.live[rid]
                out.append(self._completion(r, n, B.EOS))
        return out

    def cancel(self, rid):
        r, n = self.live.pop(rid)
        self.cancelled.append(rid)
        return self._completion(r, n, B.CANCELLED)


def drive(policy, script, n_questions, slots):
    be = ScriptedBackend(script)
    answers, events = {}, []

    def make_request(qid, prompt, l):
        return B.GenerationRequest(em.candidate_id(qid, l.depth, l.sample), tuple(prompt), l.depth, 100)

    stats = run_closed_loop(
        be, policy, [(f"q{i}", (1, 2)) for i in range(n_questions)], slots=slots, make_request=make_request,
        keyer=lambda c: script[c.request_id][1], clock=lambda: float(be.t),
        on_question=lambda q, sel, now: answers.__setitem__(q.question_id, (sel.answer_key if sel else None, now - q.arrival)),
        on_decision=lambda q, d, now, free: events.append((q.question_id, d.action, free)),
    )
    return be, stats, answers, events


def test_driver_fixed_width_respects_slots_and_answers_everything():
    script = {f"q{i}:d3:s{s}": (3 + s, "A" if s < 3 else "B") for i in range(5) for s in range(4)}
    be, stats, answers, _ = drive(al.FixedWidth(3, 4), script, 5, slots=8)
    assert be.max_live == 8 and stats["launched"] == 20 and stats["cancelled"] == 0
    assert answers == {f"q{i}": ("A", 6.0) for i in range(5)}   # latency is the slowest candidate


def test_driver_two_then_two_launches_followups_before_new_questions():
    script = {}
    for i in range(3):
        script.update({f"q{i}:d3:s0": (2, "A"), f"q{i}:d3:s1": (2, "A" if i != 0 else "B"),
                       f"q{i}:d3:s2": (2, "B"), f"q{i}:d3:s3": (2, "B")})
    be, stats, answers, events = drive(al.TwoThenTwo(3), script, 3, slots=2)
    assert be.max_live == 2
    assert stats["launched"] == 4 + 2 + 2                     # q0 needed all four
    assert answers["q0"] == ("B", 4.0) and answers["q1"][0] == "A" and answers["q2"][0] == "A"
    # q0's follow-up ran before q1 was admitted: q1 finishes after q0.
    order = [q for q, action, _ in events if action == "answer"]
    assert order == ["q0", "q1", "q2"]


def test_driver_certificate_cancels_and_frees_slots_at_once():
    script = {f"q{i}:d3:s{s}": ((2, "A") if s < 3 else (50, "B")) for i in range(2) for s in range(4)}
    be, stats, answers, _ = drive(al.CertificateCancel(3, 4), script, 2, slots=4)
    assert stats["cancelled"] == 2 and sorted(be.cancelled) == ["q0:d3:s3", "q1:d3:s3"]
    assert answers == {"q0": ("A", 2.0), "q1": ("A", 2.0)}     # not 50 steps
    assert be.active() == 0


def test_driver_rejects_a_policy_that_cannot_fit():
    with pytest.raises(ValueError):
        drive(al.FixedWidth(3, 4), {}, 1, slots=2)


def test_driver_handles_candidates_that_finish_in_the_same_step():
    # s0..s2 agree and finish together with s3: the certificate must not try to
    # cancel a candidate the backend has already returned.
    script = {f"q0:d3:s{s}": (2, "A") for s in range(4)}
    be, stats, answers, events = drive(al.CertificateCancel(3, 4), script, 1, slots=4)
    assert stats["cancelled"] == 0 and answers == {"q0": ("A", 2.0)}
    assert [a for _, a, _ in events] == ["answer"]            # one decision for the whole step
    # Three finish together, the fourth is still running: it is cancelled once.
    script = {f"q0:d3:s{s}": ((2, "A") if s < 3 else (9, "B")) for s in range(4)}
    be, stats, answers, _ = drive(al.CertificateCancel(3, 4), script, 1, slots=4)
    assert be.cancelled == ["q0:d3:s3"] and answers == {"q0": ("A", 2.0)}


# ------------------------------------------------------------------ modes, quorum, fast-then-think

def mview(sample, mode, key=None, status=al.DONE, depth=3, think=None, reason="eos"):
    return al.CandidateView(sample, depth, status, key, reason if status == al.DONE else None, 10, mode, think)


def test_quorum_cancel_answers_at_the_first_agreement_and_names_the_winner():
    p = al.QuorumCancel(3, 4)
    assert p.initial() == [al.Launch(3, s, "natural") for s in range(4)]
    one = [view(0, "A"), view(1, status=al.RUNNING), view(2, status=al.RUNNING), view(3, status=al.RUNNING)]
    assert not p.decide(one).done
    two = [view(0, "B"), view(1, status=al.RUNNING), view(2, "A"), view(3, "A")]
    d = p.decide(two)
    assert d.done and d.cancel == [1] and d.reason == "quorum" and d.select == 2
    # Two answers reaching the quorum in the same step are a tie, not a quorum.
    tie = [view(0, "A"), view(1, "A"), view(2, "B"), view(3, "B")]
    d = p.decide(tie)
    assert d.done and d.reason == "all_finished" and d.select is None and al.selected(tie).sample == 0
    none = [view(0, None), view(1, None), view(2, "A"), view(3, "B")]
    assert p.decide(none).reason == "all_finished"
    with pytest.raises(ValueError):
        al.QuorumCancel(3, 4, quorum=5)


def test_quorum_cancel_can_let_a_finished_reasoning_answer_win():
    p = al.QuorumCancel(3, 4, think_trumps=True)
    cands = [mview(0, "natural", "A", think=False), mview(1, "natural", "B", think=True),
             mview(2, "natural", status=al.RUNNING), mview(3, "natural", status=al.RUNNING)]
    d = p.decide(cands)
    assert d.done and d.reason == "think_answer" and d.select == 1 and d.cancel == [2, 3]
    capped = [mview(0, "natural", "A", think=False), mview(1, "natural", None, think=True, reason="length_cap"),
              mview(2, "natural", status=al.RUNNING), mview(3, "natural", status=al.RUNNING)]
    assert not p.decide(capped).done                         # a reasoning trace without an answer decides nothing


def test_fast_then_think_answers_on_a_direct_quorum():
    p = al.FastThenThink(3, 2, 4, 2, think_cap=4096)
    assert p.initial() == [al.Launch(3, 0, "direct"), al.Launch(3, 1, "direct")]
    assert p.max_candidates == 4
    assert not p.decide([mview(0, "direct", "A"), mview(1, "direct", status=al.RUNNING)]).done
    d = p.decide([mview(0, "direct", "A"), mview(1, "direct", "A")])
    assert d.done and d.reason == "direct_quorum" and d.select == 0 and not d.launch


def test_fast_then_think_escalates_once_and_returns_the_first_reasoning_answer():
    p = al.FastThenThink(3, 2, 4, 2, think_cap=4096)
    split = [mview(0, "direct", "A"), mview(1, "direct", "B")]
    d = p.decide(split)
    assert not d.done and d.launch == [al.Launch(4, 100, "think", 4096), al.Launch(4, 101, "think", 4096)]
    # A quorum that can no longer form escalates without waiting for the last direct answer.
    early = [mview(0, "direct", None), mview(1, "direct", status=al.RUNNING)]
    d = p.decide(early)
    assert d.launch and not d.cancel
    running = split + [mview(100, "think", status=al.RUNNING, depth=4), mview(101, "think", status=al.RUNNING, depth=4)]
    assert not p.decide(running).done and not p.decide(running).launch       # never escalates twice
    won = split + [mview(100, "think", status=al.RUNNING, depth=4), mview(101, "think", "B", depth=4)]
    d = p.decide(won)
    assert d.done and d.reason == "think_answer" and d.select == 101 and d.cancel == [100]


def test_fast_then_think_falls_back_to_the_direct_plurality():
    p = al.FastThenThink(3, 2, 4, 1)
    capped = [mview(0, "direct", "A"), mview(1, "direct", "B"), mview(100, "think", None, depth=4, reason="length_cap")]
    d = p.decide(capped)
    assert d.done and d.reason == "fallback_direct_plurality" and d.select == 0
    nothing = [mview(0, "direct", None), mview(1, "direct", None), mview(100, "think", None, depth=4)]
    d = p.decide(nothing)
    assert d.done and d.select is None
    alone = al.FastThenThink(3, 2, 4, 0)                    # no reasoning stage: direct answers only
    d = alone.decide([mview(0, "direct", "A"), mview(1, "direct", "B")])
    assert d.done and not d.launch and d.select == 0


def test_parse_policy_modes():
    p = al.parse_policy("fixed:3x4:direct")
    assert (p.depth, p.width, p.mode) == (3, 4, "direct")
    p = al.parse_policy("quorum:3x8/3:direct:trump")
    assert (p.depth, p.width, p.quorum, p.mode, p.think_trumps) == (3, 8, 3, "direct", True)
    assert al.parse_policy("quorum:2x4").quorum == 2
    p = al.parse_policy("ftt:3x4/3+4x2@4096")
    assert (p.direct_depth, p.direct_width, p.quorum, p.think_depth, p.think_width, p.think_cap) == (3, 4, 3, 4, 2, 4096)
    assert al.parse_policy("ftt:3x1+4x1").quorum == 1
    with pytest.raises(ValueError):
        al.parse_policy("quorum:3x4:sideways")
    with pytest.raises(ValueError):
        al.parse_policy("fixed:3x4:sideways")


def test_driver_returns_the_candidate_the_policy_selected():
    # Direct answers split; the reasoning candidate finishes later with B and wins
    # although A is the lowest-indexed plurality answer.
    script = {"q0:d3:s0": (2, "A"), "q0:d3:s1": (3, "B"), "q0:d4:s100": (9, "B"), "q0:d4:s101": (50, "C")}
    be = ScriptedBackend(script)
    seen, got = [], {}

    def make_request(qid, prompt, l):
        seen.append((l.mode, l.cap))
        return B.GenerationRequest(em.candidate_id(qid, l.depth, l.sample), tuple(prompt), l.depth, l.cap or 100)

    stats = run_closed_loop(
        be, al.FastThenThink(3, 2, 4, 2, think_cap=4096), [("q0", (1, 2))], slots=4, make_request=make_request,
        keyer=lambda c: script[c.request_id][1], clock=lambda: float(be.t),
        on_question=lambda q, sel, now: got.update(sample=sel.sample, key=sel.answer_key, mode=sel.mode, t=now - q.arrival),
    )
    assert seen == [("direct", None), ("direct", None), ("think", 4096), ("think", 4096)]
    assert got == {"sample": 100, "key": "B", "mode": "think", "t": 12.0}      # 3 direct steps, then 9
    assert stats["cancelled"] == 1 and be.cancelled == ["q0:d4:s101"]


def test_driver_refuses_a_selection_without_an_answer():
    class Bad(al.FixedWidth):
        def decide(self, cands):
            d = super().decide(cands)
            d.select = 1
            return d
    script = {"q0:d3:s0": (2, "A"), "q0:d3:s1": (2, None)}
    with pytest.raises(ValueError):
        drive(Bad(3, 2), script, 1, slots=2)


# ------------------------------------------------------------------ replay backend

def test_replay_backend_serves_saved_candidates_on_a_token_clock():
    from serving.looped.replay_backend import ReplayBackend
    saved = {"a": (5, B.EOS, "pa"), "b": (9, B.LENGTH_CAP, "pb"), "c": (7, B.EOS, "pc")}
    be = ReplayBackend(lambda r: saved[r.request_id], lambda r: 9)
    for rid, cap in (("a", 9), ("b", 9), ("c", 4)):
        be.submit(B.GenerationRequest(rid, (1, 2, 3), 3, cap))
    done = {}
    while be.active():
        for c in be.step():
            done[c.request_id] = c
    assert (done["a"].finish_reason, len(done["a"].token_ids), done["a"].finish_time) == (B.EOS, 5, 5.0)
    assert (done["b"].finish_reason, len(done["b"].token_ids)) == (B.LENGTH_CAP, 9)
    assert (done["c"].finish_reason, len(done["c"].token_ids), done["c"].detail["cut_by_cap"]) == (B.LENGTH_CAP, 4, True)
    assert done["a"].logical_loop_tokens == 3 * (3 + 5 - 1) and be.payload["a"] == "pa"
    be.submit(B.GenerationRequest("a", (1,), 3, 9))
    be.step(); be.step()
    c = be.cancel("a")
    assert c.finish_reason == B.CANCELLED and len(c.token_ids) == 2 and be.active() == 0
    with pytest.raises(ValueError):                           # the pool cannot say what happens past its own cap
        be.submit(B.GenerationRequest("b", (1,), 3, 10))
