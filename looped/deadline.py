"""Predeclared deadline study: when a longer answer is worth the wait.

This is the feasibility study decided on 2026-11-12. It is not a search.
The constants below are the protocol. ``configs/looped_serving/deadline_study.json``
is the same document for a reader; a run refuses the file if the two differ.

Four arms, one workload. Always-direct and always-long are the pure
allocations. The mixture is one of a fixed list, chosen on the calibration
split by correct answers delivered before the deadline, then frozen. The
queue-aware arm is one rule, not a fit: use the long answer only when this
question is the only one in service and at least half the deadline is left.

The score is the fraction of arrivals whose selected answer is correct and
whose decision time is at or before arrival + deadline. An expired question,
a late answer and a wrong answer all stay in the denominator. Nothing here
reads gold. Grading happens in ``scripts/grade_deadline_study.py``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Mapping, Optional, Sequence

from . import ensemble_state as es
from .allocation import (
    DIRECT,
    NATURAL,
    Decision,
    FixedWidth,
    Launch,
    Policy,
    quorum_winner,
    running,
    selected,
)

STUDY_NAME = "deadline_feasibility_2026_11_12"
DEPTH = 3
WIDTH = 4
DIRECT_CAP = 2048
LONG_CAP = 4096
DEADLINE_S = 240.0
RATES_PER_MIN = (4, 12)
PERIOD_QUESTIONS = 20
SLOTS = 32
KV_GB = 80
REPETITIONS = (0, 1, 2)
CALIBRATION_SEED = 0
GATE_POINTS = 2.0
LONG_SLACK_S = 120.0          # half the deadline; not estimated from either split
MAX_OTHER_UNFINISHED = 0      # long only while no other question is in service
MIXTURE_CANDIDATES = ("p25", "p50", "p75", "disagree")
MIXTURE_P = {"p25": 0.25, "p50": 0.50, "p75": 0.75}
STATIC_ARMS = ("always_direct", "always_long", "mixture")
ADAPTIVE_ARM = "queue_aware"
LOCKED_SPLITS = ("math_train_confirmation_reserve", "gsm8k_test", "math500_test")
CALIBRATION_SPLIT = "math_train_calibration"
VALIDATION_SPLIT = "math_train_validation"
LONG_BASE = 100               # natural samples of the disagree arm

STUDY = {
    "name": STUDY_NAME,
    "deadline_s": DEADLINE_S,
    "depth": DEPTH,
    "width": WIDTH,
    "direct_cap": DIRECT_CAP,
    "long_cap": LONG_CAP,
    "direct_mode": DIRECT,
    "long_mode": NATURAL,
    "rates_per_min": list(RATES_PER_MIN),
    "period_questions": PERIOD_QUESTIONS,
    "slots": SLOTS,
    "kv_gb": KV_GB,
    "repetitions": list(REPETITIONS),
    "calibration_seed": CALIBRATION_SEED,
    "gate_points": GATE_POINTS,
    "queue_aware": {"long_slack_s": LONG_SLACK_S, "max_other_unfinished": MAX_OTHER_UNFINISHED},
    "mixture_candidates": list(MIXTURE_CANDIDATES),
    "mixture_p": MIXTURE_P,
    "temperature": 0.6,
    "top_p": 0.95,
    "prompt_mode": "stock",
    "total_ut_steps": DEPTH,
    "dtype": "bfloat16",
    "execution": {"cuda_graphs": True, "static_buffers": True},
}


def study_document() -> dict:
    """The protocol as stored on disk. Key order follows ``STUDY``."""
    return json.loads(json.dumps(STUDY))


def check_study_file(path: str) -> dict:
    """Load the protocol file and refuse it if it is not the predeclared one."""
    with open(path) as f:
        got = json.load(f)
    if got != study_document():
        raise ValueError(f"{path} does not match the predeclared deadline study")
    return got


def arrival_trace(n: int, rates_per_min: Sequence[float] = RATES_PER_MIN,
                  period: int = PERIOD_QUESTIONS) -> List[float]:
    """Arrival offsets, in seconds, for ``n`` questions in list order.

    Block ``i // period`` uses ``rates_per_min[block mod len(rates)]`` questions
    per minute. Question 0 arrives at 0. The gap in front of every later
    question is one interarrival of that question's own rate, so a block of
    the fast rate really is denser. ``period`` must divide ``n`` so the
    workload is an integer number of blocks.
    """
    if n < 1 or period < 1 or not rates_per_min:
        raise ValueError("the trace needs questions, a positive period and a rate")
    if n % period != 0:
        raise ValueError(f"{n} questions do not fill blocks of {period}")
    if any(r <= 0 for r in rates_per_min):
        raise ValueError("arrival rates must be positive")
    arrivals = [0.0]
    for i in range(1, n):
        rate = rates_per_min[(i // period) % len(rates_per_min)]
        arrivals.append(arrivals[-1] + 60.0 / rate)
    return arrivals


def assigns_long(question_id: str, p_long: float) -> bool:
    """Whether a fixed mixture sends this question down the long path.

    The draw is a hash of the question id, so it does not change with the run
    seed, the arrival time or the queue. Repetitions differ only in sampling.
    """
    if not 0.0 <= p_long <= 1.0:
        raise ValueError(f"p_long must be in [0, 1], not {p_long}")
    bucket = int(hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:8], 16) % 10000
    return bucket < int(round(p_long * 10000))


class BudgetedWidth(FixedWidth):
    """Four samples at one depth, one mode and one cap. The pure arms."""

    name = "budgeted_width"

    def __init__(self, mode: str, cap: int) -> None:
        super().__init__(DEPTH, WIDTH, mode)
        if cap < 1:
            raise ValueError("cap must be positive")
        self.cap = int(cap)
        self.choice = "long" if mode == NATURAL else "direct"

    def initial(self) -> List[Launch]:
        return [Launch(self.depth, s, self.mode, self.cap) for s in range(self.width)]


class DisagreeThenLong(Policy):
    """Direct samples first. The long answer is launched only if three of the
    four direct samples can no longer agree.

    The long answer is a fresh four-sample natural vote at the long cap. It is
    the hypothesis the saved pools suggested and the calibration pools did not
    support. It does not look at the queue.
    """

    name = "disagree_then_long"
    version = "v1"

    def __init__(self) -> None:
        self.choice = "direct_then_long"

    def initial(self) -> List[Launch]:
        return [Launch(DEPTH, s, DIRECT, DIRECT_CAP) for s in range(WIDTH)]

    def decide(self, cands) -> Decision:
        direct = [c for c in cands if c.mode == DIRECT]
        long = [c for c in cands if c.mode == NATURAL]
        live = running(cands)
        if not long:
            win = quorum_winner(direct, 3)
            if win is not None:
                return Decision(cancel=[c.sample for c in live], done=True,
                                action="cancel_and_answer" if live else "answer",
                                reason="direct_quorum", select=win.sample)
            if running(direct):
                c1, _ = es.lead(es.Vote(c.sample, c.answer_key) for c in direct if c.status == "done")
                # ``lead`` ignores non-answers. A quorum of 3 can still appear
                # only if the unfinished direct samples could join the leader.
                if c1 + len(running(direct)) >= 3:
                    return Decision()
            return Decision(launch=[Launch(DEPTH, LONG_BASE + s, NATURAL, LONG_CAP) for s in range(WIDTH)],
                            action="launch", reason="no_direct_quorum")
        if running(long):
            return Decision()
        stop = [c.sample for c in live]
        winner = selected(long)
        return Decision(cancel=stop, done=True, action="cancel_and_answer" if stop else "answer",
                        reason="long_vote", select=None if winner is None else winner.sample)

    @property
    def max_candidates(self) -> int:
        return WIDTH * 2


class Arm:
    """Builds the per-question policy at admission. The queue-aware arm is the
    only one that reads ``slack_s`` and ``unfinished``."""

    name = "arm"

    def policy_for(self, question_id: str, *, slack_s: float, unfinished: int) -> Policy:
        raise NotImplementedError


class AlwaysDirect(Arm):
    name = "always_direct"

    def policy_for(self, question_id: str, *, slack_s: float, unfinished: int) -> Policy:
        return BudgetedWidth(DIRECT, DIRECT_CAP)


class AlwaysLong(Arm):
    name = "always_long"

    def policy_for(self, question_id: str, *, slack_s: float, unfinished: int) -> Policy:
        return BudgetedWidth(NATURAL, LONG_CAP)


class FractionMixture(Arm):
    name = "mixture"

    def __init__(self, candidate: str) -> None:
        if candidate not in MIXTURE_P:
            raise ValueError(f"unknown fraction mixture {candidate!r}")
        self.candidate = candidate
        self.p_long = MIXTURE_P[candidate]

    def policy_for(self, question_id: str, *, slack_s: float, unfinished: int) -> Policy:
        mode = NATURAL if assigns_long(question_id, self.p_long) else DIRECT
        return BudgetedWidth(mode, LONG_CAP if mode == NATURAL else DIRECT_CAP)


class DisagreeMixture(Arm):
    name = "mixture"

    def __init__(self) -> None:
        self.candidate = "disagree"

    def policy_for(self, question_id: str, *, slack_s: float, unfinished: int) -> Policy:
        return DisagreeThenLong()


class QueueAware(Arm):
    """Long only when nobody else is in service and half the deadline remains.

    Both thresholds are the predeclared constants. They are not fit on
    calibration or validation.
    """

    name = "queue_aware"

    def policy_for(self, question_id: str, *, slack_s: float, unfinished: int) -> Policy:
        long = slack_s >= LONG_SLACK_S and unfinished <= MAX_OTHER_UNFINISHED
        mode = NATURAL if long else DIRECT
        return BudgetedWidth(mode, LONG_CAP if long else DIRECT_CAP)


def arm_for(name: str, candidate: Optional[str] = None) -> Arm:
    """``always_direct``, ``always_long``, ``queue_aware``, or ``mixture`` with
    one of ``MIXTURE_CANDIDATES``."""
    if name == "always_direct":
        return AlwaysDirect()
    if name == "always_long":
        return AlwaysLong()
    if name == "queue_aware":
        return QueueAware()
    if name == "mixture":
        if candidate == "disagree":
            return DisagreeMixture()
        if candidate in MIXTURE_P:
            return FractionMixture(candidate)
        raise ValueError(f"mixture needs one of {MIXTURE_CANDIDATES}, not {candidate!r}")
    raise ValueError(f"unknown arm {name!r}")


def choose_mixture(runs: Sequence[Mapping]) -> str:
    """The calibration winner. Higher correct-in-time count, then fewer
    questions that actually started a long answer, then the candidate name.

    Every predeclared candidate must be present, each on the same questions.
    """
    by = {r["candidate"]: r for r in runs}
    if sorted(by) != sorted(MIXTURE_CANDIDATES) or len(by) != len(runs):
        raise ValueError(f"selection needs exactly {list(MIXTURE_CANDIDATES)}")
    ids = list(runs[0]["question_ids"])
    n = runs[0]["n"]
    for r in runs:
        if r["n"] != n or list(r["question_ids"]) != ids:
            raise ValueError("mixture runs are not the same questions in the same order")
        if r["n"] != len(r["question_ids"]):
            raise ValueError(f"{r['candidate']}: n does not match its question list")
    winner = min(runs, key=lambda r: (-r["hits"], r["long_questions"], r["candidate"]))
    return winner["candidate"]


def development_gate(reps: Sequence[Mapping[str, float]]) -> dict:
    """The continue rule. Mean advantage of the queue-aware arm over the best
    static arm, in percentage points, at least ``GATE_POINTS``, and a positive
    advantage on every repetition. A development screen, not a confirmation.
    """
    if len(reps) != len(REPETITIONS):
        raise ValueError(f"the gate wants {len(REPETITIONS)} repetitions, not {len(reps)}")
    advantages = []
    for rep in reps:
        missing = [k for k in (*STATIC_ARMS, ADAPTIVE_ARM) if k not in rep]
        if missing:
            raise ValueError(f"a repetition is missing {missing}")
        static = max(rep[name] for name in STATIC_ARMS)
        advantages.append(rep[ADAPTIVE_ARM] - static)
    mean = sum(advantages) / len(advantages)
    return {
        "advantages": advantages,
        "mean_advantage_points": 100.0 * mean,
        "gate_points": GATE_POINTS,
        "passed": 100.0 * mean >= GATE_POINTS and all(a > 0 for a in advantages),
    }


def primary_fraction(records: Sequence[Mapping], correct: Mapping[str, bool]) -> dict:
    """Hits over every arrival. ``correct`` is consulted only for an in-time
    selection; a missing id is a miss, not a skipped question."""
    if not records:
        raise ValueError("a score needs the arrivals")
    ids = [r["question_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question in the outcome list")
    hits = 0
    in_time = 0
    for r in records:
        met = bool(r["deadline_met"])
        in_time += met
        if met and correct.get(r["question_id"], False):
            hits += 1
    return {"n": len(records), "hits": hits, "in_time": in_time, "fraction": hits / len(records)}


def locked_overlap(question_ids: Sequence[str], splits: Mapping) -> Dict[str, List[str]]:
    """Ids that fall in a split this study must not touch."""
    banned = {}
    have = set(question_ids)
    for name in LOCKED_SPLITS:
        ids = [e["question_id"] for e in splits["splits"][name]["items"] if e["question_id"] in have]
        if ids:
            banned[name] = ids
    return banned
