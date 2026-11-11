"""Per-question allocation policies for a looped model: which candidates to
launch, at what depth, when to cancel and when to answer.

A policy sees only observable state: for each candidate its sample index, depth,
status, finish reason, token count and (once finished) answer key. It never
sees gold, level or subject. Every policy here is a baseline; none uses a
learned or calibrated score.

Policies are pure functions of that state, so the same object drives a live
engine (``serving/looped/driver.py``) and can be replayed on a saved pool.

``FixedWidth``         launch ``width`` candidates at one depth, answer when all
                       have finished (width 1 is the greedy candidate).
``TwoThenTwo``         launch two; answer if they agree, otherwise launch two
                       more (optionally deeper) and vote over four.
``CertificateCancel``  launch ``width``; cancel the rest as soon as the finished
                       votes already fix the full-pool plurality winner
                       (``c1 - c2 > unfinished``). With a strict lead the
                       unfinished candidates cannot change the winner, so the
                       answer key is the one the full pool would have returned.
``QuorumCancel``       launch ``width``; answer and cancel the rest as soon as
                       ``quorum`` finished candidates agree. Not a certificate:
                       the unfinished candidates could have outvoted the quorum.
``FastThenThink``      launch short direct answers first; answer on a direct
                       quorum, otherwise launch reasoning candidates and return
                       the first one that reaches an answer (see ``looped/modes.py``).

A launch can force an answer mode and carry its own token cap. ``Decision.select``
names the candidate whose answer is returned when that is not the plurality winner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import ensemble_state as es
from .modes import DIRECT, NATURAL, THINK
from .modes import POLICY_MODES as MODES

QUEUED, RUNNING, DONE, CANCELLED = "queued", "running", "done", "cancelled"
GREEDY = -1


@dataclass(frozen=True)
class Launch:
    depth: int
    sample: int          # GREEDY (-1) or a sample index, unique within a question
    mode: str = NATURAL  # looped.modes
    cap: Optional[int] = None   # token cap for this candidate; None means the run's default


@dataclass(frozen=True)
class CandidateView:
    """What a policy may know about one candidate."""

    sample: int
    depth: int
    status: str
    answer_key: Optional[str] = None
    finish_reason: Optional[str] = None
    gen_tokens: int = 0
    mode: str = NATURAL            # the mode that was requested
    think: Optional[bool] = None   # whether the answer opened a reasoning block, once known


@dataclass
class Decision:
    launch: List[Launch] = field(default_factory=list)
    cancel: List[int] = field(default_factory=list)      # sample indices
    done: bool = False
    action: str = "wait"
    reason: Optional[str] = None
    select: Optional[int] = None   # sample whose answer is returned; None means the plurality winner


def votes_of(cands: Sequence[CandidateView]) -> List[es.Vote]:
    """Votes of finished candidates, ordered by (depth, sample)."""
    done = sorted((c for c in cands if c.status == DONE), key=lambda c: (c.depth, c.sample))
    return [es.Vote((c.depth, c.sample), c.answer_key) for c in done]


def selected(cands: Sequence[CandidateView]) -> Optional[CandidateView]:
    """The candidate whose answer is returned, or ``None`` if nothing is usable."""
    win = es.select(votes_of(cands), es.LOWEST_CANDIDATE)
    if win is None:
        return None
    return next(c for c in cands if (c.depth, c.sample) == win.candidate)


def running(cands: Sequence[CandidateView]) -> List[CandidateView]:
    """Candidates that can still produce an answer: running, or launched by an
    earlier decision and waiting for a slot."""
    return [c for c in cands if c.status in (RUNNING, QUEUED)]


def quorum_winner(cands: Sequence[CandidateView], quorum: int) -> Optional[CandidateView]:
    """Lowest-indexed finished candidate whose answer at least ``quorum``
    finished candidates share and no other answer matches in count."""
    votes = votes_of(cands)
    c1, c2 = es.lead(votes)
    if c1 < quorum or c1 == c2:
        return None
    win = es.select(votes, es.LOWEST_CANDIDATE)
    return next(c for c in cands if (c.depth, c.sample) == win.candidate)


class Policy:
    name = "policy"
    version = "v1"

    def initial(self) -> List[Launch]:
        raise NotImplementedError

    def decide(self, cands: Sequence[CandidateView]) -> Decision:
        """Called each time one of the question's candidates finishes."""
        raise NotImplementedError

    @property
    def max_candidates(self) -> int:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, **{k: v for k, v in self.__dict__.items()}}


class FixedWidth(Policy):
    name = "fixed_width"

    def __init__(self, depth: int, width: int, mode: str = NATURAL) -> None:
        if width < 1:
            raise ValueError("width must be positive")
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.depth, self.width, self.mode = int(depth), int(width), mode

    def initial(self) -> List[Launch]:
        if self.width == 1:
            return [Launch(self.depth, GREEDY, self.mode)]
        return [Launch(self.depth, s, self.mode) for s in range(self.width)]

    def decide(self, cands: Sequence[CandidateView]) -> Decision:
        if running(cands):
            return Decision()
        return Decision(done=True, action="answer", reason="all_finished")

    @property
    def max_candidates(self) -> int:
        return self.width


class TwoThenTwo(Policy):
    name = "two_then_two"

    def __init__(self, depth: int, fallback_depth: Optional[int] = None) -> None:
        self.depth = int(depth)
        self.fallback_depth = int(fallback_depth if fallback_depth is not None else depth)

    def initial(self) -> List[Launch]:
        return [Launch(self.depth, 0), Launch(self.depth, 1)]

    def decide(self, cands: Sequence[CandidateView]) -> Decision:
        if running(cands):
            return Decision()
        if len(cands) == 2:
            a, b = sorted(cands, key=lambda c: c.sample)
            if a.answer_key is not None and a.answer_key == b.answer_key:
                return Decision(done=True, action="answer", reason="first_two_agree")
            return Decision(launch=[Launch(self.fallback_depth, 2), Launch(self.fallback_depth, 3)],
                            action="launch", reason="first_two_disagree")
        return Decision(done=True, action="answer", reason="all_finished")

    @property
    def max_candidates(self) -> int:
        return 4


class CertificateCancel(Policy):
    name = "certificate_cancel"

    def __init__(self, depth: int, width: int) -> None:
        if width < 2:
            raise ValueError("width must be at least 2")
        self.depth, self.width = int(depth), int(width)

    def initial(self) -> List[Launch]:
        return [Launch(self.depth, s) for s in range(self.width)]

    def decide(self, cands: Sequence[CandidateView]) -> Decision:
        live = running(cands)
        if not live:
            return Decision(done=True, action="answer", reason="all_finished")
        if es.strict_certificate(votes_of(cands), len(live)):
            return Decision(cancel=[c.sample for c in live], done=True, action="cancel_and_answer",
                            reason="strict_certificate")
        return Decision()

    @property
    def max_candidates(self) -> int:
        return self.width


class QuorumCancel(Policy):
    name = "quorum_cancel"

    def __init__(self, depth: int, width: int, quorum: int = 2, mode: str = NATURAL, think_trumps: bool = False) -> None:
        if width < 2 or not 1 <= quorum <= width:
            raise ValueError("need width >= 2 and 1 <= quorum <= width")
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.depth, self.width, self.quorum, self.mode = int(depth), int(width), int(quorum), mode
        self.think_trumps = bool(think_trumps)

    def initial(self) -> List[Launch]:
        return [Launch(self.depth, s, self.mode) for s in range(self.width)]

    def decide(self, cands: Sequence[CandidateView]) -> Decision:
        live = running(cands)
        stop = [c.sample for c in live]
        if self.think_trumps:
            thought = [c for c in cands if c.status == DONE and c.think and c.answer_key is not None]
            if thought:
                return Decision(cancel=stop, done=True, action="cancel_and_answer" if stop else "answer",
                                reason="think_answer", select=min(c.sample for c in thought))
        win = quorum_winner(cands, self.quorum)
        if win is not None:
            return Decision(cancel=stop, done=True, action="cancel_and_answer" if stop else "answer",
                            reason="quorum", select=win.sample)
        if not live:
            return Decision(done=True, action="answer", reason="all_finished")
        return Decision()

    @property
    def max_candidates(self) -> int:
        return self.width


class FastThenThink(Policy):
    """Direct answers first, reasoning only when they do not settle the question.

    ``direct_width`` direct candidates start together. As soon as ``quorum`` of
    them agree the question is answered and the rest are cancelled. Once a
    quorum can no longer form, ``think_width`` reasoning candidates are launched
    (direct candidates still running keep going: they are short and feed the
    fallback). The first reasoning candidate that finishes with an answer is
    returned. If none does, the answer is the plurality of the direct answers.
    """
    name = "fast_then_think"
    THINK_BASE = 100          # reasoning candidates use samples 100, 101, ...

    def __init__(self, direct_depth: int, direct_width: int, think_depth: int, think_width: int,
                 quorum: int = 2, think_cap: Optional[int] = None) -> None:
        if direct_width < 1 or think_width < 0 or not 1 <= quorum <= max(direct_width, 1):
            raise ValueError("need direct_width >= 1, think_width >= 0 and 1 <= quorum <= direct_width")
        self.direct_depth, self.direct_width = int(direct_depth), int(direct_width)
        self.think_depth, self.think_width = int(think_depth), int(think_width)
        self.quorum, self.think_cap = int(quorum), (int(think_cap) if think_cap else None)

    def initial(self) -> List[Launch]:
        return [Launch(self.direct_depth, s, DIRECT) for s in range(self.direct_width)]

    def decide(self, cands: Sequence[CandidateView]) -> Decision:
        direct = [c for c in cands if c.mode == DIRECT]
        think = [c for c in cands if c.mode == THINK]
        live = running(cands)
        stop = [c.sample for c in live]
        thought = [c for c in think if c.status == DONE and c.answer_key is not None]
        if thought:
            return Decision(cancel=stop, done=True, action="cancel_and_answer" if stop else "answer",
                            reason="think_answer", select=min(c.sample for c in thought))
        win = quorum_winner(direct, self.quorum)
        if win is not None:
            return Decision(cancel=stop, done=True, action="cancel_and_answer" if stop else "answer",
                            reason="direct_quorum", select=win.sample)
        if running(direct):
            c1, _ = es.lead(votes_of(direct))
            if think or c1 + len(running(direct)) >= self.quorum:
                return Decision()            # a direct quorum can still form, or reasoning is already on its way
        if not think and self.think_width:
            # No direct quorum can form any more: too few agreeing answers, or
            # every direct candidate has finished and the leading answers tie.
            return Decision(launch=[Launch(self.think_depth, self.THINK_BASE + s, THINK, self.think_cap)
                                    for s in range(self.think_width)],
                            action="launch", reason="no_direct_quorum")
        if not live:
            fallback = selected(direct)
            return Decision(done=True, action="answer", reason="fallback_direct_plurality",
                            select=None if fallback is None else fallback.sample)
        return Decision()

    @property
    def max_candidates(self) -> int:
        return self.direct_width + self.think_width


def parse_policy(spec: str) -> Policy:
    """``fixed:<depth>x<width>[:<mode>]``, ``two_then_two:<depth>[-><fallback depth>]``,
    ``cert:<depth>x<width>``, ``quorum:<depth>x<width>[/<quorum>][:<mode>][:trump]`` or
    ``ftt:<depth>x<width>[/<quorum>]+<think depth>x<think width>[@<think cap>]``."""
    kind, _, arg = spec.partition(":")
    if kind == "quorum":
        head, *flags = arg.split(":")
        shape, _, k = head.partition("/")
        d, w = shape.split("x")
        mode = next((f for f in flags if f in MODES), NATURAL)
        if any(f not in MODES and f != "trump" for f in flags):
            raise ValueError(f"unknown flag in policy spec {spec!r}")
        return QuorumCancel(int(d), int(w), int(k) if k else 2, mode, "trump" in flags)
    if kind == "ftt":
        fast, _, slow = arg.partition("+")
        shape, _, k = fast.partition("/")
        d, w = shape.split("x")
        slow, _, cap = slow.partition("@")
        td, tw = slow.split("x")
        return FastThenThink(int(d), int(w), int(td), int(tw), int(k) if k else min(2, int(w)), int(cap) if cap else None)
    if kind == "fixed":
        shape, _, mode = arg.partition(":")
        d, w = shape.split("x")
        return FixedWidth(int(d), int(w), mode or NATURAL)
    if kind == "two_then_two":
        d, _, f = arg.partition("->")
        return TwoThenTwo(int(d), int(f) if f else None)
    if kind == "cert":
        d, w = arg.split("x")
        return CertificateCancel(int(d), int(w))
    raise ValueError(f"unknown policy spec {spec!r}")
