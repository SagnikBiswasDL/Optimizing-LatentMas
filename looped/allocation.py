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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import ensemble_state as es

RUNNING, DONE, CANCELLED = "running", "done", "cancelled"
GREEDY = -1


@dataclass(frozen=True)
class Launch:
    depth: int
    sample: int          # GREEDY (-1) or a sample index


@dataclass(frozen=True)
class CandidateView:
    """What a policy may know about one candidate."""

    sample: int
    depth: int
    status: str
    answer_key: Optional[str] = None
    finish_reason: Optional[str] = None
    gen_tokens: int = 0


@dataclass
class Decision:
    launch: List[Launch] = field(default_factory=list)
    cancel: List[int] = field(default_factory=list)      # sample indices
    done: bool = False
    action: str = "wait"
    reason: Optional[str] = None


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
    return [c for c in cands if c.status == RUNNING]


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

    def __init__(self, depth: int, width: int) -> None:
        if width < 1:
            raise ValueError("width must be positive")
        self.depth, self.width = int(depth), int(width)

    def initial(self) -> List[Launch]:
        if self.width == 1:
            return [Launch(self.depth, GREEDY)]
        return [Launch(self.depth, s) for s in range(self.width)]

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


def parse_policy(spec: str) -> Policy:
    """``fixed:<depth>x<width>``, ``two_then_two:<depth>[-><fallback depth>]``
    or ``cert:<depth>x<width>``."""
    kind, _, arg = spec.partition(":")
    if kind == "fixed":
        d, w = arg.split("x")
        return FixedWidth(int(d), int(w))
    if kind == "two_then_two":
        d, _, f = arg.partition("->")
        return TwoThenTwo(int(d), int(f) if f else None)
    if kind == "cert":
        d, w = arg.split("x")
        return CertificateCancel(int(d), int(w))
    raise ValueError(f"unknown policy spec {spec!r}")
