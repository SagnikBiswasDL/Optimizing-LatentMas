"""Unweighted vote state for one question: selection, ties, and the strict
cancellation certificate.

A vote is ``(candidate, key)``. ``key`` is the vote-identity key from
``looped.answer_identity``; ``None`` means the candidate produced no usable answer
(missing box, parse failure, cancelled) and counts for nothing.

Tie rules
---------
``first_listed``      the first vote, in the order given, whose key has the top
                      count. This is the Stage-0 rule (``loopdelta_analyze.vote``).
                      Its result depends on the listing order, so in a live system
                      it would depend on which candidate happened to finish first.
``lowest_candidate``  among keys with the top count, the key held by the smallest
                      ``candidate`` value. Independent of arrival order.

Strict certificate
------------------
For a fixed pool of N unweighted candidates, with leading and runner-up counts
``c1 >= c2`` among the completed ones and ``u`` still unfinished: if
``c1 - c2 > u`` then no completion of the remaining candidates can change the
unique winner, because every rival (including an answer nobody has given yet,
which has count 0 <= c2) can reach at most ``c2 + u < c1``. That guarantees the
same winner as the full pool under this voting rule. It says nothing about the
winner being correct, and nothing about a policy that could launch candidates
beyond the fixed N.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, List, Optional, Sequence, Tuple

FIRST_LISTED = "first_listed"
LOWEST_CANDIDATE = "lowest_candidate"
TIE_RULES = (FIRST_LISTED, LOWEST_CANDIDATE)


@dataclass(frozen=True)
class Vote:
    candidate: Hashable           # stable and orderable within a question, e.g. (depth, sample)
    key: Optional[str]


def tally(votes: Iterable[Vote]) -> Counter:
    return Counter(v.key for v in votes if v.key is not None)


def select(votes: Sequence[Vote], tie_rule: str = LOWEST_CANDIDATE) -> Optional[Vote]:
    """The winning vote, or ``None`` when no candidate has a usable answer.

    Returns a vote (not just the key) so the caller can read the winning
    candidate's own record.
    """
    if tie_rule not in TIE_RULES:
        raise ValueError(f"unknown tie rule {tie_rule!r}; known: {TIE_RULES}")
    counts = tally(votes)
    if not counts:
        return None
    best = max(counts.values())
    tied = [v for v in votes if v.key is not None and counts[v.key] == best]
    if tie_rule == FIRST_LISTED:
        return tied[0]
    return min(tied, key=lambda v: v.candidate)


def lead(votes: Iterable[Vote]) -> Tuple[int, int]:
    """(leading count, runner-up count); a missing runner-up counts 0."""
    top = sorted(tally(votes).values(), reverse=True)
    return (top[0] if top else 0), (top[1] if len(top) > 1 else 0)


def strict_certificate(completed: Iterable[Vote], unfinished: int) -> bool:
    """True when the completed votes already fix the full pool's unique winner."""
    if unfinished < 0:
        raise ValueError("unfinished must be >= 0")
    c1, c2 = lead(completed)
    return c1 > 0 and c1 - c2 > unfinished


def completion_rounds(lengths: Sequence[int]) -> List[List[int]]:
    """Group candidate indices by finishing round when all start together and
    each emits one token per round (a length proxy, not a latency measurement)."""
    rounds = {}
    for i, n in enumerate(lengths):
        rounds.setdefault(int(n), []).append(i)
    return [rounds[n] for n in sorted(rounds)]
