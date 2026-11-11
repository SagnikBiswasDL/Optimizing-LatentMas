"""Vote selection, tie rules, and the strict cancellation certificate, checked
exhaustively on small pools."""
import itertools
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from looped.ensemble_state import (  # noqa: E402
    FIRST_LISTED, LOWEST_CANDIDATE, TIE_RULES, Vote, completion_rounds, lead, select,
    strict_certificate, tally,
)


def votes(keys, order=None):
    order = range(len(keys)) if order is None else order
    return [Vote(i, keys[i]) for i in order]


def test_hand_checked_selection():
    assert select(votes(["a", "b", "a", None])).key == "a"
    assert select(votes([None, None])) is None
    assert select([]) is None
    assert tally(votes(["a", None, "a", "b"])) == {"a": 2, "b": 1}
    assert lead(votes(["a", None, "a", "b"])) == (2, 1)
    assert lead(votes(["a", "a"])) == (2, 0)
    assert lead(votes([None])) == (0, 0)


def test_tie_rules_differ_only_in_order_dependence():
    keys = ["b", "a", "a", "b"]                     # 2-2 tie
    assert select(votes(keys), FIRST_LISTED).key == "b"
    assert select(votes(keys, order=[1, 0, 2, 3]), FIRST_LISTED).key == "a"      # arrival order matters
    for order in itertools.permutations(range(4)):
        assert select(votes(keys, order), LOWEST_CANDIDATE).key == "b"           # it does not here
    with pytest.raises(ValueError):
        select(votes(keys), "coin_flip")


def test_first_listed_matches_stage0_vote_rule():
    # scripts/loopdelta_analyze.vote, reimplemented literally for comparison
    def stage0(keys):
        counts = {}
        for key in keys:
            if key is not None:
                counts[key] = counts.get(key, 0) + 1
        if not counts:
            return None
        best = max(counts.values())
        return next(key for key in keys if key is not None and counts[key] == best)

    for keys in itertools.product(["a", "b", "c", None], repeat=4):
        got = select(votes(list(keys)), FIRST_LISTED)
        assert (got.key if got else None) == stage0(keys)


ALPHABET = ["a", "b", "c", "unseen", None]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_certificate_preserves_full_pool_winner_exhaustively(n):
    """Every assignment of answers (including invalid votes and an answer no
    completed candidate gave) x every set of completed candidates."""
    fired = 0
    for keys in itertools.product(ALPHABET, repeat=n):
        full = votes(list(keys))
        for mask in range(1 << n):
            done = [v for v in full if mask >> v.candidate & 1]
            if not strict_certificate(done, n - len(done)):
                continue
            fired += 1
            leader = select(done, LOWEST_CANDIDATE).key
            counts = tally(full)
            assert sum(1 for c in counts.values() if c == max(counts.values())) == 1   # unique
            for rule in TIE_RULES:
                assert select(full, rule).key == leader
    assert fired > 0


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_certificate_is_tight(n):
    """When it does not fire and work remains, some completion denies the
    leader a unique win, so no weaker margin would be safe."""
    for keys in itertools.product(["a", "b", None], repeat=n):
        full = votes(list(keys))
        for mask in range(1 << n):
            done = [v for v in full if mask >> v.candidate & 1]
            u = n - len(done)
            c1, c2 = lead(done)
            if u == 0 or c1 == 0 or strict_certificate(done, u):
                continue
            leader = select(done, LOWEST_CANDIDATE).key
            rivals = [key for key in tally(done) if key != leader]
            rival = max(rivals, key=lambda key: tally(done)[key]) if rivals else "unseen"
            adversarial = done + [Vote(100 + i, rival) for i in range(u)]
            counts = tally(adversarial)
            assert counts[rival] >= counts[leader]


def test_certificate_edge_cases():
    assert not strict_certificate([], 0)
    assert not strict_certificate(votes([None, None]), 0)       # nothing to return
    assert strict_certificate(votes(["a"]), 0)
    assert not strict_certificate(votes(["a"]), 1)              # an unseen answer could tie
    assert strict_certificate(votes(["a", "a"]), 1)
    assert not strict_certificate(votes(["a", "a", "b"]), 1)
    assert strict_certificate(votes(["a", "a", "a"]), 1)
    with pytest.raises(ValueError):
        strict_certificate(votes(["a"]), -1)


def test_completion_rounds_groups_ties():
    assert completion_rounds([5, 3, 5, 9]) == [[1], [0, 2], [3]]
    assert completion_rounds([]) == []
