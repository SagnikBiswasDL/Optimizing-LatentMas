"""Keep/discard rules for the prefix-cache loop. No model, no GPU."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoresearch.prepare import (  # noqa: E402
    CONCURRENCY,
    assemble,
    best_keep,
    cache_hit,
    decide,
)
from autoresearch.run import DummyTokenizer, check  # noqa: E402


BASELINE = {"gsm_correct": 14, "gsm_n": 16}


def row(**kwargs):
    base = {
        "crash": False,
        "hit": True,
        "e2e_s": 420.0,
        "preemptions": 0,
        "gsm_correct": 14,
        "gsm_n": 16,
    }
    base.update(kwargs)
    return base


def test_shared_prefix_is_identical_and_suffixes_differ():
    tokenizer = DummyTokenizer()
    prefix = [7] * 32
    prompts = assemble(prefix, [tokenizer.encode(f"q{i}") for i in range(CONCURRENCY)])
    assert all(prompt[:32] == prefix for prompt in prompts)
    assert len({tuple(prompt[32:]) for prompt in prompts}) == CONCURRENCY


def test_cache_hit_requires_every_request():
    assert cache_hit([32, 32, 32, 32, 32], 32)
    assert cache_hit([16, 16, 16, 16, 16], 20)
    assert not cache_hit([0, 32, 32, 32, 32], 32)
    assert not cache_hit([32, 32, 32, 32], 32)


def test_first_fit_with_accuracy_is_a_keep():
    assert decide(row(), BASELINE, None) == "keep"


def test_miss_overflow_and_accuracy_drop_are_discards():
    assert decide(row(hit=False, e2e_s=100.0), BASELINE, None) == "discard"
    assert decide(row(e2e_s=1844.0), BASELINE, None) == "discard"
    assert decide(row(preemptions=807), BASELINE, None) == "discard"
    assert decide(row(gsm_correct=13), BASELINE, None) == "discard"


def test_later_keep_needs_more_correct_or_a_real_speedup():
    best = {"gsm_correct": 14, "gsm_n": 16, "e2e_s": 420.0}
    assert decide(row(gsm_correct=15), BASELINE, best) == "keep"
    assert decide(row(e2e_s=410.0), BASELINE, best) == "discard"
    assert decide(row(e2e_s=380.0), BASELINE, best) == "keep"


def test_best_keep_reads_only_kept_rows(tmp_path):
    path = tmp_path / "results.tsv"
    path.write_text(
        "status\te2e_s\thit\tcached_min\tgsm_correct\tgsm_n\tpreemptions\tprefix_tokens\tnote\n"
        "discard\t100\t0\t0\t\t\t\t10\tmiss\n"
        "keep\t420\t1\t32\t14\t16\t0\t32\t\n"
        "keep\t500\t1\t32\t15\t16\t0\t64\t\n",
        encoding="utf-8",
    )
    assert best_keep(str(path)) == {"gsm_correct": 15, "gsm_n": 16, "e2e_s": 500.0}


def test_check_accepts_the_starting_prefix():
    check()
