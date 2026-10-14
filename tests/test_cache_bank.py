"""CPU tests for seal.cache_bank (no CUDA, no eviction)."""
import os
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seal.cache_bank import (  # noqa: E402
    CacheBank,
    clone_past,
    num_positions,
    pool_stats,
    problem_type,
    sample_synth,
    to_legacy,
)


def _fake_legacy(layers=3, heads=4, seq=32, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(layers):
        k = torch.randn(1, heads, seq, dim, generator=g)
        v = torch.randn(1, heads, seq, dim, generator=g)
        out.append((k, v))
    return out


def test_problem_type_task_and_subtype():
    item = {"question": "q", "gold": "1", "subject": "Number Theory"}
    assert problem_type(item, "math", "task") == "math"
    assert problem_type(item, "math", "subtype") == "math:number_theory"
    assert problem_type({"question": "q"}, "gsm8k", "subtype") == "gsm8k"


def test_save_load_roundtrip():
    bank = CacheBank(meta={"k": 40, "model_name": "toy"})
    leg = _fake_legacy(seed=1)
    bank.add("gsm8k", legacy=leg, donors=[{"question": "donor q", "idx": 0}],
             stats=pool_stats([leg])[0], synth_len=32)
    bank.add("medqa", legacy=_fake_legacy(seed=2), donors=[{"question": "med"}])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "bank.pt")
        bank.save(path)
        loaded = CacheBank.load(path)
    assert loaded.keys() == ["gsm8k", "medqa"]
    assert loaded.meta["k"] == 40
    assert loaded.n_pos("gsm8k") == 32
    assert loaded.resolve("gsm8k:algebra") == "gsm8k"  # parent fallback
    past = loaded.donor_cache("gsm8k", device="cpu")
    assert num_positions(past) == 32
    orig = to_legacy(past)
    assert torch.allclose(orig[0][0], leg[0][0])


def test_synth_shape_and_stats_exist():
    leg = _fake_legacy(seq=40, seed=7)
    stats, L = pool_stats([leg, _fake_legacy(seq=40, seed=8)])
    assert L == 40
    synth = sample_synth(stats, L, dtype=torch.float32, device="cpu", seed=0)
    assert num_positions(synth) == 40
    bank = CacheBank()
    bank.add("math", legacy=leg, donors=[{}], stats=stats, synth_len=L)
    s2 = bank.synth_cache("math", device="cpu", dtype=torch.float32, seed=1)
    assert num_positions(s2) == 40
    c1 = clone_past(s2)
    assert num_positions(c1) == 40
    assert not (to_legacy(c1)[0][0].data_ptr() == to_legacy(s2)[0][0].data_ptr())


def test_wrong_type_is_a_different_object():
    bank = CacheBank()
    a = _fake_legacy(seed=1)
    b = _fake_legacy(seed=2)
    bank.add("gsm8k", legacy=a, donors=[{}])
    bank.add("medqa", legacy=b, donors=[{}])
    ga = to_legacy(bank.donor_cache("gsm8k", "cpu"))
    gb = to_legacy(bank.donor_cache("medqa", "cpu"))
    assert not torch.allclose(ga[0][0], gb[0][0])
