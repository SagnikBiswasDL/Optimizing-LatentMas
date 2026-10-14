"""CPU tests for role-span KV slice / concat / eviction."""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seal.cache_bank import from_legacy, num_positions, to_legacy  # noqa: E402
from seal.segment_kv import (  # noqa: E402
    concat_seq,
    evict_by_segment,
    slice_seq,
    take_latents,
    take_roles,
)


def _fake(seq=90, seed=0):
    g = torch.Generator().manual_seed(seed)
    layers = []
    for _ in range(2):
        k = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1).repeat(1, 2, 1, 4)
        v = torch.randn(1, 2, seq, 4, generator=g)
        layers.append((k, v))
    return from_legacy(layers)


SPANS = [
    {"role": "planner", "start": 0, "end": 30},
    {"role": "critic", "start": 30, "end": 60},
    {"role": "refiner", "start": 60, "end": 90},
]


def test_slice_and_concat_roundtrip():
    past = _fake()
    parts = [slice_seq(past, s["start"], s["end"]) for s in SPANS]
    assert [num_positions(p) for p in parts] == [30, 30, 30]
    back = concat_seq(parts)
    assert num_positions(back) == 90
    a = to_legacy(past)[0][0]
    b = to_legacy(back)[0][0]
    assert torch.allclose(a, b)


def test_take_roles_order():
    past = _fake()
    c23 = take_roles(past, SPANS, ["refiner", "critic"])
    assert num_positions(c23) == 60
    # critic then refiner, not the other way around
    k = to_legacy(c23)[0][0]
    assert float(k[0, 0, 0, 0]) == 30.0
    assert float(k[0, 0, 30, 0]) == 60.0
    c1 = take_roles(past, SPANS, ["planner"])
    assert float(to_legacy(c1)[0][0][0, 0, 0, 0]) == 0.0


def test_latents_are_tails():
    past = _fake()
    lat = take_latents(past, SPANS, 5)
    assert num_positions(lat) == 15
    k = to_legacy(lat)[0][0]
    assert float(k[0, 0, 0, 0]) == 25.0
    assert float(k[0, 0, 5, 0]) == 55.0
    assert float(k[0, 0, 10, 0]) == 85.0


def test_seg_evict_equal_budget():
    past = _fake()
    out, st = evict_by_segment(past, SPANS, budget_per_role=10, sink=2)
    assert num_positions(out) == 30
    assert st["mode"] == "seg_evict"
    assert [p["out"] for p in st["parts"]] == [10, 10, 10]


def test_drop_planner_via_roles():
    past = _fake()
    out, st = evict_by_segment(
        past, SPANS, budget_per_role=10, sink=2, roles=("critic", "refiner"))
    assert num_positions(out) == 20
    assert [p["role"] for p in st["parts"]] == ["critic", "refiner"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
