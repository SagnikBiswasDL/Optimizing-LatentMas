"""CPU tests for the math ladder (no CUDA, no model download)."""
from __future__ import annotations

import os
import sys
import tempfile

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seal.cache_bank import from_legacy, mean_restore_latents, num_positions  # noqa: E402
from seal.latent_eval import split_past, stack_past  # noqa: E402
from seal.math_cache import (  # noqa: E402
    UNIFIED_MATH_PROMPT,
    append_donor,
    arm_name,
    cache_as_past,
    empty_donor_state,
    expand_ladder_arms,
    gate_verdict,
    load_donor_state_lists,
    load_unified_cache,
    mean_tapes,
    n_kept,
    parse_arm,
    parse_int_list,
    save_donor_state,
    save_unified_cache,
)
from seal.relay_compress import RelayCompressor  # noqa: E402


def _fake_legacy(layers=3, heads=2, seq=40, dim=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(layers):
        k = torch.randn(1, heads, seq, dim, generator=g)
        v = torch.randn(1, heads, seq, dim, generator=g)
        out.append((k, v))
    return from_legacy(out)


def test_parse_int_list_and_arms():
    assert parse_int_list("0,2,5") == [0, 2, 5]
    assert parse_arm("none")["kind"] == "none"
    assert parse_arm("real")["kind"] == "real"
    f = parse_arm("frozen")
    assert f["kind"] == "ours" and f["k_ttc"] == 0 and f["evict"] == 0
    k2 = parse_arm("frozen_k2")
    assert k2["k_ttc"] == 2 and k2["evict"] == 0
    live = parse_arm("k5")
    assert live["kind"] == "live" and live["k_ttc"] == 5
    e = parse_arm("frozen_k5_evict64")
    assert e["k_ttc"] == 5 and e["evict"] == 64
    assert arm_name(0, 0) == "frozen"
    assert arm_name(2, 64) == "frozen_k2_evict64"
    names = expand_ladder_arms([0, 2], [0, 64], None)
    assert names == [
        "none", "real", "frozen", "frozen_evict64", "frozen_k2", "k2",
        "frozen_k2_evict64",
    ]
    live_only = expand_ladder_arms([2, 5], [0], ["frozen_k2", "frozen_k5", "k2", "k5"])
    assert live_only == ["frozen_k2", "k2", "frozen_k5", "k5"]
    gate = expand_ladder_arms([0], [0], ["none", "frozen", "real"])
    assert gate == ["none", "real", "frozen"]


def test_mean_restore_large_n():
    g = torch.Generator().manual_seed(0)
    steps = [torch.randn(10, 32, generator=g) for _ in range(128)]
    mu = mean_restore_latents(steps)
    assert tuple(mu.shape) == (10, 32)
    # L2 restored to median donor L2 per step
    stacked = torch.stack(steps, 0)
    med = stacked.norm(dim=-1).median(dim=0).values
    got = mu.norm(dim=-1)
    assert torch.allclose(got, med, rtol=1e-4, atol=1e-4)


def test_donor_checkpoint_roundtrip():
    state = empty_donor_state(k=10, model_name="toy")
    for i in range(4):
        lat = {r: torch.randn(10, 8) for r in ("planner", "critic", "refiner")}
        append_donor(state, latents=lat, row={"idx": i}, next_idx=i + 1)
    assert n_kept(state) == 4
    tapes = mean_tapes(state)
    assert tuple(tapes["planner"].shape) == (10, 8)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "donors.pt")
        save_donor_state(state, path)
        loaded = load_donor_state_lists(path)
    assert n_kept(loaded) == 4
    assert loaded["next_idx"] == 4
    assert torch.allclose(mean_tapes(loaded)["planner"], tapes["planner"], atol=1e-5)


def test_unified_cache_roundtrip_and_evict():
    past = _fake_legacy(seq=80, seed=3)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cache.pt")
        save_unified_cache(
            path, past=past,
            meta={"n_donors": 1000, "scaffold": "replay", "k": 10},
            mean_latents={"planner": torch.randn(10, 8)},
        )
        loaded = load_unified_cache(path)
    assert loaded["meta"]["n_donors"] == 1000
    assert UNIFIED_MATH_PROMPT.startswith("Solve a mathematics problem")
    back = cache_as_past(loaded, device="cpu")
    assert num_positions(back) == 80
    out, st = RelayCompressor(mode="evict", budget=16, sink=4).compress(back)
    assert num_positions(out) == 16
    assert st.sink_retained is True
    assert st.mb_out < st.mb_in


def test_stack_split_past():
    a = _fake_legacy(seq=20, seed=1)
    b = _fake_legacy(seq=20, seed=2)
    stacked = stack_past([a, b])
    parts = split_past(stacked, 2)
    assert num_positions(parts[0]) == 20
    from seal.cache_bank import to_legacy
    assert torch.allclose(to_legacy(parts[0])[0][0], to_legacy(a)[0][0])
    assert torch.allclose(to_legacy(parts[1])[0][0], to_legacy(b)[0][0])


def test_gate_verdict():
    skip = gate_verdict({
        "arms": {
            "frozen": {"acc": 0.90},
            "real": {"acc": 0.90},
            "none": {"acc": 0.80},
        },
        "paired": {"frozen_minus_real_correct": {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}},
    })
    assert skip["recommend"] == "skip_ttc"
    ttc = gate_verdict({
        "arms": {
            "frozen": {"acc": 0.70},
            "real": {"acc": 0.90},
            "none": {"acc": 0.60},
        },
        "paired": {"frozen_minus_real_correct": {"mean": -0.2, "ci_lo": -0.3, "ci_hi": -0.1}},
    })
    assert ttc["recommend"] == "run_small_k"
    bug = gate_verdict({
        "arms": {
            "frozen": {"acc": 0.20},
            "real": {"acc": 0.90},
            "none": {"acc": 0.80},
        },
        "paired": {"frozen_minus_real_correct": {"mean": -0.7, "ci_lo": -0.8, "ci_hi": -0.5}},
    })
    assert bug["recommend"] == "bug"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
