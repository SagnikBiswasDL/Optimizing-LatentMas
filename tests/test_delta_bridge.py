"""CPU tests for DeltaBridge (no CUDA, no model download)."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seal.delta_bridge import (  # noqa: E402
    CoefPredictor,
    ResidualInjector,
    coefficients,
    derange,
    fit_pca,
    oracle_verdict,
    pick_layer_rank,
    project_residual,
    reconstruct_from_alpha,
    rms_norm,
)


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return (self.lin(x),)


class _Host(nn.Module):
    def __init__(self, d=16, n=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block(d) for _ in range(n)])


def test_pca_roundtrip_and_rank():
    g = torch.Generator().manual_seed(0)
    u_true = torch.nn.functional.normalize(torch.randn(32, 4, generator=g), dim=0)
    alpha = torch.randn(80, 4, generator=g)
    x = alpha @ u_true.T + 0.01 * torch.randn(80, 32, generator=g)
    fit = fit_pca(x, rank=4)
    assert fit["U"].shape == (32, 4)
    rec = torch.stack([project_residual(x[i], fit) for i in range(x.shape[0])])
    # rank-4 should explain most of a rank-4 signal
    assert fit["explained"] > 0.9
    err = (rec - x).norm() / x.norm()
    assert err < 0.2
    a = coefficients(x[3], fit)
    rec2 = reconstruct_from_alpha(a, fit)
    assert torch.allclose(rec2, rec[3], atol=1e-4)


def test_derange_no_fixed_points():
    g = torch.Generator().manual_seed(1)
    for n in (2, 3, 7, 30):
        perm = derange(n, g)
        assert sorted(perm) == list(range(n))
        assert all(perm[i] != i for i in range(n))


def test_oracle_verdict_go_and_shuffled():
    zero = [True] * 17 + [False] * 13
    # recover 3 of the misses, lose none
    oracle = list(zero)
    oracle[17] = True
    oracle[18] = True
    oracle[19] = True
    shuf = list(zero)
    v = oracle_verdict(zero_correct=zero, oracle_correct=oracle, shuffled_correct=shuf)
    assert v["n_recovered"] == 3
    assert v["recommend"] == "go"

    shuf2 = list(oracle)
    v2 = oracle_verdict(zero_correct=zero, oracle_correct=oracle, shuffled_correct=shuf2)
    assert v2["recommend"] == "stop_not_instance_specific"

    v3 = oracle_verdict(zero_correct=zero, oracle_correct=zero)
    assert v3["recommend"] == "retry_4token"


def test_injector_last_token_once():
    d = 8
    host = _Host(d=d, n=3)
    vec = torch.ones(d)
    inj = ResidualInjector({1: vec})
    inj.register(host)
    x = torch.randn(2, 5, d)
    inj.disable()
    y0 = host.model.layers[1](x)[0]
    inj.enable()
    y1 = host.model.layers[1](x)[0]
    assert torch.allclose(y1[:, :-1, :], y0[:, :-1, :], atol=1e-5)
    assert torch.allclose(y1[:, -1, :], y0[:, -1, :] + vec, atol=1e-5)
    # once: second forward is a no-op
    y2 = host.model.layers[1](x)[0]
    assert torch.allclose(y2, y0, atol=1e-5)
    inj.remove()


def test_coef_predictor_shapes_and_grad():
    m = CoefPredictor(d=16, n_layers=1, rank=4, hidden=32)
    q = torch.randn(5, 1, 16)
    a = m(q)
    assert tuple(a.shape) == (5, 1, 4)
    loss = a.sum()
    loss.backward()
    grads = [p.grad is not None for p in m.parameters()]
    assert any(grads)


def test_pick_prefers_layer_28_when_close():
    def fake(energy, explained):
        return {"explained": explained, "energy": energy, "n": 10, "d": 8}
    banks = {
        16: {16: fake(1.0, 0.5)},
        28: {16: fake(0.9, 0.5)},
        32: {16: fake(0.5, 0.5)},
    }
    pick = pick_layer_rank(banks, prefer_layer=28, target_rank=16)
    assert pick["layer"] == 28
    assert pick["rank"] == 16


def test_rms_norm_unit_energy():
    x = torch.randn(4, 10)
    y = rms_norm(x)
    # RMS along last dim ~ 1
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
