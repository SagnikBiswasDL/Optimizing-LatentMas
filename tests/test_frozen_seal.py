"""CPU tests for Frozen + Judger SEAL gates."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seal.frozen_seal import aime_latency_verdict, iso_acc_token_gate  # noqa: E402


def test_iso_acc_token_gate():
    go = iso_acc_token_gate(0.90, 0.90, 600.0, 480.0)
    assert go["recommend"] == "go" and go["acc_ok"] and go["tok_ok"]
    drop = iso_acc_token_gate(0.90, 0.70, 600.0, 400.0)
    assert drop["recommend"] == "stop" and not drop["acc_ok"]
    notok = iso_acc_token_gate(0.90, 0.90, 600.0, 590.0)
    assert notok["recommend"] == "stop" and notok["acc_ok"] and not notok["tok_ok"]


def test_aime_latency_verdict():
    frozen = [True, True, False, False, True, False]
    seal = [True, True, False, False, True, False]
    toks_f = [2000, 3000, 8192, 8192, 4000, 5000]
    toks_s = [1500, 2200, 6000, 7000, 3000, 4000]
    v = aime_latency_verdict(frozen, seal, toks_f, toks_s, stage="aime6")
    assert v["recommend"] == "go" and v["n_lost"] == 0
    hurt = [False, True, False, False, True, False]
    stop = aime_latency_verdict(frozen, hurt, toks_f, toks_s, stage="aime6")
    assert stop["recommend"] == "stop" and stop["n_lost"] == 1
    nocut = aime_latency_verdict(frozen, seal, toks_f, toks_f, stage="aime6")
    assert nocut["recommend"] == "stop"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
