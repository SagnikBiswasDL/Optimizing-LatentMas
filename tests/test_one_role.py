"""CPU tests for the one-role ladder (no CUDA, no model download)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from seal.one_role import (  # noqa: E402
    FORWARDS_ONE_ROLE,
    FORWARDS_REAL,
    aime_subset_verdict,
    arm_name,
    expand_one_role_arms,
    math_role_ok,
    paired_flags,
    parse_indices,
    parse_one_role_arm,
    pick_best_role,
    render_checkin,
    select_items,
    smoke_match,
    upstream_forwards,
)


def test_parse_arms():
    f = parse_one_role_arm("frozen")
    assert f["kind"] == "frozen" and f["k"] == 0 and f["role"] is None
    p = parse_one_role_arm("frozen_planner_k10")
    assert p["kind"] == "one_role" and p["role"] == "planner" and p["k"] == 10
    assert parse_one_role_arm("frozen_refiner_k10")["role"] == "refiner"
    assert parse_one_role_arm("frozen_critic_k10")["role"] == "critic"
    r = parse_one_role_arm("real")
    assert r["kind"] == "real" and r["k"] == 10
    assert arm_name("planner") == "frozen_planner_k10"
    assert arm_name(None) == "frozen"
    names = expand_one_role_arms(None)
    assert names[0] == "frozen"
    assert "frozen_planner_k10" in names
    assert "frozen_k2" not in names
    sub = expand_one_role_arms(["frozen", "frozen_planner_k10"])
    assert sub == ["frozen", "frozen_planner_k10"]
    assert upstream_forwards(p) == FORWARDS_ONE_ROLE == 11
    assert upstream_forwards(r) == FORWARDS_REAL == 33
    assert upstream_forwards(f) == 0


def test_indices_and_select():
    assert parse_indices("0,1,2,3,4,5") == list(range(6))
    assert parse_indices("0-5") == list(range(6))
    assert parse_indices("0-11") == list(range(12))
    items = [{"question": f"q{i}", "gold": str(i)} for i in range(30)]
    six = select_items(items, 6, list(range(6)))
    assert [x["idx"] for x in six] == [0, 1, 2, 3, 4, 5]
    twelve = select_items(items, 12, list(range(12)))
    assert [x["idx"] for x in twelve[:6]] == [x["idx"] for x in six]


def test_math_gate_and_pick():
    assert math_role_ok(0.75, 0.70)
    assert math_role_ok(0.75, 0.75)
    assert not math_role_ok(0.75, 0.40)
    assert not math_role_ok(0.75, 0.55, slack=0.15)  # 0.55 < 0.60
    arms = {
        "frozen": {"acc": 0.75, "tokens": {"mean": 900}},
        "frozen_planner_k10": {"acc": 0.70, "tokens": {"mean": 800}},
        "frozen_refiner_k10": {"acc": 0.80, "tokens": {"mean": 850}},
        "frozen_critic_k10": {"acc": 0.70, "tokens": {"mean": 700}},
    }
    pick = pick_best_role(arms, frozen_acc=0.75)
    assert pick["ok"] and pick["role"] == "refiner"
    dead = {
        "frozen": {"acc": 0.80},
        "frozen_planner_k10": {"acc": 0.40},
        "frozen_refiner_k10": {"acc": 0.35},
        "frozen_critic_k10": {"acc": 0.30},
    }
    assert pick_best_role(dead, frozen_acc=0.80)["ok"] is False


def test_aime_gates():
    frozen = [True, True, True, False, False, False]
    win = [True, True, True, True, False, False]  # recovered 1, lost 0
    v = aime_subset_verdict(frozen, win, stage="aime6")
    assert v["recommend"] == "go" and v["n_recovered"] == 1 and v["n_lost"] == 0
    flat = list(frozen)
    stop = aime_subset_verdict(frozen, flat, stage="aime6")
    assert stop["recommend"] == "stop"
    regress = [False, True, True, False, False, False]  # lost 1, recovered 0
    assert aime_subset_verdict(frozen, regress, stage="aime6")["recommend"] == "stop"
    persist = list(win) + [True, False, False, False, False, False]
    frozen12 = frozen + [False] * 6
    v12 = aime_subset_verdict(frozen12, persist, stage="aime12")
    assert v12["recommend"] == "go"
    flags = paired_flags(frozen, win)
    assert flags["net"] == 1
    frozen30 = [True] * 17 + [False] * 13
    live30 = [True] * 19 + [False] * 11
    v30 = aime_subset_verdict(frozen30, live30, stage="aime30")
    assert v30["recommend"] == "go"


def test_smoke_match():
    a = [{"idx": 0, "correct": True, "pred": "12"}, {"idx": 1, "correct": False, "pred": "3"}]
    b = [{"idx": 0, "correct": True, "pred": "12"}, {"idx": 1, "correct": False, "pred": "3"}]
    ok, _ = smoke_match(a, b)
    assert ok
    b2 = [{"idx": 0, "correct": True, "pred": "12"}, {"idx": 1, "correct": False, "pred": "4"}]
    ok2, _ = smoke_match(a, b2)
    assert not ok2
    md = render_checkin({"recommend": "go", "reason": "ok", "next": "AIME n=6"})
    assert "recommend" in md and "AIME n=6" in md


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
