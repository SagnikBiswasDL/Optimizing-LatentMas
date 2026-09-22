"""CPU tests for grouped Judger decoding in exp_aime_localize.

Batching the Judger is how we buy throughput, and its failure mode is silent:
if rows and batch positions drift out of alignment, every number still looks
plausible but belongs to the wrong problem. These tests stub the model out and
check the bookkeeping — alignment, timing attribution, and text filenames.
"""

import importlib.util
import os
import tempfile
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location(
    "exp_aime_localize", os.path.join(ROOT, "scripts", "exp_aime_localize.py")
)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def _stub(monkey_texts):
    """Replace every GPU/model touchpoint with an identity-ish stub."""
    E.judger_tensors = lambda wrapper, questions, ns: (questions, None)
    E.to_dev = lambda c, dev: c
    E.deep_clone = lambda c: c
    E.reset_peak = lambda: None
    E.sync = lambda: None
    E.num_positions = lambda c: 100 + int(c)
    E.kv_mb = lambda c: 1.5 * int(c)
    # gold is the item's own answer, so a correct grading proves alignment
    E.graded = lambda text, gold, task: text == f"answer-{gold}"
    E.pred_short = lambda text: text

    def fake_decode(wrapper, jids, jmask, caches, budget, **kw):
        n = len(jids)
        texts = [monkey_texts(q) for q in jids]
        return texts, [10 * (i + 1) for i in range(n)], [True] * n

    E.decode_batch_maybe_seal = fake_decode


def _wrapper():
    return types.SimpleNamespace(device="cpu")


def _args(out_dir, bs=1):
    return types.SimpleNamespace(
        out_dir=out_dir, task="aime2024", judger_budget=64, temperature=0.0,
        top_p=1.0, top_k=None, seal_on=False, decode_bs=bs,
    )


def test_group_rows_align_with_batch_positions():
    """Row i must carry batch position i's text, tokens, and grade."""
    _stub(lambda q: f"answer-{q}")
    with tempfile.TemporaryDirectory() as d:
        items = [{"idx": i, "question": f"q{i}", "gold": f"q{i}"} for i in (0, 1, 2, 4)]
        rows = E.decode_group(_wrapper(), None, _args(d), items, [1, 2, 3, 4], "real")
        assert len(rows) == 4
        assert [r["idx"] for r in rows] == [0, 1, 2, 4]
        # every item graded correct => each row read its own batch slot
        assert all(r["correct"] for r in rows), [r["pred"] for r in rows]
        assert [r["tokens"] for r in rows] == [10, 20, 30, 40]
        assert [r["cache_pos"] for r in rows] == [101, 102, 103, 104]
        for i, r in zip((0, 1, 2, 4), rows):
            p = os.path.join(d, "texts", f"{i:04d}_real.txt")
            assert os.path.isfile(p)
            assert open(p).read() == f"answer-q{i}"


def test_misalignment_is_caught():
    """Sanity-check the test itself: a shifted batch must fail the grade."""
    _stub(lambda q: "answer-q0")  # every slot returns item 0's answer
    with tempfile.TemporaryDirectory() as d:
        items = [{"idx": i, "question": f"q{i}", "gold": f"q{i}"} for i in (0, 1, 2)]
        rows = E.decode_group(_wrapper(), None, _args(d), items, [1, 2, 3], "real")
        assert [r["correct"] for r in rows] == [True, False, False]


def test_timing_attribution():
    """Grouped rows forfeit judger_s; batch_s/batch_size sums to wall clock."""
    _stub(lambda q: f"answer-{q}")
    with tempfile.TemporaryDirectory() as d:
        items = [{"idx": i, "question": f"q{i}", "gold": f"q{i}"} for i in range(3)]
        rows = E.decode_group(_wrapper(), None, _args(d), items, [1, 2, 3], "real")
        assert all(r["judger_s"] is None for r in rows)
        assert all(r["batch_size"] == 3 for r in rows)
        batch_s = rows[0]["batch_s"]
        assert all(r["batch_s"] == batch_s for r in rows)
        recovered = sum(r["batch_s"] / r["batch_size"] for r in rows)
        assert abs(recovered - batch_s) < 1e-9

        solo = E.decode_group(_wrapper(), None, _args(d), items[:1], [1], "solo")
        assert solo[0]["judger_s"] is not None
        assert solo[0]["batch_size"] == 1


def test_mixed_cache_presence_refused():
    """Padding a None cache against a real one would silently corrupt attention."""
    _stub(lambda q: f"answer-{q}")
    with tempfile.TemporaryDirectory() as d:
        items = [{"idx": i, "question": f"q{i}", "gold": f"q{i}"} for i in range(2)]
        try:
            E.decode_group(_wrapper(), None, _args(d), items, [1, None], "real")
        except ValueError:
            return
        raise AssertionError("expected ValueError on mixed cache/None batch")


def test_unbatched_tps_prefers_solo_rows():
    """Pricing must ignore grouped rows even when they dominate the run."""
    rows = [
        {"method": "real", "tokens": 100, "judger_s": 10.0},          # 10 tok/s
        {"method": "real", "tokens": 900, "judger_s": None, "batch_s": 5.0},
    ]
    assert abs(E._unbatched_tps(rows, "real") - 10.0) < 1e-9
    # arm with no solo rows falls back to speed measured elsewhere
    rows.append({"method": "real_seal40", "tokens": 500, "judger_s": None,
                 "batch_s": 3.0})
    assert abs(E._unbatched_tps(rows, "real_seal40") - 10.0) < 1e-9
    assert E._unbatched_tps([{"method": "x", "tokens": 5, "judger_s": None}], "x") == 0.0


def test_arm_table_excludes_grouped_from_per_item_latency():
    rows = [
        {"method": "real", "idx": 0, "correct": True, "tokens": 100,
         "judger_s": 10.0, "batch_s": 10.0, "batch_size": 1, "eos": True},
        {"method": "grp", "idx": 1, "correct": True, "tokens": 200,
         "judger_s": None, "batch_s": 8.0, "batch_size": 2, "eos": True},
        {"method": "grp", "idx": 2, "correct": False, "tokens": 300,
         "judger_s": None, "batch_s": 8.0, "batch_size": 2, "eos": False},
    ]
    t = E._arm_table(rows)
    assert t["real"]["n_timed"] == 1 and t["real"]["judger_s"] == 10.0
    assert t["grp"]["n_timed"] == 0
    assert t["grp"]["judger_s"] == 0.0        # reported as absent, not fabricated
    assert abs(t["grp"]["wall_s"] - 8.0) < 1e-9
    assert t["grp"]["max_batch"] == 2
    assert abs(t["grp"]["tok_per_s_wall"] - 500.0 / 8.0) < 1e-9
    notes = E._latency_notes(t)
    assert any("| grp |" in ln and "—" in ln for ln in notes)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} passed")
