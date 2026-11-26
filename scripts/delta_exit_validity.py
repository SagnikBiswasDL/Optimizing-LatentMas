#!/usr/bin/env python3
"""Validity suite for Jiayi's decode schedules.

Losslessness. 20 MATH-train questions (4 per level, seed 0), 192 new tokens.
The full-depth wavefront and the convergence diagonal at epsilon 0 must emit
the same greedy token ids as ordinary depth-4 decoding.

MATH merit. 100 MATH-train questions (20 per level, seed 0), cap 1024.
Fixed depths 1, 2, 3, and 4; Ouro's gate; token-major KV-delta at
0.05, 0.1, and 0.2; the same three cutoffs on the convergence diagonal;
second-order KV acceleration; matched-depth random halt; and KV-delta
without copy-forward.

GSM8K. 100 test questions, seed 0, cap 512. Fixed depths 3 and 4, the gate,
KV-delta at 0.1, and the convergence diagonal at 0.1.

An answer with no boxed value counts as wrong. Each row is written before
the next one starts, and a restarted process skips rows already on disk.
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_gsm8k  # noqa: E402
from looped import prompts as P  # noqa: E402
from looped.delta_exit import generate_exit  # noqa: E402
from looped.loopdelta import grade_math, load_math_train_stratified  # noqa: E402
from looped.ouro_adapter import OURO_BASE_ID, OURO_BASE_REVISION, OuroRunner  # noqa: E402

OUT = os.environ.get("OUT", "/workspace/delta_exit_validity")
LOSSLESS_PER_LEVEL = 4
LOSSLESS_TOKENS = 192
MATH_PER_LEVEL = 20
MATH_TOKENS = 1024
GSM_N = 100
GSM_TOKENS = 512


def _load_rows(path):
    rows, done = [], set()
    if not os.path.exists(path):
        return rows, done
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
            done.add((row["idx"], row["mode"]))
    return rows, done


def _write_rows(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _math_modes():
    modes = [(f"fixed_{depth}", {"mode": "fixed", "depth": depth}) for depth in (1, 2, 3, 4)]
    modes.append(("gate_0.5", {"mode": "gate", "threshold": 0.5, "depth": 4}))
    for eps in (0.05, 0.1, 0.2):
        tag = f"{eps:g}"
        modes.append((f"kv_delta_{tag}", {"mode": "kv_delta", "epsilon": eps, "depth": 4}))
        modes.append((f"diagonal_{tag}", {"mode": "diagonal", "epsilon": eps, "depth": 4}))
    modes.extend([
        ("accel_0.1", {"mode": "kv_delta", "epsilon": 0.1, "depth": 4, "rule": "accel"}),
        ("random", {"mode": "kv_delta", "epsilon": 0.1, "depth": 4, "rule": "random"}),
        ("recompute_0.1", {"mode": "kv_delta", "epsilon": 0.1, "depth": 4, "reuse": False}),
    ])
    return modes


def _gsm_items():
    rows = list(load_gsm8k("test"))
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    out = []
    for idx in order[:GSM_N]:
        row = rows[idx]
        out.append({"idx": idx, "level": 0, "question": row["question"], "gold": row["gold"]})
    return out


def _graded_row(item, name, out):
    graded = grade_math(out["text"], item["gold"])
    return {
        "idx": item["idx"],
        "level": item["level"],
        "mode": name,
        "correct": graded["correct"],
        "pred": graded["pred"],
        "gold": item["gold"],
        "hit_eos": out["hit_eos"],
        "new_tokens": len(out["token_ids"]),
        "mean_chosen_loop": out["mean_chosen_loop"],
        "mean_executed_loops": out["mean_executed_loops"],
        "rejected_loops": out.get("rejected_loops", 0),
        "wall_s": out["wall_s"],
        "text": out["text"],
    }


def _run_block(runner, items, modes, max_new_tokens, path, label, graded):
    rows, done = _load_rows(path)
    for item in items:
        prompt = runner.chat_ids(P.msgs(P.solver(item["question"])), think=False)
        for name, kw in modes:
            if (item["idx"], name) in done:
                continue
            print(f"{label} q{item['idx']} {name}", flush=True)
            out = generate_exit(runner, prompt, max_new_tokens=max_new_tokens, **kw)
            if graded:
                row = _graded_row(item, name, out)
                print(
                    f"  correct={row['correct']} tokens={row['new_tokens']} "
                    f"loops={row['mean_executed_loops']} {row['wall_s']:.0f}s",
                    flush=True,
                )
            else:
                row = {
                    "idx": item["idx"],
                    "level": item["level"],
                    "mode": name,
                    "token_ids": out["token_ids"],
                    "hit_eos": out["hit_eos"],
                    "wall_s": out["wall_s"],
                    "rejected_loops": out.get("rejected_loops", 0),
                }
                print(f"  tokens={len(out['token_ids'])} {out['wall_s']:.1f}s", flush=True)
            rows.append(row)
            done.add((item["idx"], name))
            _write_rows(path, rows)
    return rows


def _match_rate(rows, left, right, wanted):
    by = {}
    for row in rows:
        if row["idx"] not in wanted:
            continue
        by.setdefault(row["idx"], {})[row["mode"]] = row["token_ids"]
    compared = matched = 0
    first = None
    for idx, modes in sorted(by.items()):
        if left not in modes or right not in modes:
            continue
        compared += 1
        a, b = modes[left], modes[right]
        if a == b:
            matched += 1
        elif first is None:
            at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            first = {"idx": idx, "at": at, "len_left": len(a), "len_right": len(b)}
    return {"compared": compared, "matched": matched, "first_mismatch": first}


def _merit_line(rows, mode):
    got = [r for r in rows if r["mode"] == mode]
    n = len(got)
    if not n:
        return f"{mode}: no rows"
    correct = sum(1 for r in got if r["correct"])
    finished = sum(1 for r in got if r["hit_eos"])
    loops = [r["mean_executed_loops"] for r in got if r["mean_executed_loops"] is not None]
    mean_loops = sum(loops) / len(loops) if loops else float("nan")
    wall = sum(r["wall_s"] for r in got) / n
    return (
        f"{mode}: {correct}/{n} correct, {finished}/{n} finished, "
        f"mean loops {mean_loops:.2f}, mean wall {wall:.1f}s"
    )


def _summarize(loss_rows, loss_ids, math_rows, gsm_rows):
    wfd = _match_rate(loss_rows, "fixed_4", "wfd", loss_ids)
    diag0 = _match_rate(loss_rows, "fixed_4", "diagonal_eps0", loss_ids)
    lines = [
        "Losslessness (greedy token ids vs depth 4, 192 tokens)",
        f"  wavefront: {wfd['matched']}/{wfd['compared']} identical"
        + (f", first mismatch {wfd['first_mismatch']}" if wfd["first_mismatch"] else ""),
        f"  diagonal epsilon 0: {diag0['matched']}/{diag0['compared']} identical"
        + (f", first mismatch {diag0['first_mismatch']}" if diag0["first_mismatch"] else ""),
        "",
        f"MATH (cap {MATH_TOKENS})",
    ]
    for name, _ in _math_modes():
        lines.append("  " + _merit_line(math_rows, name))
    lines.append("")
    lines.append(f"GSM8K (cap {GSM_TOKENS})")
    for name in ("fixed_4", "fixed_3", "gate_0.5", "kv_delta_0.1", "diagonal_0.1"):
        lines.append("  " + _merit_line(gsm_rows, name))
    text = "\n".join(lines) + "\n"
    return {"wfd": wfd, "diagonal_eps0": diag0, "text": text}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    math_items = load_math_train_stratified(MATH_PER_LEVEL, 0)
    loss_items = []
    for level in range(1, 6):
        loss_items.extend([it for it in math_items if it["level"] == level][:LOSSLESS_PER_LEVEL])
    gsm_items = _gsm_items()
    with open(os.path.join(OUT, "protocol.json"), "w") as f:
        json.dump({
            "model": OURO_BASE_ID,
            "revision": OURO_BASE_REVISION,
            "lossless_questions": [it["idx"] for it in loss_items],
            "lossless_tokens": LOSSLESS_TOKENS,
            "math_questions": [it["idx"] for it in math_items],
            "math_tokens": MATH_TOKENS,
            "gsm_questions": [it["idx"] for it in gsm_items],
            "gsm_tokens": GSM_TOKENS,
            "math_modes": [name for name, _ in _math_modes()],
            "seed": 0,
        }, f, indent=2)
    print(
        f"math {len(math_items)} lossless {len(loss_items)} gsm {len(gsm_items)} "
        f"math_modes {len(_math_modes())}",
        flush=True,
    )
    runner = OuroRunner(
        model_id=OURO_BASE_ID, revision=OURO_BASE_REVISION, think=False, attn_implementation="sdpa",
    )
    loss_modes = [
        ("fixed_4", {"mode": "fixed", "depth": 4}),
        ("wfd", {"mode": "wfd", "depth": 4}),
        ("diagonal_eps0", {"mode": "diagonal", "epsilon": 0.0, "depth": 4}),
    ]
    gsm_modes = [
        ("fixed_4", {"mode": "fixed", "depth": 4}),
        ("fixed_3", {"mode": "fixed", "depth": 3}),
        ("gate_0.5", {"mode": "gate", "threshold": 0.5, "depth": 4}),
        ("kv_delta_0.1", {"mode": "kv_delta", "epsilon": 0.1, "depth": 4}),
        ("diagonal_0.1", {"mode": "diagonal", "epsilon": 0.1, "depth": 4}),
    ]
    loss_rows = _run_block(
        runner, loss_items, loss_modes, LOSSLESS_TOKENS,
        os.path.join(OUT, "lossless.jsonl"), "lossless", False,
    )
    math_rows = _run_block(
        runner, math_items, _math_modes(), MATH_TOKENS,
        os.path.join(OUT, "math.jsonl"), "math", True,
    )
    gsm_rows = _run_block(
        runner, gsm_items, gsm_modes, GSM_TOKENS,
        os.path.join(OUT, "gsm8k.jsonl"), "gsm8k", True,
    )
    summary = _summarize(loss_rows, {it["idx"] for it in loss_items}, math_rows, gsm_rows)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(summary["text"])
    with open(os.path.join(OUT, "DONE"), "w") as f:
        f.write("ok\n")
    print(summary["text"], flush=True)


if __name__ == "__main__":
    main()
