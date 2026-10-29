"""Frozen eval rules. The loop may not edit this file.

The October measurement on one H200, vLLM, gpu_memory_utilization 0.45:
five long prompts do not fit in the KV pool and the run takes about 30 minutes.
Five prompts that fit finish in about 7 minutes. A shared prefix is stored once,
so it can stay long and still land on the 7-minute side. Time below that cliff
is the constraint. Accuracy on a fixed GSM8K probe is the number to raise.
"""
from __future__ import annotations

import json
import os

CONCURRENCY = 5
DECODE = 32768
PROBE_N = 16
PROBE_MAX_NEW_TOKENS = 512
ACC_SLACK = 0.05
# Short side of the October cliff was ~420s. The long side was ~1800s.
FIT_E2E_S = 600.0
# Inside the fitting region, run-to-run time moves by warmup. Ignore less than this.
TIME_TIE_S = 30.0
BLOCK_SIZE = 16
MODEL = "Qwen/Qwen3-14B"
GPU_MEMORY_UTILIZATION = 0.45
MAX_MODEL_LEN = 40960
TIMEOUT_S = 45 * 60

ROOT = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(ROOT, "baseline.json")
RESULTS_PATH = os.path.join(ROOT, "results.tsv")
RUNS_PATH = os.path.join(ROOT, "runs.jsonl")

HEADER = (
    "status\te2e_s\thit\tcached_min\tgsm_correct\tgsm_n\t"
    "preemptions\tprefix_tokens\tnote\n"
)

PRESSURE_SUFFIXES = tuple(
    f"\nQuestion {i}: what is {i} plus {i}?\n" for i in range(1, CONCURRENCY + 1)
)


def assemble(prefix_ids, suffix_ids):
    """Every prompt starts with the same prefix ids. The suffix is the only difference."""
    prefix = [int(token) for token in prefix_ids]
    if len(prefix) < BLOCK_SIZE:
        raise ValueError("prefix must cover at least one cache block")
    if len(prefix) + DECODE > MAX_MODEL_LEN:
        raise ValueError("prefix plus decode does not fit max_model_len")
    prompts = []
    for suffix in suffix_ids:
        body = prefix + [int(token) for token in suffix]
        if body[: len(prefix)] != prefix:
            raise RuntimeError("prefix was not shared")
        prompts.append(body)
    return prompts


def cache_hit(cached_counts, prefix_len, block_size=BLOCK_SIZE):
    """A hit reuses every full block of the shared prefix on every request."""
    need = (int(prefix_len) // int(block_size)) * int(block_size)
    if need < block_size:
        return False
    if len(cached_counts) < CONCURRENCY:
        return False
    return all(isinstance(count, int) and count >= need for count in cached_counts)


def accuracy(correct, n):
    if not n:
        return 0.0
    return correct / n


def decide(row, baseline, best):
    """Return keep, discard, or crash.

    A keep must hit the prefix cache, finish on the short side of the cliff,
    stay within 5 points of the frozen GSM8K baseline, and beat the best keep
    on correct answers. Equal accuracy is a keep only when it is at least
    TIME_TIE_S faster.
    """
    if row.get("crash"):
        return "crash"
    if not row.get("hit"):
        return "discard"
    if row.get("e2e_s") is None or row["e2e_s"] > FIT_E2E_S:
        return "discard"
    preemptions = row.get("preemptions")
    if isinstance(preemptions, (int, float)) and preemptions > 0:
        return "discard"
    if accuracy(row["gsm_correct"], row["gsm_n"]) < accuracy(
        baseline["gsm_correct"], baseline["gsm_n"]
    ) - ACC_SLACK:
        return "discard"
    if best is None:
        return "keep"
    if row["gsm_correct"] > best["gsm_correct"]:
        return "keep"
    if (
        row["gsm_correct"] == best["gsm_correct"]
        and row["e2e_s"] < best["e2e_s"] - TIME_TIE_S
    ):
        return "keep"
    return "discard"


def load_baseline(path=BASELINE_PATH):
    with open(path, encoding="utf-8") as handle:
        baseline = json.load(handle)
    if baseline.get("gsm_n") != PROBE_N:
        raise ValueError("baseline probe size does not match PROBE_N")
    return baseline


def best_keep(path=RESULTS_PATH):
    if not os.path.exists(path):
        return None
    best = None
    with open(path, encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return None
        columns = header.rstrip("\n").split("\t")
        for line in handle:
            if not line.strip():
                continue
            row = dict(zip(columns, line.rstrip("\n").split("\t")))
            if row.get("status") != "keep":
                continue
            parsed = {
                "gsm_correct": int(row["gsm_correct"]),
                "gsm_n": int(row["gsm_n"]),
                "e2e_s": float(row["e2e_s"]),
            }
            if best is None or parsed["gsm_correct"] > best["gsm_correct"]:
                best = parsed
            elif (
                parsed["gsm_correct"] == best["gsm_correct"]
                and parsed["e2e_s"] < best["e2e_s"]
            ):
                best = parsed
    return best


def append_result(row, path=RESULTS_PATH, runs_path=RUNS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(HEADER)
    note = str(row.get("note", "")).replace("\t", " ").replace("\n", " ")
    line = (
        f"{row['status']}\t{row.get('e2e_s', '')}\t{int(bool(row.get('hit')))}\t"
        f"{row.get('cached_min', '')}\t{row.get('gsm_correct', '')}\t"
        f"{row.get('gsm_n', '')}\t{row.get('preemptions', '')}\t"
        f"{row.get('prefix_tokens', '')}\t{note}\n"
    )
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
    with open(runs_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def format_result(row):
    return (
        "RESULT"
        f" status={row['status']}"
        f" e2e_s={row.get('e2e_s', '')}"
        f" hit={int(bool(row.get('hit')))}"
        f" cached_min={row.get('cached_min', '')}"
        f" gsm_correct={row.get('gsm_correct', '')}"
        f" gsm_n={row.get('gsm_n', '')}"
        f" preemptions={row.get('preemptions', '')}"
        f" prefix_tokens={row.get('prefix_tokens', '')}"
        f" note={row.get('note', '')}"
    )
