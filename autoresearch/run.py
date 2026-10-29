#!/usr/bin/env python3
"""One autoresearch trial. Frozen. Edit scaffold.py, then run this.

  python autoresearch/run.py --check      # no GPU, checks the prefix is shared
  python autoresearch/run.py --baseline   # once, on the H200: GSM8K bar
  python autoresearch/run.py              # one trial of the current prefix
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autoresearch.prepare import (  # noqa: E402
    BASELINE_PATH,
    CONCURRENCY,
    DECODE,
    FIT_E2E_S,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MODEL,
    PRESSURE_SUFFIXES,
    PROBE_MAX_NEW_TOKENS,
    PROBE_N,
    TIMEOUT_S,
    append_result,
    assemble,
    best_keep,
    cache_hit,
    decide,
    format_result,
    load_baseline,
)
from autoresearch.scaffold import build_prefix  # noqa: E402


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [byte + 1 for byte in text.encode("utf-8")]


def grade(text, gold):
    from utils import extract_gsm8k_answer, normalize_answer

    return normalize_answer(extract_gsm8k_answer(text)) == normalize_answer(gold)


def check():
    tokenizer = DummyTokenizer()
    first = build_prefix(tokenizer)
    second = build_prefix(tokenizer)
    if first != second or len(first) < 16:
        raise SystemExit("CHECK failed: prefix is empty or not deterministic")
    suffixes = [tokenizer.encode(text) for text in PRESSURE_SUFFIXES]
    prompts = assemble(first, suffixes)
    if len(prompts) != CONCURRENCY:
        raise SystemExit("CHECK failed: expected one prompt per concurrent request")
    if any(prompt[: len(first)] != first for prompt in prompts):
        raise SystemExit("CHECK failed: prefix was not shared")
    if len({tuple(prompt[len(first):]) for prompt in prompts}) != CONCURRENCY:
        raise SystemExit("CHECK failed: suffixes were not unique")
    print(f"CHECK ok prefix_tokens={len(first)} concurrency={CONCURRENCY}")


def load_probe_questions():
    from data import load_gsm8k

    rows = []
    for item in load_gsm8k("test"):
        rows.append(item)
        if len(rows) == PROBE_N:
            break
    if len(rows) != PROBE_N:
        raise RuntimeError(f"needed {PROBE_N} GSM8K items, got {len(rows)}")
    return rows


def load_engine():
    from vllm import LLM

    started = time.perf_counter()
    engine = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=True,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",
        disable_log_stats=False,
    )
    enabled = engine.llm_engine.vllm_config.cache_config.enable_prefix_caching
    if enabled is not True:
        raise SystemExit("prefix caching did not stay enabled")
    return engine, time.perf_counter() - started


def cached_count(output):
    value = getattr(output, "num_cached_tokens", None)
    if value is None:
        return None
    return int(value)


def preemption_delta(before, after):
    def read(metrics):
        for key in ("vllm:num_preemptions_total", "vllm:num_preemptions"):
            if key in metrics and isinstance(metrics[key], (int, float)):
                return metrics[key]
        return None

    start, end = read(before), read(after)
    if start is None or end is None:
        return None
    return end - start


def metric_snapshot():
    from prometheus_client import REGISTRY

    flat = {}
    for metric in REGISTRY.collect():
        if not metric.name.startswith("vllm:"):
            continue
        for sample in metric.samples:
            if sample.name.endswith("_created"):
                continue
            flat[sample.name] = flat.get(sample.name, 0.0) + float(sample.value)
    return flat


def generate(engine, prompts, params):
    from vllm import SamplingParams

    del SamplingParams
    before = metric_snapshot()
    started = time.perf_counter()
    outputs = engine.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        params,
    )
    elapsed = time.perf_counter() - started
    return outputs, elapsed, preemption_delta(before, metric_snapshot())


def run_baseline(engine, tokenizer, questions):
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=PROBE_MAX_NEW_TOKENS)
    prompts = [tokenizer.encode(item["question"], add_special_tokens=False) for item in questions]
    outputs, elapsed, _ = generate(engine, prompts, params)
    correct = 0
    for output, item in zip(outputs, questions):
        correct += int(grade(output.outputs[0].text, item["gold"]))
    payload = {
        "gsm_correct": correct,
        "gsm_n": len(questions),
        "e2e_s": elapsed,
        "model": MODEL,
        "probe": "gsm8k-test-first-16",
        "max_new_tokens": PROBE_MAX_NEW_TOKENS,
        "note": "standalone questions, the accuracy bar for every later keep",
    }
    with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(
        f"BASELINE gsm_correct={correct} gsm_n={len(questions)} e2e_s={elapsed:.1f}"
    )


def run_trial(engine, tokenizer, questions):
    from vllm import SamplingParams

    prefix = build_prefix(tokenizer)
    pressure = assemble(prefix, [tokenizer.encode(text) for text in PRESSURE_SUFFIXES])
    probe = assemble(
        prefix,
        [tokenizer.encode("\nQuestion:\n" + item["question"] + "\n") for item in questions],
    )
    warm = SamplingParams(temperature=0.0, max_tokens=1)
    generate(engine, [prefix], warm)
    pressure_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=DECODE,
        min_tokens=DECODE,
        ignore_eos=True,
    )
    outputs, elapsed, preemptions = generate(engine, pressure, pressure_params)
    cached = [cached_count(output) for output in outputs]
    hit = cache_hit(cached, len(prefix))
    correct = ""
    if hit and elapsed <= FIT_E2E_S and not (isinstance(preemptions, (int, float)) and preemptions > 0):
        probe_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=PROBE_MAX_NEW_TOKENS
        )
        probe_out, _, _ = generate(engine, probe, probe_params)
        correct = 0
        for output, item in zip(probe_out, questions):
            correct += int(grade(output.outputs[0].text, item["gold"]))
    cached_min = min(cached) if cached and all(isinstance(c, int) for c in cached) else ""
    return {
        "e2e_s": elapsed,
        "hit": hit,
        "cached": cached,
        "cached_min": cached_min,
        "gsm_correct": correct if correct != "" else "",
        "gsm_n": PROBE_N if correct != "" else "",
        "preemptions": "" if preemptions is None else preemptions,
        "prefix_tokens": len(prefix),
    }


def finish(row):
    append_result(row)
    print(format_result(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return

    import signal

    def timeout(_signum, _frame):
        finish({
            "status": "crash",
            "crash": True,
            "note": f"timeout {TIMEOUT_S}s",
            "hit": False,
        })
        raise SystemExit(1)

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(TIMEOUT_S)
    try:
        questions = load_probe_questions()
        engine, _load_s = load_engine()
        tokenizer = engine.get_tokenizer()
        if args.baseline:
            run_baseline(engine, tokenizer, questions)
            return
        if not os.path.exists(BASELINE_PATH):
            raise SystemExit("missing autoresearch/baseline.json; run --baseline once first")
        row = run_trial(engine, tokenizer, questions)
        if row["gsm_correct"] == "":
            row["status"] = "discard"
            row["note"] = "missed the cache or the fit cutoff before the probe"
            row["crash"] = False
        else:
            baseline = load_baseline()
            row["status"] = decide(row, baseline, best_keep())
            row["note"] = ""
        finish(row)
    except SystemExit:
        raise
    except Exception as exc:
        finish({"status": "crash", "crash": True, "hit": False, "note": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
