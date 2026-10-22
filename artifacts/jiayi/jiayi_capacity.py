#!/usr/bin/env python3
"""Jiayi's two open measurements, on one vLLM engine.

reuse
  Occupy prefix-cache blocks, clear them, then show a later request
  allocates again. Also records cached tokens before and after the clear.

pressure
  One arm per process. Full prefix is 8064 tokens. Evicted prefix keeps the
  first 1024. Both then generate 32768 tokens with EOS ignored. Prefix
  caching is off unless --prefix-caching is set. Prompts differ in the first
  token so requests do not share blocks.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time

FULL_LEN = 8064
SHORT_LEN = 1024
DECODE = 32768


def metric_map():
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


def pick(metrics, *names):
    return {name: metrics.get(name) for name in names if name in metrics}


INTERESTING = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_queries_total",
    "vllm:num_preemptions",
    "vllm:num_preemptions_total",
    "vllm:request_success",
    "vllm:request_success_total",
)


def snap():
    return pick(metric_map(), *INTERESTING)


def usage_of(metrics):
    for key in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return value
    return None


def blocks_from_usage(usage, num_blocks):
    if usage is None or not num_blocks:
        return None
    used = int(round(usage * num_blocks))
    return {"used_blocks_est": used, "free_blocks_est": int(num_blocks) - used}


def cache_config(llm):
    cfg = llm.llm_engine.vllm_config.cache_config
    model = llm.llm_engine.vllm_config.model_config
    out = {}
    for key in ("enable_prefix_caching", "block_size", "num_gpu_blocks", "gpu_memory_utilization", "cache_dtype"):
        if hasattr(cfg, key):
            out[key] = getattr(cfg, key)
    out["max_model_len"] = getattr(model, "max_model_len", None)
    return out


def make_ids(tokenizer, length, salt):
    filler = tokenizer.encode("notes ", add_special_tokens=False)
    if not filler:
        filler = [1]
    header = tokenizer.encode(f"Request {salt}. ", add_special_tokens=False)
    ids = list(header)
    while len(ids) < length:
        ids.extend(filler)
    ids = ids[:length]
    # Keep salt in the first block so concurrent requests do not share a prefix.
    vocab = max(len(tokenizer), 2)
    ids[0] = (100 + salt) % (vocab - 1) + 1
    return ids


def load_llm(args):
    from vllm import LLM
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.prefix_caching,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        disable_log_stats=False,
    )
    info = {
        "load_s": time.perf_counter() - t0,
        "cache_config": cache_config(llm),
        "prefix_caching_requested": args.prefix_caching,
    }
    enabled = info["cache_config"].get("enable_prefix_caching")
    info["effective_prefix_caching"] = enabled
    if args.prefix_caching and enabled is not True:
        raise SystemExit("prefix caching did not stay enabled")
    return llm, info


def generate(llm, prompts, params, sample=False):
    before = snap()
    samples = []
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            samples.append(snap())
            stop.wait(2.0)

    thread = None
    if sample:
        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - t0
    if thread is not None:
        stop.set()
        thread.join(timeout=3)
    after = snap()
    rows = []
    for output in outputs:
        choice = output.outputs[0]
        rows.append({
            "num_cached_tokens": getattr(output, "num_cached_tokens", None),
            "prompt_len": len(output.prompt_token_ids) if output.prompt_token_ids is not None else None,
            "output_tokens": len(choice.token_ids),
            "finish_reason": choice.finish_reason,
        })
    usages = [usage_of(item) for item in samples if usage_of(item) is not None]
    return {
        "e2e_s": elapsed,
        "requests": rows,
        "before": before,
        "after": after,
        "usage_peak": max(usages) if usages else usage_of(after),
    }


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)


def phase_reuse(args):
    from vllm import SamplingParams
    llm, info = load_llm(args)
    num_blocks = int(info["cache_config"]["num_gpu_blocks"])
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=8)
    idle = snap()
    first_ids = make_ids(tokenizer, FULL_LEN, salt=1)
    second_ids = make_ids(tokenizer, FULL_LEN, salt=2)
    first = generate(llm, [{"prompt_token_ids": first_ids}], params)
    replay = generate(llm, [{"prompt_token_ids": first_ids}], params)
    reset_ok = hasattr(llm, "reset_prefix_cache")
    before_reset = snap()
    reset_result = llm.reset_prefix_cache() if reset_ok else None
    time.sleep(1.0)
    after_reset = snap()
    other = generate(llm, [{"prompt_token_ids": second_ids}], params)
    after_evict_replay = generate(llm, [{"prompt_token_ids": first_ids}], params)

    def pack(name, result):
        cached = result["requests"][0]["num_cached_tokens"]
        prompt = result["requests"][0]["prompt_len"] or FULL_LEN
        return {
            "name": name,
            "num_cached_tokens": cached,
            "prompt_len": prompt,
            "fresh_prefill_tokens": None if cached is None else prompt - int(cached),
            "e2e_s": result["e2e_s"],
            "usage": usage_of(result["after"]),
            "blocks": blocks_from_usage(usage_of(result["after"]), num_blocks),
        }

    summary = {
        "question": "prefix hits around eviction, and whether freed cache is reused",
        "num_gpu_blocks": num_blocks,
        "block_size": info["cache_config"].get("block_size"),
        "load": info,
        "idle": {"usage": usage_of(idle), "blocks": blocks_from_usage(usage_of(idle), num_blocks)},
        "steps": [
            pack("cold", first),
            pack("identical_replay_before_evict", replay),
            pack("different_request_after_evict", other),
            pack("original_replay_after_evict", after_evict_replay),
        ],
        "reset_prefix_cache": reset_ok,
        "reset_result": reset_result,
        "before_reset": {"usage": usage_of(before_reset), "blocks": blocks_from_usage(usage_of(before_reset), num_blocks)},
        "after_reset": {"usage": usage_of(after_reset), "blocks": blocks_from_usage(usage_of(after_reset), num_blocks)},
        "raw_idle": idle,
        "raw_before_reset": before_reset,
        "raw_after_reset": after_reset,
        "raw_after_reuse": other["after"],
    }
    write(os.path.join(args.out, "reuse.json"), summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)


def phase_pressure(args):
    from vllm import SamplingParams
    llm, info = load_llm(args)
    num_blocks = int(info["cache_config"]["num_gpu_blocks"])
    block_size = int(info["cache_config"].get("block_size") or 16)
    length = FULL_LEN if args.arm == "full" else SHORT_LEN
    tokenizer = llm.get_tokenizer()
    prompts = [{"prompt_token_ids": make_ids(tokenizer, length, salt=i + 1)} for i in range(args.concurrency)]
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=DECODE,
        min_tokens=DECODE,
        ignore_eos=True,
    )
    result = generate(llm, prompts, params, sample=True)
    outputs = [row["output_tokens"] for row in result["requests"]]
    pre_before = result["before"].get("vllm:num_preemptions_total")
    pre_after = result["after"].get("vllm:num_preemptions_total")
    preemptions = None
    if isinstance(pre_before, (int, float)) and isinstance(pre_after, (int, float)):
        preemptions = pre_after - pre_before
    summary = {
        "question": "full versus evicted prefix during 32768-token decode",
        "arm": args.arm,
        "prefix_tokens": length,
        "prefix_blocks": length // block_size,
        "dropped_tokens": 0 if args.arm == "full" else FULL_LEN - SHORT_LEN,
        "concurrency": args.concurrency,
        "prefix_caching": info["effective_prefix_caching"],
        "num_gpu_blocks": num_blocks,
        "block_size": block_size,
        "kv_blocks_if_resident": (length // block_size) + (DECODE // block_size),
        "output_tokens": outputs,
        "all_reached_32768": all(n == DECODE for n in outputs),
        "num_cached_tokens": [row["num_cached_tokens"] for row in result["requests"]],
        "e2e_s": result["e2e_s"],
        "preemptions": preemptions,
        "usage_before": usage_of(result["before"]),
        "usage_after": usage_of(result["after"]),
        "usage_peak": result["usage_peak"],
        "blocks_at_peak": blocks_from_usage(result["usage_peak"], num_blocks),
        "load_s": info["load_s"],
    }
    write(os.path.join(args.out, "pressure.json"), summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["reuse", "pressure"], required=True)
    parser.add_argument("--arm", choices=["full", "short"], default="full")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.phase == "reuse":
        phase_reuse(args)
    else:
        phase_pressure(args)
