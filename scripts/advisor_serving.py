#!/usr/bin/env python3
"""Advisor measurements on one vLLM engine.

Questions
  1. Prefix-cache hits before and after eviction.
  2. Whether freed cache capacity is reused.
  3. Full versus a shorter block-aligned prompt during 32768-token decode.

vLLM 0.11.1 is required. vLLM 0.11.0 turns prefix caching off when prompt
embeddings are enabled, and this script exits if that happens.

The shorter arm drops whole trailing blocks of the Judger embedding prompt
before prefill. It does not import Hugging Face KV, and it does not rewrite
an existing vLLM block table. Both arms use this same engine API.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.bench_hybrid_serving import (  # noqa: E402
    QUESTIONS,
    append_jsonl,
    cache_config,
    collect_env,
    prometheus_vllm,
    write_json,
)


def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def metric_map():
    raw = prometheus_vllm()
    flat = {}
    if isinstance(raw, dict) and "error" in raw and len(raw) == 1:
        return {"_error": raw["error"]}
    for name, samples in raw.items():
        for sample in samples:
            if sample["name"].endswith("_created"):
                continue
            flat.setdefault(sample["name"], 0.0)
            flat[sample["name"]] += float(sample["value"])
    return flat


def interesting(metrics):
    keys = (
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_queries",
        "vllm:num_preemptions",
        "vllm:num_preemptions_total",
        "vllm:gpu_cache_usage_perc",
        "vllm:request_success",
        "vllm:request_success_total",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def delta_metrics(before, after):
    out = {}
    for key, value in after.items():
        if not isinstance(value, (int, float)):
            continue
        prev = before.get(key)
        if isinstance(prev, (int, float)):
            out[key] = value - prev
    return out


def hit_rate(cached, prompt):
    if not prompt:
        return None
    return cached / prompt


def block_aligned_keep(length, block_size, drop_blocks):
    """Keep a prefix of whole blocks, dropping ``drop_blocks`` from the tail."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    full_blocks = length // block_size
    keep_blocks = max(1, full_blocks - drop_blocks)
    keep = min(length, keep_blocks * block_size)
    return keep


def make_namespace(cli):
    return SimpleNamespace(
        model_name=cli.model,
        method="latent_mas",
        task="gsm8k",
        prompt="sequential",
        think=False,
        latent_steps=cli.latent_steps,
        planner_steps=None,
        critic_steps=None,
        refiner_steps=None,
        max_new_tokens=cli.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        generate_bs=max(cli.concurrency, 1),
        use_vllm=True,
        use_second_HF_model=True,
        enable_prefix_caching=cli.prefix_caching,
        device="cuda:0",
        device2="cuda:0",
        tensor_parallel_size=1,
        gpu_memory_utilization=cli.gpu_memory_utilization,
        latent_space_realign=False,
        latent_only=False,
        sequential_info_only=False,
        agents=None,
        seed=cli.seed,
        embed_assembly="repaired",
        text_mas_context_length=-1,
        attn_implementation=None,
        max_model_len=cli.max_model_len,
    )


def load_method(cli):
    import torch
    from methods.latent_mas import LatentMASMethod
    from models import ModelWrapper
    from utils import auto_device, set_seed
    from vllm import SamplingParams

    set_seed(cli.seed)
    ns = make_namespace(cli)
    t0 = time.perf_counter()
    wrapper = ModelWrapper(cli.model, auto_device("cuda:0"), use_vllm=True, args=ns)
    if cli.max_model_len:
        # ModelWrapper does not forward max_model_len. Record the engine value.
        pass
    load_s = time.perf_counter() - t0
    method = LatentMASMethod(
        wrapper,
        latent_steps=cli.latent_steps,
        judger_max_new_tokens=cli.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        generate_bs=ns.generate_bs,
        args=ns,
    )
    method.sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=cli.max_new_tokens,
        min_tokens=cli.max_new_tokens if cli.ignore_eos else 0,
        ignore_eos=cli.ignore_eos,
    )
    info = {
        "load_s": load_s,
        "cache_config": cache_config(wrapper.vllm_engine),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    return method, info


def embeds_for(method, item):
    method._stop_before_vllm = True
    try:
        method.run_batch_vllm([item])
    finally:
        method._stop_before_vllm = False
    rows = method._last_prompt_embeds
    if not rows:
        raise RuntimeError("no prompt embeddings were captured")
    return rows[0].detach()


def _sample_metrics(stop, samples, interval=1.0):
    while not stop.is_set():
        samples.append({"t": time.time(), "metrics": interesting(metric_map())})
        stop.wait(interval)


def generate_embeds(method, tensors, params, sample=False):
    if params is None:
        params = method.sampling_params
    _sync()
    before = metric_map()
    samples = []
    stop = None
    thread = None
    if sample:
        import threading
        stop = threading.Event()
        thread = threading.Thread(target=_sample_metrics, args=(stop, samples), daemon=True)
        thread.start()
    t0 = time.perf_counter()
    outputs = method.model.vllm_engine.generate(
        [{"prompt_embeds": tensor} for tensor in tensors],
        params,
    )
    _sync()
    elapsed = time.perf_counter() - t0
    if stop is not None:
        stop.set()
        thread.join(timeout=2)
    after = metric_map()
    rows = []
    for output in outputs:
        choice = output.outputs[0]
        prompt_len = int(getattr(output, "prompt_token_ids", None) is not None and len(output.prompt_token_ids) or 0)
        # Prompt-embed requests may report length via metrics rather than token ids.
        cached = getattr(output, "num_cached_tokens", None)
        rows.append({
            "num_cached_tokens": cached,
            "output_tokens": len(choice.token_ids),
            "finish_reason": choice.finish_reason,
            "text_head": (choice.text or "")[:80],
            "prompt_token_ids_len": prompt_len,
        })
    return {
        "e2e_s": elapsed,
        "requests": rows,
        "metrics_before": interesting(before),
        "metrics_after": interesting(after),
        "metric_deltas": interesting(delta_metrics(before, after)),
        "metric_samples": samples,
    }


def short_params(n):
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, top_p=1.0, max_tokens=n)


def require_caching(info, cli, out_dir):
    enabled = info["cache_config"].get("enable_prefix_caching")
    record = {
        "requested_prefix_caching": cli.prefix_caching,
        "effective_prefix_caching": enabled,
        "cache_config": info["cache_config"],
        "load_s": info["load_s"],
        "torch": info["torch"],
        "cuda": info["cuda"],
        "env": collect_env(),
    }
    write_json(os.path.join(out_dir, "load.json"), record)
    if cli.prefix_caching and enabled is not True:
        write_json(os.path.join(out_dir, "blocker.json"), {
            "blocker": "prefix caching did not stay enabled with prompt embeddings",
            "effective_prefix_caching": enabled,
        })
        print(json.dumps(record, indent=2, default=str), flush=True)
        raise SystemExit(2)
    return record


def phase_probe(cli, method, out_dir):
    block_size = int(cache_config(method.model.vllm_engine).get("block_size") or 16)
    q0 = QUESTIONS[0]
    q1 = QUESTIONS[1]
    base = embeds_for(method, q0)
    other = embeds_for(method, q1)
    params = short_params(cli.max_new_tokens)

    cold = generate_embeds(method, [base], params)
    replay = generate_embeds(method, [base], params)
    changed = base.clone()
    mid = changed.shape[0] // 2
    changed[mid] = changed[mid] + 1
    mutated = generate_embeds(method, [changed], params)
    shared = generate_embeds(method, [other], params)

    engine = method.model.vllm_engine
    reset_ok = hasattr(engine, "reset_prefix_cache")
    usage_before_reset = metric_map()
    reset_result = None
    if reset_ok:
        reset_result = engine.reset_prefix_cache()
    usage_after_reset = metric_map()
    after_first = generate_embeds(method, [base], params)
    after_second = generate_embeds(method, [base], params)

    def pack(name, result, prompt_len):
        cached = result["requests"][0]["num_cached_tokens"]
        cached_n = int(cached or 0)
        row = {
            "probe": name,
            "prompt_len": prompt_len,
            "num_cached_tokens": cached,
            "fresh_prefill_tokens": None if cached is None else prompt_len - cached_n,
            "hit_rate": None if cached is None else hit_rate(cached_n, prompt_len),
            "output_tokens": result["requests"][0]["output_tokens"],
            "text_head": result["requests"][0]["text_head"],
            "e2e_s": result["e2e_s"],
            "metric_deltas": result["metric_deltas"],
        }
        append_jsonl(os.path.join(out_dir, "probes.jsonl"), row)
        return row

    rows = [
        pack("cold", cold, int(base.shape[0])),
        pack("identical_replay", replay, int(base.shape[0])),
        pack("changed_block", mutated, int(changed.shape[0])),
        pack("other_question", shared, int(other.shape[0])),
        pack("after_reset_first", after_first, int(base.shape[0])),
        pack("after_reset_second", after_second, int(base.shape[0])),
    ]
    summary = {
        "block_size": block_size,
        "changed_index": int(mid),
        "changed_block": int(mid) // block_size,
        "reset_prefix_cache_available": reset_ok,
        "reset_prefix_cache_result": reset_result,
        "metrics_before_reset": interesting(usage_before_reset),
        "metrics_after_reset": interesting(usage_after_reset),
        "metric_delta_reset": interesting(delta_metrics(usage_before_reset, usage_after_reset)),
        "probes": rows,
        "note": (
            "reset_prefix_cache drops cached prefix blocks. "
            "changed_block must not reuse the mutated block or its suffix. "
            "other_question may reuse only a shared instruction prefix."
        ),
    }
    write_json(os.path.join(out_dir, "probe_summary.json"), summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


def phase_pressure(cli, method, out_dir):
    import torch
    block_size = int(cache_config(method.model.vllm_engine).get("block_size") or 16)
    item = QUESTIONS[0]
    base = embeds_for(method, item)
    keep = block_aligned_keep(int(base.shape[0]), block_size, cli.drop_blocks)
    short = base[:keep].contiguous()
    from vllm import SamplingParams
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=cli.max_new_tokens,
        min_tokens=cli.max_new_tokens,
        ignore_eos=True,
    )
    arms = {
        "full": base,
        "block_aligned_shorten": short,
    }
    # Distinct first-block salt so concurrent copies do not share a prefix.
    summaries = []
    for name, tensor in arms.items():
        batch = []
        for i in range(cli.concurrency):
            row = tensor.clone()
            row[0] = row[0] + float(i + 1)
            batch.append(row)
        before = metric_map()
        if hasattr(method.model.vllm_engine, "reset_prefix_cache"):
            method.model.vllm_engine.reset_prefix_cache()
        result = generate_embeds(method, batch, params, sample=True)
        lengths = [int(t.shape[0]) for t in batch]
        cached = [r["num_cached_tokens"] for r in result["requests"]]
        outputs = [r["output_tokens"] for r in result["requests"]]
        usage = [
            sample["metrics"].get("vllm:gpu_cache_usage_perc")
            for sample in result["metric_samples"]
            if isinstance(sample["metrics"].get("vllm:gpu_cache_usage_perc"), (int, float))
        ]
        row = {
            "arm": name,
            "concurrency": cli.concurrency,
            "prefix_caching": cli.prefix_caching,
            "prompt_len": lengths[0],
            "retained_tokens": int(tensor.shape[0]),
            "dropped_tokens": int(base.shape[0] - tensor.shape[0]),
            "block_size": block_size,
            "drop_blocks": cli.drop_blocks if name != "full" else 0,
            "output_tokens": outputs,
            "all_reached_target": all(n == cli.max_new_tokens for n in outputs),
            "num_cached_tokens": cached,
            "e2e_s": result["e2e_s"],
            "metrics_before": interesting(before),
            "metric_deltas": result["metric_deltas"],
            "gpu_cache_usage_peak": max(usage) if usage else None,
            "positions": "renumbered_0_to_L" if name != "full" else "native_prefill_positions",
        }
        append_jsonl(os.path.join(out_dir, "pressure.jsonl"), row)
        summaries.append(row)
        print(json.dumps(row, indent=2, default=str), flush=True)
        torch.cuda.empty_cache()
    write_json(os.path.join(out_dir, "pressure_summary.json"), {"runs": summaries})
    return summaries


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["probe", "pressure"], default="probe")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--latent-steps", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--drop-blocks", type=int, default=4)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main():
    cli = parse_args()
    out_dir = cli.out or os.path.join(
        ROOT, "artifacts", "advisor", f"{cli.phase}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "config.json"), vars(cli))
    try:
        method, info = load_method(cli)
        require_caching(info, cli, out_dir)
        if cli.phase == "probe":
            phase_probe(cli, method, out_dir)
        else:
            phase_pressure(cli, method, out_dir)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        write_json(os.path.join(out_dir, "error.json"), {"traceback": traceback.format_exc()})
        raise SystemExit(1)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
