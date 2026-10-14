#!/usr/bin/env python3
"""One-shot answers to Jiayi's serving questions.

  1. Did LatentMAS eviction free physical GPU memory, or only shrink a tensor?
  2. What does vLLM pre-allocate (he expects on the order of 40-80 GB)?
  3. Prefix-cache hit rate before and after a real cache reset.
  4. Would evicting the latent prefix let a 32k generation stop preempting?

The latent cache is built the same way as the pipeline (Planner, Critic,
Refiner). vLLM is started only after that model is freed, so the two do not
share the GPU. No 32k decode is run: the pool size and the bytes eviction
actually frees already decide whether preemption can change.

  python scripts/diag_jiayi_cache.py --out artifacts/diag/jiayi_cache.json
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from methods import default_agents  # noqa: E402
from models import ModelWrapper  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.relay_compress import RelayCompressor, kv_mb, num_positions  # noqa: E402
from utils import auto_device, set_seed  # noqa: E402

# One contest item is enough. Memory, not accuracy, is the question.
QUESTION = (
    "Find the number of ordered pairs of positive integers (a, b) such that "
    "a + b = 100 and a * b is divisible by 6."
)

KV_BYTES_PER_TOKEN_FALLBACK = 160 * 1024  # measured earlier on this 14B model


def _gb(n: float) -> float:
    return float(n) / (1024.0 ** 3)


def _sync_alloc_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024.0 ** 2)


def _free_gb() -> float:
    torch.cuda.synchronize()
    free, _total = torch.cuda.mem_get_info()
    return _gb(free)


def _release() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


class _Grab(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(record.getMessage())
        except Exception:
            return


def _interesting(lines: List[str]) -> List[str]:
    keep = []
    for line in lines:
        low = line.lower()
        if any(s in low for s in ("kv cache", "gpu blocks", "preempt", "prefix")):
            keep.append(line)
    return keep


def build_cache(wrapper, ns, k: int):
    agents = [a for a in default_agents() if a.role != "judger"]
    past = None
    for agent in agents:
        messages = build_agent_message_sequential_latent_mas(
            role=agent.role, question=QUESTION, context="", method="latent_mas", args=ns
        )
        _, input_ids, attention_mask, _ = wrapper.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        print(f"[latent] {agent.role}  steps={k}", flush=True)
        past = wrapper.generate_latent_batch(
            input_ids, attention_mask=attention_mask, latent_steps=int(k),
            past_key_values=past,
        )
    return past


def hf_physical(args) -> Dict:
    """Shrink the latent cache the way the pipeline does, and see if the GPU notices."""
    ns = SimpleNamespace(
        model_name=args.model, task="gsm8k", prompt="sequential", think=False,
        latent_only=False, sequential_info_only=False, agents=None, use_vllm=False,
        device=args.device, device2="cuda:1", max_new_tokens=8,
        text_mas_context_length=-1, temperature=0.0, top_p=1.0, seed=args.seed,
        seal=False, kvsteer=False, ces=False, capture_acts=None, planner_steps=None,
        critic_steps=None, refiner_steps=None, latent_steps=0, latent_space_realign=False,
        enable_prefix_caching=False, gpu_memory_utilization=0.9,
    )
    wrapper = ModelWrapper(args.model, auto_device(args.device), use_vllm=False, args=ns)
    _release()
    weights_mb = _sync_alloc_mb()

    past = build_cache(wrapper, ns, args.k)
    full_pos = num_positions(past)
    full_tensor_mb = kv_mb(past)
    held_mb = _sync_alloc_mb()

    evicted, _st = RelayCompressor(
        mode="evict", budget=args.budget, sink=args.sink, importance="key_norm"
    ).compress(past)
    # Both tensors are alive here. Drop the original, which is what "evicting"
    # would have to do to return memory.
    del past
    _release()
    after_mb = _sync_alloc_mb()
    evict_pos = num_positions(evicted)
    evict_tensor_mb = kv_mb(evicted)
    per_token = (full_tensor_mb * 1024 * 1024) / max(full_pos, 1)

    del evicted
    del wrapper
    _release()

    freed_mb = held_mb - after_mb
    return {
        "weights_allocated_mb": round(weights_mb, 1),
        "full_positions": int(full_pos),
        "full_tensor_mb": round(full_tensor_mb, 2),
        "allocated_with_full_cache_mb": round(held_mb, 1),
        "evict_positions": int(evict_pos),
        "evict_tensor_mb": round(evict_tensor_mb, 2),
        "allocated_after_dropping_full_mb": round(after_mb, 1),
        "physical_mb_returned": round(freed_mb, 1),
        "bytes_per_token": int(per_token),
    }


def _pool_from_logs(lines: List[str]) -> Dict:
    pool = {"kv_cache_gib": None, "kv_cache_tokens": None, "lines": _interesting(lines)}
    for line in lines:
        low = line.lower()
        if "available kv cache memory" in low:
            # "... 54.32 GiB" or "54.32 GiB"
            for tok in line.replace(",", " ").split():
                try:
                    pool["kv_cache_gib"] = float(tok)
                    break
                except ValueError:
                    continue
        if "gpu kv cache size" in low and "token" in low:
            digits = [t for t in line.replace(",", " ").replace(":", " ").split() if t.isdigit()]
            if digits:
                pool["kv_cache_tokens"] = int(digits[0])
    return pool


def _time_one(llm, prompt: str) -> float:
    from vllm import SamplingParams

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1))
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def vllm_prefix(args) -> Dict:
    """Pre-allocation, prefix-cache hit, and whether a real reset drops the hit."""
    try:
        from vllm import LLM
    except ImportError as exc:
        return {"skipped": f"vllm not importable: {exc}"}

    grab = _Grab()
    grab.setLevel(logging.INFO)
    logging.getLogger().addHandler(grab)
    logging.getLogger("vllm").addHandler(grab)
    logging.getLogger("vllm").setLevel(logging.INFO)

    free_before = _free_gb()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
    )
    free_after = _free_gb()
    pool = _pool_from_logs(grab.lines)
    pool["process_held_gb"] = round(free_before - free_after, 2)
    pool["free_before_gb"] = round(free_before, 2)
    pool["free_after_gb"] = round(free_after, 2)

    # A different short prompt pays kernel warmup. The timed prompt is long
    # enough that a prefill miss is visibly slower than a prefix-cache hit.
    _time_one(llm, "Warmup.")
    prompt = ("Consider this competition mathematics problem carefully. " * 400).strip()
    first = _time_one(llm, prompt)
    second = _time_one(llm, prompt)
    reset_ok = None
    reset = getattr(llm, "reset_prefix_cache", None)
    if reset is None and hasattr(llm, "llm_engine"):
        reset = getattr(llm.llm_engine, "reset_prefix_cache", None)
    if reset is not None:
        try:
            reset_ok = bool(reset())
        except TypeError:
            reset_ok = bool(reset(0))
    third = _time_one(llm, prompt) if reset_ok else None

    del llm
    _release()
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    _release()

    hit = second < 0.5 * first
    miss_after_reset = third is not None and third > 0.7 * first
    return {
        "pool": pool,
        "prefill_s_first": round(first, 3),
        "prefill_s_second": round(second, 3),
        "hit_before_reset": bool(hit),
        "reset_prefix_cache": reset_ok,
        "prefill_s_after_reset": None if third is None else round(third, 3),
        "miss_after_physical_reset": bool(miss_after_reset),
    }


def verdict(hf: Dict, vllm: Dict, total_gb: float) -> List[str]:
    per = hf["bytes_per_token"] or KV_BYTES_PER_TOKEN_FALLBACK
    saved_tokens = hf["full_positions"] - hf["evict_positions"]
    saved_gb = saved_tokens * per / (1024.0 ** 3)
    need_32k_gb = 32768 * per / (1024.0 ** 3)
    lines = [
        f"GPU total is {total_gb:.0f} GB. "
        f"The latent cache is {hf['full_positions']} positions, {hf['full_tensor_mb']:.0f} MB. "
        f"After eviction ({hf['evict_positions']} positions, {hf['evict_tensor_mb']:.0f} MB) "
        f"and dropping the original, PyTorch reports {hf['physical_mb_returned']:.0f} MB returned.",
    ]
    if vllm.get("skipped"):
        lines.append(f"vLLM did not run ({vllm['skipped']}). The 40-80 GB pool and the prefix-cache hit rate were not measured.")
        return lines

    pool = vllm.get("pool") or {}
    gib = pool.get("kv_cache_gib")
    tokens = pool.get("kv_cache_tokens")
    held = pool.get("process_held_gb")
    if gib is not None:
        lines.append(f"vLLM pre-allocated {gib:.1f} GB of KV cache. The process held {held:.0f} GB after startup.")
    else:
        lines.append(f"vLLM startup held {held:.0f} GB. The KV-cache log line was not found; see pool.lines in the json.")

    if vllm.get("hit_before_reset"):
        lines.append(
            f"Prefix cache hit before any reset: second prefill {vllm['prefill_s_second']}s "
            f"vs first {vllm['prefill_s_first']}s."
        )
    else:
        lines.append(
            f"No prefix-cache hit seen: second prefill {vllm['prefill_s_second']}s "
            f"vs first {vllm['prefill_s_first']}s."
        )
    if vllm.get("miss_after_physical_reset"):
        lines.append(
            f"After vLLM's own cache reset, the same prompt missed "
            f"({vllm['prefill_s_after_reset']}s). That reset is a physical eviction. "
            f"The LatentMAS compressor never calls it."
        )
    elif vllm.get("reset_prefix_cache") is False or vllm.get("reset_prefix_cache") is None:
        lines.append("vLLM did not expose reset_prefix_cache, so the after-eviction hit rate was not measured.")

    if tokens:
        fits = tokens >= 32768
        still_fits = tokens >= max(1, 32768 - saved_tokens)
        lines.append(
            f"The pool holds {tokens} tokens. A 32k generation needs about {need_32k_gb:.1f} GB. "
            f"Evicting this latent prefix saves {saved_gb:.2f} GB. "
            + (
                "Both the full and the evicted 32k generation fit, so eviction cannot be what stops preemption."
                if fits and still_fits
                else "The pool is tighter than 32k. Eviction saves "
                f"{saved_tokens} tokens, which does not move a 32k request across that line."
                if fits == still_fits
                else "Eviction is large enough here to change whether 32k fits."
            )
        )
    else:
        lines.append(
            f"A 32k generation is about {need_32k_gb:.1f} GB of KV. "
            f"This eviction saves {saved_gb:.2f} GB, so it does not decide preemption at 32k."
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--skip_vllm", action="store_true")
    ap.add_argument("--out", default="artifacts/diag/jiayi_cache.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(2)
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    name = torch.cuda.get_device_name(0)
    total_gb = _gb(torch.cuda.get_device_properties(0).total_memory)
    print(f"[gpu] {name}  {total_gb:.1f} GB", flush=True)

    hf = hf_physical(args)
    print(
        f"[hf] cache {hf['full_tensor_mb']} MB -> {hf['evict_tensor_mb']} MB, "
        f"physical returned {hf['physical_mb_returned']} MB",
        flush=True,
    )
    vllm = {"skipped": "not run"} if args.skip_vllm else vllm_prefix(args)
    lines = verdict(hf, vllm, total_gb)

    report = {
        "gpu": name,
        "gpu_total_gb": round(total_gb, 2),
        "hf": hf,
        "vllm": vllm,
        "verdict": lines,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("", flush=True)
    for line in lines:
        print(line, flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
