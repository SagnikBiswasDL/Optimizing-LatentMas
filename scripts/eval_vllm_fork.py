#!/usr/bin/env python3
"""End-to-end comparison for the unified vLLM LatentMAS fork.

Arms (selected via --arm):
  fork_stock   unified fork, stock scheduler
  fork_tail    unified fork, TailPreemptScheduler
  hf           Hugging Face-only LatentMAS (paper path)
  hybrid       authors hybrid with repaired embed assembly
  single       single-model baseline on vLLM
  text_mas     text multi-agent on vLLM

Reports throughput, p50/p95 latency, accuracy (with paired bootstrap CI when
a paired baseline file is given), GPU memory, and preemption counts when
available. Greedy and sampled runs are separate invocations.

  python scripts/eval_vllm_fork.py --arm fork_stock --task gsm8k --n 200 \\
      --model Qwen/Qwen3-14B --concurrency 1 --greedy --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TMPDIR", "/root/tmp")

TAIL_SCHED = "serving.tail_preempt_scheduler.TailPreemptScheduler"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True,
                   choices=["fork_stock", "fork_tail", "hf", "hybrid", "single", "text_mas"])
    p.add_argument("--task", default="gsm8k",
                   choices=["gsm8k", "math500", "aime2024", "aime2025", "aime"])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--latent-steps", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--max-model-len", type=int, default=24576)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default="sequential", choices=["sequential", "hierarchical"])
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_items(task: str, n: int):
    from data import load_aime2024, load_aime2025, load_gsm8k

    items = []
    if task == "gsm8k":
        items = list(load_gsm8k(split="test"))
    elif task == "math500":
        # Prefer MATH-500; fall back to hendrycks if helper exists.
        try:
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            for row in ds:
                items.append({
                    "question": row["problem"],
                    "gold": str(row.get("answer", "")).strip(),
                    "solution": row.get("solution", ""),
                })
        except Exception as e:
            raise RuntimeError(f"MATH-500 load failed: {e}") from e
    elif task == "aime2024":
        items = list(load_aime2024(split="train"))
    elif task == "aime2025":
        items = list(load_aime2025(split="train"))
    elif task == "aime":
        for i, it in enumerate(load_aime2024(split="train")):
            it = dict(it); it["qid"] = f"aime2024-{i}"; items.append(it)
        for i, it in enumerate(load_aime2025(split="train")):
            it = dict(it); it["qid"] = f"aime2025-{i}"; items.append(it)
    else:
        raise ValueError(task)
    return items[:n]


def paired_bootstrap_ci(correct_a, correct_b, n_boot=5000, seed=0):
    import random
    n = len(correct_a)
    assert n == len(correct_b) and n > 0
    delta = sum(correct_a) / n - sum(correct_b) / n
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(correct_a[i] for i in idx) / n
        db = sum(correct_b[i] for i in idx) / n
        diffs.append(da - db)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return {"delta_pp": 100 * delta, "ci95_pp": [100 * lo, 100 * hi]}


def build_method(cli):
    from utils import set_seed
    set_seed(cli.seed)
    temp = 0.0 if cli.greedy else cli.temperature
    top_p = 1.0 if cli.greedy else cli.top_p

    if cli.arm in ("fork_stock", "fork_tail"):
        from methods.latent_mas_vllm import LatentMASVLLMMethod
        ns = SimpleNamespace(
            model_name=cli.model, task=cli.task if cli.task != "aime" else "aime2024",
            prompt=cli.prompt, think=False, latent_space_realign=False,
            latent_steps=cli.latent_steps, planner_steps=None, critic_steps=None,
            refiner_steps=None, agents=None, device="cuda",
            temperature=temp, top_p=top_p, max_new_tokens=cli.max_new_tokens,
            generate_bs=1, scheduler_cls=(TAIL_SCHED if cli.arm == "fork_tail" else None),
            max_model_len=cli.max_model_len, num_gpu_blocks_override=None,
            gpu_memory_utilization=cli.gpu_memory_utilization, tensor_parallel_size=1,
        )
        method = LatentMASVLLMMethod(
            cli.model, latent_steps=cli.latent_steps,
            judger_max_new_tokens=cli.max_new_tokens, temperature=temp, top_p=top_p,
            generate_bs=1, args=ns,
        )
        return method, "fork"

    from models import ModelWrapper
    from utils import auto_device
    from methods.latent_mas import LatentMASMethod
    from methods.baseline import BaselineMethod
    from methods.text_mas import TextMASMethod

    use_vllm = cli.arm in ("hybrid", "single", "text_mas")
    use_second = cli.arm == "hybrid"
    ns = SimpleNamespace(
        model_name=cli.model, method=("latent_mas" if cli.arm in ("hf", "hybrid") else cli.arm),
        task=cli.task if cli.task != "aime" else "aime2024", prompt=cli.prompt,
        think=False, latent_steps=cli.latent_steps, planner_steps=None, critic_steps=None,
        refiner_steps=None, max_new_tokens=cli.max_new_tokens, temperature=temp, top_p=top_p,
        generate_bs=1, use_vllm=use_vllm, use_second_HF_model=use_second,
        enable_prefix_caching=False, device="cuda:0", device2="cuda:0",
        tensor_parallel_size=1, gpu_memory_utilization=cli.gpu_memory_utilization,
        latent_space_realign=False, latent_only=False, sequential_info_only=False,
        agents=None, seed=cli.seed, embed_assembly="repaired",
        text_mas_context_length=-1, attn_implementation=None,
        max_model_len=cli.max_model_len, scheduler_cls=None,
        num_gpu_blocks_override=None, seal=False, kvsteer=False, ces=False,
        capture_acts=None,
    )
    wrapper = ModelWrapper(cli.model, auto_device("cuda:0"), use_vllm=use_vllm, args=ns)
    if cli.arm in ("hf", "hybrid"):
        method = LatentMASMethod(
            wrapper, latent_steps=cli.latent_steps, judger_max_new_tokens=cli.max_new_tokens,
            temperature=temp, top_p=top_p, generate_bs=1, args=ns,
        )
        return method, cli.arm
    if cli.arm == "single":
        method = BaselineMethod(
            wrapper, max_new_tokens=cli.max_new_tokens, temperature=temp, top_p=top_p,
            generate_bs=1, use_vllm=True, args=ns,
        )
        return method, "single"
    method = TextMASMethod(
        wrapper, max_new_tokens_each=cli.max_new_tokens, temperature=temp, top_p=top_p,
        generate_bs=1, args=ns,
    )
    return method, "text_mas"


def run_one(method, kind, item):
    t0 = time.perf_counter()
    if kind == "hybrid":
        rows = method.run_batch_vllm([item])
    else:
        rows = method.run_batch([item])
    dt = time.perf_counter() - t0
    row = rows[0]
    stats = row.get("stats") or {}
    n_out = stats.get("num_decode_tokens") or stats.get("num_output_tokens")
    if n_out is None:
        # HF / hybrid: try agent output length proxy
        agents = row.get("agents") or []
        text = (agents[-1].get("output") if agents else row.get("output")) or ""
        n_out = max(1, len(text.split()))
    return {
        "question": item.get("question", "")[:200],
        "gold": item.get("gold"),
        "prediction": row.get("prediction"),
        "correct": bool(row.get("correct")),
        "latency_s": dt,
        "output_tokens": int(n_out),
        "stats": stats,
        "output_head": (row.get("output") or "")[:200],
    }


def closed_loop(method, kind, items, concurrency: int):
    results = [None] * len(items)
    if concurrency <= 1:
        for i, item in enumerate(items):
            results[i] = run_one(method, kind, item)
            print(f"[{i+1}/{len(items)}] ok={results[i]['correct']} lat={results[i]['latency_s']:.2f}s")
        return results

    # Threaded closed loop. The unified fork engine is thread-safe for generate
    # submission via the public LLM API; HF arms should keep concurrency=1.
    inflight = set()
    idx = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        while idx < len(items) or inflight:
            while idx < len(items) and len(inflight) < concurrency:
                fut = ex.submit(run_one, method, kind, items[idx])
                inflight.add(fut)
                fut._item_index = idx  # type: ignore[attr-defined]
                idx += 1
            done = next(as_completed(inflight))
            inflight.remove(done)
            i = done._item_index  # type: ignore[attr-defined]
            results[i] = done.result()
            print(f"[{i+1}/{len(items)}] ok={results[i]['correct']} lat={results[i]['latency_s']:.2f}s")
    return results


def summarize(rows, wall_s, mem_gi_b):
    lats = [r["latency_s"] for r in rows]
    toks = [r["output_tokens"] for r in rows]
    correct = sum(1 for r in rows if r["correct"])
    n = len(rows)
    lats_sorted = sorted(lats)
    def pct(p):
        if not lats_sorted:
            return None
        k = min(len(lats_sorted) - 1, max(0, int(round(p * (len(lats_sorted) - 1)))))
        return lats_sorted[k]
    return {
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "throughput_rps": n / wall_s if wall_s > 0 else 0.0,
        "throughput_tok_s": sum(toks) / wall_s if wall_s > 0 else 0.0,
        "latency_p50_s": pct(0.50),
        "latency_p95_s": pct(0.95),
        "latency_mean_s": statistics.mean(lats) if lats else None,
        "wall_s": wall_s,
        "gpu_mem_gi_b": mem_gi_b,
    }


def main():
    cli = parse_args()
    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(cli.task, cli.n)
    print(f"Loaded {len(items)} items for {cli.task}; arm={cli.arm}")

    import torch
    torch.cuda.reset_peak_memory_stats()
    method, kind = build_method(cli)
    t0 = time.perf_counter()
    rows = closed_loop(method, kind, items, cli.concurrency)
    wall = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    summary = summarize(rows, wall, mem)
    summary.update({
        "arm": cli.arm,
        "task": cli.task,
        "model": cli.model,
        "concurrency": cli.concurrency,
        "greedy": cli.greedy,
        "latent_steps": cli.latent_steps,
        "max_new_tokens": cli.max_new_tokens,
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()
