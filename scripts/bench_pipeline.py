#!/usr/bin/env python3
"""Serial hybrid requests versus HF prep overlapped with the previous vLLM decode.

Both arms use the repaired assembly, greedy fixed-length decode, and one GPU.
The overlap arm prepares request i+1 on the Hugging Face model while vLLM
decodes request i. vLLM's GPU work runs in a child process, so the parent can
run the next latent rollout at the same time.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "bench_hybrid_serving",
    os.path.join(ROOT, "scripts", "bench_hybrid_serving.py"),
)
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
QUESTIONS = _bench.QUESTIONS
apply_stopping = _bench.apply_stopping
load_method = _bench.load_method


def _ns(**kwargs):
    class NS:
        pass
    ns = NS()
    for key, value in kwargs.items():
        setattr(ns, key, value)
    return ns


def serial(method, items):
    rows = []
    wall0 = time.perf_counter()
    for index, item in enumerate(items):
        t0 = time.perf_counter()
        prep0 = time.perf_counter()
        embeds = method.prepare_prompt_embeds([item])
        prep_s = time.perf_counter() - prep0
        torch_sync()
        gen0 = time.perf_counter()
        outputs = method.model.vllm_engine.generate(embeds, method.sampling_params)
        gen_s = time.perf_counter() - gen0
        text = outputs[0].outputs[0].text
        ntok = len(outputs[0].outputs[0].token_ids)
        rows.append({
            "arm": "serial",
            "index": index,
            "question_id": item["id"],
            "prep_s": prep_s,
            "vllm_s": gen_s,
            "e2e_s": time.perf_counter() - t0,
            "output_tokens": ntok,
            "text_head": text[:80],
        })
        print(f"serial {index} prep={prep_s:.2f} vllm={gen_s:.2f}", flush=True)
    return rows, time.perf_counter() - wall0


def pipelined(method, items):
    rows = []
    wall0 = time.perf_counter()
    prep0 = time.perf_counter()
    pending = method.prepare_prompt_embeds([items[0]])
    first_prep = time.perf_counter() - prep0
    for index, item in enumerate(items):
        box = {}

        def _generate(embeds=pending):
            try:
                torch_sync()
                t0 = time.perf_counter()
                box["outputs"] = method.model.vllm_engine.generate(embeds, method.sampling_params)
                box["vllm_s"] = time.perf_counter() - t0
            except Exception as exc:
                box["error"] = exc

        worker = threading.Thread(target=_generate)
        t0 = time.perf_counter()
        worker.start()
        prep_s = 0.0
        nxt = None
        if index + 1 < len(items):
            p0 = time.perf_counter()
            nxt = method.prepare_prompt_embeds([items[index + 1]])
            prep_s = time.perf_counter() - p0
        worker.join()
        if box.get("error"):
            raise box["error"]
        outputs = box["outputs"]
        text = outputs[0].outputs[0].text
        ntok = len(outputs[0].outputs[0].token_ids)
        rows.append({
            "arm": "pipeline",
            "index": index,
            "question_id": item["id"],
            "prep_s": first_prep if index == 0 else prep_s,
            "prep_overlapped": index + 1 < len(items),
            "vllm_s": box["vllm_s"],
            "e2e_s": time.perf_counter() - t0,
            "output_tokens": ntok,
            "text_head": text[:80],
        })
        print(
            f"pipeline {index} prep={rows[-1]['prep_s']:.2f} vllm={box['vllm_s']:.2f} "
            f"wall={rows[-1]['e2e_s']:.2f}",
            flush=True,
        )
        pending = nxt
    return rows, time.perf_counter() - wall0


def torch_sync():
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--latent-steps", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cli = _ns(
        model="Qwen/Qwen3-14B",
        latent_steps=args.latent_steps,
        max_new_tokens=args.tokens,
        temperature=0.0,
        top_p=1.0,
        stop="fixed",
        seed=0,
        prefix_caching=True,
        assembly="repaired",
        device="cuda:0",
        device2="cuda:0",
        gpu_memory_utilization=0.45,
        allow_single_gpu=True,
    )
    method, info = load_method(cli, 1)
    apply_stopping(method, "fixed", args.tokens, 0.0, 1.0)
    items = QUESTIONS[: args.n]
    # One untimed request pays for any remaining warmup inside vLLM.
    method.prepare_prompt_embeds([items[0]])
    method.model.vllm_engine.generate(
        method._pending_embeds, method.sampling_params
    )
    serial_rows, serial_wall = serial(method, items)
    pipe_rows, pipe_wall = pipelined(method, items)
    summary = {
        "load": info,
        "tokens": args.tokens,
        "n": args.n,
        "serial_wall_s": serial_wall,
        "pipeline_wall_s": pipe_wall,
        "serial_rows": serial_rows,
        "pipeline_rows": pipe_rows,
    }
    if summary["pipeline_wall_s"]:
        summary["speedup"] = summary["serial_wall_s"] / summary["pipeline_wall_s"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(json.dumps({k: summary[k] for k in ("serial_wall_s", "pipeline_wall_s", "speedup")}, indent=2))


if __name__ == "__main__":
    main()
