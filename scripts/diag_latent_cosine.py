#!/usr/bin/env python3
"""Per-step latent cosine: HF attn backends vs unified vLLM fork.

Reports a table of cosines for every latent step across Planner/Critic/Refiner.
Also runs an HF-only sdpa vs eager control to size normal kernel drift.

  python scripts/diag_latent_cosine.py --model Qwen/Qwen3-4B --n 3 --latent-steps 10
  python scripts/diag_latent_cosine.py --model Qwen/Qwen3-14B --task aime --n 3 --latent-steps 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("TMPDIR", "/root/tmp")

from models import ModelWrapper
from methods.latent_mas import LatentMASMethod
from prompts import build_agent_message_sequential_latent_mas
from utils import set_seed


def _ns(model_name: str, task: str, latent_steps: int, attn: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        model_name=model_name,
        task=task,
        prompt="sequential",
        think=False,
        latent_space_realign=False,
        latent_steps=latent_steps,
        planner_steps=None,
        critic_steps=None,
        refiner_steps=None,
        agents=None,
        device="cuda",
        device2="cuda",
        use_vllm=False,
        use_second_HF_model=False,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=64,
        generate_bs=1,
        attn_implementation=attn,
        seal=False,
        kvsteer=False,
        ces=False,
        capture_acts=None,
        embed_assembly="repaired",
        scheduler_cls=None,
        max_model_len=16384,
        num_gpu_blocks_override=None,
        gpu_memory_utilization=0.45,
        tensor_parallel_size=1,
    )


def _load_items(task: str, n: int):
    if task == "gsm8k":
        from data import load_gsm8k
        return list(load_gsm8k(split="test"))[:n]
    if task == "aime":
        from data import load_aime2024
        return list(load_aime2024(split="train"))[:n]
    raise ValueError(task)


def hf_latents(method: LatentMASMethod, item: dict) -> list[torch.Tensor]:
    past_kv = None
    latents: list[torch.Tensor] = []
    for agent in method.agents:
        if agent.role == "judger":
            break
        messages = build_agent_message_sequential_latent_mas(
            role=agent.role,
            question=item["question"],
            context="",
            method="latent_mas",
            args=method.args,
        )
        prompts, _, _, _ = method.model.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        prompt = prompts[0]
        encoded = method.model.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        ids = encoded["input_ids"].to(method.model.device)
        mask = encoded["attention_mask"].to(method.model.device)
        steps = method.steps_for_role(agent.role)
        past_kv = method.model.generate_latent_batch(
            ids,
            attention_mask=mask,
            latent_steps=steps,
            past_key_values=past_kv,
            role=agent.role,
            collect_latents=True,
        )
        collected = getattr(method.model, "last_latent_embeds", None)
        if collected is not None and steps > 0:
            for i in range(steps):
                latents.append(collected[i, 0].detach().cpu().float())
    return latents


def cosines(a: list[torch.Tensor], b: list[torch.Tensor]) -> list[float]:
    out = []
    n = min(len(a), len(b))
    for i in range(n):
        aa = a[i].float().reshape(-1)
        bb = b[i].float().reshape(-1)
        out.append(float(torch.nn.functional.cosine_similarity(aa, bb, dim=0)))
    return out


def run_hf_pair(model_name: str, task: str, items, latent_steps: int, attn_a: str, attn_b: str):
    print(f"=== HF {attn_a} vs HF {attn_b} ===")
    results = []
    ns_a = _ns(model_name, task, latent_steps, attn_a)
    print(f"Loading HF ({attn_a})...")
    model_a = ModelWrapper(model_name, torch.device("cuda"), use_vllm=False, args=ns_a)
    method_a = LatentMASMethod(
        model_a, latent_steps=latent_steps, judger_max_new_tokens=64,
        temperature=0.0, top_p=1.0, generate_bs=1, args=ns_a,
    )
    lats_a = [hf_latents(method_a, it) for it in items]
    del method_a, model_a
    torch.cuda.empty_cache()

    ns_b = _ns(model_name, task, latent_steps, attn_b)
    print(f"Loading HF ({attn_b})...")
    model_b = ModelWrapper(model_name, torch.device("cuda"), use_vllm=False, args=ns_b)
    method_b = LatentMASMethod(
        model_b, latent_steps=latent_steps, judger_max_new_tokens=64,
        temperature=0.0, top_p=1.0, generate_bs=1, args=ns_b,
    )
    lats_b = [hf_latents(method_b, it) for it in items]
    del method_b, model_b
    torch.cuda.empty_cache()

    for i, (a, b) in enumerate(zip(lats_a, lats_b)):
        c = cosines(a, b)
        print(f"item {i}: n={len(c)} min={min(c):.6f} mean={sum(c)/len(c):.6f}")
        print("  " + ", ".join(f"{x:.4f}" for x in c))
        results.append({"i": i, "cosines": c, "min": min(c), "mean": sum(c) / len(c)})
    return results, lats_a  # return sdpa (attn_a) latents for fork compare


def run_fork(model_name: str, task: str, items, latent_steps: int, hf_lats):
    from methods.latent_mas_vllm import LatentMASVLLMMethod
    from vllm.latentmas.runner_hooks import clear_collected_latents

    print("=== HF(sdpa) vs fork ===")
    ns = _ns(model_name, task, latent_steps, None)
    ns.gpu_memory_utilization = 0.45
    method = LatentMASVLLMMethod(
        model_name, latent_steps=latent_steps, judger_max_new_tokens=64,
        temperature=0.0, top_p=1.0, generate_bs=1, args=ns,
    )
    results = []
    dump = Path("/workspace/fork_runs/latents_last.pt")
    for i, item in enumerate(items):
        clear_collected_latents()
        if dump.exists():
            dump.unlink()
        method.run_batch([item], return_latents=True)
        if not dump.exists():
            print(f"item {i}: NO latent dump")
            results.append({"i": i, "cosines": [], "error": "no_dump"})
            continue
        v = list(torch.load(dump, map_location="cpu"))
        c = cosines(hf_lats[i], v)
        print(
            f"item {i}: n_hf={len(hf_lats[i])} n_v={len(v)} "
            f"min={min(c) if c else None} mean={sum(c)/len(c) if c else None}"
        )
        print("  " + ", ".join(f"{x:.4f}" for x in c))
        # Per-agent block mins
        k = latent_steps
        blocks = []
        for b in range(0, len(c), k):
            block = c[b : b + k]
            if block:
                blocks.append({"start": b, "min": min(block), "mean": sum(block) / len(block)})
        results.append({
            "i": i,
            "cosines": c,
            "min": min(c) if c else None,
            "mean": sum(c) / len(c) if c else None,
            "blocks": blocks,
            "n_hf": len(hf_lats[i]),
            "n_vllm": len(v),
        })
    del method
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "aime"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--latent-steps", type=int, default=10)
    ap.add_argument("--skip-fork", action="store_true")
    ap.add_argument("--out", default="/workspace/fork_runs/diag_latent_cosine.json")
    args = ap.parse_args()

    set_seed(42)
    task_for_prompts = "aime2024" if args.task == "aime" else args.task
    items = _load_items(args.task, args.n)
    print(f"Loaded {len(items)} {args.task} items; model={args.model} K={args.latent_steps}")

    control, sdpa_lats = run_hf_pair(
        args.model, task_for_prompts, items, args.latent_steps, "sdpa", "eager"
    )
    fork = None
    if not args.skip_fork:
        fork = run_fork(args.model, task_for_prompts, items, args.latent_steps, sdpa_lats)

    report = {
        "model": args.model,
        "task": args.task,
        "n": args.n,
        "latent_steps": args.latent_steps,
        "hf_sdpa_vs_eager": control,
        "hf_sdpa_vs_fork": fork,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("Wrote", out)

    # Summary table
    print("\n=== SUMMARY (min cosine per item) ===")
    print(f"{'item':>4}  {'sdpa_vs_eager_min':>18}  {'sdpa_vs_fork_min':>16}")
    for i in range(len(items)):
        cmin = control[i]["min"]
        fmin = fork[i]["min"] if fork else None
        print(f"{i:4d}  {cmin:18.6f}  {fmin if fmin is not None else float('nan'):16.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
