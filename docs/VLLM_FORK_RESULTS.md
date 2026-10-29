# Unified LatentMAS inside a vLLM fork

Updated 2026-10-29. One RunPod H200 (143 GB). Stack: torch 2.9.0+cu128, vLLM **fork** of 0.11.1 (`/workspace/vllm-fork`, branch `latentmas`), transformers 4.57.1.

## Claim

The contribution is a **vLLM fork** that runs the full LatentMAS pipeline — Planner / Critic / Refiner latent steps plus Judger decode — inside **one** engine, keeping KV in place across agent role prompts. Built unconditionally. Compared to the paper’s Hugging Face-only path (and hybrid / single-model / text-MAS baselines) without overclaiming.

## Design (fork)

- Request flag via `SamplingParams.extra_args["latent_mas"]` with phase schedule: `latent` → `tokens` (next agent role prompt) → … → `decode`.
- Latent step: next input is `realign(last_hidden)` into `inputs_embeds` (same realign as `models._apply_latent_realignment`; default identity + mean input-embedding norm rescale).
- Latent embeddings stored so preemption / recompute replays them.
- Agent role prompts append as ordinary tokens on the same request; Judger then decodes.
- Config: `enforce_eager=True`, async scheduling off, `enable_prompt_embeds=True`, prefix caching off.
- Wiring: `methods/latent_mas_vllm.py`, `--method latent_mas_vllm`, optional `scheduler_cls` for stock vs `TailPreemptScheduler`.

## Correctness (Qwen3-4B, GSM8K, greedy)

| Check | Result |
|---|---|
| Prompt token ids | Exact match HF vs fork |
| First latent cosine (n=5, K=4) | ≥ 0.9997 on every item |
| Judger text prefix (80 chars) | Match on 5/5 |
| Forced preemption (`gpu_memory_utilization=0.18`, 64 KV blocks, n=3) | Latents + prefix hold |
| Full greedy answer | Can diverge later (HF SDPA vs vLLM FlashAttention); not bit-identical |

Cross-stack latent agreement reported with **cosine ≥ 0.99** on early steps.

## Primary comparison: fork vs HF-only (paper path)

All greedy, Qwen3-14B, sequential prompts, `latent_steps=10`, one H200.

### GSM8K n=50, concurrency 1, max_new=2048

| Arm | Acc | tok/s | p50 s | p95 s | wall s |
|---|---:|---:|---:|---:|---:|
| **unified fork (stock)** | **0.98** | **86.4** | **5.77** | 15.1 | **350** |
| HF-only LatentMAS | 0.98 | 24.3 | 13.26 | — | 752 |
| hybrid (repaired) n=30 | 1.00 | 56.1 | 5.07 | — | 191 |
| single-model vLLM | 0.98 | 68.3 | 6.49 | — | 385 |

Paired fork vs HF: accuracy Δ = **0.0 pp**, bootstrap 95% CI **[-6.0, +6.0] pp** (agree 48/50). Per-item latency ratio HF/fork median **2.17×**.

### AIME 2024+2025 n=60, concurrency 1, max_new=8192

| Arm | Acc | tok/s | p50 s | wall s |
|---|---:|---:|---:|---:|
| unified fork (stock) | 0.383 (23/60) | 86.8 | 93.8 | 4453 |
| HF-only LatentMAS | **0.533 (32/60)** | 24.6 | 175.4 | 8871 |

Paired Δ accuracy = **−15.0 pp**, bootstrap 95% CI **[−26.7, −5.0] pp** (agree 47/60; HF-only correct on 11, fork-only on 2).

**Do not overclaim:** on AIME the fork is faster (wall 1.99×, tok/s 3.5×) but **less accurate** under the same 8192-token Judger budget. 34/60 fork items had latency >90s (≈8192 tokens at 87 tok/s — length capped). HF often stops earlier with EOS (~4600 tokens at 25 tok/s ≈ 180s). Likely causes: (1) latent drift vs HF SDPA compounding into longer Judger traces, (2) length-cap interactions. GSM8K does not show this accuracy gap.

### GSM8K concurrency (wave-batched `generate`, greedy)

| Arm | C | Acc | tok/s | p50 s | wall s |
|---|---:|---:|---:|---:|---:|
| fork_stock | 1 | 0.98 | 86.4 | 5.77 | 350 |
| fork_stock | 8 | 0.96 | 275 | 2.66 | 111 |
| fork_tail | 8 | 0.96 | 283 | 2.59 | 108 |
| fork_stock | 16 | 1.00 | 535 | 1.00 | 56 |
| fork_stock | 32 | 0.98 | 776 | 0.68 | 38 |

Stock vs tail at C=8 are within noise here (pool not stressed). Throughput scales with concurrency as expected for continuous batching.

## What we are not claiming

- Bit-identical greedy Judger text vs HF across attention backends.
- AIME accuracy parity (fork is worse under this budget; reported above).
- That tail preemption beats stock when the KV pool is not overflowing.

## Related work (short)

- LatentMAS HF path; authors’ hybrid (unreleased patch + second HF + `prompt_embeds` re-prefill).
- Coconut / continuous latent tokens as `inputs_embeds`.
- CacheBack and KV-transfer systems (CacheGen, LMCache, disaggregated prefill) — this fork **avoids** the handoff.
- Preemption / recompute scheduling (including this repo’s tail-only preemption).

## Artifacts

- Fork: `/workspace/vllm-fork` branch `latentmas`.
- Hybrid branch: `perf/vllm-fork`.
- Summaries: `/workspace/fork_runs/*/summary.json`.
- Slides: `docs/slides_vllm_fork.tex`.
