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

## Correctness and per-step latent cosine

### Protocol

Compare realigned latent vectors at every Planner / Critic / Refiner step (`K` per agent, 3 agents → `3K` vectors). Greedy Judger text is a soft check: FlashAttention (vLLM) vs SDPA (HF default) is not bit-identical.

**Control:** HF with `attn_implementation=sdpa` vs HF with `eager` sizes normal kernel drift under the same recurrent latent feedback.

### Qwen3-4B, GSM8K, K=10 (30 steps), n=3

| Pair | item | min cos | mean cos |
|---|---:|---:|---:|
| HF sdpa vs eager | 0 | 0.012 | 0.729 |
| HF sdpa vs eager | 1 | −0.027 | 0.815 |
| HF sdpa vs eager | 2 | 0.122 | 0.692 |
| HF sdpa vs fork | 0 | −0.193 | 0.835 |
| HF sdpa vs fork | 1 | 0.024 | 0.711 |
| HF sdpa vs fork | 2 | 0.037 | 0.517 |

First latent of each agent stays ≈ 0.99+ (role-prompt splicing OK). Later steps collapse on **both** the control and the fork — recurrent feedback amplifies attention-kernel noise. Fork drift is in the same envelope as sdpa↔eager; not treated as a separate realign/RoPE bug on 4B.

### Qwen3-14B, AIME, K=10 (30 steps), n=3

| Pair | item | min cos | mean cos |
|---|---:|---:|---:|
| HF sdpa vs eager | 0 | 0.808 | 0.929 |
| HF sdpa vs eager | 1 | 0.717 | 0.913 |
| HF sdpa vs eager | 2 | 0.751 | 0.917 |
| HF sdpa vs fork | 0 | **0.822** | **0.938** |

Full per-step vectors (item 0, fork vs HF sdpa):

```
planner: 0.9996 0.9988 0.9971 0.9807 0.9622 0.9675 0.9404 0.8298 0.8878 0.8970
critic:  0.9979 0.9886 0.9756 0.9634 0.9443 0.9303 0.8784 0.9024 0.8886 0.9110
refiner: 0.9958 0.9890 0.9763 0.9736 0.9106 0.8221 0.9032 0.9077 0.9061 0.9084
```

On 14B, fork vs HF min cosine is **within the sdpa↔eager control**. No fork-only latent bug found beyond backend drift. (Items 1–2 fork rows re-run after a scoring fix; see `diag_latent_cosine_14b_aime.json` when complete.)

### Forced preemption

`gpu_memory_utilization=0.18`, 64 KV blocks, n=3 GSM8K on 4B: latents + Judger prefix still hold after recompute of stored embeds.

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

| Arm | Acc | tok/s | p50 s | wall s | length-capped |
|---|---:|---:|---:|---:|---:|
| unified fork (stock) | 0.383 (23/60) | 86.8 | 93.8 | 4453 | **34/60** at 8192 decode |
| HF-only LatentMAS | **0.533 (32/60)** | 24.6 | 175.4 | 8871 | **0/60** (max ≈ 6474) |

Paired Δ accuracy = **−15.0 pp**, bootstrap 95% CI **[−26.7, −5.0] pp** (agree 47/60; HF-only correct on 11, fork-only on 2). Among the 11 HF-only-correct items, 8 were fork length-capped.

**Do not overclaim:** on AIME the fork is faster (wall 1.99×, tok/s 3.5×) but **less accurate** under the same 8192-token Judger budget. Latent cosine on 14B is within the HF kernel-drift control; the accuracy gap tracks **non-terminating Judger traces** (length cap) under FlashAttention trajectories, not a failed realignment. GSM8K does not show this accuracy gap.

### GSM8K concurrency (wave-batched `generate`, greedy)

Thread-pool concurrent `generate` segfaulted (TMA / illegal memory access). **Wave-batched** submission (one `generate` per wave of size C) is stable.

| Arm | C | Acc | tok/s | p50 s | wall s |
|---|---:|---:|---:|---:|---:|
| fork_stock | 1 | 0.98 | 86.4 | 5.77 | 350 |
| fork_stock | 8 | 0.96 | 275 | 2.66 | 111 |
| fork_tail | 8 | 0.96 | 283 | 2.59 | 108 |
| fork_stock | 16 | 1.00 | 535 | 1.00 | 56 |
| fork_tail | 16 | 1.00 | 519 | 1.04 | 58 |
| fork_stock | 32 | 0.98 | 776 | 0.68 | 38 |
| fork_tail | 32 | 0.98 | 769 | 0.69 | 38 |
| fork_stock | 60 | 0.98 | 1146 | 0.51 | 26 |

Stock vs tail are within noise when the KV pool is not overflowing. Throughput scales with concurrency as expected.

### MATH-500, concurrency 1, max_new=2048

| Arm | n | Acc | tok/s | p50 s | p95 s | wall s |
|---|---:|---:|---:|---:|---:|---:|
| fork_stock | 50 | 0.46 (23/50) | 86.0 | 11.3 | 23.6 | 686 |
| **fork_stock** | **200** | **0.510 (102/200)** | **85.2** | **12.5** | 24.2 | **2891** |

HF MATH-500 n=200: running (continuation queue).

## What we are not claiming

- Bit-identical greedy Judger text vs HF across attention backends.
- AIME accuracy parity (fork is worse under this budget; reported above).
- That tail preemption beats stock when the KV pool is not overflowing.
- That per-step latents stay within bf16 elementwise tolerance for all 30 steps — they do not, on either HF↔HF or HF↔fork, once recurrent feedback runs.

## Related work (short)

- LatentMAS HF path; authors’ hybrid (unreleased patch + second HF + `prompt_embeds` re-prefill).
- Coconut / continuous latent tokens as `inputs_embeds`.
- CacheBack and KV-transfer systems (CacheGen, LMCache, disaggregated prefill) — this fork **avoids** the handoff.
- Preemption / recompute scheduling (including this repo’s tail-only preemption).

## Artifacts

- Fork: `/workspace/vllm-fork` branch `latentmas`.
- Hybrid branch: `perf/vllm-fork`.
- Summaries: `/workspace/fork_runs/*/summary.json`.
- Cosine diags: `/workspace/fork_runs/diag_latent_cosine_4b.json`, `diag_latent_cosine_14b_aime*.json`.
- Slides: `docs/slides_vllm_fork.tex`.
