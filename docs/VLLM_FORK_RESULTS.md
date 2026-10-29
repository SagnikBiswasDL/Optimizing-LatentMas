# Unified LatentMAS inside a vLLM fork

Updated 2026-10-29. One RunPod H200 (143 GB). Stack: torch 2.9.0+cu128, vLLM **fork** of 0.11.1 (`/workspace/vllm-fork`, branch `latentmas`), transformers 4.57.1.

## Claim

The contribution is a **vLLM fork** that runs the full LatentMAS pipeline — Planner / Critic / Refiner latent steps plus Judger decode — inside **one** engine, keeping KV in place across agent role prompts. It is built unconditionally. We then compare it to the paper’s Hugging Face-only path (and the authors’ hybrid, single-model, and text-MAS baselines), without overclaiming.

## Design (fork)

- Request flag via `SamplingParams.extra_args["latent_mas"]` with an explicit phase schedule: `latent` → `tokens` (next agent role prompt) → … → `decode`.
- While latent steps remain, the V1 runner skips the sampled-token embedding and writes `realign(last_hidden)` into `inputs_embeds` for that request’s next position (same realign as `models._apply_latent_realignment`; default is identity + mean input-embedding norm rescale).
- Latent positions occupy ordinary KV blocks. Their embeddings are stored on the request so preemption / recompute replays embeds, not token ids.
- Agent transitions append role-prompt token ids to the **same** request; the scheduler prefills them. Judger then decodes normally.
- Start config: `enforce_eager=True`, async scheduling off, `enable_prompt_embeds=True`, prefix caching off. CUDA graphs left for later.

Repo wiring: `methods/latent_mas_vllm.py`, `--method latent_mas_vllm` in `run.py`, optional `scheduler_cls` for stock vs `TailPreemptScheduler`.

## Correctness (Qwen3-4B, GSM8K, greedy)

Harness: `tests/test_latent_mas_vllm_correctness.py`. Artifacts under `/workspace/fork_runs/`.

| Check | Result |
|---|---|
| Prompt token ids | Exact match HF vs fork |
| First latent cosine (n=5 items, K=4) | ≥ 0.9997 on every item |
| Early-step latents | Cosine ≥ 0.99 for the first two steps; later steps drift as HF SDPA and vLLM FlashAttention diverge |
| Judger text prefix (80 chars) | Match on all 5 items |
| Full greedy text / final boxed answer | Can diverge after the shared prefix (attn-kernel drift compounded through 12 latent steps). Not bit-identical across stacks. |
| Forced preemption (`gpu_memory_utilization=0.18`, `num_gpu_blocks_override=64`, n=3) | First-latent cosine ≥ 0.9997; prefix match; latent replay survived recompute |

Decision: report cross-stack latent agreement with **cosine ≥ 0.99** on early steps, not elementwise bf16 `allclose` across different attention backends. Same-backend bit identity was not claimed.

## Serving comparison (primary)

Primary pair: **unified fork** vs **HF-only LatentMAS** (paper setup). Additional arms: hybrid (`embed_assembly=repaired`), single-model vLLM, text-MAS vLLM. Greedy and sampled are separate. Concurrency closed-loop via `scripts/eval_vllm_fork.py`.

Numbers below are filled as `/workspace/fork_runs/*/summary.json` lands. Incomplete rows stay blank rather than invented.

### Greedy, Qwen3-14B

| Arm | Task | n | C | Acc | tok/s | p50 s | p95 s | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fork_stock | GSM8K | 50 | 1 |  |  |  |  | |
| hf | GSM8K | 50 | 1 |  |  |  |  | paper path |
| fork_stock | GSM8K | 50 | 8 |  |  |  |  | |
| fork_tail | GSM8K | 50 | 8 |  |  |  |  | |
| single | GSM8K | 50 | 8 |  |  |  |  | must report |
| hybrid | GSM8K | 30 | 1 |  |  |  |  | repaired assembly |
| fork_stock | AIME 24+25 | 60 | 1 |  |  |  |  | |
| hf | AIME 24+25 | 60 | 1 |  |  |  |  | |

### Sanity (Qwen3-4B)

| Arm | Task | n | C | Acc | wall s | notes |
|---|---|---:|---:|---:|---:|---|
| fork_stock | GSM8K | 20 | 1 |  |  | first queue job |

## Related work (short)

- **LatentMAS** and its published HF path; the authors’ hybrid depends on an unreleased vLLM patch plus a second HF copy and a `prompt_embeds` re-prefill.
- **Coconut / continuous latent tokens**: latent steps as inputs_embeds rather than discrete tokens.
- **CacheBack** and KV-transfer systems (CacheGen, LMCache, disaggregated prefill): move or compress KV across engines; this fork instead **avoids** the handoff.
- **Preemption / recompute** scheduling (including this repo’s tail-only preemption): latent embeds must survive the same path as tokens.

## What we are not claiming

- Bit-identical greedy Judger text vs HF when attention backends differ.
- That the unified fork is faster than HF before the concurrency tables are filled.
- That single-model is worse or better without reporting it side-by-side.

## Artifacts

- Fork commits on `/workspace/vllm-fork` branch `latentmas`.
- Hybrid branch `perf/vllm-fork`.
- Logs and summaries: `/workspace/fork_runs/`.
- Slides: `docs/slides_vllm_fork.tex`.
