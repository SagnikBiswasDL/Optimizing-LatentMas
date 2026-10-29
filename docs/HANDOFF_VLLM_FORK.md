# Handoff: LatentMAS in one vLLM engine

Written 2026-10-29 for a researcher picking this up cold. Branch `perf/vllm-fork`.

## What we are trying to do

Serve LatentMAS (Zou et al., `2511.20639v3.pdf` in the repo root) efficiently on reasoning tasks.

The authors run it in two programs: Hugging Face does the Planner / Critic / Refiner latent steps, then hands prompt embeddings to vLLM, which re-prefills them and decodes the Judger. That means two copies of the weights, a blocking handoff, and no batching of latent steps. Their README says they patched vLLM internals; that patch is not in their public code.

We built a vLLM fork that runs the whole pipeline in one engine: latent steps feed `realign(last_hidden)` back as the next input embedding, agent role prompts are appended to the same request, and the Judger decodes from the same in-engine KV.

Advisor context: Jiayi also asked us to look at receiver-conditioned compression of the handoff (CacheBack, `2609.32046`, and a PCA idea where each handed-off row is projected from D to d dimensions). None of that is implemented yet.

## Where things are

| What | Where |
|---|---|
| vLLM fork (2 commits on v0.11.1, tag commit `4393684`) | `serving/vllm_fork/latentmas-v0.11.1.patch` |
| Unified method | `methods/latent_mas_vllm.py`, `--method latent_mas_vllm` in `run.py` |
| Eval harness | `scripts/eval_vllm_fork.py` |
| Per-step latent drift diagnostic | `scripts/diag_latent_cosine.py` |
| Correctness test | `tests/test_latent_mas_vllm_correctness.py` |
| Raw results (`summary.json`, `rows.jsonl` per run) | `artifacts/vllm_fork/` |
| Detailed results write-up | `docs/VLLM_FORK_RESULTS.md` |
| Plain-language slides (promising results only) | `docs/slides_vllm_fork_plain.pdf` |
| Original plan | `docs/PLAN_VLLM_FORK.md` |
| Earlier findings about what the latent channel does | `docs/READ_ME_FIRST_AGENT_BRIEFING.md` |

To apply the fork: `git clone https://github.com/vllm-project/vllm && cd vllm && git checkout v0.11.1 && git am <repo>/serving/vllm_fork/latentmas-v0.11.1.patch && VLLM_USE_PRECOMPILED=1 pip install -e .` (torch 2.9.0 cu128, transformers 4.57.1).

## Results so far (Qwen3-14B, greedy, one H200)

| Task | Fork | HF-only (paper path) |
|---|---|---|
| GSM8K n=50, accuracy | 0.98 | 0.98 |
| GSM8K tokens/s | 86 | 24 |
| GSM8K median s/question | 5.8 | 13.3 |
| AIME n=60, accuracy | 0.38 | 0.53 |
| AIME answers hitting the 8192 cap | 34/60 | 0/60 |
| MATH-500 n=200, accuracy | 0.51 | not run |

The authors' hybrid on GSM8K (n=30) is 5.1 s/question at 56 tokens/s. A single model with no agents is 6.5 s at 68 tokens/s.

GSM8K throughput with many questions at once (fork): 86 / 275 / 535 / 776 / 1146 tokens/s at 1 / 8 / 16 / 32 / 50 concurrent.

## Open problems, most important first

1. **The fork's Judger writes much longer answers.** GSM8K median answer is about 510 tokens on the fork against about 300 on HF and the hybrid, and AIME hits the cap on 34/60. That is the whole AIME accuracy gap (8 of the 11 HF-only-correct items were capped on the fork) and erases the per-question speed advantage. Earlier work in this repo found the latent channel's main effect is to shorten the Judger, so the fork may be losing that effect.
   - Per-step latent cosine on 14B is within the HF sdpa-vs-eager control (`docs/VLLM_FORK_RESULTS.md`), so the latents themselves look as close to HF as HF is to itself. The cause may be downstream: position IDs of latent and role-prompt tokens, chat-template or `<think>` placement around the Judger prompt, or how the Judger's prompt differs from the HF path.
   - The cheapest test: run the HF path and the fork on the same items and diff the Judger's full input token sequence and positions, then compare answer-length distributions.
2. **Latency under concurrency is not measured correctly.** Concurrency is wave-batched (one `generate` per wave of C requests), and the per-request latency reported at C > 1 is close to wall time divided by n (median equals p95 at C = 60). Use throughput only, or rewrite the harness around an async engine with per-request timestamps.
3. **Concurrent `generate` from threads segfaulted** (illegal memory access). Wave batching sidesteps it. A real serving claim needs `AsyncLLMEngine` or the OpenAI server with continuous batching.
4. **Missing arms:** HF on MATH-500, text multi-agent on vLLM, the hybrid at concurrency, and any run where the KV pool actually overflows. Without overflow, tail preemption and stock preemption are identical.
5. **Is the method worth serving at all?** Earlier results (`docs/READ_ME_FIRST_AGENT_BRIEFING.md`): on AIME, deleting the latent channel cost 1/30 problems, and a cache from an unrelated problem gave most of the length saving. Any serving claim should be reported next to the single-model baseline.

## Compute

- RunPod pod `1a7b2e145fjhab` is stopped. Its network volume `royal_white_vole_volume` (US-NC-1) holds `/workspace`, with the fork, the repo, Qwen3-14B/4B weights, datasets, and logs.
- UCSB CSIL has a Slurm `a100` partition (one A100 80GB per node, 48 GB RAM, 24 h limit), reachable through class accounts. Home quota is 6 GB with no scratch, so Qwen3-14B does not fit without more storage. Research use of the class allocation has not been confirmed.
