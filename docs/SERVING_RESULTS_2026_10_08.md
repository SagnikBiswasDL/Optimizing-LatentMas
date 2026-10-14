# Hybrid serving results, 2026-10-08

Pod `ls1xw0slua4d6c`, one H200, driver 570.124.06. Stack unchanged: Qwen3-14B, PyTorch 2.8.0+cu128, Transformers 4.57.1, vLLM 0.11.0. Single-GPU hybrid uses `gpu_memory_utilization=0.45`. Decoding in these runs is greedy (`temperature=0`, `top_p=1`). Latent steps stay at 10. Raw rows are under `artifacts/serving/`.

Measured HBM copy bandwidth on this card: **4.26 TB/s**. A 14B bf16 weight read is about 28 GB, so the batch-1 weight-bandwidth ceiling is about **152 tokens/s**.

## Prompt-embed decode is not slower than text decode

Same engine settings, 1024 forced tokens, vLLM only (no second Hugging Face copy, utilization 0.90):

| Path | Seconds | Tokens/s |
|---|---:|---:|
| Token ids (`enable_prompt_embeds=False`) | 9.37 | 109.3 |
| Engine built with prompt embeds | 9.51 | 107.7 |

109 tokens/s is about 72% of the measured bandwidth ceiling. That is ordinary batch-1 decode, not a prompt-embed penalty. vLLM 0.11 does leave the embedding lookup outside the CUDA graph when prompt embeds are enabled; on this GPU that does not move tokens/s.

## Overlapping the next latent rollout with decode does not pay

Four repaired requests, 1024 forced tokens, one GPU. The overlap arm builds request i+1 on the Hugging Face model while vLLM decodes request i.

| Arm | Wall |
|---|---:|
| Serial | 41.52 s |
| Overlapped | 40.75 s |

Speedup **1.019x**. Decode slowed from about 9.6 s to about 10.1 s whenever the next latent prep ran on the same GPU, which cancels the ~0.8 s of prep. The first 80 characters of each greedy answer matched across arms, so the overlap did not corrupt outputs. This is a one-GPU result. The authors' two-GPU split was not available on this pod.

## Prefix caching does not engage

vLLM 0.11 turns prefix caching off when prompt embeds are enabled. The cache probe replayed the same embeddings and a copy with the middle row shifted by 1. `num_cached_tokens` was 0 both times (0.097 s vs 0.095 s). `effective_prefix_caching` is false. The first 8 tokens matched (`<think>\nOkay, let's see.`) because both generations open the same way, not because of a cache hit.

## Natural answers, separate from the fixed-length sweeps

Repaired assembly, EOS, cap 512, three saved questions. None hit the cap. All three matched the integer gold.

| Question | Tokens | Finish | Correct | End-to-end | vLLM |
|---|---:|---|---|---:|---:|
| add | 257 | stop | 42 | 3.48 s | 2.40 s |
| packs | 293 | stop | 30 | 3.56 s | 2.73 s |
| long | 499 | stop | 37 | 5.82 s | 4.86 s |

Latent rollout stays about 0.8–1.0 s. vLLM is still most of the request once the answer is a few hundred tokens.

## What this leaves in place

The repaired embedding assembly is the correctness fix (the original path still raises `ValueError` at batch size 1, and the original batched path keeps padding and promotes bf16 to float32). It is not a speed change. No further change beat that repaired baseline: decode is already at the text-prompt rate, and hiding latent prep behind decode on one GPU does not stick. The throughput lever already measured is offline batching (batch 8 at 1024 forced tokens finishes in 13.0 s versus about 10.5 s for one request).
