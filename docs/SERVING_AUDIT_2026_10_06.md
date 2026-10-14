# Hybrid serving audit — 2026-10-06

Starting commit: `53f1b42c31261d912c08fb0bdb7caf2aa7c14824` on `feature/seal-token-efficiency`.
That commit matches `origin/feature/seal-token-efficiency`. No newer remote commits.
Work is on `perf/hybrid-serving`. `baseline/latentmas-fork` (`153ea35`, upstream `9a9e4d3`) is untouched.

The published LatentMAS numbers are the Hugging Face path. The hybrid path (HF Planner/Critic/Refiner, vLLM Judger) is the optional accelerated path and is the performance baseline.

## What the code does

`run_batch_vllm` runs the three latent agents on a second HF model, then sends **prompt embeddings** to `vllm_engine.generate`. It does not pass HF KV tensors. The HF rollout is `@torch.no_grad()` on `generate_latent_batch_hidden_state`, not `inference_mode`. Realignment still runs: with `--latent_space_realign` off, the map is the identity and the latent vector is rescaled to the mean embedding norm.

`run.py` forces `--use_second_HF_model` and `--enable_prefix_caching` whenever `--use_vllm` is set for `latent_mas`. The pipeline is a blocking offline loop: one batch finishes HF preparation and then one synchronous `LLM.generate` before the next batch starts. HF and vLLM do not overlap. This is not an online server.

No vLLM patch is in this repo. The README says the authors modified vLLM internals. On stock vLLM 0.11.0, `enable_prompt_embeds=True` together with prefix caching makes vLLM **disable prefix caching** and log a warning (`vllm/engine/arg_utils.py` around the 0.11.0 release). Block hashes in that version do not include embedding bytes. Unsafe reuse is therefore unlikely on 0.11.0 because caching is off, not because the hash is embedding-aware. Later vLLM added embedding-aware prefix caching (PR 27219). The installed version has to be read at runtime; the historical note of vLLM 0.11.0 is not a measurement of this machine.

`scripts/diag_jiayi_cache.py` still runs HF and vLLM separately and infers cache hits from timing. It is not a concurrent hybrid measurement.

## Defects in the authors' assembly

Confirmed against `baseline/latentmas-fork` as well as this tree. The original lines are still the default (`--embed_assembly original`).

1. **Batch size 1 crashes.** `embedding_layer(...)` returns `[1, L, H]`. `.squeeze(0)` makes it `[L, H]`, and `B, L, H = ...` fails. One request cannot run.
2. **A padded batch is not the same work as separate requests.** The Judger tokenize uses the tokenizer's padding side and then discards the attention mask. Pad embeddings are inserted into the vLLM prompt. Shorter rows are then padded with `torch.zeros`, which is float32, so the bf16 prompt is promoted to float32.
3. **Prompt embeddings were tied to the prefix-caching flag.** Turning caching off also built an engine without `enable_prompt_embeds`. The constructor now requests prompt embeddings for every hybrid LatentMAS load and passes the caching flag through. On vLLM 0.11.0 the effective cache state is still off when embeddings are enabled.

`--embed_assembly repaired` is only the repair above: keep the batch dimension, drop masked positions, preserve dtype, pass each request at its own length. It is not a performance change. The insertion index still uses the original `tokenizer(left)` call, which defaults to `add_special_tokens=True` while the prompt was tokenized with `add_special_tokens=False`. Both lengths are recorded. An overshoot is clamped the way the original slice clamps.

## Not run

This machine has no NVIDIA GPU, no CUDA toolkit, and no PyTorch. `nvidia-smi` is absent. Nsight Systems and Nsight Compute are absent. Nothing below has a latency number.

## First commands once a GPU is attached

Two GPUs are the authors' setup (vLLM on `cuda:0`, HF on `cuda:1`). One 80GB+ GPU can be tried with `--allow-single-gpu`. Qwen3-14B in bf16 is about 28GB per copy.

```bash
python scripts/bench_hybrid_serving.py --phase env
python scripts/bench_hybrid_serving.py --phase one --max-new-tokens 32 --latent-steps 10
python scripts/bench_hybrid_serving.py --phase one-repaired --max-new-tokens 32 --latent-steps 10 --no-warmup
```

`one` should record the squeeze crash and keep it. `one-repaired` is the first successful request. Decoding in the harness is greedy (`temperature 0`). `run.py` defaults to temperature 0.6, top_p 0.95; do not compare those policies as a speedup.

After that request succeeds: `--phase lengths`, then `--phase batch`, then a Nsight trace of `one-repaired`. The trace has to follow vLLM's child process:

```bash
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
  -o artifacts/serving/nsys_one_repaired --gpu-metrics-devices=all \
  python scripts/bench_hybrid_serving.py --phase one-repaired --max-new-tokens 128 --no-warmup
```

`--stop fixed` is a synthetic fixed-length run. `--stop eos` is the natural-answer run. Report them separately.
