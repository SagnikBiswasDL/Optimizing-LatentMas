# Lock-in — Frozen Mean-Replay + Judger SEAL

**This is the only GPU campaign.** Everything else is stopped.

## Why this, not another hunt

We already have two positives that do not need more invention:

1. **Frozen MATH-1k Mean-Replay** = Real K=10 on GSM8K (0.92) and MATH (0.76 vs 0.74). Zero upstream forwards. AIME is the exception (56.7 vs 66.7).
2. **Judger SEAL** (GSM8K exec−reflection, L28, coef 40) cuts tokens **−17% iso-acc** on live LatentMAS. Coef 80 is −39%.

One-role (Planner/Refiner/Critic at K=10) recovered MATH instances and **zero extra AIME solves**. DeltaBridge oracle residual **destroyed AIME**. Native Judger vectors are worse than generic SEAL. MedQA SEAL already costs accuracy.

AIME extra Real solves are recurrent, not a one-token residual and not a one-role patch. **Do not hunt AIME accuracy.** SEAL is a **latency / token** head. If Frozen+SEAL keeps GSM8K/MATH accuracy and shortens the Judger, that is the paper result. If AIME n=6 loses net solves or does not cut tokens, **stop the pod**.

## The experiment

| | |
|---|---|
| Model | Qwen3-14B only |
| Cache | Frozen MATH-1k Mean-Replay, 422 pos, ~66MB. Restore, do not rebuild. |
| Arms | `frozen` vs `frozen_seal` |
| Vector | `artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt` (fallback `gsm8k_layer28.pt`) |
| Coef / layer | 40 / 28, Judger decode only |
| Upstream | **0** live agents. No DeltaBridge. No MLP. No PCA. |

Driver: `bash scripts/run_frozen_seal.sh`

## Gates (kill, do not iterate)

| Stage | n | B | T | Continue iff |
|---|---|---|---|---|
| GSM8K | 40 | 4 | 1024 | acc ≥ frozen − 0.05 **and** tokens ≤ 90% of frozen |
| MATH | 40 | 4 | 2048 | same |
| AIME24 | 6 | 1 | 8192 | lost ≤ recovered **and** tokens drop ≥ 5% |
| AIME n=12 / 30 | | | | only if n=6 says `go`. Stop if lost > recovered |

Outputs: `artifacts/frozen_seal/{gate.json,CHECKIN.md,<stage>/report.json}`

## Explicitly forbidden

- Training `g_φ` / any MLP
- Live Planner / Critic / Refiner
- New SEAL vectors, coef sweeps, native Judger vectors
- Tuning on recovered AIME items
- Restarting one-role or DeltaBridge for AIME n=30

## GPU

Do not start until a pod is given. Restore `math1k/cache.pt` from `/workspace` if present. Then:

```bash
bash /workspace/recover_env.sh   # git pull then venv; upload new files AFTER this
bash scripts/run_frozen_seal.sh
```

If `gate.json` recommend is `stop`, power off. Do not invent a follow-up on that GPU.
