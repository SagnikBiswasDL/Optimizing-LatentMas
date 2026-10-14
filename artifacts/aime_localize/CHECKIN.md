# AIME localization check-in

**recommend:** `collect_views`

need real / none / c3 before calling a hop wasteful

## Where the latency is (per agent)

| Agent | n | seconds | % of e2e | KV positions written | peak MB |
|---|---:|---:|---:|---:|---:|
| planner | 36 | 0.326 | 0.21% | 209 (KV positions) | 28925 |
| critic | 36 | 0.296 | 0.19% | 250 (KV positions) | 28673 |
| refiner | 36 | 0.288 | 0.18% | 236 (KV positions) | 28765 |
| judger | 6 | 155.420 | 99.42% | 6591 (tokens emitted) | - |

Silent agents together cost 0.910s of the 156.3s item. Anything that only removes silent-agent work is capped at that number — the Judger decode is the budget that matters.

## Latency per arm

Real end-to-end **156.4s/item**: Judger decode 155.4s (99.4%), silent agents 1.00s (0.6%). (n_timed=6 of 6, unbatched only.)
Real Judger emits 6591 tokens at 42.4 tok/s; 67% of items stop on EOS (the rest burn the full budget).

Per-item latency (unbatched rows only; `—` means the arm was run grouped):

| Arm | acc | judger_s | upstream_s | e2e_s | vs Real | tokens | eos | n(timed) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real__bs16 | 0.333 | — | 1.00 | — | — | 6816 | 50% | 0/6 |
| real_seal60 | 1.000 | 114.7 | 1.24 | 116.0 | -40.4s (-25.8%) | 4896 | 50% | 2/2 |
| real | 0.667 | 155.4 | 1.00 | 156.4 | +0.0s (+0.0%) | 6591 | 67% | 6/6 |
| real_seal40 | 0.500 | 158.7 | 1.00 | 159.7 | +3.3s (+2.1%) | 6745 | 33% | 6/6 |
| none | 0.333 | 172.7 | 0.00 | 172.7 | +16.3s (+10.4%) | 7348 | 17% | 6/6 |

GPU cost actually paid per arm (valid for grouped runs too — `batch_s/batch_size` sums back to true wall clock):

| Arm | n | max batch | wall_s | tok/s (aggregate) |
|---|---:|---:|---:|---:|
| none | 6 | 1 | 1036 | 42.6 |
| real_seal40 | 6 | 1 | 952 | 42.5 |
| real | 6 | 1 | 933 | 42.4 |
| real__bs16 | 6 | 6 | 533 | 76.8 |
| real_seal60 | 2 | 1 | 229 | 42.7 |

Upstream is charged honestly: in the growing tape `c3` still needs Planner and Critic to have run, so only `none`/`c1`/`c2`/`isolated` can bank silent-agent time.

## Accuracy per arm

Critical items (Real-only vs Frozen in the locked ladder): [4, 10, 18]

| Arm | n | acc | correct items | critical kept | lost vs Real | recovered vs none |
|---|---:|---:|---|---|---|---|
| real_seal60 | 2 | 1.000 | [0, 1] | [] | [4, 10, 18] | [1] |
| real | 6 | 0.667 | [0, 4, 10, 18] | [4, 10, 18] | [] | [4, 10] |
| real_seal40 | 6 | 0.500 | [0, 1, 4] | [4] | [10, 18] | [1, 4] |
| real__bs16 | 6 | 0.333 | [0, 18] | [18] | [4, 10] | [] |
| none | 6 | 0.333 | [0, 18] | [18] | [4, 10] | [] |

Jiayi map: `real` = judger(c1+c2+c3). `c1`/`c2`/`c3` = private writes. `isolated` = a1(c1')→… with empty/frozen c', judger still gets concat writes. `evict_seg` = SnapKV budget per role span.

