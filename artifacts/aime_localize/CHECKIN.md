# AIME localization check-in

**recommend:** `collect_views`

need real / none / c3 before calling a hop wasteful

## Where the latency is (per agent)

| Agent | n | seconds | % of e2e | KV positions written | peak MB |
|---|---:|---:|---:|---:|---:|
| planner | 6 | 0.409 | 0.26% | 210 (KV positions) | 29624 |
| critic | 6 | 0.300 | 0.19% | 251 (KV positions) | 28657 |
| refiner | 6 | 0.289 | 0.18% | 237 (KV positions) | 28756 |
| judger | 6 | 155.420 | 99.36% | 6591 (tokens emitted) | - |

Silent agents together cost 0.998s of the 156.4s item. Anything that only removes silent-agent work is capped at that number — the Judger decode is the budget that matters.

## Latency per arm

Real end-to-end **156.4s/item**: Judger decode 155.4s (99.4%), silent agents 1.00s (0.6%).
Real Judger emits 6591 tokens at 42.4 tok/s; 67% of items stop on EOS (the rest burn the full budget).

| Arm | acc | judger_s | upstream_s | e2e_s | vs Real | tokens | eos |
|---|---:|---:|---:|---:|---:|---:|---:|
| real | 0.667 | 155.4 | 1.00 | 156.4 | +0.0s (+0.0%) | 6591 | 67% |

Upstream is charged honestly: in the growing tape `c3` still needs Planner and Critic to have run, so only `none`/`c1`/`c2`/`isolated` can bank silent-agent time.

## Accuracy per arm

Critical items (Real-only vs Frozen in the locked ladder): [4, 10, 18]

| Arm | n | acc | correct items | critical kept | lost vs Real | recovered vs none |
|---|---:|---:|---|---|---|---|
| real | 6 | 0.667 | [0, 4, 10, 18] | [4, 10, 18] | [] | [0, 4, 10, 18] |

Jiayi map: `real` = judger(c1+c2+c3). `c1`/`c2`/`c3` = private writes. `isolated` = a1(c1')→… with empty/frozen c', judger still gets concat writes. `evict_seg` = SnapKV budget per role span.

