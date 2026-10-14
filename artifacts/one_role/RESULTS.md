# One-role ladder — running lab notebook

Pod `0aa4o7fovbcdq9` (H200). Job finished 01:40 UTC. GPU now **idle (0 MiB)**.
Recommend: **stop**. Do not run AIME n=30 or Judger SEAL.

## Checkpoint 1 — smoke — GO

Two Frozen MATH passes matched exactly. Cache pos=422.

## Checkpoint 2 — MATH n=20 — GO (all three roles)

| Arm | Acc | Recovered vs Frozen | Lost | Tokens | Forwards |
|---|---:|---:|---:|---:|---:|
| frozen | 0.60 | — | — | 1456 | 0 |
| frozen_planner_k10 | 0.80 | 4 | 0 | 1420 | 11 |
| frozen_refiner_k10 | 0.80 | 4 | 0 | 1394 | 11 |
| **frozen_critic_k10** | **0.85** | **5** | **0** | 1393 | 11 |

Winner by the pre-registered rule: **Critic**. This is a real MATH result: one live K=10 block on Mean-Replay beats Frozen by +20–25 pts and loses nothing. Same family as the DeltaBridge MATH oracle (handoff helps MATH).

## Checkpoint 3 — AIME n=6 Frozen vs Critic — borderline

Both 2/6. Recovered item 4, lost item 5. Net 0. Continued to n=12 as pre-registered.

## Checkpoint 4 — AIME n=12 — STOP

Both **7/12 (58.3%)**. Same swap: recovered 4, lost 5. Items 6–11: Critic matched Frozen on every item (5 keep, 1 both-miss). Tokens went the wrong way (6173 vs Frozen 5523). Extra 11 forwards did not buy AIME solves.

AIME n=30 was not run. SEAL not run.

## Read

Keeping K=10 but dropping to one live role does **not** recover AIME’s extra Real solves. The AIME pairing is a 1–1 coin flip on two items, then a perfect copy of Frozen on the rest. That is not “Critic almost works.”

MATH still likes a live K=10 block on the frozen prefix. That is consistent with Frozen≈Real on MATH and with DeltaBridge beating Real on MATH: contest-easy math can use a cheap instance residual; AIME cannot.

## Optional next (only if you want more GPU)

1. **Planner-K10 on AIME n=6** — the original favorite; MATH picked Critic by accuracy, which may be the wrong AIME role. Cheap (~30 min). Kill at n=6 if net ≤ 0.
2. Otherwise **stop the pod**. Idle H200 is just billing. Next mechanism is not another frozen_k* variant.

Locked reminder: Frozen AIME 17/30, Real K=10 20/30.
