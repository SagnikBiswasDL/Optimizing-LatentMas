# Promotion decision

Rule: keep **every** baseline solve **and** cut total tokens by **≥10%** across items [0, 1, 2, 4, 10, 18].

Baseline `real`: solves [0, 1, 4, 10, 18], total **45291** tokens at cap 16384, nothing censored.

| candidate | solves kept | solves lost | total tok | saving (all items) | saving (baseline's solves) | verdict |
|---|---|---|---:|---:|---:|---|
| `real_seal-40` | 0,10,18 | 1,4 | 59979 | ≤-32.4% | -27.5% | reject: lost 1,4 |
| `real_seal-60` | 0,1,10,18 | 4 | 70112 | ≤-54.8% | -57.1% | reject: lost 4 |

A `≤` on the saving means the candidate was still censored somewhere, so its true token count is higher and its true saving is lower.

## Nothing promoted

No candidate both kept every baseline solve and cleared the token bar. Check the near-miss note above before concluding steering cannot work: the binding constraint so far has been termination, not token count.

