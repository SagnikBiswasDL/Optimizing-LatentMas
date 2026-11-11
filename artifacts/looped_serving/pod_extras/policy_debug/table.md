| depth | policy | accuracy % (95% CI) | candidates/q | answered % | capped % | loop-tokens/q (charged) | latency mean s | p50 s | p95 s | tokens/s | questions/min | peak KV GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | SC@4 (live driver) | 12.5 (0.0, 37.5) | 4.00 | 3.1 | 96.9 | 3672 | 4.1 | 4.1 | 4.2 | 371 | 29.0 | 1.5 |
| 3 | SC@4 + certificate cancel (live) | 12.5 (0.0, 37.5) | 4.00 | 3.1 | 96.9 | 3672 | 4.1 | 4.1 | 4.2 | 370 | 29.0 | 1.5 |
| 3 | two-then-two (live) | 12.5 (0.0, 37.5) | 4.00 | 3.1 | 96.9 | 3672 | 8.4 | 8.4 | 8.5 | 365 | 28.6 | 1.5 |
