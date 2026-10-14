# Degenerate-tail (loop) detection

Detector: 8-gram repetition over a 256-token window, threshold 0.8, two consecutive windows required.

Aborting at the onset costs accuracy only for rows marked `would_lose_solve` — those were correct, so the abort would have cut a healthy generation and the detector is too aggressive for them.

## `real`

Abort-at-onset saves **0 of 39542 tokens (0.0%)**, about 0s at 42.4 tok/s. Solves lost: none.

| item | tokens | EOS | correct | loop onset | tokens saved | would lose solve |
|---:|---:|---|---|---:|---:|---|
| 0 | 2599 | yes | yes | — | 0 | no |
| 1 | 8191 | no | no | — | 0 | no |
| 2 | 8192 | no | no | — | 0 | no |
| 4 | 7652 | yes | yes | — | 0 | no |
| 10 | 5847 | yes | yes | — | 0 | no |
| 18 | 7061 | yes | yes | — | 0 | no |

