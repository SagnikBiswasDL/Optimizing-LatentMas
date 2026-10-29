# Prefix-cache loop

You are running one experiment at a time on one H200. Edit only `autoresearch/scaffold.py`. Do not edit `prepare.py`, `run.py`, `baseline.json`, or this file during the loop.

The prefix you build is copied onto all five requests. vLLM stores that shared opening once. Five unshared long prompts do not fit in this GPU's KV pool and take about 30 minutes. A shared prefix that still fits finishes in about 7 minutes. The loop keeps a prefix only when it lands on the 7-minute side, the cache actually hits, and the fixed 16-question GSM8K probe stays within 5 points of `baseline.json`. Among those, a later run is kept only if it gets more probe questions right, or the same number right and at least 30 seconds faster.

## Once, before the loop

```bash
python autoresearch/run.py --baseline
```

## Every trial

1. Edit `scaffold.py`.
2. Commit that file.
3. Run `python autoresearch/run.py`.
4. Read the `RESULT` line.
5. On `keep`, leave the commit.
6. On `discard` or `crash`, `git reset --hard HEAD~1`.
7. Do not stop to ask. A miss is discarded before the accuracy probe, so a bad prefix is cheap.

Out of scope: steering vectors, a different model, a different concurrency, a shorter forced decode, salting the prefix per request, and turning prefix caching off. Those change the measurement rather than the prefix.

The four comparison arms (full KV and eviction, cache on and off) are already recorded under `artifacts/jiayi/`. This loop is the fifth arm: one shared prefix, cache on.
