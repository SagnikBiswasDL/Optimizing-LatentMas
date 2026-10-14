"""Why is StaticCache slow? Separating allocation from true length.

The briefing attributes the StaticCache penalty to attending over every allocated
slot. That cannot be right. This model's KV cache is 160.0 KB/token (measured
directly from `cache_mb` across 112 pipeline rows, exactly consistent). Attending
over 9,216 allocated slots instead of the 956 real ones therefore reads 1.35 GB
more per step, which at 4.8 TB/s is 0.28 ms/token. The measured penalty is 53.9
ms/token -- 191x larger. Whatever is costing the time, it is not KV bandwidth.

Every existing measurement confounds two variables, because a probe that allocates
L and generates only a few hundred tokens has allocation >> true length:

  * ALLOCATION: `max_cache_len`, fixed for the run.
  * TRUE LENGTH: slots actually filled, which grows every step.

If cost tracks allocation, a long run at a large allocation is uniformly slow and
the ladder in the briefing (allocate tight, grow on demand) is the right fix. If
cost tracks true length, then a large allocation is harmless once the run actually
fills it, the ladder buys nothing, and the pipeline's loss came from somewhere
else entirely. These predict opposite things, and the per-step time series
distinguishes them directly: flat means allocation, ramping means true length.

Also recorded, because both would masquerade as a scaling law:
  * torch.compile / CUDA-graph re-entry counts, in case the cost is recapture.
  * time of the first few steps, which is graph capture, reported separately from
    the steady state rather than averaged into it.

Run on the pod (see scripts/rp.sh):
  python scripts/diag_static_mechanism.py --out artifacts/diag/static_mechanism.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics as st
import sys
import time
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KV_KB_PER_TOKEN = 160.0  # measured from the pipeline's own cache_mb field
HBM_TBS = {"H200": 4.8, "H100": 3.35, "A100": 2.04, "L40": 0.86, "A6000": 0.77}


def bandwidth_tbs() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    for k, v in HBM_TBS.items():
        if k in name:
            return v
    return None


class StepTicker:
    """Timestamps every decode step.

    Implemented as a LogitsProcessor because that is the one hook called exactly
    once per generated token, inside generate(), on both the compiled and eager
    paths. It synchronises so each timestamp reflects completed GPU work; the
    ~20us cost is negligible against the 13-67 ms/step being measured, and it is
    paid identically by every configuration.
    """

    def __init__(self) -> None:
        self.t: List[float] = []

    def __call__(self, input_ids, scores):
        torch.cuda.synchronize()
        self.t.append(time.perf_counter())
        return scores

    def steps_ms(self) -> List[float]:
        return [1e3 * (b - a) for a, b in zip(self.t, self.t[1:])]


def compile_counters() -> Dict[str, int]:
    try:
        from torch._dynamo.utils import counters
        return {k: int(sum(v.values())) if isinstance(v, dict) else int(v)
                for k, v in counters.items()}
    except Exception:
        return {}


def load(model_name: str, attn: str = "sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation=attn)
    m.to("cuda").eval()
    m.config.use_cache = True
    return m, tok


def make_prompt(tok, n_tokens: int) -> torch.Tensor:
    filler = ("Consider the following competition mathematics problem and reason "
              "carefully about each step before answering. ")
    ids = tok(filler * 400, return_tensors="pt").input_ids[:, :n_tokens]
    return ids.to("cuda")


@torch.no_grad()
def run_one(model, tok, prefix_len: int, new_tokens: int,
            max_cache_len: Optional[int]) -> Dict:
    """One configuration. max_cache_len None means the dynamic cache."""
    ids = make_prompt(tok, prefix_len)
    ticker = StepTicker()
    kw = dict(
        input_ids=ids,
        max_new_tokens=int(new_tokens),
        # Pinned so every configuration emits exactly the same number of tokens;
        # an early EOS would otherwise make the series lengths incomparable.
        min_new_tokens=int(new_tokens),
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
        logits_processor=[ticker],
    )
    if max_cache_len is not None:
        from transformers import StaticCache
        kw["past_key_values"] = StaticCache(
            config=model.config, max_batch_size=1,
            max_cache_len=int(max_cache_len), device=model.device,
            dtype=model.dtype)

    before = compile_counters()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**kw)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    after = compile_counters()

    produced = int(out.shape[1] - ids.shape[1])
    steps = ticker.steps_ms()
    # The first steps are prefill and graph capture; averaging them into the rate
    # is how a capture cost gets mistaken for a per-token cost.
    warm = steps[8:] if len(steps) > 16 else steps
    decile = []
    if len(warm) >= 10:
        n = len(warm) // 10
        decile = [round(st.median(warm[i * n:(i + 1) * n]), 3) for i in range(10)]
    return {
        "max_cache_len": max_cache_len,
        "cache": "dynamic" if max_cache_len is None else "static",
        "prefix_len": prefix_len,
        "true_len_end": prefix_len + produced,
        "tokens": produced,
        "wall_s": round(wall, 3),
        "tok_per_s_wall": round(produced / wall, 2),
        "step_ms_first8": [round(x, 3) for x in steps[:8]],
        "step_ms_median": round(st.median(warm), 3) if warm else None,
        "step_ms_p10": round(sorted(warm)[len(warm) // 10], 3) if len(warm) >= 10 else None,
        "step_ms_p90": round(sorted(warm)[9 * len(warm) // 10], 3) if len(warm) >= 10 else None,
        "step_ms_deciles": decile,
        "ramp_ratio": (round(decile[-1] / decile[0], 3)
                       if decile and decile[0] else None),
        "compile_delta": {k: after.get(k, 0) - before.get(k, 0)
                          for k in set(after) | set(before)
                          if after.get(k, 0) - before.get(k, 0) != 0},
    }


def price_ladder(by: Dict[str, Dict], prefix: int = 900, new: int = 7000,
                 ladder: Sequence[int] = (2048, 4096, 8192, 16384)) -> None:
    """Fit cost(allocation) from the static runs and price the growing-ladder fix.

    "Cost tracks allocation" makes the ladder the right *shape* of fix, but says
    nothing about whether it is worth building. It is worth building only if the
    integral of cost over a realistic generation beats the dynamic path, and the
    catch is that a tight allocation stops being tight: true length grows into it,
    so a tightly-allocated static cache gets more expensive exactly as the run
    goes on, while the dynamic path is flat in length.
    """
    pts = {r["max_cache_len"]: r["step_ms_median"]
           for r in by.values()
           if r.get("cache") == "static" and r.get("max_cache_len")
           and r.get("step_ms_median")}
    dyn = [r["step_ms_median"] for r in by.values()
           if r.get("cache") == "dynamic" and r.get("step_ms_median")]
    if len(pts) < 2 or not dyn:
        print("  (need two static allocations and one dynamic run to price the "
              "ladder; skipping)")
        return
    (l1, c1), (l2, c2) = sorted(pts.items())[:2]
    slope = (c2 - c1) / (l2 - l1)
    base = c1 - slope * l1
    d = max(dyn)  # the long dynamic run, the honest comparator
    print(f"  fitted: static(L) = {base:.2f} + {slope:.6f}*L ms/step; "
          f"dynamic = {d:.2f} ms/step flat")
    if slope > 0:
        print(f"  break-even allocation: L = {(d - base) / slope:.0f} slots "
              f"(above this, static loses to dynamic)")
    dyn_s = new * d / 1e3
    ideal = sum(base + slope * (prefix + t) for t in range(new)) / 1e3
    print(f"  on {prefix}+{new} tokens: dynamic {dyn_s:.1f}s, "
          f"static at perfect tight allocation {ideal:.1f}s "
          f"-> {dyn_s / ideal:.2f}x (an unachievable ceiling)")
    t, total = prefix, 0.0
    for L in ladder:
        room = min(L - t, prefix + new - t)
        if room <= 0:
            continue
        total += room * (base + slope * L)
        t += room
    if total:
        print(f"  with the {list(ladder)} ladder: {total / 1e3:.1f}s "
              f"-> {dyn_s / (total / 1e3):.2f}x")
    # Graph capture is per distinct cache shape, so a finer ladder trades the
    # allocation saving for recompilation.
    cap = [r for r in by.values()
           if r.get("cache") == "static" and r.get("wall_s") and r.get("tokens")
           and r.get("step_ms_median")]
    if cap:
        r = cap[0]
        cost = r["wall_s"] - r["tokens"] * r["step_ms_median"] / 1e3
        print(f"  compile/capture per distinct shape: ~{cost:.0f}s, so a "
              f"{len(ladder)}-rung ladder pays ~{len(ladder) * cost:.0f}s "
              f"against a best-case {dyn_s - ideal:.1f}s saving per item")
    bw = 160.0 * 1024 / 4.8e12 * 1e3
    print(f"  per-slot overhead: {slope:.6f} ms vs {bw:.7f} ms required by "
          f"bandwidth -> {slope / bw:.0f}x. Removing THAT, not growing the "
          f"allocation, is where the {d / base:.1f}x would come from.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--prefix_len", type=int, default=900)
    ap.add_argument("--short", type=int, default=400,
                    help="token count for the allocation-isolating configs")
    ap.add_argument("--long", type=int, default=7000,
                    help="token count for the config that fills its allocation")
    ap.add_argument("--out", default="artifacts/diag/static_mechanism.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    bw = bandwidth_tbs()
    hdr = {
        "device": torch.cuda.get_device_name(0),
        "bandwidth_tbs": bw,
        "kv_kb_per_token": KV_KB_PER_TOKEN,
        "attn": args.attn,
        "torch": torch.__version__,
    }
    print(json.dumps(hdr, indent=2), flush=True)
    if bw:
        for L in (1024, 9216, 16384):
            ms = 1e3 * (L * KV_KB_PER_TOKEN * 1024) / (bw * 1e12)
            print(f"# predicted KV read at {L:>5} slots: {ms:.3f} ms/step", flush=True)

    P, S, Lg = args.prefix_len, args.short, args.long
    # Chosen so allocation and true length are varied independently:
    #   - dynamic short/long give the reference and show dynamic's own length scaling
    #   - static 1024 short:  allocation tight,  true length tight
    #   - static 9216 short:  allocation large,  true length tight   <- the "trap"
    #   - static 9216 long:   allocation large,  true length large   <- never measured
    configs = [
        ("dynamic, short", None, S),
        ("static 1024, short", 1024, S),
        ("static 9216, short", 9216, S),
        ("static 16384, short", 16384, S),
        ("dynamic, long", None, Lg),
        ("static 9216, long (fills its allocation)", 9216, Lg),
    ]

    model, tok = load(args.model, args.attn)
    rows: List[Dict] = []
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def save() -> None:
        with open(args.out, "w") as fh:
            json.dump({"header": hdr, "runs": rows}, fh, indent=2)

    for name, L, n in configs:
        need = P + n + 8
        if L is not None and L < need:
            print(f"\n## {name}: SKIPPED (max_cache_len {L} < prefix+new {need})",
                  flush=True)
            continue
        print(f"\n## {name}", flush=True)
        try:
            r = run_one(model, tok, P, n, L)
            r["name"] = name
            rows.append(r)
            print(f"   {r['tokens']} tok in {r['wall_s']}s = "
                  f"{r['tok_per_s_wall']} tok/s", flush=True)
            print(f"   steady-state step: median {r['step_ms_median']} ms "
                  f"(p10 {r['step_ms_p10']}, p90 {r['step_ms_p90']})", flush=True)
            print(f"   first 8 steps: {r['step_ms_first8']}", flush=True)
            print(f"   deciles: {r['step_ms_deciles']}", flush=True)
            print(f"   ramp (last decile / first): {r['ramp_ratio']}", flush=True)
            if r["compile_delta"]:
                print(f"   compile counters: {r['compile_delta']}", flush=True)
        except Exception as exc:
            rows.append({"name": name, "max_cache_len": L, "tokens": n,
                         "error": f"{type(exc).__name__}: {exc}"})
            print(f"   FAILED: {type(exc).__name__}: {exc}", flush=True)
        finally:
            # Flushed per config: inductor can take the interpreter down with no
            # traceback, and losing the whole file to the last config is worse.
            save()
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 68)
    print("VERDICT")
    by = {r.get("name"): r for r in rows if "error" not in r}
    trap = by.get("static 9216, short")
    fills = by.get("static 9216, long (fills its allocation)")
    tight = by.get("static 1024, short")
    dyn_s = by.get("dynamic, short")
    if trap and fills:
        print(f"  static@9216 with 400 real tokens:  "
              f"{trap['step_ms_median']} ms/step, ramp {trap['ramp_ratio']}")
        print(f"  static@9216 filling its allocation: "
              f"{fills['step_ms_median']} ms/step, ramp {fills['ramp_ratio']}")
        if (trap["ramp_ratio"] or 1) < 1.25 and (fills["ramp_ratio"] or 1) < 1.25:
            print("  => cost tracks ALLOCATION, not true length (both flat).")
            price_ladder(by)
        elif (fills["ramp_ratio"] or 1) >= 1.25:
            print("  => cost tracks TRUE LENGTH (the long run ramps). A large "
                  "allocation is NOT itself the problem, and the ladder buys "
                  "little; find the real per-step cost.")
    if tight and dyn_s:
        g = dyn_s["step_ms_median"] / tight["step_ms_median"]
        print(f"  tight static vs dynamic, same work: {g:.2f}x "
              f"({dyn_s['step_ms_median']} -> {tight['step_ms_median']} ms/step)")
        print("  (this is the CUDA-graph gain, and the ceiling on any ladder)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
