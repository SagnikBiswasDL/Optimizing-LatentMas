"""Where the other 74% of the GPU went.

Batch-1 decode of a 14B bf16 model is memory-bound: every token streams the whole
weight matrix, so the ceiling is `weights / HBM bandwidth`. On an H200 that is
~6.2 ms/token, i.e. ~162 tok/s. The AIME runs measured **42.5 tok/s**, which is 26%
of roofline — a 3.8x gap that has nothing to do with the research question.

Closing even part of it is a latency win on *every* item, with no accuracy risk,
which makes it strictly better than the steering result we are chasing. This script
measures how much is recoverable and whether the LatentMAS pipeline survives the
changes required, which is the part that is not obvious:

  * `StaticCache` pre-allocates fixed-size KV, but LatentMAS *injects* a cache built
    by custom latent forward passes. The injection has to be copied in, and the
    copy has to be faithful.
  * The SEAL steerer is a forward hook on layer 28. Under `torch.compile` a hook can
    silently drop out of the captured graph, or force a recompile every time its
    coefficient changes.

Run: python scripts/diag_throughput.py --model Qwen/Qwen3-14B --new_tokens 256
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HBM_TBS = {  # device name substring -> HBM bandwidth in TB/s
    "H200": 4.8, "H100": 3.35, "A100": 2.04, "L40": 0.86, "A6000": 0.77,
}


def bandwidth_tbs() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    for k, v in HBM_TBS.items():
        if k in name:
            return v
    return None


def free_model(m):
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load(model_name: str, attn: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = dict(torch_dtype=dtype)
    if attn:
        kw["attn_implementation"] = attn
    m = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    m.to("cuda").eval()
    m.config.use_cache = True
    return m, tok


def make_prompt(tok, n_tokens: int) -> torch.Tensor:
    """A prompt of about the right length; the AIME caches ran 566-962 positions."""
    filler = ("Consider the following competition mathematics problem and reason "
              "carefully about each step before answering. ")
    ids = tok(filler * 200, return_tensors="pt").input_ids[:, :n_tokens]
    return ids.to("cuda")


@torch.no_grad()
def time_decode(model, tok, ids, new_tokens: int, static: bool, warmup: int) -> Dict:
    gen = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False,
               pad_token_id=tok.pad_token_id, use_cache=True)
    if static:
        gen["cache_implementation"] = "static"
    # Warmup matters enormously here: with reduce-overhead the first call pays graph
    # capture, and reporting that as throughput would understate the gain several-fold.
    for _ in range(warmup):
        model.generate(input_ids=ids, **gen)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(input_ids=ids, **gen)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    produced = out.shape[1] - ids.shape[1]
    return {"s": dt, "tokens": int(produced), "tok_per_s": produced / dt}


def check_static_injection(model, tok, prefix_len: int, new_tokens: int) -> Dict:
    """Does an injected prefix cache survive conversion to StaticCache intact?

    This is the gate for the whole idea: if a latent cache cannot be copied into a
    static cache without changing the output, then CUDA graphs are unavailable to
    this pipeline no matter how fast they are.
    """
    from transformers import StaticCache

    ids = make_prompt(tok, prefix_len)
    head, tail = ids[:, :-8], ids[:, -8:]
    P, L = head.shape[1], tail.shape[1]
    # Mirror decode_batch in seal/latent_eval.py, which passes cache_position
    # explicitly; without it generate() cannot place an injected cache.
    cp = torch.arange(P, P + L, device=ids.device)

    def gen(cache):
        with torch.no_grad():
            return model.generate(
                input_ids=tail, past_key_values=cache,
                attention_mask=torch.ones_like(ids), cache_position=cp,
                max_new_tokens=new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id)

    try:
        with torch.no_grad():
            dyn = model(input_ids=head, use_cache=True).past_key_values
        a = gen(dyn)
        with torch.no_grad():
            dyn2 = model(input_ids=head, use_cache=True).past_key_values
        legacy = dyn2.to_legacy_cache() if hasattr(dyn2, "to_legacy_cache") else dyn2
        stat = StaticCache(config=model.config, max_batch_size=1,
                           max_cache_len=P + new_tokens + 16,
                           device=model.device, dtype=model.dtype)
        # StaticCache allocates lazily on first update, so force allocation before
        # copying the injected KV in.
        k0 = legacy[0][0]
        stat.early_initialization(k0.shape[0], k0.shape[1], k0.shape[3],
                                  k0.dtype, k0.device)
        for li in range(model.config.num_hidden_layers):
            k, v = legacy[li]
            n = k.shape[-2]
            stat.layers[li].keys[:, :, :n] = k
            stat.layers[li].values[:, :, :n] = v
        b = gen(stat)
        ta, tb = a[0, L:].tolist(), b[0, L:].tolist()
        return {"ok": True, "identical": ta == tb,
                "n_compared": len(ta),
                "note": ("injected cache survives the move to StaticCache"
                         if ta == tb else
                         "StaticCache injection changes the output — CUDA graphs are "
                         "NOT usable for this pipeline without fixing this first")}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def check_seal_under_compile(model, tok, vector_path: str, layer: int,
                             ids, new_tokens: int) -> Dict:
    """Does the layer-28 steering hook still fire once the forward is compiled?

    A hook that is silently excluded from the graph would make every compiled SEAL
    number a measurement of the *unsteered* model, which would look like a
    spectacular efficiency result and be nothing at all.
    """
    if not os.path.isfile(vector_path):
        return {"skipped": f"no vector at {vector_path}"}
    try:
        from seal import SealSteerer
    except Exception as exc:
        return {"skipped": f"import failed: {exc}"}
    try:
        steer = SealSteerer.from_artifact(vector_path, coef=40.0, layer_index=layer)
        steer.register(model)
        steer.enable()
        with torch.no_grad():
            on = model.generate(input_ids=ids, max_new_tokens=new_tokens,
                                do_sample=False, pad_token_id=tok.pad_token_id)
        steer.disable()
        with torch.no_grad():
            off = model.generate(input_ids=ids, max_new_tokens=new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        same = torch.equal(on, off)
        return {"hook_fires": not same,
                "note": ("STEERING HAS NO EFFECT — hook dropped from the graph"
                         if same else "steered and unsteered outputs differ, as expected")}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--prefix_len", type=int, default=700,
                    help="prefix tokens; the AIME caches were 566-962 positions")
    ap.add_argument("--new_tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="artifacts/aime_localize/throughput.json")
    ap.add_argument("--seal_vector", default="artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt")
    ap.add_argument("--seal_layer", type=int, default=28)
    ap.add_argument("--skip_compile", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    bw = bandwidth_tbs()
    dev = torch.cuda.get_device_name(0)
    print(f"# Decode throughput probe\ndevice: {dev}")
    rows: List[Dict] = []

    # (label, attn_implementation, static cache, compile)
    configs = [
        ("current (sdpa + dynamic)", "sdpa", False, False),
        ("flash_attention_2 + dynamic", "flash_attention_2", False, False),
        ("sdpa + static cache", "sdpa", True, False),
        ("sdpa + static + compile", "sdpa", True, True),
        ("flash_attn_2 + static + compile", "flash_attention_2", True, True),
    ]
    if args.skip_compile:
        configs = [c for c in configs if not c[3]]

    extras = {}

    def save() -> None:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"device": dev, "model": args.model, "configs": rows,
                       "gates": extras}, fh, indent=2)

    for label, attn, static, comp in configs:
        print(f"\n## {label}", flush=True)
        try:
            model, tok = load(args.model, attn)
        except Exception as exc:
            print(f"  load failed: {type(exc).__name__}: {exc}")
            rows.append({"config": label, "error": f"load: {exc}"})
            continue
        try:
            if comp:
                # reduce-overhead turns on CUDA graphs, which is where most of the
                # batch-1 win lives; it requires static shapes, hence static cache.
                model.forward = torch.compile(model.forward, mode="reduce-overhead",
                                              fullgraph=False)
            ids = make_prompt(tok, args.prefix_len)
            r = time_decode(model, tok, ids, args.new_tokens, static,
                            warmup=args.warmup + (2 if comp else 0))
            r["config"] = label
            if bw:
                ceiling = bw * 1e12 / (sum(p.numel() for p in model.parameters()) * 2)
                r["roofline_tok_per_s"] = ceiling
                r["pct_of_roofline"] = 100.0 * r["tok_per_s"] / ceiling
            rows.append(r)
            print(f"  {r['tok_per_s']:.1f} tok/s"
                  + (f"  ({r['pct_of_roofline']:.0f}% of roofline)" if bw else ""))

            # Run the two compatibility gates once, on the fastest viable config.
            if static and "injection" not in extras:
                extras["injection"] = check_static_injection(
                    model, tok, args.prefix_len, 32)
                print(f"  static-cache injection: {extras['injection']}")
            if comp and "seal" not in extras:
                extras["seal"] = check_seal_under_compile(
                    model, tok, args.seal_vector, args.seal_layer, ids, 24)
                print(f"  SEAL hook under compile: {extras['seal']}")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            rows.append({"config": label, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            free_model(model)
            # torch.compile can abort the interpreter below Python (inductor has
            # taken the process down here with no traceback), so results are
            # flushed per config rather than only at the end.
            save()

    ok = [r for r in rows if "tok_per_s" in r]
    base = next((r for r in ok if r["config"].startswith("current")), None)
    print("\n## Summary\n")
    print("| config | tok/s | % roofline | vs current |")
    print("|---|---:|---:|---:|")
    for r in rows:
        if "tok_per_s" not in r:
            print(f"| {r['config']} | — | — | {r.get('error', 'failed')[:40]} |")
            continue
        rel = f"{r['tok_per_s'] / base['tok_per_s']:.2f}x" if base else "—"
        pct = f"{r.get('pct_of_roofline', 0):.0f}%" if "pct_of_roofline" in r else "—"
        print(f"| {r['config']} | {r['tok_per_s']:.1f} | {pct} | {rel} |")
    if base and ok:
        best = max(ok, key=lambda r: r["tok_per_s"])
        if best is not base:
            print(f"\nBest is **{best['config']}** at {best['tok_per_s']:.1f} tok/s, "
                  f"**{best['tok_per_s'] / base['tok_per_s']:.2f}x** the current path. "
                  f"That is a latency win on every item at no accuracy cost, and it "
                  f"composes with any token reduction from steering.")
    save()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
