"""Why StaticCache lost: decode rate as a function of allocated cache length.

`scripts/diag_throughput.py` measured StaticCache at 1.70x the dynamic path and
that number did not survive contact with the pipeline, where the same path ran
at 0.56x. The probe generated 256 tokens from a 700-token prefix, so it
allocated `max_cache_len ~= 964`; the real Judger allocates
`prefix + judger_budget ~= 8970` because the budget is 8192.

StaticCache attends over every allocated slot on every step, masked, rather than
over the slots actually filled. So its cost scales with the allocation while the
dynamic cache's scales with the true length. Any probe that allocates a short
cache measures a speedup that vanishes at the length that matters.

This sweeps the allocation with everything else fixed, which separates "static
vs dynamic" from "short cache vs long cache".

Usage:
  python scripts/diag_static_cache_len.py --lens 1024,2048,4096,9216,16384
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, StaticCache


def build_static(model, max_cache_len: int, device, dtype) -> StaticCache:
    cache = StaticCache(config=model.config, max_batch_size=1,
                        max_cache_len=int(max_cache_len), device=device, dtype=dtype)
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads)
    # StaticCache allocates lazily, so the tensors do not exist until this call.
    cache.early_initialization(1, model.config.num_key_value_heads, head_dim,
                               dtype, device)
    return cache


@torch.no_grad()
def decode_rate(model, ids: torch.Tensor, new_tokens: int, cache) -> float:
    """Tokens per second for a hand-rolled greedy loop, prefill excluded."""
    device = ids.device
    P = ids.shape[1]
    out = model(input_ids=ids, past_key_values=cache,
                cache_position=torch.arange(P, device=device), use_cache=True)
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(new_tokens):
        out = model(input_ids=nxt, past_key_values=cache,
                    cache_position=torch.tensor([P + i], device=device), use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    return new_tokens / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--prefix_len", type=int, default=770)
    ap.add_argument("--new_tokens", type=int, default=64)
    ap.add_argument("--lens", default="1024,2048,4096,9216,16384")
    ap.add_argument("--out", default="artifacts/aime_localize/static_cache_len.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device, dtype = torch.device("cuda:0"), torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cuda:0", attn_implementation="sdpa").eval()

    ids = torch.randint(1000, 5000, (1, args.prefix_len), device=device)
    print(f"# StaticCache cost vs allocated length\ndevice: "
          f"{torch.cuda.get_device_name(0)}\nprefix: {args.prefix_len} tokens, "
          f"decoding {args.new_tokens}\n", flush=True)

    # Warm the kernels so the first measured config is not paying autotune.
    decode_rate(model, ids, 8, DynamicCache())

    rows: List[Dict] = []
    dyn = decode_rate(model, ids, args.new_tokens, DynamicCache())
    rows.append({"cache": "dynamic", "max_cache_len": None, "tok_per_s": dyn})
    print(f"dynamic (grows to true length)      {dyn:6.1f} tok/s   1.00x", flush=True)

    for L in [int(x) for x in args.lens.split(",") if x]:
        if L < args.prefix_len + args.new_tokens + 2:
            print(f"static  max_cache_len={L:<6} skipped (shorter than prefix+decode)")
            continue
        rate = decode_rate(model, ids, args.new_tokens,
                           build_static(model, L, device, dtype))
        rows.append({"cache": "static", "max_cache_len": L, "tok_per_s": rate})
        print(f"static  max_cache_len={L:<6}      {rate:6.1f} tok/s   "
              f"{rate / dyn:.2f}x", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"device": torch.cuda.get_device_name(0), "model": args.model,
                   "prefix_len": args.prefix_len, "new_tokens": args.new_tokens,
                   "rows": rows}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
