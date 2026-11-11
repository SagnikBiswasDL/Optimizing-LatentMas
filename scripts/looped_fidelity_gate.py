#!/usr/bin/env python
"""Backend fidelity gate: does the pinned engine compute what the evidence assumed?

Compares the pinned, unmodified vllm-rlt engine with the Hugging Face reference
harness that produced the saved Stage-0 pools. Needs a CUDA GPU.

Sections (each writes one key of ``fidelity.json``):

``depth``      Greedy generations from the engine at each depth, scored position
               by position against the reference under teacher forcing, for two
               reference semantics: uniform depth (prompt and decode at ``r``)
               and full-depth prompt (prompt at 4, decode at ``r``). Arms:
               ``stock``, ``boundary_shift``, ``shared_kv`` and a depth-limited
               engine (``total_ut_steps=r``).
``sequence``   Token-for-token comparison with reference greedy generation.
``batching``   Same request alone, in a mixed-depth batch, and in a different
               submission order: greedy and seeded sampling.
``cancel``     Abort requests mid-decode and before the first step; KV blocks
               must be released at once and surviving requests must not change.
``kv``         KV blocks one request reserves at each depth and layout.

Teacher forcing is used because two correct implementations in different
arithmetic stop producing the same greedy sequence after the first near-tie; a
mismatch is only evidence of a semantic difference if the reference margin
between its own choice and the engine's token is large.

    python scripts/looped_fidelity_gate.py --model <snapshot dir> \
        --items artifacts/looped_serving/splits/math_train_smoke.items.jsonl \
        --out artifacts/looped_serving/fidelity_gate
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped import experiment_manifest as em  # noqa: E402
from looped import prompts as P  # noqa: E402
from looped.ouro_adapter import OURO_BASE_ID, OURO_BASE_REVISION, OuroRunner  # noqa: E402
from serving.looped.backend import GenerationRequest, run_all  # noqa: E402
from serving.looped.reference_backend import ReferenceBackend, agreement  # noqa: E402
from serving.looped.rlt_backend import ENGINE_SHA, RltBackend  # noqa: E402

GB = 1 << 30


def free(*objs) -> None:
    for o in objs:
        try:
            o.close()
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()


def engine(args, dtype, **kw) -> RltBackend:
    kw.setdefault("kv_cache_memory_bytes", args.kv_gb * GB)
    kw.setdefault("max_num_seqs", 64)
    return RltBackend(args.model, stop_ids=args.stop_ids, dtype=dtype, attention_backend=args.attention, **kw)


def summarize(rows) -> dict:
    """Aggregate per-request agreement dicts."""
    pos = sum(r["positions"] for r in rows)
    miss = sum(r["mismatches"] for r in rows)
    return {
        "requests": len(rows), "positions": pos, "mismatches": miss,
        "mismatch_rate": miss / pos if pos else None,
        "first_token_matches": sum(r["first_token_match"] for r in rows),
        "max_first_token_margin": max(r["first_token_margin"] for r in rows),
        "max_mismatch_margin": max(r["max_mismatch_margin"] for r in rows),
        "requests_with_any_mismatch": sum(r["mismatches"] > 0 for r in rows),
        "median_top2_gap": sorted(r["median_top2_gap"] for r in rows)[len(rows) // 2],
    }


# ---------------------------------------------------------------------------- depth

def depth_section(args, ref: ReferenceBackend, prompts, dtype: str) -> dict:
    full = ref.runner.max_depth
    arms = {
        "stock": dict(prompt_mode="stock"),
        "boundary_shift": dict(prompt_mode="boundary_shift"),
        "shared_kv": dict(prompt_mode="boundary_shift", kv_layout="shared"),
    }
    for d in args.limited_depths:
        arms[f"depth_limited_engine_{d}"] = dict(prompt_mode="stock", total_ut_steps=d)
    out = {}
    for arm, kw in arms.items():
        depths = [kw["total_ut_steps"]] if "total_ut_steps" in kw else args.depths
        try:
            be = engine(args, dtype, **kw)
            reqs = [GenerationRequest(f"p{i}:d{d}", tuple(p), d, args.cap) for d in depths for i, p in enumerate(prompts)]
            t0 = time.perf_counter()
            done = run_all(be, reqs)
            wall = time.perf_counter() - t0
            leftover = be.stats()["used_blocks"]
            sem = dict(be.semantics)
            free(be)
            be = None
            free()
            per_depth = {}
            for d in depths:
                uni, hyb, errors, toks = [], [], 0, 0
                for i, p in enumerate(prompts):
                    c = done[f"p{i}:d{d}"]
                    if c.finish_reason == "error":
                        errors += 1
                        continue
                    toks += len(c.token_ids)
                    uni.append(agreement(ref.teacher_forced(p, c.token_ids, d), c.token_ids))
                    hyb.append(agreement(ref.teacher_forced(p, c.token_ids, d, prompt_depth=full), c.token_ids))
                per_depth[str(d)] = {"errors": errors, "generated_tokens": toks,
                                     "vs_uniform_depth": summarize(uni), "vs_full_depth_prompt": summarize(hyb)}
            out[arm] = {"semantics": sem, "wall_s": round(wall, 2), "used_blocks_after": leftover, "depths": per_depth}
        except Exception:
            out[arm] = {"error": traceback.format_exc()}
            be = None
            free()
        print(f"[depth/{dtype}] {arm}: " + json.dumps(
            {d: {"uni": (v["vs_uniform_depth"]["mismatches"], round(v["vs_uniform_depth"]["max_mismatch_margin"], 4)),
                 "full": (v["vs_full_depth_prompt"]["mismatches"], round(v["vs_full_depth_prompt"]["max_mismatch_margin"], 4)),
                 "err": v["errors"]}
             for d, v in out[arm].get("depths", {}).items()} or out[arm]), flush=True)
    return out


# ---------------------------------------------------------------------------- sequence

def sequence_section(args, ref: ReferenceBackend, prompts, dtype: str) -> dict:
    prompts = prompts[: args.n_sequence]
    reqs = [GenerationRequest(f"p{i}:d{d}", tuple(p), d, args.cap) for d in args.depths for i, p in enumerate(prompts)]
    want = run_all(ref, reqs)
    be = engine(args, dtype, prompt_mode="boundary_shift")
    got = run_all(be, reqs)
    free(be)
    be = None
    free()
    rows = []
    for r in reqs:
        a, b = want[r.request_id].token_ids, got[r.request_id].token_ids
        k = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        rows.append({"request": r.request_id, "reference_tokens": len(a), "engine_tokens": len(b),
                     "common_prefix": k, "identical": a == b,
                     "same_finish": want[r.request_id].finish_reason == got[r.request_id].finish_reason})
    out = {"requests": len(rows), "identical": sum(r["identical"] for r in rows), "rows": rows}
    print(f"[sequence/{dtype}] identical {out['identical']}/{out['requests']}", flush=True)
    return out


# ---------------------------------------------------------------------------- batching

def sampled(prompts, tag, depths, cap, temperature=0.6, top_p=0.95):
    reqs = []
    for i, p in enumerate(prompts):
        d = depths[i % len(depths)]
        cid = em.candidate_id(f"{tag}{i}", d, 0)
        reqs.append(GenerationRequest(cid, tuple(p), d, cap, temperature, top_p, em.candidate_seed(0, f"{tag}{i}", cid)))
    return reqs


def batching_section(args, prompts, dtype: str) -> dict:
    n = args.n_batch
    main, other = prompts[:n], prompts[n: 2 * n]
    out = {}
    for label, temp in (("greedy", 0.0), ("sampled", 0.6)):
        reqs = sampled(main, "a", args.depths, args.cap, temp, 0.95 if temp else 1.0)
        extra = sampled(other, "b", args.depths, args.cap, temp, 0.95 if temp else 1.0)
        be = engine(args, dtype, prompt_mode="boundary_shift")
        alone = {}
        for r in reqs:                                   # one request at a time
            alone.update(run_all(be, [r]))
        batch = run_all(be, reqs)                        # all together, mixed depths
        mixed = run_all(be, list(reversed(extra + reqs)))  # other neighbours, other order
        free(be)
        be = None
        free()
        be2 = engine(args, dtype, prompt_mode="boundary_shift")  # a fresh engine
        again = run_all(be2, reqs)
        free(be2)
        be2 = None
        free()
        out[label] = {
            "requests": len(reqs),
            "alone_equals_batch": sum(alone[r.request_id].token_ids == batch[r.request_id].token_ids for r in reqs),
            "batch_equals_reordered_with_neighbours": sum(
                batch[r.request_id].token_ids == mixed[r.request_id].token_ids for r in reqs),
            "batch_equals_fresh_engine": sum(batch[r.request_id].token_ids == again[r.request_id].token_ids for r in reqs),
            "mean_tokens": sum(len(batch[r.request_id].token_ids) for r in reqs) / len(reqs),
        }
        print(f"[batching/{dtype}] {label}: {out[label]}", flush=True)
    return out


# ---------------------------------------------------------------------------- cancel

def cancel_section(args, prompts, dtype: str) -> dict:
    n = args.n_batch
    keep = sampled(prompts[:n], "keep", args.depths, args.cap)
    victims = sampled(prompts[n: 2 * n], "victim", args.depths, args.cap)
    be = engine(args, dtype, prompt_mode="boundary_shift")
    control = run_all(be, keep)
    assert be.stats()["used_blocks"] == 0

    # cancel before the engine ever stepped
    be.submit(victims[0])
    before = be.stats()["used_blocks"]
    c0 = be.cancel(victims[0].request_id)
    early = {"finish_reason": c0.finish_reason, "tokens": len(c0.token_ids),
             "used_blocks_before": before, "used_blocks_after": be.stats()["used_blocks"]}

    for r in keep + victims:
        be.submit(r)
    done, live = {}, {v.request_id for v in victims}
    while live and not all(len(be.partial(v)) >= args.cancel_after for v in live):
        for c in be.step():
            done[c.request_id] = c
            live.discard(c.request_id)
    blocks, times, tokens_at_cancel = [be.stats()["used_blocks"]], [], []
    for v in sorted(live):
        tokens_at_cancel.append(len(be.partial(v)))
        t0 = time.perf_counter()
        done[v] = be.cancel(v)
        times.append(time.perf_counter() - t0)
        blocks.append(be.stats()["used_blocks"])
    steps_after = 0
    while be.active():
        steps_after += 1
        for c in be.step():
            done[c.request_id] = c
    out = {
        "cancel_before_first_step": early,
        "cancelled_mid_decode": len(times),
        "victims_finished_before_cancel": len(victims) - len(times),
        "tokens_at_cancel": tokens_at_cancel,
        "used_blocks_trace": blocks,
        "blocks_strictly_decrease": all(a > b for a, b in zip(blocks, blocks[1:])),
        "cancel_call_ms_max": max(times) * 1e3 if times else None,
        "cancel_call_ms_mean": sum(times) / len(times) * 1e3 if times else None,
        "cancelled_tokens_frozen": all(len(done[v].token_ids) == t for v, t in zip(sorted(live), tokens_at_cancel)),
        "survivors": len(keep),
        "survivors_identical_to_control": sum(done[r.request_id].token_ids == control[r.request_id].token_ids for r in keep),
        "used_blocks_after_all": be.stats()["used_blocks"],
        "engine_requests_after_all": be.stats()["engine_requests"],
    }
    free(be)
    be = None
    free()
    print(f"[cancel/{dtype}] {json.dumps({k: v for k, v in out.items() if k != 'used_blocks_trace'})}", flush=True)
    return out


# ---------------------------------------------------------------------------- kv

def kv_section(args, prompts, dtype: str) -> dict:
    p = prompts[0]
    out = {}
    arms = {"last_exited": dict(), "shared": dict(kv_layout="shared"),
            "last_exited_incremental": dict(incremental_allocation=True),
            "depth_limited_engine_2": dict(total_ut_steps=2)}
    for arm, kw in arms.items():
        try:
            be = engine(args, dtype, prompt_mode="stock", **kw)
            cache = be.engine.cache_manager
            cfg = be.engine.model.config
            # The manager counts one block per stored depth plane.
            per_block = (cache.block_size * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * 2
                         * torch.empty((), dtype=getattr(torch, dtype)).element_size())
            row = {"storage_depths": int(cache.storage_depths), "block_size": int(cache.block_size),
                   "num_blocks": int(cache.num_blocks), "bytes_per_block": int(per_block), "by_depth": {}}
            for d in ([2] if "total_ut_steps" in kw else [2, 4]):
                be.submit(GenerationRequest(f"kv:d{d}", tuple(p), d, args.kv_cap))
                be.step()
                at_admission = be.stats()["used_blocks"]
                while be.active() and len(be.partial(f"kv:d{d}")) < 64:
                    be.step()
                after_64 = be.stats()["used_blocks"]
                if be.active():
                    be.cancel(f"kv:d{d}")
                row["by_depth"][str(d)] = {
                    "prompt_tokens": len(p), "max_new_tokens": args.kv_cap,
                    "used_blocks_after_first_step": at_admission, "used_blocks_after_64_tokens": after_64,
                    "reserved_mb_after_first_step": round(at_admission * per_block / 2 ** 20, 1),
                    "used_blocks_after_cancel": be.stats()["used_blocks"],
                }
            free(be)
            be = None
            free()
            out[arm] = row
        except Exception:
            out[arm] = {"error": traceback.format_exc()}
            be = None
            free()
        print(f"[kv/{dtype}] {arm}: {json.dumps(out[arm])[:600]}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local snapshot directory of the pinned checkpoint")
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", default="artifacts/looped_serving/fidelity_gate")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--depths", default="1,2,3,4")
    ap.add_argument("--limited_depths", default="2,3")
    ap.add_argument("--n", type=int, default=16, help="prompts in the depth section")
    ap.add_argument("--n_sequence", type=int, default=4)
    ap.add_argument("--n_batch", type=int, default=8)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--kv_cap", type=int, default=2048)
    ap.add_argument("--cancel_after", type=int, default=48)
    ap.add_argument("--kv_gb", type=int, default=40)
    ap.add_argument("--attention", default="triton")
    ap.add_argument("--sections", default="depth,sequence,batching,cancel,kv")
    args = ap.parse_args()
    args.depths = [int(x) for x in args.depths.split(",")]
    args.limited_depths = [int(x) for x in args.limited_depths.split(",") if x]
    sections = args.sections.split(",")
    os.makedirs(args.out, exist_ok=True)

    items = [json.loads(line) for line in open(args.items)]
    result = {
        "engine_sha": ENGINE_SHA, "model": OURO_BASE_ID, "model_revision": OURO_BASE_REVISION,
        "items_sha256": em.file_sha256(args.items), "args": {k: v for k, v in vars(args).items()},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "dtypes": {},
    }
    for dtype in args.dtypes.split(","):
        runner = OuroRunner(OURO_BASE_ID, dtype=dtype)
        args.stop_ids = list(runner.stop_ids)
        prompts = [runner.chat_ids(P.msgs(P.solver(it["text"])), think=runner.think) for it in items]
        result["prompt_last_token"] = sorted({p[-1] for p in prompts})
        result["stop_ids"] = args.stop_ids
        ref = ReferenceBackend(runner)
        block = {}
        fns = {
            "depth": lambda: depth_section(args, ref, prompts[: args.n], dtype),
            "sequence": lambda: sequence_section(args, ref, prompts, dtype),
            "batching": lambda: batching_section(args, prompts, dtype),
            "cancel": lambda: cancel_section(args, prompts, dtype),
            "kv": lambda: kv_section(args, prompts, dtype),
        }
        for name in sections:
            t0 = time.perf_counter()
            try:
                block[name] = fns[name]()
            except Exception:
                block[name] = {"error": traceback.format_exc()}
                print(f"[{name}/{dtype}] FAILED\n{block[name]['error']}", flush=True)
                free()
            block.setdefault("_wall_s", {})[name] = round(time.perf_counter() - t0, 1)
        result["dtypes"][dtype] = block
        with open(os.path.join(args.out, "fidelity.json"), "w") as f:
            json.dump(result, f, indent=1)
        del ref, runner
        free()
    print("wrote", os.path.join(args.out, "fidelity.json"))


if __name__ == "__main__":
    main()
