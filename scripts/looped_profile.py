#!/usr/bin/env python
"""Closed-loop latency/throughput profile of fixed-depth, fixed-width arms.

Each arm ``<depth>x<width>`` answers every question of a frozen split with
``width`` candidates at one depth (width 1 is the greedy candidate; wider arms
are seeded samples ``s0..s{width-1}``). Questions are admitted first come, first
served whenever ``width`` of the ``--slots`` candidate slots are free, so every
arm runs at the same candidate concurrency on the same engine. A question's
latency is the time from its admission to the finish of its last candidate.

This script never sees gold answers. Grade with ``scripts/grade_looped_profile.py``.

A candidate's seed depends only on (run seed, question id, candidate id), so
sample ``s0..s3`` of a width-8 arm are the candidates of the width-4 arm, and
``--extend_from`` can rerun only the capped candidates of a finished arm under a
larger ``--cap`` and check that the old tokens are reproduced as a prefix.

    python scripts/looped_profile.py --model <snapshot dir> \
        --items artifacts/looped_serving/splits/math_train_smoke.items.jsonl \
        --arms 3x1,3x4,3x8 --out artifacts/looped_serving/profile_smoke
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import statistics
import sys
import time
from collections import deque

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped import answer_identity as ai  # noqa: E402
from looped import experiment_manifest as em  # noqa: E402
from looped import prompts as P  # noqa: E402
from serving.looped.backend import GenerationRequest, run_all, strip_stop  # noqa: E402
from serving.looped.rlt_backend import ENGINE_SHA, RltBackend  # noqa: E402

GB = 1 << 30
MODEL_ID, MODEL_REV = "ByteDance/Ouro-1.4B", "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
FINISH = {"eos": "eos", "length_cap": "length_cap", "cancelled": "cancelled", "error": "error"}


def stop_ids_for(tok) -> list:
    """Same rule as ``OuroRunner._stop_ids``."""
    ids = {int(tok.eos_token_id)} if tok.eos_token_id is not None else set()
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return sorted(ids)


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs) -> dict:
    return {"mean": statistics.fmean(xs), "p50": pct(xs, .5), "p90": pct(xs, .9), "p95": pct(xs, .95),
            "max": max(xs)} if xs else {}


def candidates_for(qid: str, depth: int, width: int, args) -> list:
    """(candidate_id, sample, temperature, top_p, seed) for one question."""
    if width == 1:
        cid = em.candidate_id(qid, depth, -1)
        return [(cid, -1, 0.0, 1.0, em.candidate_seed(args.run_seed, qid, cid))]
    out = []
    for s in range(width):
        cid = em.candidate_id(qid, depth, s)
        out.append((cid, s, args.temperature, args.top_p, em.candidate_seed(args.run_seed, qid, cid)))
    return out


def manifest_for(args, be, arm: dict, prompt_hash: str, split_hash: str) -> em.RunManifest:
    return em.RunManifest(
        code_sha=args.code_sha, engine_sha=ENGINE_SHA, model_revision=f"{MODEL_ID}@{MODEL_REV}",
        tokenizer_revision=f"{MODEL_ID}@{MODEL_REV}", dataset_revision=args.dataset_revision,
        prompt_hash=prompt_hash, split_hash=split_hash,
        decode_config={**arm, "temperature": args.temperature, "top_p": args.top_p, "stop_ids": be.semantics["stop_ids"],
                       "slots": args.slots, "items_sha256": em.file_sha256(args.items), "limit": args.limit},
        cache_semantics=(f"prompt_mode={be.semantics['prompt_mode']}; prompt_depth={be.semantics['prompt_depth']}; "
                         f"kv_layout={be.semantics['kv_layout']} x{be.semantics['kv_storage_depths']} planes"),
        attention_backend=be.semantics["attention_backend"], dtype=be.semantics["dtype"],
        dependency_versions={"torch": torch.__version__, "transformers": md.version("transformers"),
                             "vllm-rlt": md.version("vllm-rlt"), "triton": md.version("triton")},
        hardware=torch.cuda.get_device_name(0), seed_scheme=em.SEED_SCHEME, run_seed=args.run_seed,
        identity_version=ai.STRUCTURAL, grader={"name": "none", "note": "graded offline"},
        extra={"engine": {k: be.semantics[k] for k in ("scheduler_mode", "max_num_seqs", "max_num_batched_tokens",
                                                       "kv_num_blocks", "execution", "exit_mode")},
               "driver": "closed loop, FIFO admission by free candidate slots"},
    )


def record(run_id, qid, cand, comp, text, t0) -> dict:
    cid, sample, _, _, seed = cand
    ans, status = ai.extract_final_answer(text)
    ident = ai.identify(ans) if ans is not None else None
    rel = lambda t: None if t is None else round(t - t0, 4)  # noqa: E731
    return {
        "run_id": run_id, "question_id": qid, "candidate_id": cid, "depth": comp.depth, "sample": sample, "seed": seed,
        "prompt_tokens": comp.prompt_tokens, "generated_token_ids": comp.token_ids, "gen_tokens": len(comp.token_ids),
        "finish_reason": FINISH[comp.finish_reason], "text": text, "pred": ans,
        "answer_key": ident.key if ident is not None else None,
        "parse_status": ident.status if ident is not None else status,
        "scalar_confidence_definition": None, "confidence_summary": None,
        "logical_loop_tokens": comp.logical_loop_tokens, "executed_loop_tokens": comp.charged_loop_tokens, "probe_work": 0,
        "submit_time": rel(comp.submit_time), "first_token_time": rel(comp.first_token_time),
        "finish_time": rel(comp.finish_time), "cancel_time": None, "retire_time": rel(comp.finish_time),
        "detail": comp.detail,
    }


def run_arm(be, tok, questions, arm: dict, args, out_dir: str, manifest: em.RunManifest, work=None) -> dict:
    """``work`` overrides the candidate list: [(qid, prompt, [candidate tuples])]."""
    depth, width, cap = arm["depth"], arm["width"], arm["cap"]
    if work is None:
        work = [(q["question_id"], q["prompt"], candidates_for(q["question_id"], depth, width, args)) for q in questions]
    pending = deque(work)
    stop = set(be.semantics["stop_ids"])
    live, q_open, q_rows, rows = {}, {}, [], []
    peak_blocks = 0
    torch.cuda.reset_peak_memory_stats()
    cand_path = os.path.join(out_dir, "candidates.jsonl")
    with open(cand_path, "w") as f:
        t0 = time.perf_counter()
        while pending or live:
            while pending and len(live) + len(pending[0][2]) <= args.slots:
                qid, prompt, cands = pending.popleft()
                q_open[qid] = {"arrival": time.perf_counter(), "left": len(cands), "candidates": [c[0] for c in cands]}
                for c in cands:
                    be.submit(GenerationRequest(c[0], tuple(prompt), depth, cap, c[2], c[3], c[4]))
                    live[c[0]] = (qid, c)
            if pending and not live:
                raise RuntimeError(f"a question needs {len(pending[0][2])} slots but only {args.slots} exist")
            for comp in be.step():
                qid, c = live.pop(comp.request_id)
                text = tok.decode(strip_stop(comp.token_ids, stop), skip_special_tokens=True)
                row = record(manifest.run_id, qid, c, comp, text, t0)
                em.check_record(row, em.CANDIDATE_RECORD_FIELDS)
                em.assert_no_gold(row)
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                q = q_open[qid]
                q["left"] -= 1
                if not q["left"]:
                    q_rows.append({"question_id": qid, "scheduled_arrival": round(q["arrival"] - t0, 4),
                                   "completion_time": round(comp.finish_time - t0, 4),
                                   "latency_s": round(comp.finish_time - q["arrival"], 4),
                                   "all_candidate_ids": q["candidates"]})
                    del q_open[qid]
            peak_blocks = max(peak_blocks, be.stats()["used_blocks"])
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    with open(os.path.join(out_dir, "questions.jsonl"), "w") as f:
        for r in q_rows:
            f.write(json.dumps(r) + "\n")
    toks = sum(r["gen_tokens"] for r in rows)
    reasons = {}
    for r in rows:
        reasons[r["finish_reason"]] = reasons.get(r["finish_reason"], 0) + 1
    summary = {
        "run_id": manifest.run_id, "arm": arm, "questions": len(q_rows), "candidates": len(rows), "slots": args.slots,
        "wall_s": round(wall, 2), "generated_tokens": toks, "tokens_per_s": toks / wall,
        "questions_per_min": 60 * len(q_rows) / wall,
        "logical_loop_tokens": sum(r["logical_loop_tokens"] for r in rows),
        "charged_loop_tokens": sum(r["executed_loop_tokens"] for r in rows),
        "charged_loop_tokens_per_s": sum(r["executed_loop_tokens"] for r in rows) / wall,
        "finish_reasons": reasons, "mean_gen_tokens": toks / len(rows),
        "question_latency_s": dist([r["latency_s"] for r in q_rows]),
        "candidate_latency_s": dist([r["finish_time"] - r["submit_time"] for r in rows]),
        "first_token_s": dist([r["first_token_time"] - r["submit_time"] for r in rows if r["first_token_time"] is not None]),
        "peak_used_blocks": peak_blocks, "num_blocks": be.stats()["num_blocks"],
        "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated() / GB,
        "used_blocks_after": be.stats()["used_blocks"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="", help="comma list of <depth>x<width>")
    ap.add_argument("--extend_from", default="", help="arm directory whose capped candidates are rerun at --cap")
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--slots", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--run_seed", type=int, default=0)
    ap.add_argument("--prompt_mode", default="boundary_shift")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attention", default="triton")
    ap.add_argument("--scheduler_mode", default="refill")
    ap.add_argument("--kv_layout", default="last_exited")
    ap.add_argument("--kv_gb", type=int, default=70)
    ap.add_argument("--max_num_batched_tokens", type=int, default=4096)
    ap.add_argument("--execution", default="{}", help="JSON for vllm_rlt.ExecutionConfig")
    ap.add_argument("--exit_mode", default="ouro")
    ap.add_argument("--code_sha", default=open("CODE_SHA").read().strip() if os.path.exists("CODE_SHA") else "unknown")
    ap.add_argument("--splits", default="configs/looped_serving/splits.json")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    items = [json.loads(line) for line in open(args.items)]
    if args.limit:
        items = items[: args.limit]
    for it in items:
        em.assert_no_gold(it)
        it["prompt"] = tok.apply_chat_template(P.msgs(P.solver(it["text"])), tokenize=True, add_generation_prompt=True)
    spec = json.load(open(args.splits))
    args.dataset_revision = f"{spec['datasets']['math_train']['repo']}@{spec['datasets']['math_train']['revision']}"
    prompt_hash = em.sha256_hex(em.canonical_json([P.solver("{question}"), tok.chat_template]))[:16]

    be = RltBackend(
        args.model, stop_ids=stop_ids_for(tok), prompt_mode=args.prompt_mode, dtype=args.dtype,
        attention_backend=args.attention, kv_layout=args.kv_layout, max_num_seqs=args.slots,
        max_num_batched_tokens=args.max_num_batched_tokens, scheduler_mode=args.scheduler_mode,
        kv_cache_memory_bytes=args.kv_gb * GB, execution=json.loads(args.execution), exit_mode=args.exit_mode,
    )
    # Warm every depth so kernel compilation is not charged to the first arm.
    warm = [GenerationRequest(f"warm:{d}:{i}", tuple(items[i % len(items)]["prompt"]), d, 48, 0.6, 0.95, i)
            for d in range(1, be.full_depth + 1) for i in range(min(args.slots, 8))]
    t0 = time.perf_counter()
    run_all(be, warm)
    print(f"engine ready; warm-up {time.perf_counter() - t0:.1f}s; {json.dumps(be.semantics)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    plan = []
    for a in [x for x in args.arms.split(",") if x]:
        d, w = (int(v) for v in a.split("x"))
        plan.append(({"depth": d, "width": w, "cap": args.cap, "kind": "fixed"}, None))
    if args.extend_from:
        src = [json.loads(line) for line in open(os.path.join(args.extend_from, "candidates.jsonl"))]
        src_manifest = json.load(open(os.path.join(args.extend_from, em.MANIFEST_FILE)))
        capped = [r for r in src if r["finish_reason"] == "length_cap"]
        by_q, prompts = {}, {it["question_id"]: it["prompt"] for it in items}
        for r in capped:
            by_q.setdefault(r["question_id"], []).append(
                (r["candidate_id"], r["sample"], 0.0 if r["sample"] < 0 else args.temperature,
                 1.0 if r["sample"] < 0 else args.top_p, r["seed"]))
        depth = src_manifest["decode_config"]["depth"]
        arm = {"depth": depth, "width": src_manifest["decode_config"]["width"], "cap": args.cap, "kind": "extend_capped",
               "from_run": src_manifest["config_hash"], "from_cap": src_manifest["decode_config"]["cap"]}
        plan.append((arm, ([(q, prompts[q], cs) for q, cs in by_q.items()],
                           {r["candidate_id"]: r["generated_token_ids"] for r in capped})))

    for arm, extra in plan:
        name = (f"d{arm['depth']}_w{arm['width']}_cap{arm['cap']}" + ("_extend" if arm["kind"] != "fixed" else "")
                + (f"_{args.tag}" if args.tag else ""))
        out_dir = os.path.join(args.out, name)
        manifest = manifest_for(args, be, arm, prompt_hash, spec["split_hash"])
        state = em.open_run_dir(out_dir, manifest)
        if state == "resume":
            if os.path.exists(os.path.join(out_dir, "summary.json")):
                print(f"{name}: already complete ({manifest.run_id}); skipping", flush=True)
                continue
            raise SystemExit(f"{out_dir} holds an unfinished arm; move it aside and rerun")
        summary = run_arm(be, tok, items, arm, args, out_dir, manifest, work=extra[0] if extra else None)
        if extra:
            new = {json.loads(line)["candidate_id"]: json.loads(line)["generated_token_ids"]
                   for line in open(os.path.join(out_dir, "candidates.jsonl"))}
            summary["prefix_reproduced"] = sum(new[c][: len(old)] == old for c, old in extra[1].items())
            json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
        print(f"{name}: " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items()
                                        if k in ("wall_s", "candidates", "tokens_per_s", "questions_per_min",
                                                 "finish_reasons", "mean_gen_tokens", "peak_used_blocks",
                                                 "prefix_reproduced")})
              + f" latency_p50={summary['question_latency_s'].get('p50'):.1f}s"
              + f" p95={summary['question_latency_s'].get('p95'):.1f}s", flush=True)
    be.close()


if __name__ == "__main__":
    main()
