#!/usr/bin/env python
"""Closed-loop latency/throughput profile of fixed-depth, fixed-width arms.

Each arm ``<depth>x<width>[:<mode>][@<cap>]`` answers every question of a frozen
split with ``width`` candidates at one depth (width 1 is the greedy candidate;
wider arms are seeded samples ``s0..s{width-1}``). ``mode`` is ``natural``
(default), ``direct``, ``think`` or ``nothink`` (``looped/modes.py``); a forced mode needs
``--prompt_mode stock`` and, to stay the uniform-depth computation, an engine
whose depth equals the arm's (``--total_ut_steps``). Questions are admitted first come, first
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
from looped import modes as M  # noqa: E402
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


def mode_candidate_id(qid: str, depth: int, sample: int, mode: str = M.NATURAL) -> str:
    """Natural candidates keep the plain id; a forced mode is part of the id, so
    its seed differs from the natural candidate with the same sample index."""
    return em.candidate_id(qid, depth, sample) + ("" if mode == M.NATURAL else f":{mode}")


def parse_arm(spec: str, default_cap: int) -> dict:
    """``<depth>x<width>[:<mode>][@<cap>]``."""
    body, _, cap = spec.partition("@")
    shape, _, mode = body.partition(":")
    d, w = (int(v) for v in shape.split("x"))
    mode = mode or M.NATURAL
    if mode not in M.MODES:
        raise ValueError(f"unknown mode in arm {spec!r}")
    return {"depth": d, "width": w, "cap": int(cap) if cap else default_cap, "kind": "fixed", "mode": mode}


def arm_name(arm: dict, tag: str = "") -> str:
    mode = arm.get("mode", M.NATURAL)
    return (f"d{arm['depth']}_w{arm['width']}" + ("" if mode == M.NATURAL else f"_{mode}") + f"_cap{arm['cap']}"
            + ("_extend" if arm["kind"] != "fixed" else "") + (f"_{tag}" if tag else ""))


def check_mode_semantics(be, depth: int, mode: str, allow_full_depth_prefix_state: bool = False) -> None:
    """A forced prefix is part of the engine's prompt, so the token after it is
    read from the engine's prompt state."""
    if mode == M.NATURAL:
        return
    if be.semantics["prompt_mode"] != "stock":
        raise SystemExit("a forced mode needs --prompt_mode stock: boundary_shift would require the engine to "
                         "emit the forced token itself")
    if depth != be.semantics["full_depth"] and not allow_full_depth_prefix_state:
        raise SystemExit(f"depth {depth} with a forced mode on a depth-{be.semantics['full_depth']} engine reads the "
                         "token after the prefix from the full-depth state; use --total_ut_steps or pass "
                         "--allow_full_depth_prefix_state")


def candidates_for(qid: str, depth: int, width: int, args, mode: str = M.NATURAL) -> list:
    """(candidate_id, sample, temperature, top_p, seed) for one question."""
    if width == 1:
        cid = mode_candidate_id(qid, depth, -1, mode)
        return [(cid, -1, 0.0, 1.0, em.candidate_seed(args.run_seed, qid, cid))]
    out = []
    for s in range(width):
        cid = mode_candidate_id(qid, depth, s, mode)
        out.append((cid, s, args.temperature, args.top_p, em.candidate_seed(args.run_seed, qid, cid)))
    return out


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PINS = {}


def verified_pins(model_path: str) -> dict:
    """What is actually installed, read from disk rather than assumed: the
    source tree's content hash, the model snapshot directory and the engine's
    checked-out commit. Stops when one of them is not the pinned revision."""
    if _PINS:
        return _PINS
    import subprocess

    import vllm_rlt
    snapshot = os.path.basename(os.path.realpath(model_path).rstrip("/"))
    try:
        head = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(vllm_rlt.__file__))),
                               "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        head = None
    if len(snapshot) == 40 and snapshot != MODEL_REV:
        raise SystemExit(f"the model directory is snapshot {snapshot}, not the pinned {MODEL_REV}")
    if head is not None and head != ENGINE_SHA:
        raise SystemExit(f"the installed engine is at {head}, not the pinned {ENGINE_SHA}")
    _PINS.update(source_tree_sha256=em.source_tree_hash(ROOT),
                 model_snapshot=snapshot if len(snapshot) == 40 else "unverified (not a snapshot directory)",
                 engine_head=head or "unverified (no git checkout)")
    return _PINS


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
                                                       "kv_num_blocks", "execution", "exit_mode", "full_depth",
                                                       "checkpoint_depth", "first_token_depth")},
               "driver": "closed loop, FIFO admission by free candidate slots",
               "verified": verified_pins(args.model)},
    )


# What an extension may change relative to its source arm: the cap, and the
# number of slots (a longer cap needs more KV per candidate). Everything else
# that defines the candidates must be the source's.
EXTENSION_MUST_MATCH = ("temperature", "top_p", "stop_ids", "items_sha256")


def extension_arm(src_manifest: dict, cap: int) -> dict:
    """The arm that reruns the capped candidates of ``src_manifest`` at ``cap``:
    same depth, width and answer mode, so the old candidate ids and seeds still
    name the same intervention."""
    cfg = src_manifest["decode_config"]
    if cfg.get("kind") != "fixed":
        raise SystemExit("only a fixed arm can be extended")
    if cap <= cfg["cap"]:
        raise SystemExit(f"the extension cap {cap} is not above the source cap {cfg['cap']}")
    return {"depth": cfg["depth"], "width": cfg["width"], "cap": cap, "kind": "extend_capped",
            "mode": cfg.get("mode", M.NATURAL), "from_run": src_manifest["config_hash"], "from_cap": cfg["cap"]}


def check_extension(src_manifest: dict, new: em.RunManifest) -> None:
    """Refuse an extension whose generation settings differ from its source."""
    cfg, now = src_manifest["decode_config"], new.payload()
    diff = [f"decode_config.{k}: {cfg.get(k)!r} -> {now['decode_config'].get(k)!r}"
            for k in EXTENSION_MUST_MATCH if cfg.get(k) != now["decode_config"].get(k)]
    diff += [f"{k}: {src_manifest.get(k)!r} -> {now.get(k)!r}"
             for k in ("engine_sha", "model_revision", "tokenizer_revision", "prompt_hash", "split_hash",
                       "cache_semantics", "attention_backend", "dtype", "run_seed", "seed_scheme")
             if src_manifest.get(k) != now.get(k)]
    if diff:
        raise SystemExit("the extension does not match its source arm: " + "; ".join(diff))


def extension_plan(args, items):
    """(arm, work, old tokens) for ``--extend_from``."""
    src = [json.loads(line) for line in open(os.path.join(args.extend_from, "candidates.jsonl"))]
    src_manifest = json.load(open(os.path.join(args.extend_from, em.MANIFEST_FILE)))
    arm = extension_arm(src_manifest, args.cap)
    capped = [r for r in src if r["finish_reason"] == "length_cap"]
    if any(r.get("mode", M.NATURAL) != arm["mode"] for r in capped):
        raise SystemExit("the source rows were not generated in the source manifest's mode")
    by_q, prompts = {}, {it["question_id"]: it["prompt"] for it in items}
    for r in capped:
        by_q.setdefault(r["question_id"], []).append(
            (r["candidate_id"], r["sample"], 0.0 if r["sample"] < 0 else args.temperature,
             1.0 if r["sample"] < 0 else args.top_p, r["seed"]))
    return arm, [(q, prompts[q], cs) for q, cs in by_q.items()], {r["candidate_id"]: r["generated_token_ids"] for r in capped}


def answer_tokens(prefix, comp) -> list:
    """The answer as the model would have written it: forced prefix, then what it generated."""
    return list(prefix) + list(comp.token_ids)


def record(run_id, qid, cand, comp, text, t0, mode: str = M.NATURAL, prefix=(), think_token=None) -> dict:
    cid, sample, _, _, seed = cand
    tokens = answer_tokens(prefix, comp)
    ans, ident = ai.answer_from_text(text)
    rel = lambda t: None if t is None else round(t - t0, 4)  # noqa: E731
    return {
        "run_id": run_id, "question_id": qid, "candidate_id": cid, "depth": comp.depth, "sample": sample, "seed": seed,
        "prompt_tokens": comp.prompt_tokens - len(prefix), "generated_token_ids": tokens, "gen_tokens": len(tokens),
        "mode": mode, "prefix_tokens": len(prefix),
        "think": None if think_token is None else M.observed_think(tokens, think_token),
        "finish_reason": FINISH[comp.finish_reason], "text": text, "pred": ans,
        "answer_key": ident.key, "parse_status": ident.status,
        "scalar_confidence_definition": None, "confidence_summary": None,
        "logical_loop_tokens": comp.logical_loop_tokens, "executed_loop_tokens": comp.charged_loop_tokens, "probe_work": 0,
        "submit_time": rel(comp.submit_time), "first_token_time": rel(comp.first_token_time),
        "finish_time": rel(comp.finish_time), "cancel_time": None, "retire_time": rel(comp.finish_time),
        "detail": comp.detail,
    }


def draw_openers(be, work, depth: int, banned: int, args, max_attempts: int = 64):
    """First answer token of each candidate, drawn from the model's own
    first-token distribution conditioned on not being ``banned``.

    Each candidate gets one-token requests with independent seeds until one
    returns another token (rejection sampling, so the accepted token has
    exactly the conditional distribution). A greedy candidate cannot be redrawn
    and is refused. Returns ``{candidate id: (token, attempts, charged loop-tokens)}``;
    the work of every attempt is charged to the candidate as ``probe_work``.
    """
    todo = {c[0]: (qid, prompt, c) for qid, prompt, cands in work for c in cands}
    if any(c[1] < 0 for _, _, c in todo.values()):
        raise SystemExit("nothink needs sampled candidates: a greedy first token cannot be redrawn")
    out = {cid: [None, 0, 0] for cid in todo}
    for attempt in range(max_attempts):
        if not todo:
            break
        ids = list(todo)
        for i in range(0, len(ids), args.slots):
            batch = []
            for cid in ids[i:i + args.slots]:
                qid, prompt, c = todo[cid]
                rid = f"{cid}:opener{attempt}"
                batch.append(GenerationRequest(rid, tuple(prompt), depth, 1, c[2], c[3],
                                               em.candidate_seed(args.run_seed, qid, rid)))
            for rid, comp in run_all(be, batch).items():
                cid = rid.rsplit(":opener", 1)[0]
                out[cid][1] += 1
                out[cid][2] += comp.charged_loop_tokens
                if comp.token_ids and int(comp.token_ids[0]) in be.semantics.get("stop_ids", ()):
                    raise SystemExit(f"{cid}: the model's first token is a stop token; nothing to prefill")
                if comp.token_ids and int(comp.token_ids[0]) != banned:
                    out[cid][0] = int(comp.token_ids[0])
                    del todo[cid]
    if todo:
        raise SystemExit(f"{len(todo)} candidates returned the banned first token {max_attempts} times")
    return {cid: tuple(v) for cid, v in out.items()}


def run_arm(be, tok, questions, arm: dict, args, out_dir: str, manifest: em.RunManifest, work=None) -> dict:
    """``work`` overrides the candidate list: [(qid, prompt, [candidate tuples])]."""
    depth, width, cap = arm["depth"], arm["width"], arm["cap"]
    mode = arm.get("mode", M.NATURAL)
    think_token = M.think_id(tok)
    check_mode_semantics(be, depth, mode, getattr(args, "allow_full_depth_prefix_state", False))
    if work is None:
        work = [(q["question_id"], q["prompt"], candidates_for(q["question_id"], depth, width, args, mode))
                for q in questions]
    t_open = time.perf_counter()
    openers = draw_openers(be, work, depth, think_token, args) if mode == M.NOTHINK else {}
    opener_wall = time.perf_counter() - t_open       # not part of the latency columns
    fixed_prefix = () if mode == M.NOTHINK else M.prefix_ids(tok, mode)
    prefix_of = lambda cid: (openers[cid][0],) if openers else fixed_prefix  # noqa: E731
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
                    prefix = prefix_of(c[0])
                    be.submit(GenerationRequest(c[0], tuple(prompt) + prefix, depth, cap - len(prefix), c[2], c[3], c[4]))
                    live[c[0]] = (qid, c)
            if pending and not live:
                raise RuntimeError(f"a question needs {len(pending[0][2])} slots but only {args.slots} exist")
            for comp in be.step():
                qid, c = live.pop(comp.request_id)
                prefix = prefix_of(c[0])
                text = tok.decode(strip_stop(answer_tokens(prefix, comp), stop), skip_special_tokens=True)
                row = record(manifest.run_id, qid, c, comp, text, t0, mode, prefix, think_token)
                if openers:
                    row["opener_attempts"], row["probe_work"] = openers[c[0]][1], openers[c[0]][2]
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
        "probe_work": sum(r["probe_work"] for r in rows),
        "opener_attempts": sum(r.get("opener_attempts", 0) for r in rows), "opener_wall_s": round(opener_wall, 2),
        "finish_reasons": reasons, "mean_gen_tokens": toks / len(rows),
        "think_share": sum(bool(r["think"]) for r in rows) / len(rows),
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
    ap.add_argument("--arms", default="", help="comma list of <depth>x<width>[:<mode>][@<cap>]")
    ap.add_argument("--total_ut_steps", type=int, default=0,
                    help="run a depth-limited engine: the same weights with this many loops everywhere")
    ap.add_argument("--allow_full_depth_prefix_state", action="store_true")
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
        total_ut_steps=args.total_ut_steps or None,
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
        arm = parse_arm(a, args.cap)
        check_mode_semantics(be, arm["depth"], arm["mode"], args.allow_full_depth_prefix_state)
        plan.append((arm, None))
    if args.extend_from:
        arm, work, old_tokens = extension_plan(args, items)
        plan.append((arm, (work, old_tokens)))

    for arm, extra in plan:
        name = arm_name(arm, args.tag)
        out_dir = os.path.join(args.out, name)
        manifest = manifest_for(args, be, arm, prompt_hash, spec["split_hash"])
        if extra:
            check_extension(json.load(open(os.path.join(args.extend_from, em.MANIFEST_FILE))), manifest)
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
                                                 "finish_reasons", "mean_gen_tokens", "think_share", "peak_used_blocks",
                                                 "prefix_reproduced")})
              + f" latency_p50={summary['question_latency_s'].get('p50'):.1f}s"
              + f" p95={summary['question_latency_s'].get('p95'):.1f}s", flush=True)
    be.close()


if __name__ == "__main__":
    main()
