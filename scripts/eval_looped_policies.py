#!/usr/bin/env python
"""Run allocation policies live on the pinned engine and record what happened.

Each policy (``looped/allocation.py``) answers every question of a frozen split
under the same closed-loop load as ``scripts/looped_profile.py``: first come,
first served, at most ``--slots`` candidates in flight. Unlike a replay, the
policy really launches and cancels requests, so the latency columns are
measurements of the policy, including the wait for a slot before a follow-up
launch.

Policy specs (``looped.allocation.parse_policy``): ``fixed:<depth>x<width>[:<mode>]``,
``two_then_two:<depth>[-><depth>]``, ``cert:<depth>x<width>``,
``quorum:<depth>x<width>[/<quorum>][:<mode>][:trump]`` and
``ftt:<depth>x<width>[/<quorum>]+<think depth>x<think width>[@<think cap>]``.
A policy that forces an answer mode needs ``--prompt_mode stock``.

Candidate ids and seeds are the ones ``looped_profile.py`` uses, so sample
``s0`` of a policy run is the same request as sample ``s0`` of a fixed arm. In
bf16 the tokens can still differ, because bf16 results depend on batch
composition (see the fidelity gate).

This script never sees gold answers. Grade with ``scripts/grade_looped_profile.py``.

    python scripts/eval_looped_policies.py --model <snapshot dir> \
        --items artifacts/looped_serving/splits/math_train_smoke.items.jsonl \
        --policies two_then_two:3,cert:3x4 --out artifacts/looped_serving/profile_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import looped_profile as LP  # noqa: E402
from looped import allocation as al  # noqa: E402
from looped import answer_identity as ai  # noqa: E402
from looped import experiment_manifest as em  # noqa: E402
from looped import modes as M  # noqa: E402
from looped import prompts as P  # noqa: E402
from serving.looped.backend import GenerationRequest, run_all, strip_stop  # noqa: E402
from serving.looped.driver import run_closed_loop  # noqa: E402
from serving.looped.rlt_backend import RltBackend  # noqa: E402

GB = 1 << 30


def run_policy(be, tok, items, policy: al.Policy, args, out_dir: str, manifest: em.RunManifest) -> dict:
    stop = set(be.semantics["stop_ids"])
    rows, q_rows, texts, prefixes, reasons_final = [], [], {}, {}, {}
    think_token = M.think_id(tok)
    peak_blocks = [0]
    torch.cuda.reset_peak_memory_stats()
    cand_f = open(os.path.join(out_dir, "candidates.jsonl"), "w")
    dec_f = open(os.path.join(out_dir, "decisions.jsonl"), "w")
    t0 = time.perf_counter()

    def make_request(qid, prompt, l: al.Launch) -> GenerationRequest:
        cid = LP.mode_candidate_id(qid, l.depth, l.sample, l.mode)
        greedy = l.sample == al.GREEDY
        LP.check_mode_semantics(be, l.depth, l.mode, args.allow_full_depth_prefix_state)
        prefix = M.prefix_ids(tok, l.mode)
        prefixes[cid] = (l.mode, prefix)
        return GenerationRequest(cid, tuple(prompt) + prefix, l.depth, (l.cap or args.cap) - len(prefix),
                                 0.0 if greedy else args.temperature, 1.0 if greedy else args.top_p,
                                 em.candidate_seed(args.run_seed, qid, cid))

    def answer_text(comp):
        return tok.decode(strip_stop(LP.answer_tokens(prefixes[comp.request_id][1], comp), stop), skip_special_tokens=True)

    def keyer(comp):
        text = answer_text(comp)
        texts[comp.request_id] = text
        if comp.finish_reason == "error":
            return None
        ans, _ = ai.extract_final_answer(text)
        return ai.identify(ans).key if ans is not None else None

    def on_candidate(q, sample, comp, cancel_time):
        req = q.candidates[sample]["request"]
        text = texts.pop(comp.request_id, None)
        if text is None:
            text = answer_text(comp)
        mode, prefix = prefixes[comp.request_id]
        row = LP.record(manifest.run_id, q.question_id, (req.request_id, sample, None, None, req.seed), comp, text, t0,
                        mode, prefix, think_token)
        if cancel_time is not None:
            row["cancel_time"] = round(cancel_time - t0, 4)
            row["answer_key"], row["parse_status"], row["pred"] = None, "cancelled", None
        em.check_record(row, em.CANDIDATE_RECORD_FIELDS)
        em.assert_no_gold(row)
        rows.append(row)
        cand_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        peak_blocks[0] = max(peak_blocks[0], be.stats()["used_blocks"])

    def on_decision(q, decision, now, free_slots):
        views = q.views()
        state = [[v.sample, v.depth, v.status, v.answer_key, v.finish_reason, v.gen_tokens] for v in views]
        ev = {"question_id": q.question_id, "timestamp": round(now - t0, 4),
              "observable_state_hash": em.sha256_hex(em.canonical_json(state))[:16], "action": decision.action,
              "candidates_considered": [em.candidate_id(q.question_id, v.depth, v.sample) for v in views],
              "estimated_gain": None, "estimated_incremental_ms": None, "deadline_slack": None,
              "KV_budget": {"free_slots": free_slots}, "policy_version": f"{policy.name}-{policy.version}",
              "fallback_reason": decision.reason}
        em.check_record(ev, em.DECISION_EVENT_FIELDS)
        em.assert_no_gold(ev)
        dec_f.write(json.dumps(ev) + "\n")
        if decision.done:
            reasons_final[q.question_id] = decision.reason

    def on_question(q, sel, now):
        q_rows.append({
            "question_id": q.question_id, "scheduled_arrival": round(q.arrival - t0, 4),
            "completion_time": round(now - t0, 4), "latency_s": round(now - q.arrival, 4),
            "selected_candidate_id": None if sel is None else q.candidates[sel.sample]["request_id"],
            "decision_reason": reasons_final.get(q.question_id),
            "selected_answer": None if sel is None else sel.answer_key,
            "all_candidate_ids": [c["request_id"] for _, c in sorted(q.candidates.items())],
            "candidates_launched": len(q.candidates),
            "candidates_cancelled": sum(c["status"] == al.CANCELLED for c in q.candidates.values()),
        })

    stats = run_closed_loop(be, policy, [(it["question_id"], it["prompt"]) for it in items], slots=args.slots,
                            make_request=make_request, keyer=keyer, on_candidate=on_candidate,
                            think_of=lambda comp, l: M.observed_think(
                                LP.answer_tokens(prefixes[comp.request_id][1], comp), think_token),
                            on_decision=on_decision, on_question=on_question)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    cand_f.close()
    dec_f.close()
    with open(os.path.join(out_dir, "questions.jsonl"), "w") as f:
        for r in q_rows:
            f.write(json.dumps(r) + "\n")
    toks = sum(r["gen_tokens"] for r in rows)
    reasons = {}
    for r in rows:
        reasons[r["finish_reason"]] = reasons.get(r["finish_reason"], 0) + 1
    summary = {
        "run_id": manifest.run_id, "policy": policy.describe(), "questions": len(q_rows), "candidates": len(rows),
        "slots": args.slots, "cap": args.cap, "wall_s": round(wall, 2), "generated_tokens": toks,
        "tokens_per_s": toks / wall, "questions_per_min": 60 * len(q_rows) / wall,
        "logical_loop_tokens": sum(r["logical_loop_tokens"] for r in rows),
        "charged_loop_tokens": sum(r["executed_loop_tokens"] for r in rows),
        "finish_reasons": reasons, "mean_gen_tokens": toks / len(rows),
        "candidates_per_question": len(rows) / len(q_rows),
        "cancelled_candidates": stats["cancelled"],
        "decision_reasons": {k: sum(v == k for v in reasons_final.values()) for k in sorted(set(reasons_final.values()), key=str)},
        "think_share": sum(bool(r["think"]) for r in rows) / len(rows),
        "question_latency_s": LP.dist([r["latency_s"] for r in q_rows]),
        "candidate_latency_s": LP.dist([r["finish_time"] - r["submit_time"] for r in rows]),
        "first_token_s": LP.dist([r["first_token_time"] - r["submit_time"] for r in rows
                                  if r["first_token_time"] is not None]),
        "peak_used_blocks": peak_blocks[0], "num_blocks": be.stats()["num_blocks"],
        "peak_live_candidates": stats["peak_live"],
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
    ap.add_argument("--policies", required=True)
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
    ap.add_argument("--execution", default="{}")
    ap.add_argument("--exit_mode", default="ouro")
    ap.add_argument("--total_ut_steps", type=int, default=0)
    ap.add_argument("--allow_full_depth_prefix_state", action="store_true")
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
        args.model, stop_ids=LP.stop_ids_for(tok), prompt_mode=args.prompt_mode, dtype=args.dtype,
        attention_backend=args.attention, kv_layout=args.kv_layout, max_num_seqs=args.slots,
        max_num_batched_tokens=args.max_num_batched_tokens, scheduler_mode=args.scheduler_mode,
        kv_cache_memory_bytes=args.kv_gb * GB, execution=json.loads(args.execution), exit_mode=args.exit_mode,
        total_ut_steps=args.total_ut_steps or None,
    )
    warm = [GenerationRequest(f"warm:{d}:{i}", tuple(items[i % len(items)]["prompt"]), d, 48, 0.6, 0.95, i)
            for d in range(1, be.full_depth + 1) for i in range(min(args.slots, 8))]
    run_all(be, warm)
    print(f"engine ready; {json.dumps(be.semantics)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    for spec_str in [s for s in args.policies.split(",") if s]:
        policy = al.parse_policy(spec_str)
        arm = {"kind": "policy", "policy": policy.describe(), "spec": spec_str, "cap": args.cap}
        name = ("policy_" + spec_str.replace(":", "_").replace("->", "to").replace("/", "q").replace("+", "_then_")
                .replace("@", "_cap") + f"_cap{args.cap}" + (f"_{args.tag}" if args.tag else ""))
        out_dir = os.path.join(args.out, name)
        manifest = LP.manifest_for(args, be, arm, prompt_hash, spec["split_hash"])
        if em.open_run_dir(out_dir, manifest) == "resume":
            if os.path.exists(os.path.join(out_dir, "summary.json")):
                print(f"{name}: already complete ({manifest.run_id}); skipping", flush=True)
                continue
            raise SystemExit(f"{out_dir} holds an unfinished run; move it aside and rerun")
        s = run_policy(be, tok, items, policy, args, out_dir, manifest)
        print(f"{name}: " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()
                                        if k in ("wall_s", "candidates", "candidates_per_question", "cancelled_candidates",
                                                 "tokens_per_s", "questions_per_min", "finish_reasons")})
              + f" latency_p50={s['question_latency_s']['p50']:.1f}s p95={s['question_latency_s']['p95']:.1f}s", flush=True)
    be.close()


if __name__ == "__main__":
    main()
