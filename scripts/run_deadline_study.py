#!/usr/bin/env python
"""Run one arm of the predeclared deadline study on the pinned engine.

The workload, the caps, the queue-aware rule and the mixture list live in
``looped/deadline.py``. This script does not choose among them. Calibration
runs one mixture candidate at seed 0. Validation runs a pure arm, the
queue-aware arm, or the mixture named by ``selection.json``, at seeds 0, 1
and 2. A ``--smoke N`` run is labelled so the grader will not treat it as
evidence.

    python scripts/run_deadline_study.py --model <snapshot> \
        --items artifacts/looped_serving/splits/math_train_calibration.items.jsonl \
        --split math_train_calibration --arm mixture --candidate p50 --run-seed 0 \
        --out artifacts/looped_serving/deadline_study/calibration
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import looped_profile as LP  # noqa: E402
from looped import answer_identity as ai  # noqa: E402
from looped import deadline as D  # noqa: E402
from looped import experiment_manifest as em  # noqa: E402
from looped import modes as M  # noqa: E402
from looped import prompts as P  # noqa: E402
from serving.looped.backend import GenerationRequest, run_all, strip_stop  # noqa: E402
from serving.looped.deadline_driver import run_deadline  # noqa: E402
from serving.looped.rlt_backend import RltBackend  # noqa: E402

GB = 1 << 30
STUDY_FILE = "configs/looped_serving/deadline_study.json"
REPORT_EVERY_S = 60.0


def run_name(arm: str, candidate: str, seed: int, smoke: bool) -> str:
    name = arm if arm != "mixture" else f"mixture_{candidate}"
    return f"{name}_seed{seed}" + ("_smoke" if smoke else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--split", required=True, choices=(D.CALIBRATION_SPLIT, D.VALIDATION_SPLIT))
    ap.add_argument("--arm", required=True, choices=("always_direct", "always_long", "mixture", "queue_aware"))
    ap.add_argument("--candidate", default="")
    ap.add_argument("--selection", default="")
    ap.add_argument("--run-seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--study", default=STUDY_FILE)
    ap.add_argument("--splits", default="configs/looped_serving/splits.json")
    ap.add_argument("--code_sha", default=open("CODE_SHA").read().strip() if os.path.exists("CODE_SHA") else "unknown")
    args = ap.parse_args()
    D.check_study_file(args.study)
    if args.split == D.CALIBRATION_SPLIT and args.run_seed != D.CALIBRATION_SEED:
        raise SystemExit(f"calibration selection uses seed {D.CALIBRATION_SEED}")
    if args.split == D.VALIDATION_SPLIT and args.run_seed not in D.REPETITIONS:
        raise SystemExit(f"validation repetitions are {list(D.REPETITIONS)}")
    if args.arm == "mixture" and args.split == D.CALIBRATION_SPLIT:
        if args.candidate not in D.MIXTURE_CANDIDATES:
            raise SystemExit(f"--candidate must be one of {list(D.MIXTURE_CANDIDATES)}")
    elif args.arm == "mixture":
        if args.candidate:
            raise SystemExit("validation does not take --candidate; the frozen selection names the mixture")
        if not args.selection:
            raise SystemExit("validation of the mixture needs --selection")
        chosen = json.load(open(args.selection))
        if chosen.get("study") != D.STUDY_NAME or chosen.get("chosen") not in D.MIXTURE_CANDIDATES:
            raise SystemExit("selection.json is not a frozen mixture for this study")
        args.candidate = chosen["chosen"]
    elif args.candidate:
        raise SystemExit(f"{args.arm} does not take --candidate")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    splits = em.load_splits(args.splits)
    split_ids = [e["question_id"] for e in splits["splits"][args.split]["items"]]
    items = [json.loads(line) for line in open(args.items)]
    for it in items:
        em.assert_no_gold(it)
    if args.smoke:
        if args.smoke < 1 or args.smoke > len(items):
            raise SystemExit("--smoke is a positive prefix of the items file")
        items = items[: args.smoke]
    elif [it["question_id"] for it in items] != split_ids:
        raise SystemExit(f"{args.items} is not {args.split} in split order; the arrival trace is that order")
    banned = D.locked_overlap([it["question_id"] for it in items], splits)
    if banned:
        raise SystemExit(f"refusing questions from a locked split: {sorted(banned)}")
    for it in items:
        it["prompt"] = tok.apply_chat_template(P.msgs(P.solver(it["text"])), tokenize=True, add_generation_prompt=True)
    if args.smoke:
        arrivals = [i * 60.0 / D.RATES_PER_MIN[0] for i in range(len(items))]
    else:
        arrivals = D.arrival_trace(len(items))

    spec = splits
    args.dataset_revision = f"{spec['datasets']['math_train']['repo']}@{spec['datasets']['math_train']['revision']}"
    args.temperature, args.top_p = D.STUDY["temperature"], D.STUDY["top_p"]
    args.slots, args.limit, args.run_seed = D.SLOTS, 0, args.run_seed
    prompt_hash = em.sha256_hex(em.canonical_json([P.solver("{question}"), tok.chat_template]))[:16]
    be = RltBackend(
        args.model, stop_ids=LP.stop_ids_for(tok), prompt_mode="stock", dtype="bfloat16",
        attention_backend="triton", kv_layout="last_exited", max_num_seqs=D.SLOTS,
        max_num_batched_tokens=4096, scheduler_mode="refill",
        kv_cache_memory_bytes=D.KV_GB * GB, execution=dict(D.STUDY["execution"]), exit_mode="ouro",
        total_ut_steps=D.DEPTH,
    )
    warm = [GenerationRequest(f"warm:{i}", tuple(items[i % len(items)]["prompt"]), D.DEPTH, 48, 0.6, 0.95, i)
            for i in range(min(D.SLOTS, 8))]
    run_all(be, warm)
    print(f"engine ready; {json.dumps(be.semantics)}", flush=True)

    arm = D.arm_for(args.arm, args.candidate or None)
    name = run_name(args.arm, args.candidate, args.run_seed, bool(args.smoke))
    out_dir = os.path.join(args.out, name)
    manifest = LP.manifest_for(
        args, be,
        {"kind": "deadline", "arm": args.arm, "candidate": args.candidate or None, "study": D.STUDY_NAME,
         "deadline_s": D.DEADLINE_S, "smoke": bool(args.smoke), "n_questions": len(items)},
        prompt_hash, spec["split_hash"])
    if em.open_run_dir(out_dir, manifest) == "resume":
        if os.path.exists(os.path.join(out_dir, "questions.jsonl")):
            print(f"{name}: already complete ({manifest.run_id}); skipping", flush=True)
            be.close()
            return
        raise SystemExit(f"{out_dir} holds an unfinished run; move it aside and rerun")

    stop = set(be.semantics["stop_ids"])
    think_token = M.think_id(tok)
    prefixes, texts, rows = {}, {}, []
    cand_f = open(os.path.join(out_dir, "candidates.jsonl"), "w")
    # Question outcomes are appended as each question closes. questions.jsonl is
    # still the ordered file the grader reads, written once the arm finishes.
    outcome_f = open(os.path.join(out_dir, "outcomes.jsonl"), "w")
    io_lock = threading.Lock()
    partial = []
    n_candidates = [0]
    t0 = time.perf_counter()
    study_dir = os.path.dirname(os.path.abspath(args.out))

    def utc_now():
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def write_status(phase, sync_candidates=False):
        with io_lock:
            met = sum(bool(row["deadline_met"]) for row in partial)
            long_n = sum(bool(row["launched_long"]) for row in partial)
            payload = {
                "study": D.STUDY_NAME, "phase": phase, "arm": args.arm,
                "candidate": args.candidate or None, "seed": args.run_seed,
                "split": args.split, "smoke": bool(args.smoke), "updated_at": utc_now(),
                "wall_s": round(time.perf_counter() - t0, 1),
                "outcomes": len(partial), "n_questions": len(items),
                "deadline_met": met, "launched_long": long_n,
                "candidates_flushed": n_candidates[0],
            }
            _replace_json(os.path.join(out_dir, "status.json"), payload)
            text = (
                f"{payload['study']}\n"
                f"phase: {phase}\n"
                f"arm: {args.arm}"
                + (f" candidate {args.candidate}" if args.candidate else "")
                + f" seed {args.run_seed}\n"
                f"split: {args.split}\n"
                f"updated: {payload['updated_at']}\n"
                f"wall_min: {payload['wall_s'] / 60:.1f}\n"
                f"outcomes: {len(partial)}/{len(items)}  in_time: {met}  long_launched: {long_n}\n"
                f"candidates_flushed: {n_candidates[0]}\n"
                f"partial outcomes: {out_dir}/outcomes.jsonl\n"
            )
            _replace_text(os.path.join(study_dir, "STATUS.txt"), text)
            with open(os.path.join(study_dir, "status_log.txt"), "a") as log:
                log.write(
                    f"{payload['updated_at']} {phase} {name} "
                    f"outcomes={len(partial)}/{len(items)} in_time={met} "
                    f"long={long_n} candidates={n_candidates[0]}\n")
                log.flush()
                os.fsync(log.fileno())
            if sync_candidates and not cand_f.closed:
                cand_f.flush()
                os.fsync(cand_f.fileno())
        print(
            f"{name}: {phase} {len(partial)}/{len(items)} in time, "
            f"{n_candidates[0]} candidates flushed",
            flush=True)

    stop_report = threading.Event()

    def report_loop():
        while not stop_report.wait(REPORT_EVERY_S):
            write_status("running", sync_candidates=True)

    reporter = threading.Thread(target=report_loop, name="deadline-status", daemon=True)

    def make_request(qid, prompt, launch):
        cid = LP.mode_candidate_id(qid, launch.depth, launch.sample, launch.mode)
        LP.check_mode_semantics(be, launch.depth, launch.mode)
        prefix = M.prefix_ids(tok, launch.mode)
        prefixes[cid] = (launch.mode, prefix)
        cap = launch.cap if launch.cap is not None else D.DIRECT_CAP
        return GenerationRequest(cid, tuple(prompt) + prefix, launch.depth, cap - len(prefix),
                                 args.temperature, args.top_p, em.candidate_seed(args.run_seed, qid, cid))

    def answer_text(comp):
        return tok.decode(strip_stop(LP.answer_tokens(prefixes[comp.request_id][1], comp), stop), skip_special_tokens=True)

    def keyer(comp):
        text = answer_text(comp)
        texts[comp.request_id] = text
        if comp.finish_reason == "error":
            return None
        return ai.answer_from_text(text)[1].key

    def on_candidate(q, sample, comp, cancel_time):
        req = q.candidates[sample]["request"]
        text = texts.pop(comp.request_id, None)
        if text is None:
            text = answer_text(comp)
        mode, prefix = prefixes[comp.request_id]
        row = LP.record(manifest.run_id, q.question_id, (req.request_id, sample, None, None, req.seed), comp, text, t0,
                        mode, prefix, think_token)
        if comp.finish_reason == "cancelled":
            row["answer_key"], row["parse_status"], row["pred"] = None, "cancelled", None
        em.check_record(row, em.CANDIDATE_RECORD_FIELDS)
        em.assert_no_gold(row)
        rows.append(row)
        with io_lock:
            cand_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            cand_f.flush()
            n_candidates[0] += 1

    def on_outcome(q):
        row = q.outcome
        em.assert_no_gold(row)
        with io_lock:
            outcome_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            outcome_f.flush()
            os.fsync(outcome_f.fileno())
            partial.append(row)
        write_status("running")

    questions = [(it["question_id"], it["prompt"], arr) for it, arr in zip(items, arrivals)]
    write_status("running")
    reporter.start()
    try:
        result = run_deadline(
            be, arm, questions, slots=D.SLOTS, deadline_s=D.DEADLINE_S, clock=time.perf_counter, sleeper=time.sleep,
            make_request=make_request, keyer=keyer, on_candidate=on_candidate, on_outcome=on_outcome,
            think_of=lambda comp, launch: M.observed_think(LP.answer_tokens(prefixes[comp.request_id][1], comp), think_token),
        )
    except BaseException:
        write_status("failed", sync_candidates=True)
        raise
    finally:
        stop_report.set()
        reporter.join(timeout=1)
        with io_lock:
            cand_f.flush()
            os.fsync(cand_f.fileno())
            cand_f.close()
            outcome_f.flush()
            os.fsync(outcome_f.fileno())
            outcome_f.close()
    torch.cuda.synchronize()
    be.close()
    with open(os.path.join(out_dir, "questions.jsonl"), "w") as f:
        for row in result["questions"]:
            em.assert_no_gold(row)
            f.write(json.dumps(row) + "\n")
    info = {"study": D.STUDY_NAME, "arm": args.arm, "candidate": args.candidate or None,
            "seed": args.run_seed, "split": args.split, "smoke": bool(args.smoke),
            "question_ids": [it["question_id"] for it in items], "arrivals": arrivals,
            "run_id": manifest.run_id}
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"run_id": manifest.run_id, "wall_s": round(time.perf_counter() - t0, 2), **result["stats"],
                   "questions": len(result["questions"]),
                   "deadline_met": sum(bool(q["deadline_met"]) for q in result["questions"]),
                   "launched_long": sum(bool(q["launched_long"]) for q in result["questions"])}, f, indent=2)
        f.write("\n")
    write_status("complete", sync_candidates=True)
    print(f"{name}: met {sum(bool(q['deadline_met']) for q in result['questions'])}/{len(items)} "
          f"expired {result['stats']['expired']}", flush=True)


def _replace_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _replace_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
