#!/usr/bin/env python3
"""LoopDelta stage 0b: does Δ predict the depth a question needs, and does a
solver's own Δ predict whether its answer is right?

For a difficulty-stratified MATH-Train subset (same seeded shuffle as stage 0a,
so its items are a prefix of 0a's and join on ``idx``), at every depth r:

  1. generate one greedy answer and ``--n_samples`` sampled answers (T, top_p)
  2. re-prefill prompt + answer at depth r in one pass and record, over the
     answer positions only, the KV deltas Δ_2..Δ_r (the solver's late-loop delta
     is the last entry) and the token log-probabilities of the answer under the
     loop-r readout (the standard confidence baseline for vote weighting)
  3. grade against the boxed gold answer

Generations are checkpointed per depth and reused on restart. The analysis
(min sufficient depth, AUROC, vote and policy simulations) lives in
scripts/loopdelta_analyze.py.

    python scripts/loopdelta_stage0b.py --per_level 60 --depths 1 2 3 4 --out artifacts/loopdelta/stage0b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped import prompts as P  # noqa: E402
from looped.harness import environment, read_jsonl, write_json, write_jsonl  # noqa: E402
from looped.loopdelta import (answer_key, fingerprint, grade_math, kv_deltas,  # noqa: E402
                              load_math_train_stratified, summarize)
from looped.ouro_adapter import OURO_BASE_ID, OuroRunner  # noqa: E402


def generate_depth(runner, prompts, items, r, args):
    """Greedy + sampled generations at depth ``r``; one row per (question, sample)."""
    slots = min(args.slots_cap, args.slots * runner.max_depth // r)
    rows = []
    t0 = time.perf_counter()
    g = runner.generate_pool(prompts, depth=r, max_new_tokens=args.max_new, slots=slots)
    wall_greedy = time.perf_counter() - t0
    for q, it in enumerate(items):
        rows.append({"q": q, "sample": -1, "ids": g.token_ids[q], "text": g.texts[q],
                     "hit_eos": g.hit_eos[q], "work": g.work[q].as_dict()})
    N = args.n_samples
    rep = [p for p in prompts for _ in range(N)]
    t0 = time.perf_counter()
    s = runner.generate_pool(rep, depth=r, max_new_tokens=args.max_new, slots=slots,
                             temperature=args.temperature, top_p=args.top_p, seed=1000 + r)
    wall_sample = time.perf_counter() - t0
    for j in range(len(rep)):
        q, k = divmod(j, N)
        rows.append({"q": q, "sample": k, "ids": s.token_ids[j], "text": s.texts[j],
                     "hit_eos": s.hit_eos[j], "work": s.work[j].as_dict()})
    timing = {"slots": slots, "wall_greedy_s": round(wall_greedy, 2), "wall_sampled_s": round(wall_sample, 2),
              "token_loops_greedy": g.executed_token_loops, "token_loops_sampled": s.executed_token_loops}
    return rows, timing


@torch.no_grad()
def rescore(runner, prompts, gens, r, bs):
    """Teacher-forced pass over prompt + answer at depth ``r``."""
    stop = set(runner.stop_ids)
    out = [None] * len(gens)
    order = sorted(range(len(gens)), key=lambda j: len(prompts[gens[j]["q"]]) + len(gens[j]["ids"]))
    for s in range(0, len(order), bs):
        chunk = order[s:s + bs]
        seqs, glens = [], []
        for j in chunk:
            ans = [t for t in gens[j]["ids"] if t not in stop]
            seqs.append(list(prompts[gens[j]["q"]]) + ans)
            glens.append(len(ans))
        ids, mask = runner._pad_left(seqs)
        pos = (mask.cumsum(-1) - 1).clamp_min(0)
        B, S = ids.shape
        _, h, cache = runner.forward(depth=r, input_ids=ids, position_ids=pos,
                                     attention_mask=mask if bool((mask == 0).any()) else None)
        gmask = torch.zeros_like(mask)
        for b, gl in enumerate(glens):
            if gl:
                gmask[b, S - gl:] = 1
        d = kv_deltas(cache, runner.L, r, gmask) if r >= 2 else None
        for b, j in enumerate(chunk):
            gl = glens[b]
            rec = {}
            if d is not None:
                rec.update(summarize(d[b]))
            if gl:
                hp = h[b, S - gl - 1:S - 1]                       # positions that predict the answer tokens
                tgt = ids[b, S - gl:]
                lp_sum, lp_min, ent_sum = 0.0, 0.0, 0.0
                for c in range(0, gl, 512):
                    lp = torch.log_softmax(runner.model.lm_head(hp[c:c + 512]).float(), -1)
                    tok = lp.gather(-1, tgt[c:c + 512, None]).squeeze(-1)
                    lp_sum += float(tok.sum())
                    lp_min = min(lp_min, float(tok.min()))
                    ent_sum += float(-(lp.exp() * lp).sum(-1).sum())
                rec.update(mean_logprob=lp_sum / gl, min_logprob=lp_min, mean_entropy=ent_sum / gl)
            out[j] = rec
        del cache, h
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_level", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--slots", type=int, default=32, help="pool slots at full depth; scaled by max_depth // r")
    ap.add_argument("--slots_cap", type=int, default=96)
    ap.add_argument("--score_bs", type=int, default=8)
    ap.add_argument("--model", default=OURO_BASE_ID)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="artifacts/loopdelta/stage0b")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    items = load_math_train_stratified(args.per_level, args.seed)
    runner = OuroRunner(model_id=args.model, attn_implementation=args.attn, device=args.device)
    env = environment(runner, args)
    env.update(subset=fingerprint(items), n=len(items))
    write_json(os.path.join(args.out, "env.json"), env)
    prompts = [runner.chat_ids(P.msgs(P.solver(it["question"])), think=runner.think) for it in items]

    timing_path = os.path.join(args.out, "timing.json")
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}
    for r in args.depths:
        gpath = os.path.join(args.out, f"gen_r{r}.jsonl")
        expected = len(items) * (1 + args.n_samples)
        if os.path.exists(gpath) and len(read_jsonl(gpath)) == expected:
            gens = read_jsonl(gpath)
            print(f"r={r}: reusing {gpath}", flush=True)
        else:
            gens, timing[f"r{r}"] = generate_depth(runner, prompts, items, r, args)
            write_jsonl(gpath, gens)
            write_json(timing_path, timing)
            print(f"r={r}: generated {len(gens)} in {timing[f'r{r}']}", flush=True)

        t0 = time.perf_counter()
        scores = rescore(runner, prompts, gens, r, args.score_bs)
        rows = []
        for g, sc in zip(gens, scores):
            it = items[g["q"]]
            gr = grade_math(g["text"], it["gold"])
            rows.append({"idx": it["idx"], "level": it["level"], "subject": it["subject"], "depth": r,
                         "sample": g["sample"], "prompt_tokens": len(prompts[g["q"]]),
                         "gen_tokens": len(g["ids"]), "hit_eos": g["hit_eos"],
                         "loop_tokens": g["work"]["loop_tokens"], "pred": gr["pred"],
                         "key": answer_key(gr["pred"]), "correct": gr["correct"],
                         "str_ok": gr["str_ok"], "mv_ok": gr["mv_ok"], **sc})
        write_jsonl(os.path.join(args.out, f"scored_r{r}.jsonl"), rows)
        acc = sum(x["correct"] for x in rows if x["sample"] == -1) / len(items)
        print(f"r={r}: scored in {time.perf_counter() - t0:.0f}s, greedy acc {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
