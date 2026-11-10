#!/usr/bin/env python3
"""Stage 5a: is "keep the first r loop-cache sets" a faithful depth-r history?

On the real checkpoint. The history is the planner turn *including the planner's
own generated reasoning* (generated at full depth), and the continuation is the
next agent's turn. For each r < full depth, three depth-r histories are compared:

  recompute  process the full token history at depth r in one pass
  reuse      process it at depth r, store the cache, continue later (split pass)
  truncate   process it at full depth, keep loop-cache sets [0, r)

and for latent handoffs (planner prompt + k latent steps generated at full depth):

  replay     depth r, feeding the *saved* full-depth latent vectors
  truncate   full-depth latent cache, first r sets       (should equal replay)
  regen      depth r, regenerating latents from loop-r readouts (a different input)

Reported: cache shapes/lengths, per-loop K/V max |diff|, next-token logit max
|diff|, top-1 / top-5 agreement, and greedy-continuation agreement.

  python scripts/looped_cache_check.py --device cuda --dtype bfloat16
  python scripts/looped_cache_check.py --device cpu --dtype float32
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from looped import prompts as P  # noqa: E402
from looped.harness import environment, write_json  # noqa: E402
from looped.ouro_adapter import OuroRunner, cache_summary, clone_cache, slice_cache  # noqa: E402
from looped.tasks import load_gsm8k_subset  # noqa: E402


def kv_diff(a, b, L):
    out = []
    for loop in range(len(a.key_cache) // L):
        m = 0.0
        for j in range(loop * L, (loop + 1) * L):
            for x, y in ((a.key_cache[j], b.key_cache[j]), (a.value_cache[j], b.value_cache[j])):
                if x.shape != y.shape:
                    return {"shape_mismatch": [list(x.shape), list(y.shape)]}
                m = max(m, (x.float() - y.float()).abs().max().item())
        out.append(m)
    return {"per_loop_max_abs": out}


def logit_cmp(a, b):
    a, b = a.float().view(-1), b.float().view(-1)
    t5a = set(a.topk(5).indices.tolist())
    t5b = set(b.topk(5).indices.tolist())
    return {"max_abs": (a - b).abs().max().item(), "top1_same": int(a.argmax()) == int(b.argmax()),
            "top5_overlap": len(t5a & t5b)}


def greedy(runner, cache, first_logits, depth, n):
    """Greedy continuation from an already-filled cache (cache is consumed)."""
    toks = []
    lg = first_logits
    for _ in range(n):
        t = int(lg.argmax(-1))
        toks.append(t)
        if t in runner.stop_ids:
            break
        lg, _, cache = runner.forward(depth=depth, input_ids=torch.tensor([[t]]), cache=cache)
    return toks


def agree(a, b):
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return {"same_prefix_tokens": k, "len_a": len(a), "len_b": len(b), "identical": a == b}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n_questions", type=int, default=3)
    ap.add_argument("--planner_tokens", type=int, default=96)
    ap.add_argument("--latent_steps", type=int, default=8)
    ap.add_argument("--cont_tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runner = OuroRunner(device=args.device, dtype=args.dtype)
    L, T = runner.L, runner.max_depth
    out = args.out or f"artifacts/looped/cache_check_{runner.device.type}_{args.dtype}.json"
    report = {"env": environment(runner, args), "questions": []}

    for it in load_gsm8k_subset(args.n_questions, args.seed):
        q = it["question"]
        plan_ids = runner.chat_ids(P.msgs(P.planner(q)))
        g = runner.generate([plan_ids], depth=T, max_new_tokens=args.planner_tokens)
        gen = [t for t in g.token_ids[0] if t not in runner.stop_ids]
        history = plan_ids + gen
        # close the planner turn and open the next agent's turn (ChatML)
        turn = ("<|im_end|>\n<|im_start|>user\n" + P.plan_critic(q, "(the plan above)")
                + "<|im_end|>\n<|im_start|>assistant\n<think>\n")
        cont = runner.tok(turn, add_special_tokens=False).input_ids
        H = torch.tensor([history])
        C = torch.tensor([cont])
        qrep = {"idx": it["idx"], "history_tokens": len(history), "planner_generated": len(gen),
                "cont_tokens": len(cont), "token": {}, "latent": {}}

        _, _, full = runner.forward(depth=T, input_ids=H)
        qrep["full_cache"] = cache_summary(full, L)
        for r in range(1, T):
            rec_lg, _, rec_c = runner.forward(depth=r, input_ids=torch.cat([H, C], 1))
            _, _, reuse_c = runner.forward(depth=r, input_ids=H)
            stored = clone_cache(reuse_c, L)
            reuse_lg, _, reuse_after = runner.forward(depth=r, input_ids=C, cache=clone_cache(stored, L))
            cut = slice_cache(full, r, L)
            kvd = kv_diff(stored, cut, L)
            trunc_lg, _, trunc_after = runner.forward(depth=r, input_ids=C, cache=clone_cache(cut, L))
            g_rec = greedy(runner, rec_c, rec_lg, r, args.cont_tokens)
            g_reuse = greedy(runner, reuse_after, reuse_lg, r, args.cont_tokens)
            g_trunc = greedy(runner, trunc_after, trunc_lg, r, args.cont_tokens)
            qrep["token"][str(r)] = {
                "cache_reuse": cache_summary(stored, L), "cache_truncate": cache_summary(cut, L),
                "kv_reuse_vs_truncate": kvd,
                "logits_recompute_vs_reuse": logit_cmp(rec_lg, reuse_lg),
                "logits_reuse_vs_truncate": logit_cmp(reuse_lg, trunc_lg),
                "greedy_recompute_vs_reuse": agree(g_rec, g_reuse),
                "greedy_reuse_vs_truncate": agree(g_reuse, g_trunc),
            }

        # ---- latent handoff
        Pq = torch.tensor([plan_ids])
        _, h4, c4 = runner.forward(depth=T, input_ids=Pq)
        c4, _, e4 = runner.latent_rollout(c4, h4[:, -1], depth=T, steps=args.latent_steps)
        for r in range(1, T):
            _, hr, cr = runner.forward(depth=r, input_ids=Pq)
            replay, _, _ = runner.latent_rollout(cr, hr[:, -1], depth=r, steps=args.latent_steps, forced=e4)
            cut = slice_cache(c4, r, L)
            _, hg, cg = runner.forward(depth=r, input_ids=Pq)
            regen, _, eg = runner.latent_rollout(cg, hg[:, -1], depth=r, steps=args.latent_steps)
            cos = torch.nn.functional.cosine_similarity(e4.float(), eg.float(), dim=-1).tolist()
            lg_rep, _, rep_after = runner.forward(depth=r, input_ids=C, cache=clone_cache(replay, L))
            lg_cut, _, cut_after = runner.forward(depth=r, input_ids=C, cache=clone_cache(cut, L))
            lg_reg, _, reg_after = runner.forward(depth=r, input_ids=C, cache=clone_cache(regen, L))
            gr = greedy(runner, rep_after, lg_rep, r, args.cont_tokens)
            gc = greedy(runner, cut_after, lg_cut, r, args.cont_tokens)
            gg = greedy(runner, reg_after, lg_reg, r, args.cont_tokens)
            qrep["latent"][str(r)] = {
                "kv_replay_vs_truncate": kv_diff(replay, cut, L),
                "logits_replay_vs_truncate": logit_cmp(lg_rep, lg_cut),
                "greedy_replay_vs_truncate": agree(gr, gc),
                "latent_cos_fulldepth_vs_regen": [round(x, 4) for x in cos],
                "logits_truncate_vs_regen": logit_cmp(lg_cut, lg_reg),
                "greedy_truncate_vs_regen": agree(gc, gg),
            }
        report["questions"].append(qrep)
        print({k: v for k, v in qrep.items() if k in ("idx", "history_tokens")},
              {r: (v["logits_reuse_vs_truncate"]["max_abs"], v["greedy_reuse_vs_truncate"]["identical"])
               for r, v in qrep["token"].items()}, flush=True)
        write_json(out, report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
