#!/usr/bin/env python3
"""LoopDelta stage 0a: does the prefill Δ trajectory track MATH difficulty?

Prefill only (no decoding). Each MATH-Train question is wrapped in the solver
prompt and prefilled once at ``--depth`` loops. Per question we record:

  - KV deltas Δ_t (t=2..R) over all prompt positions and over the last position
  - hidden-state deltas between consecutive loops (after the shared final norm)
  - per-loop next-token entropy, top-1 agreement with the last loop, and the KL
    between consecutive loops at the last position (logit-space baselines)
  - Ouro's own exit gate per loop and its expected exit step (native baseline)
  - prompt length in tokens and question length in characters (confound)

Depths beyond the trained 4 loops need ``--extrapolate``; they are reported
separately because the model never ran them in training.

    python scripts/loopdelta_stage0a.py --per_level 200 --depth 4 --out artifacts/loopdelta/stage0a_r4
    python scripts/loopdelta_stage0a.py --per_level 200 --depth 8 --extrapolate --out artifacts/loopdelta/stage0a_r8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looped import prompts as P  # noqa: E402
from looped.harness import environment, write_json, write_jsonl  # noqa: E402
from looped.loopdelta import fingerprint, kv_deltas, load_math_train_stratified, summarize  # noqa: E402
from looped.ouro_adapter import OURO_BASE_ID, OuroRunner, sync  # noqa: E402


@torch.no_grad()
def prefill_batch(runner: OuroRunner, prompts, depth: int):
    ids, mask = runner._pad_left(prompts)
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    cache = runner.new_cache(depth)
    S = ids.shape[1]
    with runner.depth(depth):
        out = runner.model.model(
            input_ids=ids, attention_mask=mask if bool((mask == 0).any()) else None,
            position_ids=pos, past_key_values=cache, use_cache=True,
            cache_position=torch.arange(S, device=runner.device),
        )
    base, hs_list, gate_list = out
    return ids, mask, base.past_key_values, hs_list, gate_list


@torch.no_grad()
def row_stats(runner: OuroRunner, mask, cache, hs_list, gate_list, depth: int):
    B, S = mask.shape
    last = torch.zeros_like(mask)
    last[:, -1] = 1                                            # left padding: last column is real
    d_all = kv_deltas(cache, runner.L, depth, mask)
    d_last = kv_deltas(cache, runner.L, depth, last)

    m = mask.float().unsqueeze(-1)
    h_delta = []
    for t in range(1, depth):
        cur, prev = hs_list[t].float(), hs_list[t - 1].float()
        num = (((cur - prev) ** 2) * m).sum((1, 2))
        den = ((prev ** 2) * m).sum((1, 2)).clamp_min(1e-12)
        h_delta.append((num / den).cpu())
    h_delta = torch.stack(h_delta, 1) if h_delta else torch.zeros(B, 0)

    logp = [torch.log_softmax(runner.model.lm_head(h[:, -1]).float(), -1) for h in hs_list]
    ent = torch.stack([-(lp.exp() * lp).sum(-1) for lp in logp], 1).cpu()
    top1 = torch.stack([lp.argmax(-1) for lp in logp], 1).cpu()
    agree = (top1 == top1[:, -1:]).float()
    kl = [((logp[t].exp() * (logp[t] - logp[t - 1])).sum(-1)).cpu() for t in range(1, depth)]
    kl = torch.stack(kl, 1) if kl else torch.zeros(B, 0)

    lam = torch.stack([torch.sigmoid(g[:, -1, 0].float()) for g in gate_list], 1).cpu()
    remaining = torch.ones(B)
    pdf = []
    for t in range(depth):
        p = lam[:, t] * remaining if t < depth - 1 else remaining
        remaining = remaining * (1 - lam[:, t])
        pdf.append(p)
    pdf = torch.stack(pdf, 1)
    exp_exit = (pdf * torch.arange(1, depth + 1).float()).sum(1)

    rows = []
    for b in range(B):
        s_all, s_last = summarize(d_all[b]), summarize(d_last[b])
        rows.append({
            **s_all,
            **{f"{k}_lastpos": v for k, v in s_last.items()},
            "delta_layers_k": d_all[b, ..., 0].tolist(),
            "delta_layers_v": d_all[b, ..., 1].tolist(),
            "h_delta": h_delta[b].tolist(),
            "entropy": ent[b].tolist(),
            "top1_agree_last": agree[b].tolist(),
            "kl_consec": kl[b].tolist(),
            "gate_lambda": lam[b].tolist(),
            "gate_exit_pdf": pdf[b].tolist(),
            "gate_expected_exit": float(exp_exit[b]),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_level", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--extrapolate", action="store_true", help="allow depth beyond the trained loops")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--model", default=OURO_BASE_ID)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="artifacts/loopdelta/stage0a_r4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    items = load_math_train_stratified(args.per_level, args.seed)
    runner = OuroRunner(model_id=args.model, attn_implementation=args.attn, device=args.device)
    if args.depth > runner.max_depth:
        if not args.extrapolate:
            raise SystemExit(f"depth {args.depth} > trained {runner.max_depth}; pass --extrapolate")
        runner.max_depth = args.depth
    env = environment(runner, args)
    env.update(subset=fingerprint(items), n=len(items), trained_depth=int(runner.model.config.total_ut_steps))
    write_json(os.path.join(args.out, "env.json"), env)

    prompts = [runner.chat_ids(P.msgs(P.solver(it["question"])), think=runner.think) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))
    rows = [None] * len(items)
    sync(runner.device)
    t0 = time.perf_counter()
    for s in range(0, len(order), args.bs):
        chunk = order[s:s + args.bs]
        _, mask, cache, hs_list, gate_list = prefill_batch(runner, [prompts[i] for i in chunk], args.depth)
        stats = row_stats(runner, mask, cache, hs_list, gate_list, args.depth)
        for i, st in zip(chunk, stats):
            it = items[i]
            rows[i] = {"idx": it["idx"], "level": it["level"], "subject": it["subject"],
                       "prompt_tokens": len(prompts[i]), "question_chars": len(it["question"]), **st}
        del cache, hs_list, gate_list
        print(f"{min(s + args.bs, len(order))}/{len(order)}  {time.perf_counter() - t0:.0f}s", flush=True)
    sync(runner.device)
    write_jsonl(os.path.join(args.out, "rows.jsonl"), rows)
    print(f"wrote {len(rows)} rows to {args.out} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
