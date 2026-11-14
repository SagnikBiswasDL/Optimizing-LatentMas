#!/usr/bin/env python3
"""Loop-selection features for each question, computed before its answer.

Two kinds, both read off depth-4 passes of the reference Hugging Face model:

  prefill  the question's solver prompt prefilled once (as in LoopDelta stage 0a):
           KV and hidden-state deltas between loops, per-loop next-token entropy,
           KL between loops, top-1 agreement with the last loop, Ouro's exit gate.
  probe    the first ``--probe`` tokens of a greedy depth-4 answer, then prompt +
           probe prefilled again; the same signals averaged over the first k probe
           positions for each k in ``--ks``. This is LoopDelta's untested fallback:
           the loop-to-loop change while the model starts solving, not while it reads.

The prompt is the one ``scripts/looped_profile.py`` sends, so features and labels
describe the same input. Gold answers are never read.

    python scripts/loop_probe_features.py --model $M \
        --items artifacts/looped_serving/splits/math_train_calibration.items.jsonl \
        --only_questions artifacts/looped_serving/splits/router_pilot_questions.txt \
        --out /root/out/router_pilot/features
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
from looped.loopdelta import kv_deltas, summarize  # noqa: E402
from looped.ouro_adapter import OuroRunner, sync  # noqa: E402
from scripts.loopdelta_stage0a import prefill_batch, row_stats  # noqa: E402

THINK = "<think>"


@torch.no_grad()
def span_stats(runner: OuroRunner, mask, cache, hs_list, gate_list, depth: int, span: torch.Tensor):
    """Loop-to-loop signals averaged over the positions marked in ``span`` [B, S]."""
    B = span.shape[0]
    d = kv_deltas(cache, runner.L, depth, span)
    tail = int(span.any(0).nonzero().min())              # probe positions sit at the right end
    w = span[:, tail:].float()
    n = w.sum(-1).clamp_min(1.0)
    hs_list = [h[:, tail:] for h in hs_list]
    gate_list = [g[:, tail:] for g in gate_list]
    logp = [torch.log_softmax(runner.model.lm_head(h).float(), -1) for h in hs_list]     # [B, P, V] per loop
    ent = torch.stack([(-(lp.exp() * lp).sum(-1) * w).sum(-1) / n for lp in logp], 1)
    top1 = torch.stack([lp.argmax(-1) for lp in logp], 1)                                # [B, R, S]
    agree = torch.stack([((top1[:, t] == top1[:, -1]).float() * w).sum(-1) / n for t in range(depth)], 1)
    kl = torch.stack([(((logp[t].exp() * (logp[t] - logp[t - 1])).sum(-1)) * w).sum(-1) / n
                      for t in range(1, depth)], 1)
    lam = torch.stack([((torch.sigmoid(g[..., 0].float()) * w).sum(-1) / n) for g in gate_list], 1)
    del logp
    rows = []
    for b in range(B):
        s = summarize(d[b])
        rows.append({**s, "delta_layers_k": d[b, ..., 0].tolist(), "delta_layers_v": d[b, ..., 1].tolist(),
                     "entropy": ent[b].cpu().tolist(), "top1_agree_last": agree[b].cpu().tolist(),
                     "kl_consec": kl[b].cpu().tolist(), "gate_lambda": lam[b].cpu().tolist()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--only_questions", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--probe", type=int, default=64)
    ap.add_argument("--ks", default="16,32,64")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    os.makedirs(args.out, exist_ok=True)

    items = [json.loads(line) for line in open(args.items)]
    if args.only_questions:
        keep = {line.strip() for line in open(args.only_questions) if line.strip()}
        items = [it for it in items if it["question_id"] in keep]
        if len(items) != len(keep):
            raise SystemExit("--only_questions names questions that are not in --items")
    runner = OuroRunner(model_id=args.model, attn_implementation=args.attn)
    tok = runner.tok
    think_id = tok.convert_tokens_to_ids(THINK)
    prompts = [tok.apply_chat_template(P.msgs(P.solver(it["text"])), tokenize=True, add_generation_prompt=True)
               for it in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))
    out_path = os.path.join(args.out, "features.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(line)["question_id"] for line in open(out_path)}
    f = open(out_path, "a")
    sync(runner.device)
    t0 = time.perf_counter()
    for s in range(0, len(order), args.bs):
        chunk = [i for i in order[s:s + args.bs] if items[i]["question_id"] not in done]
        if not chunk:
            continue
        batch = [prompts[i] for i in chunk]
        _, mask, cache, hs_list, gate_list = prefill_batch(runner, batch, args.depth)
        pre = row_stats(runner, mask, cache, hs_list, gate_list, args.depth)
        del cache, hs_list, gate_list

        gen = runner.generate(batch, depth=args.depth, max_new_tokens=args.probe)
        probes = []
        for ids in gen.token_ids:
            ids = [t for t in ids if t not in runner.stop_ids] or ids[:1]
            probes.append(ids)
        seqs = [p + q for p, q in zip(batch, probes)]
        _, mask2, cache2, hs2, gates2 = prefill_batch(runner, seqs, args.depth)
        S = mask2.shape[1]
        by_k = {}
        for k in ks:
            span = torch.zeros_like(mask2)
            for b, q in enumerate(probes):
                n = min(k, len(q))
                start = S - len(q)
                span[b, start:start + n] = 1
            by_k[k] = span_stats(runner, mask2, cache2, hs2, gates2, args.depth, span)
        del cache2, hs2, gates2
        for j, i in enumerate(chunk):
            row = {"question_id": items[i]["question_id"], "prompt_tokens": len(prompts[i]),
                   "question_chars": len(items[i]["text"]), "probe_tokens": len(probes[j]),
                   "probe_hit_eos": bool(gen.hit_eos[j]), "probe_first_is_think": probes[j][0] == think_id,
                   "prefill": pre[j], "probe": {str(k): by_k[k][j] for k in ks}}
            f.write(json.dumps(row) + "\n")
        f.flush()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"{min(s + args.bs, len(order))}/{len(order)}  {time.perf_counter() - t0:.0f}s", flush=True)
    f.close()
    sync(runner.device)
    print(f"wrote {out_path} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
