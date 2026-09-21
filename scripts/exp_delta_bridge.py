#!/usr/bin/env python3
"""DeltaBridge driver: collect, fit PCA, oracle eval, cheap predictor.

Restart-safe. Writes artifacts/delta_bridge/CHECKIN.md after each stage so a
human can decide whether to keep spending GPU time.

  python scripts/exp_delta_bridge.py --mode collect
  python scripts/exp_delta_bridge.py --mode fit
  python scripts/exp_delta_bridge.py --mode eval --task math --n 100
  python scripts/exp_delta_bridge.py --mode eval --task aime2024 --n 30
  python scripts/exp_delta_bridge.py --mode train_coef
  python scripts/exp_delta_bridge.py --mode gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import load_aime2024, load_aime2025, load_gsm8k, load_math  # noqa: E402
from methods import default_agents  # noqa: E402
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.cache_bank import kv_mb, num_positions  # noqa: E402
from seal.delta_bridge import (  # noqa: E402
    DEFAULT_LAYERS_14B,
    DEFAULT_RANKS,
    ORACLE_ARMS,
    CoefPredictor,
    ResidualInjector,
    coefficients,
    continue_forward,
    deltas_from_residual,
    derange,
    fit_pca,
    forward_bridge,
    oracle_verdict,
    parse_int_tuple,
    pick_layer_rank,
    reconstruct_from_alpha,
)
from seal.latent_eval import (  # noqa: E402
    build_upstream_timed,
    decode_batch,
    deep_clone,
    graded,
    make_ns,
    mean_ci,
    pred_short,
    reset_peak,
    sync,
    to_cpu,
    to_dev,
)
from seal.math_cache import cache_as_past, cache_path, load_unified_cache  # noqa: E402
from utils import auto_device, set_seed  # noqa: E402


def load_task(task: str, split: str):
    if task == "aime2024":
        return list(load_aime2024(split="train"))
    if task == "aime2025":
        return list(load_aime2025(split="train"))
    if task == "math":
        return list(load_math(split=split))
    return list(load_gsm8k(split=split))


def maybe_load_model(args, ns):
    from models import ModelWrapper

    print(f"[load] {args.model_name} device={args.device}", flush=True)
    return ModelWrapper(args.model_name, auto_device(args.device), use_vllm=False, args=ns)


def default_n(task: str, smoke: bool) -> int:
    if smoke:
        return 2
    if task.startswith("aime"):
        return 30
    return 100


def default_budget(task: str) -> int:
    if task.startswith("aime"):
        return 8192
    if task == "math":
        return 2048
    return 1024


def cpu_vec(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2:
        t = t[0]
    return t.detach().float().cpu().contiguous().view(-1)


def pack_slot(slot: Dict[int, torch.Tensor], layers: Sequence[int]) -> Dict[str, torch.Tensor]:
    return {str(li): cpu_vec(slot[int(li)]) for li in layers}


def unpack_slot(row: Dict[str, torch.Tensor], layers: Sequence[int]) -> Dict[int, torch.Tensor]:
    return {int(li): row[str(li)].float() for li in layers}


def residuals_path(out_dir: str) -> str:
    return os.path.join(out_dir, "residuals.pt")


def bank_path(out_dir: str) -> str:
    return os.path.join(out_dir, "bank.pt")


def states_path(out_dir: str) -> str:
    return os.path.join(out_dir, "eval_states.pt")


def predictor_path(out_dir: str) -> str:
    return os.path.join(out_dir, "predictor.pt")


def judger_tensors(wrapper, questions, ns):
    jmsgs = [
        build_agent_message_sequential_latent_mas(
            role="judger", question=q, context="", method="latent_mas", args=ns)
        for q in questions
    ]
    _, jids, jmask, _ = wrapper.prepare_chat_batch(jmsgs, add_generation_prompt=True)
    return jids, jmask


def load_frozen(args, dtype=None):
    cache = args.cache or cache_path(os.path.join(ROOT, "artifacts/math_ladder/math1k"))
    if not os.path.isfile(cache):
        raise SystemExit(
            f"missing cache {cache}; scp from the pod or "
            "run: bash scripts/run_math_ladder.sh build"
        )
    payload = load_unified_cache(cache)
    past = cache_as_past(payload, device="cpu", dtype=dtype)
    return payload, past, cache


def save_collect_ckpt(path, meta, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"meta": meta, "rows": rows}, path)


def run_collect(args):
    os.makedirs(args.out_dir, exist_ok=True)
    layers = parse_int_tuple(args.layers, DEFAULT_LAYERS_14B)
    n_bridge = int(args.n_bridge)
    ckpt = residuals_path(args.out_dir)
    rows: List[Dict[str, Any]] = []
    if os.path.isfile(ckpt) and not args.force:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        rows = list(blob.get("rows") or [])
        print(f"[collect] resume n={len(rows)} from {ckpt}", flush=True)

    want = int(args.stat_n)
    if args.smoke:
        want = min(want, 2)
    pool = list(load_math(split="train"))
    if args.stat_pool > 0:
        pool = pool[: int(args.stat_pool)]
    done = {int(r["idx"]) for r in rows}
    remaining_idx = [i for i in range(len(pool)) if i not in done]
    remaining_idx = remaining_idx[: max(0, want - len(rows))]
    print(
        f"[collect] have={len(rows)} want={want} remaining={len(remaining_idx)} "
        f"layers={layers} n_bridge={n_bridge}",
        flush=True,
    )
    if len(rows) >= want:
        print("[collect] already complete", flush=True)
        return ckpt

    ns = make_ns(args, task="math")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for --mode collect")
    wrapper = maybe_load_model(args, ns)
    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass
    dtype = next(wrapper.model.parameters()).dtype
    payload, frozen_cpu, cache = load_frozen(args, dtype=dtype)
    frozen_cpu = cache_as_past(payload, device="cpu", dtype=dtype)
    up_agents = [a for a in default_agents() if a.role != "judger"]
    meta = {
        "model_name": args.model_name,
        "k": int(args.k),
        "layers": list(layers),
        "n_bridge": n_bridge,
        "cache": cache,
        "n_donors_cache": (payload.get("meta") or {}).get("n_donors"),
    }
    t0 = time.time()
    for idx in remaining_idx:
        it = pool[idx]
        q = it["question"]
        jids, jmask = judger_tensors(wrapper, [q], ns)
        # Real K=10
        past_r, up_t, _ = build_upstream_timed(wrapper, [q], args.k, ns, up_agents)
        past_r, h_real_slots = forward_bridge(
            wrapper, past_r, n_bridge=n_bridge, record_layers=layers,
        )
        del past_r
        # Frozen + judger prefill + bridge
        past_f = to_dev(deep_clone(frozen_cpu), wrapper.device)
        past_f, q_hid = continue_forward(
            wrapper, jids, jmask, past_f, record_layers=layers,
        )
        past_f, h_fr_slots = forward_bridge(
            wrapper, past_f, n_bridge=n_bridge, record_layers=layers,
        )
        del past_f
        torch.cuda.empty_cache()
        row = {
            "idx": int(idx),
            "gold": str(it.get("gold") or ""),
            "q": pack_slot(q_hid, layers),
            "h_real": [pack_slot(s, layers) for s in h_real_slots],
            "h_frozen": [pack_slot(s, layers) for s in h_fr_slots],
            "upstream_s": float(up_t.get("upstream") or 0.0),
        }
        rows.append(row)
        if (len(rows) % int(args.ckpt_every) == 0) or (len(rows) >= want):
            save_collect_ckpt(ckpt, {**meta, "n": len(rows)}, rows)
            print(
                f"[collect] {len(rows)}/{want} idx={idx} "
                f"up={row['upstream_s']:.2f}s elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
        else:
            print(
                f"[collect] {len(rows)}/{want} idx={idx} "
                f"up={row['upstream_s']:.2f}s elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
    save_collect_ckpt(ckpt, {**meta, "n": len(rows)}, rows)
    print(f"[collect] wrote {ckpt} n={len(rows)}", flush=True)
    write_checkin(args.out_dir)
    print("DELTA_BRIDGE_COLLECT_DONE", flush=True)
    return ckpt


def _slot_matrix(rows, key, slot, layer) -> torch.Tensor:
    xs = []
    for r in rows:
        t = r[key][int(slot)][str(int(layer))]
        xs.append(t.float().view(-1))
    return torch.stack(xs, 0)


def run_fit(args):
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = residuals_path(args.root_dir)
    if not os.path.isfile(ckpt):
        ckpt = residuals_path(args.out_dir)
    if not os.path.isfile(ckpt):
        raise SystemExit(f"missing {ckpt}; run --mode collect")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    rows = list(blob.get("rows") or [])
    meta = dict(blob.get("meta") or {})
    layers = tuple(int(x) for x in (meta.get("layers") or DEFAULT_LAYERS_14B))
    n_bridge = int(meta.get("n_bridge") or args.n_bridge)
    ranks = parse_int_tuple(args.ranks, DEFAULT_RANKS)
    print(f"[fit] n={len(rows)} layers={layers} ranks={ranks} n_bridge={n_bridge}", flush=True)
    if len(rows) < 4:
        raise SystemExit(f"need more residuals (have {len(rows)})")

    banks: Dict[str, Any] = {
        "meta": {**meta, "n": len(rows), "ranks": list(ranks)},
        "slots": {},
    }
    pick_slot = 0
    layer_banks_for_pick: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for slot in range(n_bridge):
        slot_blk: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for li in layers:
            h_r = _slot_matrix(rows, "h_real", slot, li)
            h_f = _slot_matrix(rows, "h_frozen", slot, li)
            d = h_r - h_f
            rank_blk = {}
            for r in ranks:
                fit = fit_pca(d, int(r))
                # keep tensors
                rank_blk[int(r)] = fit
                print(
                    f"[fit] slot={slot} L{li} r={r} explained={fit['explained']:.3f} "
                    f"energy={fit['energy']:.4f}",
                    flush=True,
                )
            slot_blk[int(li)] = rank_blk
        banks["slots"][str(slot)] = slot_blk
        if slot == pick_slot:
            layer_banks_for_pick = slot_blk

    pick = pick_layer_rank(
        layer_banks_for_pick,
        prefer_layer=int(args.prefer_layer),
        target_rank=int(args.target_rank),
    )
    banks["pick"] = pick
    out = bank_path(args.out_dir)
    torch.save(banks, out)
    with open(os.path.join(args.out_dir, "fit_report.json"), "w") as f:
        json.dump(_fit_json(banks), f, indent=2)
    print(f"[fit] pick {pick['reason']}", flush=True)
    print(f"[fit] wrote {out}", flush=True)
    write_checkin(args.out_dir)
    print("DELTA_BRIDGE_FIT_DONE", flush=True)
    return out


def _fit_json(banks):
    out = {"meta": banks.get("meta"), "pick": banks.get("pick"), "explained": {}}
    for slot, slot_blk in (banks.get("slots") or {}).items():
        out["explained"][str(slot)] = {}
        for li, rank_blk in slot_blk.items():
            out["explained"][str(slot)][str(li)] = {
                str(r): {
                    "explained": float(v["explained"]),
                    "energy": float(v["energy"]),
                    "n": int(v["n"]),
                }
                for r, v in rank_blk.items()
            }
    return out


def _load_bank(args):
    path = args.bank or bank_path(args.root_dir) or bank_path(args.out_dir)
    if not args.bank:
        for cand in (bank_path(args.root_dir), bank_path(args.out_dir)):
            if os.path.isfile(cand):
                path = cand
                break
        else:
            path = bank_path(args.root_dir)
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}; run --mode fit")
    return torch.load(path, map_location="cpu", weights_only=False), path


def _picked_bank(banks, slot: int = 0) -> Dict[int, Dict[str, Any]]:
    pick = banks["pick"]
    layer = int(pick["layer"])
    rank = int(pick["rank"])
    slot_blk = banks["slots"][str(int(slot))]
    return {layer: slot_blk[layer][rank]}, layer, rank


def parse_arms(raw: str) -> List[str]:
    if not raw or not str(raw).strip():
        return list(ORACLE_ARMS)
    want = [a.strip() for a in str(raw).split(",") if a.strip()]
    bad = [a for a in want if a not in ORACLE_ARMS and a not in ("predicted",)]
    if bad:
        raise SystemExit(f"unknown arms {bad}; known {ORACLE_ARMS + ('predicted',)}")
    return want


def _load_eval_states(path):
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _done_idx_arms(jsonl_path: str) -> Dict[int, set]:
    found = defaultdict(set)
    if not os.path.isfile(jsonl_path):
        return found
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            found[int(rec["idx"])].add(str(rec["method"]))
    return found


def collect_eval_states(args, wrapper, ns, frozen_cpu, items, layers, n_bridge):
    """Real vs Frozen bridge states for the eval split (no Judger decode)."""
    path = states_path(args.out_dir)
    blob = _load_eval_states(path) if not args.force else None
    have = {}
    if blob and blob.get("task") == args.task:
        have = {int(r["idx"]): r for r in blob.get("rows") or []}
    up_agents = [a for a in default_agents() if a.role != "judger"]
    rows = [have[i] for i in sorted(have)] if have else []
    t0 = time.time()
    for it in items:
        idx = int(it["idx"])
        if idx in have:
            continue
        q = it["question"]
        jids, jmask = judger_tensors(wrapper, [q], ns)
        past_r, _, _ = build_upstream_timed(wrapper, [q], args.k, ns, up_agents)
        _, h_real_slots = forward_bridge(
            wrapper, past_r, n_bridge=n_bridge, record_layers=layers,
        )
        del past_r
        past_f = to_dev(deep_clone(frozen_cpu), wrapper.device)
        past_f, q_hid = continue_forward(
            wrapper, jids, jmask, past_f, record_layers=layers,
        )
        _, h_fr_slots = forward_bridge(
            wrapper, past_f, n_bridge=n_bridge, record_layers=layers,
        )
        del past_f
        torch.cuda.empty_cache()
        row = {
            "idx": idx,
            "gold": str(it.get("gold") or ""),
            "q": pack_slot(q_hid, layers),
            "h_real": [pack_slot(s, layers) for s in h_real_slots],
            "h_frozen": [pack_slot(s, layers) for s in h_fr_slots],
        }
        have[idx] = row
        rows = [have[i] for i in sorted(have)]
        torch.save(
            {"task": args.task, "layers": list(layers), "n_bridge": n_bridge, "rows": list(have.values())},
            path,
        )
        print(
            f"[states] {args.task} {len(have)}/{len(items)} idx={idx} "
            f"elapsed={time.time()-t0:.0f}s",
            flush=True,
        )
    torch.save(
        {"task": args.task, "layers": list(layers), "n_bridge": n_bridge, "rows": list(have.values())},
        path,
    )
    return have


def _residual_at(state_row, layer: int, slot: int = 0) -> torch.Tensor:
    h_r = state_row["h_real"][slot][str(int(layer))].float().view(-1)
    h_f = state_row["h_frozen"][slot][str(int(layer))].float().view(-1)
    return h_r - h_f


def _arm_deltas(
    arm: str,
    state_row,
    donor_row,
    layer: int,
    bank_one: Dict[int, Dict[str, Any]],
    predictor=None,
) -> Optional[Dict[int, torch.Tensor]]:
    if arm == "zero":
        return None
    if arm == "predicted":
        if predictor is None:
            raise RuntimeError("predicted arm needs a predictor")
        q = state_row["q"][str(int(layer))].float().view(1, 1, -1)
        with torch.no_grad():
            alpha = predictor(q)
        rec = reconstruct_from_alpha(alpha.view(-1), bank_one[layer])
        return {int(layer): rec}
    src = donor_row if arm == "shuffled" else state_row
    dmap = {int(layer): _residual_at(src, layer, 0)}
    if arm == "oracle_full":
        return deltas_from_residual(dmap, layers=[layer], mode="full")
    if arm == "oracle_r":
        return deltas_from_residual(dmap, layers=[layer], bank=bank_one, mode="oracle_r")
    if arm == "shuffled":
        return deltas_from_residual(dmap, layers=[layer], bank=bank_one, mode="oracle_r")
    raise ValueError(arm)


def run_eval(args):
    os.makedirs(args.out_dir, exist_ok=True)
    arms = parse_arms(args.arms)
    banks, bpath = _load_bank(args)
    bank_one, layer, rank = _picked_bank(banks, slot=0)
    layers_all = tuple(int(x) for x in (banks["meta"].get("layers") or DEFAULT_LAYERS_14B))
    n_bridge = int(banks["meta"].get("n_bridge") or args.n_bridge)
    items = load_task(args.task, args.split)[: int(args.n)]
    for i, it in enumerate(items):
        it["idx"] = i
    print(
        f"[eval] task={args.task} n={len(items)} arms={arms} "
        f"layer={layer} rank={rank} n_bridge={n_bridge} bank={bpath}",
        flush=True,
    )

    ns = make_ns(args, task=args.task)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for --mode eval")
    wrapper = maybe_load_model(args, ns)
    try:
        wrapper.tokenizer.padding_side = "left"
    except Exception:
        pass
    dtype = next(wrapper.model.parameters()).dtype
    payload, _, cache = load_frozen(args, dtype=dtype)
    frozen_cpu = cache_as_past(payload, device="cpu", dtype=dtype)

    have_states = collect_eval_states(
        args, wrapper, ns, frozen_cpu, items, layers_all, n_bridge,
    )
    order = [int(it["idx"]) for it in items]
    rng = torch.Generator().manual_seed(int(args.seed) + 17)
    perm = derange(len(order), rng)
    shuffle_of = {order[i]: order[perm[i]] for i in range(len(order))}

    pred = None
    if "predicted" in arms:
        pp = args.predictor or predictor_path(args.root_dir)
        if not os.path.isfile(pp):
            pp = predictor_path(args.out_dir)
        if not os.path.isfile(pp):
            raise SystemExit(f"missing predictor {pp}")
        blob = torch.load(pp, map_location="cpu", weights_only=False)
        pred = CoefPredictor(
            d=int(blob["d"]), n_layers=int(blob["n_layers"]), rank=int(blob["rank"]),
            hidden=int(blob.get("hidden") or 256),
        )
        pred.load_state_dict(blob["state_dict"])
        pred.eval()

    tok = None
    from seal.delta_bridge import bridge_token_id
    tok = bridge_token_id(wrapper.tokenizer)
    dump = os.path.join(args.out_dir, "latency_rows.jsonl")
    if args.force and os.path.isfile(dump):
        os.remove(dump)
    done = _done_idx_arms(dump)
    dump_f = open(dump, "a")
    injector = ResidualInjector()
    injector.ensure_layers(wrapper.model, [layer])
    t_wall = time.time()
    rows_new: List[Dict[str, Any]] = []

    for it in items:
        idx = int(it["idx"])
        need = [a for a in arms if a not in done.get(idx, set())]
        if not need:
            continue
        gold = it["gold"]
        q = it["question"]
        jids, jmask = judger_tensors(wrapper, [q], ns)
        past_f = to_dev(deep_clone(frozen_cpu), wrapper.device)
        past_j, _ = continue_forward(wrapper, jids, jmask, past_f)
        del past_f
        past_j_cpu = to_cpu(past_j)
        del past_j
        bridge_ids = torch.tensor([[tok]], dtype=torch.long, device=wrapper.device)
        bridge_mask = torch.ones_like(bridge_ids)
        state_row = have_states[idx]
        donor = have_states[shuffle_of[idx]]
        for arm in need:
            deltas = _arm_deltas(arm, state_row, donor, layer, bank_one, predictor=pred)
            past_use = to_dev(deep_clone(past_j_cpu), wrapper.device)
            injector.set_deltas(deltas or {})
            if deltas:
                injector.enable()
            else:
                injector.disable()
            reset_peak()
            sync()
            t1 = time.perf_counter()
            texts, ntoks, eoss = decode_batch(
                wrapper, bridge_ids, bridge_mask, [past_use],
                args.judger_budget, temperature=args.temperature, top_p=args.top_p,
            )
            sync()
            t_j = time.perf_counter() - t1
            injector.disable()
            rec = {
                "idx": idx,
                "method": arm,
                "task": args.task,
                "layer": int(layer),
                "rank": int(rank),
                "n_bridge": n_bridge,
                "correct": bool(graded(texts[0], gold, args.task)),
                "pred": pred_short(texts[0]),
                "tokens": int(ntoks[0]),
                "eos": bool(eoss[0]),
                "judger_s": float(t_j),
                "e2e_s": float(t_j),
                "cache_pos": int(num_positions(past_use)),
                "cache_mb": float(kv_mb(past_use)),
                "shuffle_src": int(shuffle_of[idx]) if arm == "shuffled" else idx,
            }
            rows_new.append(rec)
            dump_f.write(json.dumps(rec) + "\n")
            dump_f.flush()
            done[idx].add(arm)
            print(
                f"[eval] {args.task} idx={idx} {arm} correct={rec['correct']} "
                f"tok={rec['tokens']} J={t_j:.1f}s elapsed={time.time()-t_wall:.0f}s",
                flush=True,
            )
            del past_use, texts
            torch.cuda.empty_cache()
        del past_j_cpu

    dump_f.close()
    report = summarize_eval(dump, arms, args, layer, rank)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print_eval_table(report)
    write_checkin(args.root_dir)
    print(f"[done] {os.path.join(args.out_dir, 'report.json')}", flush=True)
    print("DELTA_BRIDGE_EVAL_DONE", flush=True)
    return report


def summarize_eval(jsonl_path, arms, args, layer, rank):
    by = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            by[str(rec["method"])].append(rec)
    out: Dict[str, Any] = {
        "config": {
            "task": args.task, "n": args.n, "k": args.k,
            "judger_budget": args.judger_budget, "seed": args.seed,
            "layer": layer, "rank": rank, "n_bridge": args.n_bridge,
            "arms": arms,
        },
        "arms": {},
    }
    for arm in arms:
        rs = sorted(by.get(arm, []), key=lambda r: r["idx"])
        if not rs:
            continue
        blk = {
            "n": len(rs),
            "acc": float(np.mean([r["correct"] for r in rs])),
            "tokens": mean_ci([r["tokens"] for r in rs], seed=args.seed),
            "judger_s": mean_ci([r["judger_s"] for r in rs], seed=args.seed),
            "correct": [bool(r["correct"]) for r in rs],
            "idx": [int(r["idx"]) for r in rs],
        }
        out["arms"][arm] = blk
    z = out["arms"].get("zero")
    o = out["arms"].get("oracle_r")
    s = out["arms"].get("shuffled")
    fl = out["arms"].get("oracle_full")
    if z and o and z["n"] == o["n"]:
        out["verdict"] = oracle_verdict(
            zero_correct=z["correct"],
            oracle_correct=o["correct"],
            shuffled_correct=(s["correct"] if s else None),
            full_correct=(fl["correct"] if fl else None),
        )
    return out


def print_eval_table(report):
    print("\n=== DELTABRIDGE EVAL ===", flush=True)
    for name, a in report.get("arms", {}).items():
        tok = (a.get("tokens") or {}).get("mean") or 0.0
        print(f"  {name:<14} acc={a.get('acc', 0):.3f} n={a.get('n')} tok={tok:.0f}", flush=True)
    v = report.get("verdict") or {}
    if v:
        print(f"  recommend={v.get('recommend')}", flush=True)
        print(f"  {v.get('reason')}", flush=True)


def run_train_coef(args):
    """Cheap q -> alpha regression on collected MATH-train residuals. No 14B."""
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = residuals_path(args.root_dir)
    if not os.path.isfile(ckpt):
        ckpt = residuals_path(args.out_dir)
    banks, _ = _load_bank(args)
    if not os.path.isfile(ckpt):
        raise SystemExit(f"missing {ckpt}")
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    rows = list(blob["rows"])
    bank_one, layer, rank = _picked_bank(banks, slot=0)
    q = torch.stack([r["q"][str(layer)].float().view(-1) for r in rows], 0)
    d = _slot_matrix(rows, "h_real", 0, layer) - _slot_matrix(rows, "h_frozen", 0, layer)
    alpha = torch.stack([coefficients(d[i], bank_one[layer]) for i in range(d.shape[0])], 0)
    n = q.shape[0]
    n_dev = max(1, min(int(0.1 * n), n - 1))
    q_tr, q_dv = q[n_dev:], q[:n_dev]
    a_tr, a_dv = alpha[n_dev:], alpha[:n_dev]
    if q_tr.shape[0] == 0:
        q_tr, a_tr = q, alpha
    model = CoefPredictor(d=q.shape[1], n_layers=1, rank=rank, hidden=int(args.hidden))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    model.train()
    t0 = time.time()
    best = 1e9
    best_state = None
    for step in range(int(args.steps)):
        perm = torch.randperm(q_tr.shape[0])[: int(args.bs)]
        pred = model(q_tr[perm].unsqueeze(1))[:, 0, :]
        loss = F.mse_loss(pred, a_tr[perm])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == int(args.steps) - 1:
            model.eval()
            with torch.no_grad():
                dv = F.mse_loss(model(q_dv.unsqueeze(1))[:, 0, :], a_dv).item()
            model.train()
            if dv < best:
                best = dv
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[train_coef] step={step} train={loss.item():.5f} dev={dv:.5f}", flush=True)
    model.load_state_dict(best_state or model.state_dict())
    dest = args.root_dir or args.out_dir
    out = predictor_path(dest)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "d": q.shape[1],
            "n_layers": 1,
            "rank": rank,
            "layer": layer,
            "hidden": int(args.hidden),
            "dev_mse": best,
        },
        out,
    )
    with open(os.path.join(dest, "train_coef_report.json"), "w") as f:
        json.dump(
            {"n": n, "layer": layer, "rank": rank, "dev_mse": best, "steps": args.steps,
             "elapsed_s": time.time() - t0},
            f, indent=2,
        )
    print(f"[train_coef] wrote {out} dev_mse={best:.5f}", flush=True)
    write_checkin(dest)
    print("DELTA_BRIDGE_TRAIN_COEF_DONE", flush=True)
    return out


def _read_report(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_checkin(root: str) -> str:
    """Human-readable status for a 2-hour check-in."""
    root = root or os.path.join(ROOT, "artifacts/delta_bridge")
    os.makedirs(root, exist_ok=True)
    collect_p = residuals_path(root)
    fit_p = os.path.join(root, "fit_report.json")
    math_p = os.path.join(root, "eval_math", "report.json")
    gsm_p = os.path.join(root, "eval_gsm8k", "report.json")
    aime_p = os.path.join(root, "eval_aime24", "report.json")
    pred_p = os.path.join(root, "train_coef_report.json")

    n_collect = 0
    if os.path.isfile(collect_p):
        try:
            n_collect = len(torch.load(collect_p, map_location="cpu", weights_only=False).get("rows") or [])
        except Exception:
            n_collect = -1
    fit = _read_report(fit_p)
    math = _read_report(math_p)
    gsm = _read_report(gsm_p)
    aime = _read_report(aime_p)
    pred = _read_report(pred_p)

    recommend = "running"
    reason = "still collecting or waiting on a stage"
    if aime and aime.get("verdict"):
        recommend = aime["verdict"].get("recommend") or recommend
        reason = aime["verdict"].get("reason") or reason
    elif math and math.get("verdict"):
        zacc = (math.get("arms") or {}).get("zero", {}).get("acc")
        if zacc is not None and zacc < 0.60:
            recommend = "stop_protocol"
            reason = (
                f"MATH zero acc={zacc:.3f} << frozen 0.76; dummy bridge broke the "
                "protocol. Do not spend AIME hours until this is fixed."
            )
        else:
            recommend = "continue_aime"
            reason = "MATH protocol looks usable; AIME oracle is the next spend."

    status = {
        "recommend": recommend,
        "reason": reason,
        "collect_n": n_collect,
        "fit": (fit or {}).get("pick"),
        "math": _arm_accs(math),
        "gsm8k": _arm_accs(gsm),
        "aime": _arm_accs(aime),
        "aime_verdict": (aime or {}).get("verdict"),
        "train_coef": pred,
        "next": _next_step(recommend, n_collect, fit, math, gsm, aime, pred),
    }
    with open(os.path.join(root, "gate.json"), "w") as f:
        json.dump(status, f, indent=2)
    lines = [
        "# DeltaBridge check-in",
        "",
        f"**recommend:** `{recommend}`",
        "",
        reason,
        "",
        f"- collect n = {n_collect}",
        f"- pick = {status['fit']}",
        f"- MATH = {status['math']}",
        f"- GSM8K = {status['gsm8k']}",
        f"- AIME = {status['aime']}",
        f"- next = {status['next']}",
        "",
        "Locked baselines: Frozen AIME 56.7% (17/30), Real K=10 66.7% (20/30).",
        "Do not tune on recovered AIME items. Skip train/SEAL/evict unless recommend=go.",
        "",
    ]
    path = os.path.join(root, "CHECKIN.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[checkin] {recommend}: {reason}", flush=True)
    return path


def _arm_accs(report):
    if not report:
        return None
    return {k: round(float(v.get("acc")), 4) for k, v in (report.get("arms") or {}).items()}


def _next_step(recommend, n_collect, fit, math, gsm, aime, pred):
    if n_collect <= 0:
        return "collect MATH-train residuals"
    if not fit:
        return "fit PCA"
    if not math:
        return "eval MATH n=100 oracle arms"
    if recommend == "stop_protocol":
        return "fix the dummy-bridge protocol; do not run AIME"
    if not gsm:
        return "eval GSM8K n=100 (cheap sanity)"
    if not aime:
        return "eval AIME24 n=30 oracle arms (~1.2h/arm)"
    if recommend == "go" and not pred:
        return "train_coef then eval --arms predicted"
    if recommend == "retry_4token":
        return "re-collect with --n_bridge 4 and rerun oracle"
    if recommend and str(recommend).startswith("stop"):
        return "stop the campaign; write the negative"
    return "see gate.json"


def run_gate(args):
    path = write_checkin(args.out_dir)
    print(f"[gate] {path}", flush=True)
    print("DELTA_BRIDGE_GATE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", default="gate",
        choices=["collect", "fit", "eval", "train_coef", "gate", "checkin"],
    )
    ap.add_argument("--model_name", default="Qwen/Qwen3-14B")
    ap.add_argument("--task", default="math",
                    choices=["gsm8k", "math", "aime2024", "aime2025"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--stat_n", type=int, default=1000)
    ap.add_argument("--stat_pool", type=int, default=0)
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--judger_budget", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--layers", default="16,24,28,32")
    ap.add_argument("--ranks", default="4,8,16,32")
    ap.add_argument("--n_bridge", type=int, default=1)
    ap.add_argument("--prefer_layer", type=int, default=28)
    ap.add_argument("--target_rank", type=int, default=16)
    ap.add_argument("--arms", default="zero,oracle_full,oracle_r,shuffled")
    ap.add_argument("--cache", default="")
    ap.add_argument("--bank", default="")
    ap.add_argument("--predictor", default="")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="artifacts/delta_bridge")
    ap.add_argument("--root_dir", default="artifacts/delta_bridge")
    args = ap.parse_args()

    if args.n <= 0:
        args.n = default_n(args.task, args.smoke)
    if args.judger_budget <= 0:
        args.judger_budget = 64 if args.smoke else default_budget(args.task)
    if args.smoke:
        args.k = min(args.k, 4)
        args.stat_n = min(args.stat_n, 2)
        print(f"[smoke] k={args.k} n={args.n} donors={args.stat_n} T={args.judger_budget}", flush=True)

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.root_dir, exist_ok=True)

    if args.mode == "collect":
        run_collect(args)
    elif args.mode == "fit":
        run_fit(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "train_coef":
        run_train_coef(args)
    else:
        run_gate(args)


if __name__ == "__main__":
    main()
