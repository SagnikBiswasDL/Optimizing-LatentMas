#!/usr/bin/env python3
"""Serve LatentMAS Judger requests at fixed concurrency on one vLLM engine.

One process per scheduler mode, because the mode is read when vLLM builds its
scheduler:
  stock  vLLM's own preemption (free everything, recover by prefix cache)
  tail   serving/tail_preempt_scheduler.py (pin the head, trim the tail)

Both modes load the Hugging Face latent model and the vLLM Judger on the same
GPU, as LatentMAS runs in practice. The latent agents run once per question and
the Judger prompt embeddings are cached to --embeds, so every mode and
concurrency level decodes the same prompts.

At each concurrency level C, all questions are served closed-loop with at most
C requests in flight: a new one is submitted as soon as one finishes.

  python scripts/serve_scale.py --mode stock --concurrency 8,16,32,60 --out DIR
  python scripts/serve_scale.py --mode tail  --concurrency 16,32,60 --out DIR
  python scripts/serve_scale.py --mode tail --smoke --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# The vLLM engine core runs in a child process and imports the scheduler by name.
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

SCHEDULER = "serving.tail_preempt_scheduler.TailPreemptScheduler"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["stock", "tail"], required=True)
    p.add_argument("--concurrency", default="8,16,32,60")
    p.add_argument("--model", default="Qwen/Qwen3-14B")
    p.add_argument("--latent-steps", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=20000)
    p.add_argument("--max-model-len", type=int, default=24576)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="first N questions only (0 = all 60)")
    p.add_argument("--level-timeout-s", type=float, default=0, help="abort a level after this long (0 = none)")
    p.add_argument("--embeds", default="", help="cache of Judger prompt embeddings (default OUT/embeds.pt)")
    p.add_argument("--smoke", action="store_true",
                   help="6 questions, concurrency 6, 1024 tokens, 400 KV blocks: forces preemption in about a minute")
    p.add_argument("--num-gpu-blocks-override", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_items(limit):
    from data import load_aime2024, load_aime2025
    items = []
    for source, loader in (("aime2024", load_aime2024), ("aime2025", load_aime2025)):
        for i, item in enumerate(loader(split="train")):
            row = dict(item)
            row["qid"] = f"{source}-{i}"
            items.append(row)
    return items[:limit] if limit else items


def load_method(cli):
    from methods.latent_mas import LatentMASMethod
    from models import ModelWrapper
    from utils import auto_device, set_seed

    set_seed(cli.seed)
    ns = SimpleNamespace(
        model_name=cli.model, method="latent_mas", task="aime2024", prompt="sequential",
        think=False, latent_steps=cli.latent_steps, planner_steps=None, critic_steps=None,
        refiner_steps=None, max_new_tokens=cli.max_new_tokens, temperature=0.0, top_p=1.0,
        generate_bs=1, use_vllm=True, use_second_HF_model=True, enable_prefix_caching=True,
        device="cuda:0", device2="cuda:0", tensor_parallel_size=1,
        gpu_memory_utilization=cli.gpu_memory_utilization, latent_space_realign=False,
        latent_only=False, sequential_info_only=False, agents=None, seed=cli.seed,
        embed_assembly="repaired", text_mas_context_length=-1, attn_implementation=None,
        max_model_len=cli.max_model_len, scheduler_cls=SCHEDULER,
        num_gpu_blocks_override=cli.num_gpu_blocks_override or None,
    )
    t0 = time.perf_counter()
    wrapper = ModelWrapper(cli.model, auto_device("cuda:0"), use_vllm=True, args=ns)
    load_s = time.perf_counter() - t0
    method = LatentMASMethod(wrapper, latent_steps=cli.latent_steps,
                             judger_max_new_tokens=cli.max_new_tokens, temperature=0.0,
                             top_p=1.0, generate_bs=1, args=ns)
    return method, load_s


def build_embeds(method, items, path):
    """Latent agents once per question; Judger prompt embeddings to disk."""
    import torch
    if os.path.exists(path):
        saved = torch.load(path, map_location="cpu")
        if saved["qids"][: len(items)] == [it["qid"] for it in items]:
            n = len(items)
            return saved["embeds"][:n], saved["latent_s"][:n]
        print(f"[serve] {path} does not match the question list; rebuilding", flush=True)
    embeds, latent_s = [], []
    for item in items:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        method._stop_before_vllm = True
        try:
            method.run_batch_vllm([item])
        finally:
            method._stop_before_vllm = False
        torch.cuda.synchronize()
        latent_s.append(time.perf_counter() - t0)
        embeds.append(method._last_prompt_embeds[0].detach().to("cpu").contiguous())
        print(f"[serve] embeds {item['qid']}: {embeds[-1].shape[0]} positions, {latent_s[-1]:.2f}s", flush=True)
    torch.save({"qids": [it["qid"] for it in items], "embeds": embeds, "latent_s": latent_s}, path)
    return embeds, latent_s


def grade(text, gold):
    from utils import extract_gsm8k_answer, normalize_answer
    pred = normalize_answer(extract_gsm8k_answer(text))
    try:
        return pred, pred is not None and int(pred) == int(str(gold).strip())
    except (TypeError, ValueError):
        return pred, False


def metrics():
    from prometheus_client import REGISTRY
    flat = {}
    for metric in REGISTRY.collect():
        if not metric.name.startswith("vllm:"):
            continue
        for s in metric.samples:
            if s.name.endswith("_created"):
                continue
            flat[s.name] = flat.get(s.name, 0.0) + float(s.value)
    return flat


def kv_usage(m):
    for key in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
        if key in m:
            return m[key]
    return None


def preemptions(m):
    for key in ("vllm:num_preemptions_total", "vllm:num_preemptions"):
        if key in m:
            return m[key]
    return None


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    k = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[k]


def serve_level(engine, items, embeds, latent_s, cli, conc, max_tokens, rows_path):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    queue = list(range(len(items)))
    inflight = {}
    done = []
    samples = []
    before = metrics()
    t_start = time.time()
    next_sample = 0.0
    aborted = False
    while queue or inflight:
        while queue and len(inflight) < conc:
            i = queue.pop(0)
            rid = f"{cli.mode}-c{conc}-{items[i]['qid']}"
            params = SamplingParams(
                temperature=cli.temperature, top_p=cli.top_p, top_k=cli.top_k,
                max_tokens=max_tokens, seed=cli.seed * 1000 + i,
                output_kind=RequestOutputKind.DELTA,
            )
            engine.add_request(rid, {"prompt_embeds": embeds[i]}, params)
            inflight[rid] = {"i": i, "submit": time.time(), "first": None, "tokens": 0, "text": []}
        for out in engine.step():
            st = inflight.get(out.request_id)
            if st is None:
                continue
            comp = out.outputs[0] if out.outputs else None
            if comp is not None and comp.token_ids:
                if st["first"] is None:
                    st["first"] = time.time()
                st["tokens"] += len(comp.token_ids)
                st["text"].append(comp.text)
            if out.finished:
                st["finish"] = time.time()
                st["finish_reason"] = comp.finish_reason if comp is not None else None
                st["num_cached_tokens"] = getattr(out, "num_cached_tokens", None)
                done.append((out.request_id, inflight.pop(out.request_id)))
        now = time.time() - t_start
        if now >= next_sample:
            m = metrics()
            samples.append({"t": now, "kv_usage": kv_usage(m), "preemptions": preemptions(m),
                             "running": len(inflight)})
            next_sample = now + 2.0
        if cli.level_timeout_s and now > cli.level_timeout_s and not aborted:
            engine.abort_request(list(inflight))
            for rid, st in inflight.items():
                st["finish"] = time.time()
                st["finish_reason"] = "aborted_timeout"
                done.append((rid, st))
            inflight.clear()
            queue.clear()
            aborted = True
    wall = time.time() - t_start
    after = metrics()

    rows = []
    for rid, st in done:
        item = items[st["i"]]
        text = "".join(st["text"])
        pred, ok = grade(text, item["gold"])
        rows.append({
            "mode": cli.mode, "concurrency": conc, "request_id": rid, "qid": item["qid"],
            "prompt_positions": int(embeds[st["i"]].shape[0]), "latent_s": latent_s[st["i"]],
            "output_tokens": st["tokens"], "finish_reason": st.get("finish_reason"),
            "latency_s": st["finish"] - st["submit"],
            "ttft_s": None if st["first"] is None else st["first"] - st["submit"],
            "submit_t": st["submit"] - t_start, "finish_t": st["finish"] - t_start,
            "num_cached_tokens": st.get("num_cached_tokens"),
            "gold": str(item["gold"]), "pred": pred, "correct": ok, "text": text,
        })
    with open(rows_path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    finished = [r for r in rows if r["finish_reason"] != "aborted_timeout"]
    lat = [r["latency_s"] for r in finished]
    usage = [s["kv_usage"] for s in samples if isinstance(s["kv_usage"], (int, float))]
    pre_b, pre_a = preemptions(before), preemptions(after)
    total_tokens = sum(r["output_tokens"] for r in rows)
    return {
        "mode": cli.mode, "concurrency": conc, "n": len(rows), "n_finished": len(finished),
        "aborted": aborted, "wall_s": wall,
        "requests_per_hour": len(finished) / wall * 3600 if wall else None,
        "output_tokens": total_tokens, "output_tok_per_s": total_tokens / wall if wall else None,
        "latency_p50_s": pct(lat, 0.5), "latency_p95_s": pct(lat, 0.95),
        "latency_mean_s": statistics.mean(lat) if lat else None,
        "ttft_p50_s": pct([r["ttft_s"] for r in finished if r["ttft_s"] is not None], 0.5),
        "accuracy": sum(r["correct"] for r in finished) / len(finished) if finished else None,
        "hit_cap": sum(r["finish_reason"] == "length" for r in finished),
        "preemptions": (pre_a - pre_b) if pre_a is not None and pre_b is not None else None,
        "kv_usage_peak": max(usage) if usage else None,
        "t_start": t_start, "t_end": t_start + wall,
        "samples": samples,
    }


def summarize_events(path, t0, t1):
    out = {"preempt": 0, "resume": 0, "trim_events": 0, "trimmed_blocks": 0,
           "recompute_tokens": 0, "hit_tokens": 0, "pinned_blocks_at_preempt": 0}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            ev = json.loads(line)
            if not (t0 <= ev["t"] <= t1):
                continue
            kind = ev["event"]
            if kind == "preempt":
                out["preempt"] += 1
                out["pinned_blocks_at_preempt"] += ev.get("pinned_blocks", 0)
            elif kind == "resume":
                out["resume"] += 1
                out["recompute_tokens"] += ev["recompute_tokens"]
                out["hit_tokens"] += ev["hit_tokens"]
            elif kind == "trim":
                out["trim_events"] += 1
                out["trimmed_blocks"] += ev["blocks"]
    return out


def main():
    cli = parse_args()
    os.makedirs(cli.out, exist_ok=True)
    if cli.smoke:
        cli.limit, levels, max_tokens = 6, [6], 1024
        cli.num_gpu_blocks_override = cli.num_gpu_blocks_override or 400
        cli.max_model_len = 4096
    else:
        levels, max_tokens = [int(x) for x in cli.concurrency.split(",") if x], cli.max_new_tokens
    tag = cli.mode + ("_smoke" if cli.smoke else "")
    events_path = os.path.join(cli.out, f"events_{tag}.jsonl")
    os.environ["TAIL_PREEMPT_MODE"] = cli.mode
    os.environ["TAIL_PREEMPT_LOG"] = events_path

    items = load_items(cli.limit)
    method, load_s = load_method(cli)
    engine_llm = method.model.vllm_engine
    cfg = engine_llm.llm_engine.vllm_config
    load = {
        "load_s": load_s, "mode": cli.mode,
        "enable_prefix_caching": cfg.cache_config.enable_prefix_caching,
        "num_gpu_blocks": cfg.cache_config.num_gpu_blocks,
        "block_size": cfg.cache_config.block_size,
        "max_model_len": cfg.model_config.max_model_len,
        "scheduler_cls": str(cfg.scheduler_config.scheduler_cls),
        "args": vars(cli),
    }
    try:
        import vllm
        load["vllm"] = vllm.__version__
    except Exception:
        pass
    with open(os.path.join(cli.out, f"load_{tag}.json"), "w") as handle:
        json.dump(load, handle, indent=2, default=str)
    print(json.dumps(load, indent=2, default=str), flush=True)
    if load["enable_prefix_caching"] is not True:
        raise SystemExit("prefix caching is off; both arms need it on")

    embeds_path = cli.embeds or os.path.join(cli.out, "embeds_smoke.pt" if cli.smoke else "embeds.pt")
    embeds, latent_s = build_embeds(method, items, embeds_path)

    rows_path = os.path.join(cli.out, f"rows_{tag}.jsonl")
    summary_path = os.path.join(cli.out, f"summary_{tag}.json")
    summaries = []
    if os.path.exists(summary_path) and not cli.smoke:
        summaries = json.load(open(summary_path))["levels"]
    done_levels = {s["concurrency"] for s in summaries}
    engine = engine_llm.llm_engine
    for conc in levels:
        if conc in done_levels and not cli.smoke:
            print(f"[serve] {cli.mode} c={conc} already done; skipping", flush=True)
            continue
        engine_llm.reset_prefix_cache()
        print(f"[serve] {cli.mode} c={conc} start {time.strftime('%H:%M:%S')}", flush=True)
        level = serve_level(engine, items, embeds, latent_s, cli, conc, max_tokens, rows_path)
        level["events"] = summarize_events(events_path, level["t_start"], level["t_end"])
        summaries.append(level)
        brief = {k: v for k, v in level.items() if k != "samples"}
        print("[serve] " + json.dumps(brief, default=str), flush=True)
        with open(summary_path, "w") as handle:
            json.dump({"load": load, "levels": summaries}, handle, indent=2, default=str)
    print(f"[serve] {cli.mode} complete", flush=True)


if __name__ == "__main__":
    main()
