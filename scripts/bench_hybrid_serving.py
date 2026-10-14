#!/usr/bin/env python3
"""Benchmark the optional hybrid LatentMAS path.

Hugging Face runs Planner, Critic, and Refiner. vLLM runs the Judger.
The authors' published results use Hugging Face for every agent. This
script measures the hybrid path, and it says so in every output file.

Phases
  env           Machine, packages, commit. No model load.
  one           One request on the authors' assembly. Batch size 1 crashes.
  one-repaired  Same request after the baseline repair.
  lengths       Three real prompt lengths, one at a time.
  batch         Offline batches 1, 2, 4, 8.
  cache         Repeated embeds versus a perturbed embed of the same length.
  offered       Bounded queue in front of the blocking pipeline.
  summarize     Aggregate a results directory. Keeps failures.

``--stop fixed`` forces the output length and is a synthetic speed run.
``--stop eos`` is the natural-answer run. Do not mix them in one speed claim.

Example
  python scripts/bench_hybrid_serving.py --phase env
  python scripts/bench_hybrid_serving.py --phase one --max-new-tokens 32
  python scripts/bench_hybrid_serving.py --phase one-repaired --max-new-tokens 32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import queue
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Fixed questions. Lengths differ on purpose. Gold is the integer the
# question asks for, so a short smoke run can be scored without GSM8K.
QUESTIONS = [
    {"id": "add", "question": "What is 17 + 25?", "gold": "42", "solution": "42"},
    {"id": "packs", "question": "A store sells pencils in packs of 12. Maya buys 3 packs and gives 6 pencils to her brother. How many pencils does Maya have left?", "gold": "30", "solution": "30"},
    {"id": "rows", "question": "A garden has 9 rows of plants. Each row has 8 plants. A storm destroys 14 plants. How many plants remain?", "gold": "58", "solution": "58"},
    {"id": "tickets", "question": "Tickets cost 7 dollars each. A class buys 15 tickets and gets a 5 dollar discount on the total. How many dollars does the class pay?", "gold": "100", "solution": "100"},
    {"id": "books", "question": "Noah reads 18 pages a day for 6 days, then 12 pages a day for 4 days. How many pages does he read in all?", "gold": "156", "solution": "156"},
    {"id": "tiles", "question": "A floor is 14 tiles wide and 9 tiles long. Each tile covers 1 square foot. A rug covers 26 square feet. How many square feet of floor remain uncovered?", "gold": "100", "solution": "100"},
    {"id": "beads", "question": "Lina has 8 bags of 24 beads. She uses 51 beads for a bracelet and 37 beads for a necklace. How many beads are left?", "gold": "104", "solution": "104"},
    {"id": "long", "question": (
        "Read the following classroom notes before solving. "
        + ("The notes mention supplies, seating, and a field trip, and none of those details change the arithmetic. " * 40)
        + "Using only the last sentence, compute 3 times 15 minus 8."
    ), "gold": "37", "solution": "37"},
]


def _run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=30)
        return out.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def git_commit():
    return _run(["git", "-C", ROOT, "rev-parse", "HEAD"])


def collect_env():
    env = {
        "commit": git_commit(),
        "branch": _run(["git", "-C", ROOT, "rev-parse", "--abbrev-ref", "HEAD"]),
        "baseline_label": "optional hybrid HF+vLLM path, not the HF-only published configuration",
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    versions = {}
    for name in ("torch", "transformers", "vllm", "numpy"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    env["packages"] = versions
    try:
        import torch
        env["torch_cuda"] = torch.version.cuda
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["gpu_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": prop.name,
                    "total_gb": round(prop.total_memory / (1024 ** 3), 2),
                    "capability": f"{prop.major}.{prop.minor}",
                })
        env["gpus"] = gpus
    except Exception as exc:
        env["cuda_available"] = False
        env["torch_error"] = str(exc)
    env["nvidia_smi"] = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current",
        "--format=csv",
    ])
    env["nvlink"] = _run(["nvidia-smi", "nvlink", "--status"])
    return env


class GpuPoller(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.wait(0.5):
            text = _run([
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            self.samples.append({"t": time.time(), "raw": text})

    def stop(self):
        self._halt.set()
        self.join(timeout=2)


def prometheus_vllm():
    try:
        from prometheus_client import REGISTRY
    except Exception as exc:
        return {"error": str(exc)}
    out = {}
    for metric in REGISTRY.collect():
        if not metric.name.startswith("vllm:"):
            continue
        out[metric.name] = [
            {"name": s.name, "labels": dict(s.labels), "value": s.value}
            for s in metric.samples
        ]
    return out


def cache_config(llm):
    found = {}
    engine = getattr(llm, "llm_engine", None)
    cfg = getattr(engine, "vllm_config", None)
    cache = getattr(cfg, "cache_config", None) if cfg is not None else None
    if cache is not None:
        for key in (
            "enable_prefix_caching",
            "block_size",
            "num_gpu_blocks",
            "gpu_memory_utilization",
            "cache_dtype",
        ):
            if hasattr(cache, key):
                found[key] = getattr(cache, key)
    model_cfg = getattr(cfg, "model_config", None) if cfg is not None else None
    if model_cfg is not None:
        for key in ("enable_prompt_embeds", "dtype", "max_model_len", "revision"):
            if hasattr(model_cfg, key):
                value = getattr(model_cfg, key)
                found[key] = str(value) if not isinstance(value, (int, float, bool, str, type(None))) else value
    attn = getattr(cfg, "attention_config", None) if cfg is not None else None
    if attn is not None and hasattr(attn, "backend"):
        found["attention_backend"] = str(attn.backend)
    return found


def _make_grabber():
    import logging

    class Grab(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines = []

        def emit(self, record):
            try:
                self.lines.append(record.getMessage())
            except Exception:
                return

    grab = Grab()
    grab.setLevel(logging.INFO)
    for name in ("vllm", "vllm.engine", ""):
        logger = logging.getLogger(name) if name else logging.getLogger()
        logger.addHandler(grab)
        logger.setLevel(logging.INFO)
    return grab


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_rows(rows):
    ok = [r for r in rows if not r.get("error")]
    bad = [r for r in rows if r.get("error")]
    steady = [r for r in ok if not r.get("warmup")]
    lat = [r["e2e_s"] for r in steady if r.get("e2e_s") is not None]
    walls = [r["e2e_s"] for r in steady if r.get("count_wall", True) and r.get("e2e_s") is not None]
    toks = [r.get("output_tokens") or 0 for r in steady]
    wall = sum(walls)
    summary = {
        "n_rows": len(rows),
        "n_error": len(bad),
        "n_warmup": sum(1 for r in ok if r.get("warmup")),
        "n_steady": len(steady),
        "errors": [r.get("error") for r in bad],
        "output_tokens_total": sum(toks),
        "output_tokens_mean": (sum(toks) / len(toks)) if toks else None,
        "e2e_s_mean": (sum(lat) / len(lat)) if lat else None,
        "e2e_s_min": min(lat) if lat else None,
        "e2e_s_max": max(lat) if lat else None,
        "requests_per_s": (len(steady) / wall) if wall else None,
        "output_tokens_per_s": (sum(toks) / wall) if wall else None,
    }
    if len(lat) >= 3:
        summary["e2e_s_p50"] = percentile(lat, 0.50)
    if len(lat) >= 20:
        summary["e2e_s_p95"] = percentile(lat, 0.95)
    else:
        summary["e2e_s_p95"] = None
        summary["p95_note"] = "omitted; fewer than 20 steady requests"
    return summary


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, default=str) + "\n")


def make_namespace(cli, batch_size):
    return SimpleNamespace(
        model_name=cli.model,
        method="latent_mas",
        task="gsm8k",
        prompt="sequential",
        think=False,
        latent_steps=cli.latent_steps,
        planner_steps=None,
        critic_steps=None,
        refiner_steps=None,
        max_new_tokens=cli.max_new_tokens,
        temperature=cli.temperature,
        top_p=cli.top_p,
        generate_bs=max(batch_size, 1),
        use_vllm=True,
        use_second_HF_model=True,
        enable_prefix_caching=cli.prefix_caching,
        device=cli.device,
        device2=cli.device2,
        tensor_parallel_size=1,
        gpu_memory_utilization=cli.gpu_memory_utilization,
        latent_space_realign=False,
        latent_only=False,
        sequential_info_only=False,
        agents=None,
        seed=cli.seed,
        embed_assembly=cli.assembly,
        text_mas_context_length=-1,
        attn_implementation=None,
    )


def apply_stopping(method, stop, max_tokens, temperature, top_p):
    from vllm import SamplingParams
    if stop == "fixed":
        method.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            min_tokens=max_tokens,
            ignore_eos=True,
        )
        method.temperature = 0.0
        method.top_p = 1.0
    else:
        method.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )


def load_method(cli, batch_size):
    import torch
    from methods.latent_mas import LatentMASMethod
    from models import ModelWrapper
    from utils import auto_device, set_seed

    set_seed(cli.seed)
    ns = make_namespace(cli, batch_size)
    grab = _make_grabber()
    t0 = time.perf_counter()
    wrapper = ModelWrapper(cli.model, auto_device(cli.device), use_vllm=True, args=ns)
    load_s = time.perf_counter() - t0
    method = LatentMASMethod(
        wrapper,
        latent_steps=cli.latent_steps,
        judger_max_new_tokens=cli.max_new_tokens,
        temperature=cli.temperature,
        top_p=cli.top_p,
        generate_bs=ns.generate_bs,
        args=ns,
    )
    apply_stopping(method, cli.stop, cli.max_new_tokens, cli.temperature, cli.top_p)
    info = {
        "load_s": load_s,
        "cache_config": cache_config(wrapper.vllm_engine),
        "init_logs": [line for line in grab.lines if any(
            s in line.lower() for s in ("prefix", "kv cache", "gpu blocks", "prompt embed", "dtype")
        )],
        "padding_side": wrapper.tokenizer.padding_side,
        "torch_dtype": str(getattr(wrapper.HF_model, "dtype", None)),
    }
    return method, info


def run_once(method, items, *, warmup, repeat, batch_size, stop):
    method.stage_log = []
    poller = GpuPoller()
    poller.start()
    t_submit = time.time()
    t0 = time.perf_counter()
    error = None
    results = []
    try:
        results = method.run_batch_vllm(items)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    e2e = time.perf_counter() - t0
    try:
        poller.stop()
    except Exception:
        traceback.print_exc()
    stages = list(method.stage_log)
    rows = []
    if error and not results:
        rows.append({
            "error": error,
            "warmup": warmup,
            "repeat": repeat,
            "batch_size": batch_size,
            "stop": stop,
            "e2e_s": e2e,
            "count_wall": True,
            "t_submit": t_submit,
            "n_items": len(items),
            "question_ids": [item["id"] for item in items],
            "stages": stages,
            "gpu_samples": poller.samples,
        })
        return rows
    for offset, (item, result) in enumerate(zip(items, results)):
        metrics = result.get("metrics") or {}
        first = metrics.get("first_token_time")
        arrival = metrics.get("arrival_time")
        row = {
            "error": error,
            "warmup": warmup,
            "repeat": repeat,
            "batch_size": batch_size,
            "stop": stop,
            "question_id": item["id"],
            "gold": item["gold"],
            "prediction": result.get("prediction"),
            "correct": result.get("correct"),
            "finish_reason": result.get("finish_reason"),
            "output_tokens": result.get("output_tokens"),
            "prompt_embed_tokens": result.get("prompt_embed_tokens"),
            "num_cached_tokens": result.get("num_cached_tokens"),
            "embed_assembly": result.get("embed_assembly"),
            "e2e_s": e2e,
            "count_wall": offset == 0,
            "t_submit": t_submit,
            "ttft_from_vllm_arrival_s": (
                None if first is None or arrival is None else first - arrival
            ),
            "vllm_metrics": metrics,
            "stages": stages,
            "raw_prediction": result.get("raw_prediction"),
        }
        rows.append(row)
    if rows:
        rows[0]["gpu_samples"] = poller.samples
    return rows


def require_cuda(cli):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Phase env can run without a GPU.")
    count = torch.cuda.device_count()
    if count >= 2 and cli.device == "auto":
        cli.device, cli.device2 = "cuda:0", "cuda:1"
    elif count == 1 and cli.allow_single_gpu:
        cli.device = cli.device2 = "cuda:0"
        if cli.gpu_memory_utilization > 0.5:
            cli.gpu_memory_utilization = 0.45
            print("Single GPU: lowering vLLM memory utilization to 0.45 so the HF copy can load.", flush=True)
    elif count == 1:
        raise SystemExit(
            "One GPU is visible. The hybrid path keeps a second HF copy. "
            "Re-run with --allow-single-gpu on an 80GB+ GPU, or provide two GPUs."
        )
    elif cli.device == "auto":
        cli.device, cli.device2 = "cuda:0", "cuda:1"


def measured_calls(cli, method, groups):
    rows = []
    if cli.warmup and cli.phase != "one":
        got = run_once(
            method, groups[0],
            warmup=True, repeat=-1, batch_size=len(groups[0]), stop=cli.stop,
        )
        rows.extend(got)
        if any(r.get("error") for r in got) and not cli.keep_going:
            return rows
    for repeat in range(cli.repeats):
        for items in groups:
            got = run_once(
                method, items,
                warmup=False, repeat=repeat, batch_size=len(items), stop=cli.stop,
            )
            rows.extend(got)
            if any(r.get("error") for r in got) and not cli.keep_going:
                return rows
    return rows


def phase_cache(cli, method, out_dir):
    """Distinct embeddings, same length, to see whether prefix caching reuses them."""
    import torch
    items = [QUESTIONS[0]]
    rows = measured_calls(cli, method, [items])
    embeds = getattr(method, "_last_prompt_embeds", None)
    if not embeds:
        append_jsonl(os.path.join(out_dir, "rows.jsonl"), {
            "error": "no prompt embeddings were captured; cache probe did not run",
            "phase": "cache",
        })
        return rows
    base = embeds[0].detach()
    changed = base.clone()
    changed[changed.shape[0] // 2] = changed[changed.shape[0] // 2] + 1
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8)
    probes = [("repeat_identical", base), ("distinct_embed", changed)]
    for name, tensor in probes:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = method.model.vllm_engine.generate([{"prompt_embeds": tensor}], params)
        elapsed = time.perf_counter() - t0
        choice = outputs[0].outputs[0]
        row = {
            "phase": "cache",
            "probe": name,
            "e2e_s": elapsed,
            "output_tokens": len(choice.token_ids),
            "text": choice.text,
            "num_cached_tokens": getattr(outputs[0], "num_cached_tokens", None),
            "finish_reason": choice.finish_reason,
            "warmup": False,
            "stop": "eos",
        }
        rows.append(row)
        append_jsonl(os.path.join(out_dir, "rows.jsonl"), row)
    texts = [r.get("text") for r in rows if r.get("probe")]
    note = {
        "identical_and_distinct_text_match": len(texts) == 2 and texts[0] == texts[1],
        "effective_prefix_caching": cache_config(method.model.vllm_engine).get("enable_prefix_caching"),
    }
    write_json(os.path.join(out_dir, "cache_probe.json"), note)
    return rows


def phase_offered(cli, method, out_dir):
    """Offered-rate probe of the blocking pipeline. The engine does not overlap requests."""
    pending = queue.Queue(maxsize=cli.queue_limit)
    dropped = []
    items = QUESTIONS[: cli.offered_requests]
    interval = 1.0 / cli.offered_rate if cli.offered_rate > 0 else 0.0

    def produce():
        for item in items:
            try:
                pending.put(item, timeout=0.01)
            except queue.Full:
                dropped.append(item["id"])
            if interval:
                time.sleep(interval)

    producer = threading.Thread(target=produce, daemon=True)
    t0 = time.perf_counter()
    producer.start()
    rows = []
    seen = 0
    while seen < len(items):
        try:
            item = pending.get(timeout=interval + 5)
        except queue.Empty:
            if not producer.is_alive() and pending.empty():
                break
            continue
        depth = pending.qsize()
        got = run_once(method, [item], warmup=False, repeat=0, batch_size=1, stop=cli.stop)
        for row in got:
            row["queue_depth_after_take"] = depth
            row["phase"] = "offered"
        rows.extend(got)
        seen += 1
    producer.join(timeout=2)
    wall = time.perf_counter() - t0
    write_json(os.path.join(out_dir, "offered.json"), {
        "offered_rate_rps": cli.offered_rate,
        "queue_limit": cli.queue_limit,
        "submitted": len(items),
        "dropped": dropped,
        "wall_s": wall,
        "achieved_rps": None if wall == 0 else (len(items) - len(dropped)) / wall,
        "note": "Upstream LLM.generate is synchronous. This queue sits in the client. HF preparation and vLLM generation do not overlap.",
    })
    return rows


def run_phase(cli):
    out_dir = cli.out or os.path.join(
        ROOT, "artifacts", "serving", f"{cli.phase}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)
    env = collect_env()
    if cli.phase == "one":
        cli.assembly = "original"
    elif cli.phase in {"one-repaired", "lengths", "batch", "cache", "offered"}:
        cli.assembly = "repaired"
    write_json(os.path.join(out_dir, "env.json"), env)
    config = vars(cli).copy()
    config["out"] = out_dir
    config["commit"] = env.get("commit")
    write_json(os.path.join(out_dir, "config.json"), config)
    if cli.phase == "env":
        print(json.dumps(env, indent=2, default=str))
        print(f"wrote {out_dir}")
        return 0 if env.get("cuda_available") else 2

    require_cuda(cli)
    batch_size = 8 if cli.phase == "batch" else 1
    method, load_info = load_method(cli, batch_size)
    write_json(os.path.join(out_dir, "load.json"), load_info)
    print(json.dumps(load_info, indent=2, default=str), flush=True)

    if cli.phase in {"one", "one-repaired"}:
        groups = [[QUESTIONS[0]]]
        rows = measured_calls(cli, method, groups)
    elif cli.phase == "lengths":
        groups = [[q] for q in (QUESTIONS[0], QUESTIONS[1], QUESTIONS[-1])]
        rows = measured_calls(cli, method, groups)
    elif cli.phase == "batch":
        groups = [QUESTIONS[:n] for n in (1, 2, 4, 8)]
        rows = measured_calls(cli, method, groups)
    elif cli.phase == "cache":
        rows = phase_cache(cli, method, out_dir)
    elif cli.phase == "offered":
        rows = phase_offered(cli, method, out_dir)
    else:
        raise SystemExit(f"unknown phase {cli.phase}")

    for row in rows:
        if row.get("phase") == "cache" and row.get("probe"):
            continue
        append_jsonl(os.path.join(out_dir, "rows.jsonl"), row)
    summary = summarize_rows(rows)
    summary["load_s"] = load_info.get("load_s")
    summary["effective_prefix_caching"] = load_info.get("cache_config", {}).get("enable_prefix_caching")
    write_json(os.path.join(out_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out_dir}")
    return 1 if summary["n_error"] else 0


def summarize_dir(path):
    rows = []
    row_path = os.path.join(path, "rows.jsonl")
    with open(row_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    summary = summarize_rows(rows)
    print(json.dumps(summary, indent=2))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="env", choices=[
        "env", "one", "one-repaired", "lengths", "batch", "cache", "offered", "summarize",
    ])
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--latent-steps", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Greedy. run.py's default hybrid command is temperature 0.6; this harness pins 0 unless overridden.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", choices=["eos", "fixed"], default="eos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--assembly", choices=["original", "repaired"], default="repaired")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device2", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--allow-single-gpu", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--offered-rate", type=float, default=0.2)
    parser.add_argument("--offered-requests", type=int, default=4)
    parser.add_argument("--queue-limit", type=int, default=2)
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary-dir", default=None)
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.phase == "summarize":
        if not cli.summary_dir:
            raise SystemExit("--summary-dir is required")
        summarize_dir(cli.summary_dir)
        return 0
    return run_phase(cli)


if __name__ == "__main__":
    sys.exit(main())
