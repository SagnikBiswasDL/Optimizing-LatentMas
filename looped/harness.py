"""Batched agent calls, per-item ledgers, and run metadata shared by the scripts."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from looped import prompts as P
from looped.ouro_adapter import OuroRunner, allocated_bytes, peak_bytes, reset_peak, sync


def environment(runner: OuroRunner, args) -> Dict:
    import transformers

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        commit, dirty = None, None
    dev = runner.device
    return {
        "model_id": runner.model_id,
        "model_revision": runner.revision,
        "dtype": str(runner.model.dtype),
        "attn_implementation": runner.model.config._attn_implementation,
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "git_commit": commit,
        "git_dirty": dirty,
        "args": vars(args),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def warmup(runner: OuroRunner, depths: Iterable[int]) -> float:
    t0 = time.perf_counter()
    ids = runner.chat_ids(P.msgs(P.solver("What is 2 + 3?")))
    for r in depths:
        runner.generate([ids, ids], depth=r, max_new_tokens=8)
    sync(runner.device)
    return time.perf_counter() - t0


def run_agent(
    runner: OuroRunner,
    texts: Sequence[str],
    *,
    depth: int,
    max_new_tokens: int,
    batch_size: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
    think: Optional[bool] = None,
) -> Dict:
    """One role over many prompts through a ``batch_size``-slot pool.

    Only arm-level wall time is meaningful (throughput); per-request latency is
    not defined under pooled decoding.
    """
    if think is None:
        think = bool(getattr(runner, "think", False))
    prompts = [runner.chat_ids(P.msgs(t), think=think) for t in texts]
    reset_peak(runner.device)
    g = runner.generate_pool(prompts, depth=depth, max_new_tokens=max_new_tokens, slots=batch_size,
                             temperature=temperature, top_p=top_p, seed=seed)
    wall, executed = g.wall_s, g.executed_token_loops
    peak_seen = allocated_bytes(runner.device)
    recs: List[Dict] = []
    for j, ids in enumerate(g.token_ids):
        reasoning, answer, closed = runner.split_think(ids)
        # No thinking mode: the whole generation is the answer, not an unclosed thought.
        if not think and not closed:
            answer, closed = g.texts[j], True
        recs.append({
            "prompt_tokens": len(prompts[j]),
            "gen_tokens": len(ids),
            "hit_eos": g.hit_eos[j],
            "closed_think": closed,
            "answer_text": answer,
            "full_text": g.texts[j],
            "work": g.work[j].as_dict(),
        })
    pk = peak_bytes(runner.device)
    return {
        "records": recs,
        "wall_s": wall,
        "executed_token_loops": executed,
        "peak_mem_gb": round((pk if pk is not None else peak_seen) / 2 ** 30, 3),
        "peak_mem_kind": "torch.cuda.max_memory_allocated" if pk is not None else "max sampled allocation",
    }


def handoff(rec: Dict, tail_chars: int = 2000) -> str:
    """Text passed downstream: the post-``</think>`` answer, or the reasoning tail
    when the agent hit its budget before closing its thinking."""
    if rec["closed_think"] and rec["answer_text"]:
        return rec["answer_text"]
    return rec["full_text"][-tail_chars:]


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def add_work(*ws: Dict) -> Dict:
    out: Dict = {}
    for w in ws:
        for k, v in w.items():
            out[k] = out.get(k, 0) + v
    return out
