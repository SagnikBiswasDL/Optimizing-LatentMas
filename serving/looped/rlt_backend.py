"""vllm-rlt (pinned, unmodified) behind the looped backend contract.

The engine is driven in process through ``StageEngine`` so the scheduler and KV
manager can be inspected. Nothing in the engine source is changed.

Depth semantics. The engine always prefills a prompt at the model's full depth
and reads the first generated token from that full-depth state; ``max_loops``
only bounds decode loops (``stage_engine.py``: "Prefill always runs the model's
full depth"). Two prompt modes are offered on top of that:

``stock``
    Submit the prompt as is. ``semantics["prompt_depth"] == "full"``.

``boundary_shift``
    Withhold the last prompt token. The engine prefills the rest at full depth
    and its first output must be the withheld token; that token is then processed
    as an ordinary decode token at the requested depth, so every token the caller
    sees is read from a depth-``r`` state. In Ouro, loop ``i`` of a token reads
    only loop ``i`` keys of earlier tokens, so the first ``r`` KV planes of a
    full-depth prompt are the planes a depth-``r`` prefill would have written.
    The result is therefore the uniform-depth computation, at the price of
    ``(full - r) * (prompt - 1)`` extra loop-tokens of prefill. If the engine
    emits anything other than the withheld token the request is aborted and
    reported as ``error``; it is never silently accepted.

Stop tokens are applied by the engine itself: ``Scheduler.eos_token_ids`` is a
tuple and is set here to the caller's stop ids.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .backend import (CANCELLED, EOS, ERROR, LENGTH_CAP, Backend, Completion, GenerationRequest,
                      logical_loop_tokens)

ENGINE_SHA = "e0a5515ec970d634546d2e1d38b3ba3602072884"
PROMPT_MODES = ("stock", "boundary_shift")


class RltBackend(Backend):
    name = "vllm-rlt"

    def __init__(
        self,
        model_path: str,
        *,
        stop_ids: Sequence[int],
        prompt_mode: str = "boundary_shift",
        dtype: str = "bfloat16",
        device: str = "cuda",
        attention_backend: str = "triton",
        kv_layout: str = "last_exited",
        max_num_seqs: int = 32,
        max_num_batched_tokens: int = 4096,
        scheduler_mode: str = "refill",
        prefill_chunk_size: int = 512,
        num_blocks: Optional[int] = None,
        kv_cache_memory_bytes: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        incremental_allocation: bool = False,
        total_ut_steps: Optional[int] = None,
        execution: Optional[Dict] = None,
        exit_mode: str = "ouro",
        clock=time.perf_counter,
    ) -> None:
        import torch
        from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
        from vllm_rlt.engine.stage_engine import StageEngine
        from vllm_rlt.models import OuroConfig, OuroForCausalLM

        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}")
        self._SamplingParams = SamplingParams
        self.prompt_mode = prompt_mode
        self.clock = clock
        config = OuroConfig.from_dict(json.loads((Path(model_path) / "config.json").read_text()))
        self.checkpoint_depth = int(config.total_ut_steps)
        if total_ut_steps is not None:
            # A depth-limited engine: the same weights with fewer loops everywhere.
            config = dataclasses.replace(config, total_ut_steps=int(total_ut_steps))
        self.full_depth = int(config.total_ut_steps)
        model = OuroForCausalLM.from_pretrained(model_path, device=device, dtype=getattr(torch, dtype), config=config)
        self.engine = StageEngine(
            model,
            cache_config=CacheConfig(
                num_blocks=num_blocks, layout=kv_layout, gpu_memory_utilization=gpu_memory_utilization,
                kv_cache_memory_bytes=kv_cache_memory_bytes, incremental_allocation=incremental_allocation,
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                mode=scheduler_mode, prefill_chunk_size=prefill_chunk_size,
            ),
            execution_config=ExecutionConfig(**(execution or {})),
            # Every request pins min_loops == max_loops with exit_threshold 1, so the
            # gate never decides anything; the mode only selects the scheduling path
            # (async scheduling needs "ouro_delayed").
            exit_config=ExitConfig(mode=exit_mode),
            attention_backend=attention_backend,
        )
        self.stop_ids = tuple(int(t) for t in stop_ids)
        self.engine.scheduler.eos_token_ids = self.stop_ids
        self._live: Dict[str, Dict] = {}
        cache = self.engine.cache_manager
        self.semantics = {
            "engine": "InferMatrixLab/vllm-rlt", "engine_sha": ENGINE_SHA, "driver": "in-process StageEngine",
            "prompt_mode": prompt_mode,
            "prompt_depth": "full" if prompt_mode == "stock" else "request",
            "prefill_executed_depth": self.full_depth,
            "first_token_depth": "full" if prompt_mode == "stock" else "request",
            "kv_layout": kv_layout, "kv_storage_depths": int(cache.storage_depths),
            "kv_block_size": int(cache.block_size), "kv_num_blocks": int(cache.num_blocks),
            "dtype": dtype, "attention_backend": attention_backend, "scheduler_mode": scheduler_mode,
            "max_num_seqs": max_num_seqs, "max_num_batched_tokens": max_num_batched_tokens,
            "full_depth": self.full_depth, "checkpoint_depth": self.checkpoint_depth,
            "stop_ids": list(self.stop_ids), "execution": execution or {}, "exit_mode": exit_mode,
        }

    # ------------------------------------------------------------------ contract

    def submit(self, request: GenerationRequest) -> None:
        if request.request_id in self._live:
            raise ValueError(f"duplicate live request {request.request_id}")
        if request.depth > self.full_depth:
            raise ValueError(f"depth {request.depth} exceeds engine depth {self.full_depth}")
        shift = self.prompt_mode == "boundary_shift"
        if shift and len(request.prompt_ids) < 2:
            raise ValueError("boundary_shift needs a prompt of at least two tokens")
        prompt = list(request.prompt_ids[:-1] if shift else request.prompt_ids)
        params = self._SamplingParams(
            max_tokens=request.max_new_tokens + (1 if shift else 0),
            temperature=float(request.temperature), top_p=float(request.top_p), seed=int(request.seed),
            min_loops=int(request.depth), max_loops=int(request.depth), exit_threshold=1.0,
        )
        self._live[request.request_id] = {
            "request": request, "submit": self.clock(), "first": None, "seen": 0,
            "expect": int(request.prompt_ids[-1]) if shift else None, "tokens": [], "depths": [],
        }
        try:
            self.engine.add_request(request.request_id, prompt, params)
        except BaseException:
            del self._live[request.request_id]
            raise

    def step(self) -> List[Completion]:
        outputs = self.engine.step()
        now = self.clock()
        done: List[Completion] = []
        for out in outputs:
            st = self._live.get(out.request_id)
            if st is None:
                continue
            st["tokens"], st["depths"] = list(out.token_ids), list(out.exit_depths)
            skip = 1 if st["expect"] is not None else 0
            if st["expect"] is not None and st["tokens"] and st["tokens"][0] != st["expect"]:
                if not out.finished:
                    self.engine.abort_request(out.request_id)
                done.append(self._finish(out.request_id, ERROR, now,
                                         {"boundary_violation": {"expected": st["expect"], "got": st["tokens"][0]}}))
                continue
            if st["first"] is None and len(st["tokens"]) > skip:
                st["first"] = now
            if out.finished:
                reason = {"stop": EOS, "length": LENGTH_CAP, "abort": CANCELLED}[out.finish_reason]
                done.append(self._finish(out.request_id, reason, now, {}))
        return done

    def cancel(self, request_id: str) -> Completion:
        st = self._live[request_id]
        out = self.engine.abort_request(request_id)
        st["tokens"], st["depths"] = list(out.token_ids), list(out.exit_depths)
        return self._finish(request_id, CANCELLED, self.clock(), {})

    def active(self) -> int:
        return len(self._live)

    def partial(self, request_id: str) -> List[int]:
        st = self._live[request_id]
        return st["tokens"][1 if st["expect"] is not None else 0:]

    def stats(self) -> Dict:
        cache = self.engine.cache_manager
        return {"used_blocks": int(cache.num_used_blocks), "num_blocks": int(cache.num_blocks),
                "engine_requests": len(self.engine.scheduler.requests), "live": len(self._live)}

    def close(self) -> None:
        self.engine.close()

    # ------------------------------------------------------------------ helpers

    def _finish(self, request_id: str, reason: str, now: float, detail: Dict) -> Completion:
        st = self._live.pop(request_id)
        req: GenerationRequest = st["request"]
        skip = 1 if st["expect"] is not None else 0
        tokens, depths = st["tokens"][skip:], st["depths"]
        submitted_prompt = len(req.prompt_ids) - skip
        # depths[j] is the depth of the state token j was read from. Token 0 comes
        # from prefill; every later token cost that many loops on its predecessor.
        charged = self.full_depth * submitted_prompt + sum(depths[1:])
        detail = dict(detail, exit_depth_histogram=_histogram(depths), engine_tokens=len(st["tokens"]))
        return Completion(
            request_id=request_id, depth=req.depth, prompt_tokens=len(req.prompt_ids), token_ids=tokens,
            finish_reason=reason, submit_time=st["submit"], first_token_time=st["first"], finish_time=now,
            logical_loop_tokens=logical_loop_tokens(req.depth, len(req.prompt_ids), len(tokens)),
            charged_loop_tokens=int(charged), detail=detail,
        )


def _histogram(values: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out
