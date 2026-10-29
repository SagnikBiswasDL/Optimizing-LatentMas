"""Tail-only preemption for the vLLM V1 scheduler (written against vLLM 0.11.1).

Stock vLLM preempts by freeing every block of the lowest-priority running
request and later recomputing it, relying on prefix caching to recover
whatever blocks were not overwritten in the meantime.

Here a preempted request keeps a reference on its hashed (full) blocks, so the
pool cannot evict them. When another request needs room, the pinned tails are
released one block at a time, most recently preempted request first, before
any running request is preempted. On resume, the ordinary prefix-cache lookup
finds the surviving head by hash, and only the released tail is recomputed.
With prefix caching off, the pinned head is handed back to the request directly.
A resumed request waits until the pool can hold its whole remaining recompute,
so it is not readmitted only to be preempted again.

Plug in with ``LLM(..., scheduler_cls="serving.tail_preempt_scheduler.TailPreemptScheduler")``.

Env:
  TAIL_PREEMPT_MODE  ``tail`` (default) or ``stock``. ``stock`` keeps vLLM's
                     behaviour and only records events, so both arms produce
                     the same logs.
  TAIL_PREEMPT_LOG   JSONL path for preempt/resume/trim events. Optional.
"""
from __future__ import annotations

import json
import os
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

logger = init_logger(__name__)


class TailPreemptScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tp_mode = os.environ.get("TAIL_PREEMPT_MODE", "tail").strip().lower()
        if self._tp_mode not in ("tail", "stock"):
            raise ValueError(f"TAIL_PREEMPT_MODE must be tail or stock, got {self._tp_mode!r}")
        self._tp_log_path = os.environ.get("TAIL_PREEMPT_LOG") or None
        self._tp_headroom = int(os.environ.get("TAIL_PREEMPT_HEADROOM", "0") or 0)
        # req_id -> hashed head blocks, oldest first. Insertion order is
        # preemption order, so the last entry is the most recent victim.
        self._tp_pins: dict[str, list] = {}
        self._tp_trimmed: dict[str, int] = {}

        km = self.kv_cache_manager
        reasons = []
        # Without prefix caching the pinned head cannot be found by hash, so it is
        # handed back to the request directly on resume.
        self._tp_caching = bool(km.enable_caching)
        if len(km.coordinator.single_type_managers) != 1:
            reasons.append("more than one KV cache group")
        if self.num_spec_tokens:
            reasons.append("speculative decoding")
        if self.connector is not None:
            reasons.append("KV connector")
        if getattr(self.scheduler_config, "async_scheduling", False):
            reasons.append("async scheduling")
        if self._tp_mode == "tail" and reasons:
            logger.warning("TailPreemptScheduler falling back to stock: %s", ", ".join(reasons))
            self._tp_mode = "stock"

        self._tp_manager = km.coordinator.single_type_managers[0]
        self._tp_pool = km.block_pool
        self._tp_orig_free = km.free
        self._tp_orig_alloc = km.allocate_slots
        self._tp_orig_get_computed = km.get_computed_blocks
        km.free = self._tp_free
        km.allocate_slots = self._tp_allocate_slots
        km.get_computed_blocks = self._tp_get_computed_blocks
        self._tp_event("init", mode=self._tp_mode, num_gpu_blocks=self.cache_config.num_gpu_blocks,
                       block_size=self.block_size, prefix_caching=self._tp_caching,
                       fallback_reasons=reasons)

    # ------------------------------------------------------------------ log
    def _tp_event(self, event: str, **fields) -> None:
        if not self._tp_log_path:
            return
        row = {"t": time.time(), "event": event, **fields}
        with open(self._tp_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    # ----------------------------------------------------------------- free
    def _tp_free(self, request) -> None:
        rid = request.request_id
        # The scheduler frees a RUNNING request only when preempting it; finished
        # and aborted requests are freed after their status changes.
        preempting = request.status == RequestStatus.RUNNING
        if not preempting:
            self._tp_orig_free(request)
            self._tp_release_pins(rid)
            return
        free_before = self._tp_pool.get_num_free_blocks()
        if self._tp_mode == "stock":
            self._tp_orig_free(request)
            self._tp_event("preempt", req=rid, mode="stock", num_tokens=request.num_tokens,
                           num_computed_tokens=request.num_computed_tokens, pinned_blocks=0,
                           free_before=free_before, free_after=self._tp_pool.get_num_free_blocks())
            return
        blocks = self._tp_manager.req_to_blocks.pop(rid, [])
        num_hashed = self._tp_manager.num_cached_block.pop(rid, 0)
        num_keep = num_hashed if self._tp_caching else request.num_computed_tokens // self.block_size
        head, tail = blocks[:num_keep], blocks[num_keep:]
        # Unhashed blocks cannot be found again on resume, so they go back now.
        self._tp_pool.free_blocks(reversed(tail))
        if head:
            self._tp_pins[rid] = head
        self._tp_event("preempt", req=rid, mode="tail", num_tokens=request.num_tokens,
                       num_computed_tokens=request.num_computed_tokens, pinned_blocks=len(head),
                       free_before=free_before, free_after=self._tp_pool.get_num_free_blocks())

    def _tp_release_pins(self, rid: str) -> None:
        head = self._tp_pins.pop(rid, None)
        if head:
            self._tp_pool.free_blocks(reversed(head))

    def _tp_trim_one(self, exclude_rid: str, protected_ids: set) -> bool:
        """Release the last pinned block of the lowest-priority other victim."""
        def rank(rid):
            req = self.requests.get(rid)
            return (req.priority, req.arrival_time) if req is not None else (0, 0.0)

        for rid in sorted(self._tp_pins, key=rank, reverse=True):
            if rid == exclude_rid:
                continue
            head = self._tp_pins[rid]
            while head:
                block = head[-1]
                if block.block_id in protected_ids:
                    break
                head.pop()
                shared = block.ref_cnt > 1
                self._tp_pool.free_blocks([block])
                if not shared:
                    if not head:
                        del self._tp_pins[rid]
                    self._tp_trimmed[rid] = self._tp_trimmed.get(rid, 0) + 1
                    return True
            if not head:
                self._tp_pins.pop(rid, None)
        return False

    # --------------------------------------------------------------- lookup
    def _tp_get_computed_blocks(self, request):
        blocks, num_tokens = self._tp_orig_get_computed(request)
        head = self._tp_pins.get(request.request_id)
        if self._tp_caching or not head:
            return blocks, num_tokens
        # The last token must be recomputed to produce logits.
        keep = min(len(head), (request.num_tokens - 1) // self.block_size)
        return blocks, keep * self.block_size

    # -------------------------------------------------------------- allocate
    def _tp_allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
                           new_computed_blocks=None, *args, **kwargs):
        rid = request.request_id
        resuming = request.status == RequestStatus.PREEMPTED and request.num_computed_tokens == 0
        hit_blocks = new_computed_blocks.blocks[0] if new_computed_blocks is not None else ()
        handoff = resuming and not self._tp_caching and rid in self._tp_pins

        # Waiting only helps while something running can still finish and free blocks.
        if resuming and rid in self._tp_pins and self.running:
            need = (-(-request.num_tokens // self.block_size)
                    - num_new_computed_tokens // self.block_size + self._tp_headroom)
            if self._tp_pool.get_num_free_blocks() < need:
                return None

        if handoff:
            keep = num_new_computed_tokens // self.block_size
            head = self._tp_pins[rid]
            if len(head) > keep:
                self._tp_pool.free_blocks(reversed(head[keep:]))
                del head[keep:]
            # The request takes over the pin's reference; no ref count changes.
            self._tp_manager.req_to_blocks[rid] = list(head)

        result = self._tp_orig_alloc(request, num_new_tokens, num_new_computed_tokens,
                                     new_computed_blocks, *args, **kwargs)
        if result is None and self._tp_mode == "tail" and self._tp_pins:
            self._tp_trimmed = {}
            protected = {b.block_id for b in hit_blocks}
            while result is None and self._tp_trim_one(rid, protected):
                result = self._tp_orig_alloc(request, num_new_tokens, num_new_computed_tokens,
                                             new_computed_blocks, *args, **kwargs)
            for victim, n in self._tp_trimmed.items():
                self._tp_event("trim", req=victim, blocks=n, for_req=rid,
                               remaining_pinned=len(self._tp_pins.get(victim, ())))

        if handoff:
            if result is None:
                self._tp_manager.req_to_blocks.pop(rid, None)
            else:
                self._tp_pins.pop(rid, None)
        if result is not None and resuming:
            self._tp_release_pins(rid)
            self._tp_event("resume", req=rid, mode=self._tp_mode, num_tokens=request.num_tokens,
                           hit_tokens=num_new_computed_tokens,
                           recompute_tokens=request.num_tokens - num_new_computed_tokens,
                           num_preemptions=request.num_preemptions)
        return result
