"""Unified LatentMAS inside a single vLLM engine (no Hugging Face latent path).

Uses the forked V1 latent-step mode: Planner/Critic/Refiner latent steps and
Judger decode share one in-engine KV. Scheduler is selectable via
``args.scheduler_cls`` (stock vs TailPreemptScheduler).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from . import agents_from_spec, default_agents
from prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
)
from utils import extract_gsm8k_answer, normalize_answer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    _HAS_VLLM = False


class LatentMASVLLMMethod:
    """LatentMAS with all agents on one vLLM fork engine."""

    def __init__(
        self,
        model_name: str,
        *,
        latent_steps: int = 10,
        planner_steps: Optional[int] = None,
        critic_steps: Optional[int] = None,
        refiner_steps: Optional[int] = None,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        generate_bs: int = 1,
        args=None,
    ) -> None:
        if not _HAS_VLLM:
            raise RuntimeError("vLLM is required for LatentMASVLLMMethod")
        self.args = args
        self.model_name = model_name
        self.latent_steps = int(latent_steps)
        if args is not None:
            if planner_steps is None:
                planner_steps = getattr(args, "planner_steps", None)
            if critic_steps is None:
                critic_steps = getattr(args, "critic_steps", None)
            if refiner_steps is None:
                refiner_steps = getattr(args, "refiner_steps", None)
        self.planner_steps = int(planner_steps) if planner_steps is not None else None
        self.critic_steps = int(critic_steps) if critic_steps is not None else None
        self.refiner_steps = int(refiner_steps) if refiner_steps is not None else None
        self.judger_max_new_tokens = int(judger_max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.generate_bs = max(1, int(generate_bs))
        self.agents = (
            agents_from_spec(getattr(args, "agents", None)) if args else default_agents()
        )
        self.method_name = "latent_mas_vllm"
        self.task = getattr(args, "task", "gsm8k")
        self.think = bool(getattr(args, "think", False)) if args else False
        self.latent_space_realign = (
            bool(getattr(args, "latent_space_realign", False)) if args else False
        )
        self.prompt_style = getattr(args, "prompt", "sequential") if args else "sequential"
        self._last_stats: List[Dict[str, Any]] = []

        tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
        gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
        engine_extra: Dict[str, Any] = {
            "enable_prompt_embeds": True,
            "enforce_eager": True,
            "disable_log_stats": False,
            # Latent embeds are not hashable as token-id prefixes.
            "enable_prefix_caching": False,
        }
        # Prefer async scheduling off when the installed EngineArgs supports it.
        from vllm.engine.arg_utils import EngineArgs

        _ea_fields = getattr(EngineArgs, "__dataclass_fields__", {})
        if "async_scheduling" in _ea_fields:
            engine_extra["async_scheduling"] = False
        if "disable_async_output_proc" in _ea_fields:
            engine_extra["disable_async_output_proc"] = True
        if getattr(args, "scheduler_cls", None):
            engine_extra["scheduler_cls"] = args.scheduler_cls
        if getattr(args, "max_model_len", None):
            engine_extra["max_model_len"] = int(args.max_model_len)
        if getattr(args, "num_gpu_blocks_override", None):
            engine_extra["num_gpu_blocks_override"] = int(args.num_gpu_blocks_override)

        print(
            f"[latent_mas_vllm] Loading {model_name} "
            f"(gpu_memory_utilization={gpu_util}, enforce_eager=True)"
        )
        self.engine = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_util,
            **engine_extra,
        )
        self.tokenizer = self.engine.get_tokenizer()

    def steps_for_role(self, role: str) -> int:
        role = (role or "").lower()
        override = {
            "planner": self.planner_steps,
            "critic": self.critic_steps,
            "refiner": self.refiner_steps,
        }.get(role)
        if override is not None:
            return int(override)
        return int(self.latent_steps)

    def _build_messages(self, role: str, question: str):
        if self.prompt_style == "hierarchical":
            return build_agent_message_hierarchical_latent_mas(
                role=role,
                question=question,
                context="",
                method="latent_mas",
                args=self.args,
            )
        return build_agent_message_sequential_latent_mas(
            role=role,
            question=question,
            context="",
            method="latent_mas",
            args=self.args,
        )

    def _render(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _encode_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_request(self, item: Dict, *, return_latents: bool = False) -> Tuple[str, SamplingParams, int]:
        """Return (planner_prompt, sampling_params, num_internal_tokens)."""
        question = item["question"]
        phases: List[Dict[str, Any]] = []
        first_prompt: Optional[str] = None
        num_internal = 0

        for agent in self.agents:
            messages = self._build_messages(agent.role, question)
            prompt = self._render(messages, add_generation_prompt=True)
            if self.think:
                prompt = f"{prompt}<think>"

            if agent.role != "judger":
                steps = self.steps_for_role(agent.role)
                if first_prompt is None:
                    first_prompt = prompt
                    phases.append({"kind": "latent", "steps": steps})
                    num_internal += steps
                else:
                    token_ids = self._encode_ids(prompt)
                    phases.append({"kind": "tokens", "token_ids": token_ids})
                    phases.append({"kind": "latent", "steps": steps})
                    num_internal += len(token_ids) + steps
            else:
                if first_prompt is None:
                    # Judger-only ablation: no latent phases.
                    first_prompt = prompt
                    phases.append({"kind": "decode"})
                else:
                    token_ids = self._encode_ids(prompt)
                    phases.append({"kind": "tokens", "token_ids": token_ids})
                    phases.append({"kind": "decode"})
                    num_internal += len(token_ids)

        assert first_prompt is not None
        max_tokens = num_internal + int(self.judger_max_new_tokens)
        sp = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
            extra_args={
                "latent_mas": {
                    "phases": phases,
                    "latent_space_realign": self.latent_space_realign,
                    "return_latents": bool(return_latents),
                }
            },
        )
        return first_prompt, sp, num_internal

    def run_batch(
        self,
        items: List[Dict],
        *,
        return_latents: bool = False,
    ) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        prompts: List[str] = []
        params_list: List[SamplingParams] = []
        internals: List[int] = []
        for item in items:
            prompt, sp, n_int = self._build_request(item, return_latents=return_latents)
            prompts.append(prompt)
            params_list.append(sp)
            internals.append(n_int)

        t0 = time.time()
        outputs = self.engine.generate(prompts, params_list)
        wall = time.time() - t0

        results: List[Dict] = []
        self._last_stats = []
        for idx, (item, out, n_int) in enumerate(zip(items, outputs, internals)):
            completion = out.outputs[0]
            all_ids = list(completion.token_ids)
            decode_ids = all_ids[n_int:] if n_int < len(all_ids) else []
            text = self.tokenizer.decode(decode_ids, skip_special_tokens=True).strip()
            # Fallback if slicing failed: use engine text (may include junk).
            if not text and completion.text:
                text = completion.text.strip()

            pred = normalize_answer(extract_gsm8k_answer(text))
            gold = item.get("gold", "")
            if self.task in ("aime2024", "aime2025"):
                try:
                    ok = int(pred) == int(str(gold).strip())
                except ValueError:
                    ok = False
            else:
                ok = (pred == gold) if (pred and gold) else False

            stats = {
                "num_output_tokens": len(all_ids),
                "num_internal_tokens": n_int,
                "num_decode_tokens": len(decode_ids),
                "finish_reason": completion.finish_reason,
                "wall_s": wall / max(1, len(items)),
            }
            if hasattr(out, "metrics") and out.metrics is not None:
                stats["metrics"] = str(out.metrics)
            self._last_stats.append(stats)

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item.get("solution", ""),
                    "prediction": pred,
                    "correct": ok,
                    "output": text,
                    "agents": [
                        {
                            "name": "Unified",
                            "role": "latent_mas_vllm",
                            "input": prompts[idx][:200] + "...",
                            "output": text,
                            "latent_steps": self.latent_steps,
                        }
                    ],
                    "stats": stats,
                }
            )
        return results
