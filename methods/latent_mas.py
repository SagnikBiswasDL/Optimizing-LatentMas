import time
from typing import Dict, List, Optional, Tuple

from . import default_agents, agents_from_spec
from models import ModelWrapper, _past_length
from seal.kv_steer import HANDOFF_MODES
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import torch
import argparse
try:
    from vllm import SamplingParams
    _HAS_VLLM = True
except ImportError:
    SamplingParams = None
    _HAS_VLLM = False
import pdb

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        planner_steps: Optional[int] = None,
        critic_steps: Optional[int] = None,
        refiner_steps: Optional[int] = None,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = int(latent_steps)
        # Per-agent budgets; None means fall back to latent_steps (original interface).
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
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        # Agent chain; subset-able for ablations via --agents (Judger always last).
        self.agents = agents_from_spec(getattr(args, "agents", None)) if args else default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        # SamplingParams is only needed for the vLLM decode path; build it lazily
        # so HF-only runs work without vLLM installed.
        self.sampling_params = None
        if _HAS_VLLM and bool(getattr(args, "use_vllm", False)):
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=args.max_new_tokens,
            )
        self.task = args.task

    def steps_for_role(self, role: str) -> int:
        """Latent continuation steps for an upstream agent role."""
        role = (role or "").lower()
        override = {
            "planner": self.planner_steps,
            "critic": self.critic_steps,
            "refiner": self.refiner_steps,
        }.get(role)
        if override is not None:
            return int(override)
        return int(self.latent_steps)

    def latent_forwards(self) -> int:
        """Actual latent-stage forwards: sum_role (K_role + 1) over upstream agents."""
        total = 0
        for agent in self.agents:
            if agent.role == "judger":
                continue
            total += self.steps_for_role(agent.role) + 1
        return total

    def budget_tuple(self) -> Tuple[int, int, int]:
        return (
            self.steps_for_role("planner"),
            self.steps_for_role("critic"),
            self.steps_for_role("refiner"),
        )

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        final_output_tokens = [0 for _ in range(batch_size)]

        # In-pipeline activation capture (native-vector program): per item, one
        # mean layer-L state per agent, tagged later with final correctness.
        capture = bool(getattr(self.args, "capture_acts", None)) and getattr(self.model, "act_recorder", None) is not None
        agent_acts: List[Dict[str, "torch.Tensor"]] = [dict() for _ in range(batch_size)]

        for agent in self.agents:

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]


            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)
                role_steps = self.steps_for_role(agent.role)

                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                if capture:
                    self.model._record_enable()
                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=role_steps,
                    past_key_values=past_kv,
                    role=agent.role,
                )
                if capture:
                    mean_acts = self.model._record_pop()
                    if mean_acts is not None:
                        for idx in range(batch_size):
                            agent_acts[idx][agent.role] = mean_acts[idx].clone()
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = role_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": role_steps,
                            "output": "",
                        }
                    )
            else:

                past_for_decoding = past_kv if (_past_length(past_kv) > 0) else None

                if self.args.think:
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                # One-shot KV-cache steering of the handoff the Judger consumes
                # (arXiv:2507.08799). Handoff modes edit past_for_decoding in place
                # before decoding; the judger_token control arm steers the Judger's
                # own final prompt token during a dedicated decode.
                kvsteer = getattr(self.model, "kvsteer", None)
                kv_on = kvsteer is not None and kvsteer.has_effect
                kv_judger_token = kv_on and kvsteer.positions == "judger_token"
                if kv_on and kvsteer.positions in HANDOFF_MODES and past_for_decoding is not None:
                    n_ed = kvsteer.apply_to_handoff(past_for_decoding)
                    if n_ed == 0:
                        print("[KVsteer][warn] handoff edit affected 0 layers")

                if capture:
                    self.model._record_enable()
                if kv_judger_token:
                    generated_batch, _ = self.model.generate_text_batch_kv_judger(
                        judger_prompts,
                        max_new_tokens=self.judger_max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        past_key_values=past_for_decoding,
                    )
                else:
                    generated_batch, _ = self.model.generate_text_batch(
                        judger_ids,
                        judger_mask,
                        max_new_tokens=self.judger_max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        past_key_values=past_for_decoding,
                        role=agent.role,
                    )
                if capture:
                    mean_acts = self.model._record_pop()
                    if mean_acts is not None:
                        for idx in range(batch_size):
                            agent_acts[idx][agent.role] = mean_acts[idx].clone()
                judger_token_counts = list(getattr(self.model, "last_gen_token_counts", []))
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    if idx < len(judger_token_counts):
                        final_output_tokens[idx] = int(judger_token_counts[idx])
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
                        }
                    )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
                # print(f'=========================================')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None
            
            res = {
                "question": item["question"],
                "gold": gold,
                "solution": item["solution"],
                "prediction": pred,
                "raw_prediction": final_text,
                "agents": agent_traces[idx],
                "correct": ok,
                "output_tokens": final_output_tokens[idx],
            }
            if capture:
                res["agent_acts"] = agent_acts[idx]
            results.append(res)
        return results
    
    def _sync_devices(self) -> None:
        if not torch.cuda.is_available():
            return
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)

    def _stage(self, name: str):
        """Time one serial stage. No-op unless the caller set ``stage_log``.

        Synchronization is correct while Hugging Face preparation and vLLM
        generation do not overlap. An overlapped pipeline must not use this.
        """
        outer = self

        class _Span:
            def __enter__(self):
                self.enabled = getattr(outer, "stage_log", None) is not None
                if self.enabled:
                    outer._sync_devices()
                    self.t0 = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                if self.enabled:
                    outer._sync_devices()
                    outer.stage_log.append({
                        "stage": name,
                        "seconds": time.perf_counter() - self.t0,
                        "error": None if exc_type is None else exc_type.__name__,
                    })
                return False

        return _Span()

    def _vllm_request_stats(self, outputs) -> List[Dict]:
        stats = []
        for out in outputs:
            choice = out.outputs[0]
            metrics = getattr(out, "metrics", None)
            metric_dict = None
            if metrics is not None:
                metric_dict = {
                    key: getattr(metrics, key)
                    for key in (
                        "arrival_time",
                        "first_scheduled_time",
                        "first_token_time",
                        "last_token_time",
                        "finished_time",
                        "queued_time",
                    )
                    if hasattr(metrics, key)
                }
            stats.append({
                "output_tokens": len(getattr(choice, "token_ids", []) or []),
                "finish_reason": getattr(choice, "finish_reason", None),
                "num_cached_tokens": getattr(out, "num_cached_tokens", None),
                "metrics": metric_dict,
            })
        return stats

    def _run_judger_repaired(self, judger_prompts: List[str], batch_size: int):
        """Judger embeddings with padding removed and the batch dimension kept.

        This is a baseline repair, not a speed optimization. Batch size 1
        cannot run the original ``squeeze(0)`` path. A padded batch otherwise
        sends pad rows and float32 zeros that a lone request does not.
        """
        from serving.embed_assembly import assemble_repaired, left_insert_lengths

        assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

        past_per_item = []
        for i in range(batch_size):
            parts = self._repaired_parts[i]
            if not parts:
                raise RuntimeError("repaired assembly has no latent embeddings")
            past_per_item.append(torch.cat(parts, dim=0))

        with self._stage("judger_tokenize_embed"):
            encoded = self.model.tokenizer(
                judger_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            judger_ids = encoded["input_ids"].to(self.model.HF_device)
            judger_mask = encoded["attention_mask"].to(self.model.HF_device)
            curr_prompt_emb = self.model.embedding_layer(judger_ids)
            insert_lengths = left_insert_lengths(self.model.tokenizer, judger_prompts)
            insert_lengths_no_special = left_insert_lengths(
                self.model.tokenizer, judger_prompts, add_special_tokens=False,
            )
            # The original slice clamps. An extra special token must not raise.
            valid_lengths = [int(judger_mask[i].sum().item()) for i in range(batch_size)]
            insert_used = [min(n, valid) for n, valid in zip(insert_lengths, valid_lengths)]

        with self._stage("judger_assemble_transfer"):
            rows = assemble_repaired(
                curr_prompt_emb, judger_mask, past_per_item, insert_used,
            )
            rows = [row.to(self.vllm_device) for row in rows]
            self._last_prompt_embeds = rows
            prompt_embeds_list = [{"prompt_embeds": row} for row in rows]
            self._pending_embeds = prompt_embeds_list

        if getattr(self, "_stop_before_vllm", False):
            return [""] * batch_size

        with self._stage("vllm_generate"):
            outputs = self.model.vllm_engine.generate(
                prompt_embeds_list,
                self.sampling_params,
            )
        self._last_vllm_stats = self._vllm_request_stats(outputs)
        self._last_embed_meta = {
            "assembly": "repaired",
            "prompt_lengths": [int(row.shape[0]) for row in rows],
            "embed_dtype": str(rows[0].dtype),
            "insert_lengths": insert_lengths,
            "insert_lengths_used": insert_used,
            "insert_lengths_no_special": insert_lengths_no_special,
            "valid_prompt_lengths": valid_lengths,
        }
        return [out.outputs[0].text.strip() for out in outputs]

    def prepare_prompt_embeds(self, items: List[Dict]):
        """Latent agents plus Judger embedding assembly, without vLLM decode.

        Used to overlap preparation of a later request with vLLM answering
        an earlier one. The embeddings match ``run_batch_vllm`` in repaired mode.
        """
        if getattr(self.args, "embed_assembly", "original") != "repaired":
            raise RuntimeError("prepare_prompt_embeds requires embed_assembly=repaired")
        self._stop_before_vllm = True
        try:
            self.run_batch_vllm(items)
            return list(self._pending_embeds)
        finally:
            self._stop_before_vllm = False

    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        self._last_vllm_stats = None
        self._last_embed_meta = None
        self._last_prompt_embeds = None
        assembly = getattr(self.args, "embed_assembly", "original")
        self._repaired_parts = [[] for _ in range(batch_size)] if assembly == "repaired" else None

        embedding_record = []
        for agent in self.agents:
            
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]

            with self._stage(f"prepare_{agent.role}"):
                prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                    batch_messages, add_generation_prompt=True
                )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)
                role_steps = self.steps_for_role(agent.role)

                # to wrap all latent thoughts from previous agents
                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                with self._stage(f"tokenize_{agent.role}"):
                    wrapped_encoded = self.model.tokenizer(
                        wrapped_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                    wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                    wrapped_tokens_batch: List[List[str]] = []
                    for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                        active_ids = ids_row[mask_row.bool()].tolist()
                        wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                with self._stage(f"latent_{agent.role}"):
                    past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=role_steps,
                        past_key_values=past_kv,
                    )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = role_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if role_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -role_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]

                if self._repaired_parts is not None:
                    if self.latent_only:
                        parts = [
                            previous_hidden_embedding[i].contiguous()
                            for i in range(batch_size)
                        ]
                    else:
                        from serving.embed_assembly import valid_agent_embeds
                        parts = valid_agent_embeds(previous_hidden_embedding, wrapped_mask)
                    for i, part in enumerate(parts):
                        self._repaired_parts[i].append(part)
                    if self.sequential_info_only or self.latent_only:
                        for i in range(batch_size):
                            self._repaired_parts[i] = self._repaired_parts[i][-1:]

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": role_steps,
                            "output": "",
                        }
                    )
            else:
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    judger_prompts = prompts

                if assembly == "repaired":
                    generated_texts = self._run_judger_repaired(judger_prompts, batch_size)
                else:
                    # Original hybrid semantics. Batch size 1 crashes on squeeze(0).
                    with self._stage("judger_assemble_transfer"):
                        past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                        judger_encoded = self.model.tokenizer(
                            judger_prompts,
                            return_tensors="pt",
                            padding=True,
                            add_special_tokens=False,
                        )
                        judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                        curr_prompt_emb = self.model.embedding_layer(judger_encoded).squeeze(0).to(self.vllm_device)

                        assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                        len_of_left = []
                        for p in judger_prompts:
                            idx = p.find("<|im_start|>user\n")
                            left = p[: idx + len("<|im_start|>user\n")]
                            len_of_left.append(len(self.model.tokenizer(left)['input_ids']))

                        B, L, H = curr_prompt_emb.shape
                        _, Lp, H = past_embedding.shape

                        whole_prompt_emb_list = []
                        for i in range(B):
                            insert_idx = len_of_left[i]
                            left_emb = curr_prompt_emb[i, :insert_idx, :]
                            right_emb = curr_prompt_emb[i, insert_idx:, :]
                            combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                            whole_prompt_emb_list.append(combined)

                        max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                        whole_prompt_emb = torch.stack([
                            torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                            for x in whole_prompt_emb_list
                        ])
                        prompt_embeds_list = [
                            {"prompt_embeds": embeds} for embeds in whole_prompt_emb
                        ]
                        self._last_embed_meta = {
                            "assembly": "original",
                            "prompt_lengths": [int(whole_prompt_emb.shape[1])] * B,
                            "embed_dtype": str(whole_prompt_emb.dtype),
                            "insert_lengths": [int(n) for n in len_of_left],
                        }

                    with self._stage("vllm_generate"):
                        outputs = self.model.vllm_engine.generate(
                            prompt_embeds_list,
                            self.sampling_params,
                        )
                    self._last_vllm_stats = self._vllm_request_stats(outputs)
                    generated_texts = [out.outputs[0].text.strip() for out in outputs]

                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )


        results: List[Dict] = []
        stats = self._last_vllm_stats or []
        meta = self._last_embed_meta or {}
        prompt_lengths = meta.get("prompt_lengths") or []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item["gold"]
            ok = (pred == gold) if (pred and gold) else False
            row = {
                "question": item["question"],
                "gold": gold,
                "solution": item["solution"],
                "prediction": pred,
                "raw_prediction": final_text,
                "agents": agent_traces[idx],
                "correct": ok,
                "embed_assembly": meta.get("assembly", assembly),
            }
            if idx < len(stats):
                row.update(stats[idx])
            if idx < len(prompt_lengths):
                row["prompt_embed_tokens"] = prompt_lengths[idx]
            results.append(row)
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
