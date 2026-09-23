import os
import csv
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval() 
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        # Attention kernel is a load-time choice and materially changes decode speed
        # at the 8k-16k contexts this pipeline runs at. Default stays whatever
        # transformers picks, so existing results are unaffected unless asked.
        _attn = getattr(args, "attn_implementation", None) or os.environ.get("SEAL_ATTN_IMPL") or ""
        _extra = {"attn_implementation": _attn} if _attn else {}
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                **_extra,
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

        # In-pipeline activation capture (native-vector program): a read-only
        # recorder hook at a deep layer. Toggled per agent-role from the pipeline;
        # the recorded current-token states become correctness-contrastive vectors.
        self.act_recorder = None
        if args is not None and getattr(args, "capture_acts", None):
            from seal.capture import ActivationRecorder
            cap_layer = getattr(args, "capture_layer", None)
            if cap_layer is None or cap_layer < 0:
                cap_layer = getattr(args, "seal_layer", -1)
            if cap_layer is None or cap_layer < 0:
                cap_layer = 28
            self.act_recorder = ActivationRecorder(int(cap_layer))
            self.act_recorder.register(self.model)
            self.act_recorder.disable()
            print(f"[capture] activation recorder enabled at layer {self.act_recorder.layer_index}")

        # SEAL steering (token-efficiency): HF backend only. The hook is registered
        # once (persistent) and toggled per agent-role via `seal_active_roles`, so we
        # can steer any subset of {planner, critic, refiner, judger}.
        self.seal = None
        self.seal_active_roles = set()
        self.last_gen_token_counts: List[int] = []
        if args is not None and bool(getattr(args, "seal", False)):
            from seal import SealSteerer
            seal_layer = getattr(args, "seal_layer", -1)
            self.seal = SealSteerer.from_artifact(
                getattr(args, "seal_vector"),
                coef=float(getattr(args, "seal_coef", 0.0)),
                layer_index=(seal_layer if seal_layer is not None and seal_layer >= 0 else None),
                apply_to=getattr(args, "seal_apply_to", "last"),
            )
            roles_raw = getattr(args, "seal_agents", "judger") or "judger"
            if roles_raw.strip().lower() == "all":
                self.seal_active_roles = {"planner", "critic", "refiner", "judger"}
            else:
                self.seal_active_roles = {r.strip().lower() for r in roles_raw.split(",") if r.strip()}
            # Register once; keep disabled until a matching agent runs.
            self.seal.register(self.model)
            self.seal.disable()
            print(
                f"[SEAL] enabled: layer={self.seal.layer_index} coef={self.seal.coef} "
                f"apply_to={self.seal.apply_to} agents={sorted(self.seal_active_roles)} "
                f"vector={getattr(args, 'seal_vector')}"
            )

        # One-shot KV-cache steering of the LatentMAS handoff (HF backend only).
        # Unlike SEAL (per-step residual-stream hook), this makes a single edit to
        # the shared K/V cache the Judger consumes, at all layers, right before the
        # Judger decodes. It targets the memory channel the latent agents pass
        # forward. Applied by the pipeline (methods/latent_mas.py) for the Judger.
        self.kvsteer = None
        if args is not None and bool(getattr(args, "kvsteer", False)):
            from seal.kv_steer import KVCacheSteerer
            self.kvsteer = KVCacheSteerer.from_artifact(
                getattr(args, "kvsteer_vector"),
                c_v=float(getattr(args, "kvsteer_cv", 0.0)),
                c_k=float(getattr(args, "kvsteer_ck", 0.0)),
                positions=getattr(args, "kvsteer_positions", "handoff_last"),
                last_k=int(getattr(args, "kvsteer_last_k", 40)),
            )
            print(
                f"[KVsteer] enabled: {self.kvsteer.summary()} "
                f"vector={getattr(args, 'kvsteer_vector')}"
            )

        # CES / K-budget trainable residual steerer (HF backend). Separate from SEAL.
        # Primary experiment: apply only during recurrent latent steps (not prefill).
        self.ces = None
        if args is not None and bool(getattr(args, "ces", False)):
            self.attach_ces_from_args(args)

    def attach_ces_from_args(self, args) -> None:
        """Construct and register a TrainableSteerer from CLI / Namespace attrs."""
        from seal.ces_steerer import TrainableSteerer

        hidden = int(self.model.config.hidden_size)
        layer = int(getattr(args, "ces_layer", 28))
        if layer < 0:
            layer = max(0, int(getattr(self.model.config, "num_hidden_layers", 1)) - 2)
        roles_raw = getattr(args, "ces_agents", "planner,critic,refiner") or "planner,critic,refiner"
        if str(roles_raw).strip().lower() == "all":
            agents = {"planner", "critic", "refiner", "judger"}
        else:
            agents = {r.strip().lower() for r in str(roles_raw).split(",") if r.strip()}
        steer_phase = getattr(args, "ces_steer_phase", "latent_only")
        init_std = float(getattr(args, "ces_init_std", 0.0))
        coef = float(getattr(args, "ces_coef", 1.0))
        vec_path = getattr(args, "ces_vector", None)
        if vec_path:
            self.ces = TrainableSteerer.from_artifact(
                vec_path, coef=coef, agents=agents, steer_phase=steer_phase
            )
            if layer >= 0:
                self.ces.layer_index = layer
        else:
            self.ces = TrainableSteerer(
                hidden_size=hidden,
                layer_index=layer,
                coef=coef,
                init_std=init_std,
                agents=agents,
                steer_phase=steer_phase,
            )
        self.ces.register(self.model)
        self.ces.to(self.device)
        self.ces.disable()
        print(
            f"[CES] enabled: layer={self.ces.layer_index} coef={self.ces.coef} "
            f"phase={self.ces.steer_phase} agents={sorted(self.ces.agents)} "
            f"vector={vec_path or 'random/zero'}"
        )

    def attach_ces(
        self,
        *,
        layer_index: int,
        coef: float = 1.0,
        init_std: float = 0.0,
        agents=None,
        steer_phase: str = "latent_only",
        vector: Optional[torch.Tensor] = None,
    ):
        """Programmatic CES attach (smoke tests / trainers)."""
        from seal.ces_steerer import TrainableSteerer

        hidden = int(self.model.config.hidden_size)
        self.ces = TrainableSteerer(
            hidden_size=hidden,
            layer_index=int(layer_index),
            coef=float(coef),
            init_std=float(init_std),
            agents=set(agents or {"planner", "critic", "refiner"}),
            steer_phase=steer_phase,
        )
        if vector is not None:
            with torch.no_grad():
                self.ces.v.copy_(vector.float().view(-1).to(self.ces.v.device))
        self.ces.register(self.model)
        self.ces.to(self.device)
        self.ces.disable()
        return self.ces

    def _seal_activate_for(self, role: Optional[str]) -> bool:
        """Enable the steerer iff SEAL is on and this role is targeted.

        role=None (e.g. single-agent baseline) counts as active. Returns whether
        the hook was enabled so the caller can disable it afterwards.
        """
        seal = getattr(self, "seal", None)
        if seal is None:
            return False
        active = (role is None) or (role in self.seal_active_roles)
        if not active:
            seal.disable()
            return False
        # Select this role's vector/coef (supports per-role native vectors).
        seal.set_active_role(role)
        if not seal.has_effect_for(role):
            seal.disable()
            return False
        seal.enable()
        return True

    def _record_enable(self) -> bool:
        """Start recording layer-L current-token states (in-pipeline capture)."""
        rec = getattr(self, "act_recorder", None)
        if rec is None:
            return False
        rec.enable()
        return True

    def _record_pop(self):
        """Stop recording and return the per-run mean state [B, D] (or None)."""
        rec = getattr(self, "act_recorder", None)
        if rec is None:
            return None
        mean = rec.pop_mean()
        rec.disable()
        return mean

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if getattr(self.args, "latent_space_realign", False):
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
        role: Optional[str] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        use_seal = self._seal_activate_for(role)
        # temperature <= 0 → greedy / deterministic decoding for Gate 1 curves.
        do_sample = float(temperature) > 0.0
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        try:
            outputs = self.model.generate(**gen_kwargs)
        finally:
            if use_seal:
                self.seal.disable()
        sequences = outputs.sequences
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        # Generated tokens always begin after the (padded) input width, uniform
        # across rows. Slicing by per-row prompt length would start inside the
        # padding region (corrupting both the text and the token count).
        gen_start = input_ids.shape[1]
        generations: List[str] = []
        token_counts: List[int] = []
        for idx in range(sequences.shape[0]):
            generated_ids = sequences[idx, gen_start:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
            # Count real generated tokens: stop at first EOS; ignore trailing pad.
            cnt = 0
            for tok in generated_ids.tolist():
                if eos_id is not None and tok == eos_id:
                    cnt += 1
                    break
                if pad_id is not None and pad_id != eos_id and tok == pad_id:
                    break
                cnt += 1
            token_counts.append(cnt)
        self.last_gen_token_counts = token_counts
        return generations, outputs.past_key_values

    def _count_generated_tokens(self, generated_ids: List[int]) -> int:
        """Count real generated tokens: stop at first EOS; ignore trailing pad."""
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        cnt = 0
        for tok in generated_ids:
            if eos_id is not None and tok == eos_id:
                cnt += 1
                break
            if pad_id is not None and pad_id != eos_id and tok == pad_id:
                break
            cnt += 1
        return cnt

    @torch.no_grad()
    def generate_text_batch_kv_judger(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        """Judger decode with one-shot cache steering of the Judger's own final
        prompt token (the ``judger_token`` control arm).

        Following arXiv:2507.08799 (Appendix C.5), we append a neutral offset
        token so the real last prompt token (the chat aggregation token) lands in
        the cache; we prefill everything up to that offset, steer the last cached
        position across all layers, then generate from the offset token onward.
        Left padding is used so the aggregation token is the last column for every
        row. This steers the Judger's context rather than the latent handoff, so
        any gain the handoff arms show over this arm is attributable to the memory
        channel specifically.
        """
        kvsteer = getattr(self, "kvsteer", None)
        if kvsteer is None or not kvsteer.has_effect:
            raise RuntimeError("generate_text_batch_kv_judger requires an active kvsteer")

        # Left-pad tokenize with an appended neutral offset token.
        offset_str = "\n"
        offset_id = self.tokenizer(offset_str, add_special_tokens=False)["input_ids"]
        offset_id = offset_id[-1] if offset_id else self.tokenizer.eos_token_id
        texts = [p + offset_str for p in prompts]
        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            )
        finally:
            self.tokenizer.padding_side = old_side
        input_ids = enc["input_ids"].to(self.device)      # [B, L]  (last col = offset)
        attn = enc["attention_mask"].to(self.device)
        B, L = input_ids.shape
        past_len = _past_length(past_key_values) if past_key_values is not None else 0

        # Prefill everything except the offset token onto the handoff cache.
        prefill_ids = input_ids[:, :-1]                   # [B, L-1]
        prefill_attn = attn[:, :-1]
        # Left padding -> real judger tokens are flush-right; position ids continue
        # from the handoff length, pads get position 0 (masked out anyway).
        valid = prefill_attn.sum(dim=1)                   # real judger tokens per row
        Lp = prefill_ids.shape[1]
        position_ids = torch.zeros((B, Lp), dtype=torch.long, device=self.device)
        for i in range(B):
            v = int(valid[i].item())
            if v > 0:
                position_ids[i, Lp - v:] = torch.arange(
                    past_len, past_len + v, device=self.device
                )
        if past_len > 0:
            past_mask = torch.ones((B, past_len), dtype=attn.dtype, device=self.device)
            prefill_full_mask = torch.cat([past_mask, prefill_attn], dim=-1)
        else:
            prefill_full_mask = prefill_attn
        cache_position = torch.arange(
            past_len, past_len + Lp, dtype=torch.long, device=self.device
        )
        out = self.model(
            input_ids=prefill_ids,
            attention_mask=prefill_full_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
        )
        cache = out.past_key_values

        # Steer the last cached position (the aggregation token) across all layers.
        steer_pos = _past_length(cache) - 1
        kvsteer.apply(cache, [steer_pos])

        # Generate from the offset token, continuing the steered cache.
        offset_ids = input_ids[:, -1:]                    # [B, 1]
        gen_full_mask = torch.cat(
            [prefill_full_mask, torch.ones((B, 1), dtype=attn.dtype, device=self.device)],
            dim=-1,
        )
        gen_cache_position = torch.arange(
            past_len + Lp, past_len + Lp + 1, dtype=torch.long, device=self.device
        )
        outputs = self.model.generate(
            input_ids=offset_ids,
            attention_mask=gen_full_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=cache,
            cache_position=gen_cache_position,
        )
        sequences = outputs.sequences
        # sequences = [offset token | generated...]; skip the offset prefix.
        gen_start = offset_ids.shape[1]
        generations: List[str] = []
        token_counts: List[int] = []
        for idx in range(sequences.shape[0]):
            generated_ids = sequences[idx, gen_start:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
            token_counts.append(self._count_generated_tokens(generated_ids.tolist()))
        self.last_gen_token_counts = token_counts
        return generations, None

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    def _ces_prepare(self, role: Optional[str]) -> bool:
        """Enable CES for this role if configured; return whether it was armed."""
        ces = getattr(self, "ces", None)
        if ces is None:
            return False
        role_l = (role or "").lower()
        if role_l and role_l not in ces.agents:
            ces.disable()
            return False
        ces.set_active_role(role)
        ces.enable()
        return True

    def _ces_finish(self, armed: bool) -> None:
        ces = getattr(self, "ces", None)
        if armed and ces is not None:
            ces.disable()

    def _generate_latent_batch_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        role: Optional[str] = None,
        detach_debug: bool = True,
        collect_latents: bool = False,
        forced_latents: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Shared latent rollout. Caller controls torch.no_grad / enable_grad.

        CES: by default steerer is active only during recurrent latent steps
        (phase='latent'), not during textual prefill. Prefill+latent is selected
        via ces.steer_phase == 'prefill_and_latent'.

        ``collect_latents`` stores the realigned embeddings actually fed at each
        latent step on ``self.last_latent_embeds`` with shape ``[K, B, D]``.
        ``forced_latents`` teacher-forces those embeddings (``[K, D]`` or
        ``[K, B, D]``) instead of recycling ``last_hidden``.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        _seal_on = self._seal_activate_for(role)
        _ces_on = self._ces_prepare(role)
        ces = getattr(self, "ces", None)
        collected: List[torch.Tensor] = []
        try:
            if ces is not None and _ces_on:
                ces.set_phase("prefill")

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            if ces is not None and _ces_on:
                ces.set_phase("latent")

            forced = None
            if forced_latents is not None:
                forced = forced_latents
                if forced.dim() == 2:
                    forced = forced.unsqueeze(1)
                if int(forced.shape[0]) != int(latent_steps):
                    raise ValueError(
                        f"forced_latents first dim {forced.shape[0]} != latent_steps {latent_steps}"
                    )

            for step in range(latent_steps):
                source_model = self.HF_model if hasattr(self, "HF_model") else self.model
                latent_vec = self._apply_latent_realignment(last_hidden, source_model)
                if forced is not None:
                    latent_vec = forced[step].to(device=latent_vec.device, dtype=latent_vec.dtype)
                if collect_latents:
                    collected.append(latent_vec.detach().float().cpu())
                if detach_debug:
                    # Side-channel only; do not break the live latent chain.
                    _ = latent_vec.detach()
                latent_embed = latent_vec.unsqueeze(1)

                past_len = _past_length(past)
                latent_mask = torch.ones(
                    (latent_embed.shape[0], past_len + 1),
                    dtype=torch.long,
                    device=self.device,
                )
                outputs = self.model(
                    inputs_embeds=latent_embed,
                    attention_mask=latent_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past = outputs.past_key_values
                last_hidden = outputs.hidden_states[-1][:, -1, :]
        finally:
            if _seal_on:
                self.seal.disable()
            self._ces_finish(_ces_on)
        self.last_latent_embeds = (
            torch.stack(collected, 0) if collect_latents and collected else None
        )
        return past

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        role: Optional[str] = None,
        collect_latents: bool = False,
        forced_latents: Optional[torch.Tensor] = None,
    ) -> Tuple:
        return self._generate_latent_batch_impl(
            input_ids,
            attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
            role=role,
            detach_debug=True,
            collect_latents=collect_latents,
            forced_latents=forced_latents,
        )

    def generate_latent_batch_grad(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        role: Optional[str] = None,
    ) -> Tuple:
        """Latent rollout with gradients enabled (CES training / Gate 2).

        Host weights should be frozen; only the CES vector should require grad.
        Does not use gradient checkpointing by default — validate separately if
        enabling checkpointing with differentiable KV caches.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                return self._generate_latent_batch_impl(
                    input_ids,
                    attention_mask,
                    latent_steps=latent_steps,
                    past_key_values=past_key_values,
                    role=role,
                    detach_debug=False,
                )
        finally:
            if was_training:
                self.model.train()

    def teacher_force_nll(
        self,
        past_key_values: Optional[Tuple],
        prompt_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        role: Optional[str] = None,
        steer_judger: bool = False,
    ) -> torch.Tensor:
        """Length-normalized NLL of target_ids after prompt under past_kv.

        Used for Gate 2/3 answer-NLL smoke and for CES energy E_v(y|x).
        By default CES is not applied during Judger teacher-force (primary claim
        steers upstream latent steps only).
        """
        from seal.ces import length_normalized_nll_from_logits

        if prompt_ids.dim() != 2 or target_ids.dim() != 2:
            raise ValueError("prompt_ids and target_ids must be [B, T]")
        if prompt_ids.shape[0] != target_ids.shape[0]:
            raise ValueError("batch size mismatch")

        # Optionally arm CES for judger (smoke / ablation only).
        ces_armed = False
        if steer_judger:
            ces_armed = self._ces_prepare(role or "judger")
            ces = getattr(self, "ces", None)
            if ces is not None and ces_armed:
                ces.set_phase("latent")
        try:
            full_ids = torch.cat([prompt_ids, target_ids], dim=1)
            attn = torch.ones_like(full_ids, device=full_ids.device)
            if past_key_values is not None:
                past_len = _past_length(past_key_values)
                if past_len > 0:
                    past_mask = torch.ones(
                        (attn.shape[0], past_len),
                        dtype=attn.dtype,
                        device=attn.device,
                    )
                    attn = torch.cat([past_mask, attn], dim=-1)
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=attn,
                past_key_values=past_key_values,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs.logits  # [B, prompt+target, V]
            # Causal: predict token t from position t-1.
            # Targets occupy the last |target| positions of full_ids; their
            # predicting logits are at indices [prompt_len-1 : prompt_len+target_len-1]
            # relative to the prompt+target segment (which starts at index 0 of logits).
            prompt_len = prompt_ids.shape[1]
            target_len = target_ids.shape[1]
            pred_logits = logits[:, prompt_len - 1 : prompt_len + target_len - 1, :]
            return length_normalized_nll_from_logits(pred_logits, target_ids)
        finally:
            self._ces_finish(ces_armed)

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings

