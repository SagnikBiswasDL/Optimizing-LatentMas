import argparse
import json
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct

# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"], required=True,
                        help="Which multi-agent method to run: 'baseline', 'text_mas', or 'latent_mas'.")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"],
                        help="Model choices to use for experiments (e.g. 'Qwen/Qwen3-14B').")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0, help="Default latent steps for all upstream agents")
    parser.add_argument("--planner_steps", type=int, default=None,
                        help="Override Planner latent steps (default: --latent_steps)")
    parser.add_argument("--critic_steps", type=int, default=None,
                        help="Override Critic latent steps (default: --latent_steps)")
    parser.add_argument("--refiner_steps", type=int, default=None,
                        help="Override Refiner latent steps (default: --latent_steps)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agents", type=str, default=None,
                        help="LatentMAS agent chain for ablations: comma-separated subset of planner,critic,refiner (Judger is always appended last). e.g. 'planner,critic' or 'full'. Default: full 4-agent chain.")

    # SEAL steering (token efficiency): applied to the Judger's text decoding (HF backend)
    parser.add_argument("--seal", action="store_true", help="Enable SEAL steering during Judger decoding")
    parser.add_argument("--seal_vector", type=str, default=None, help="Path to the SEAL steering-vector artifact (.pt)")
    parser.add_argument("--seal_layer", type=int, default=-1, help="Decoder layer to steer; -1 uses the artifact's layer_index")
    parser.add_argument("--seal_coef", type=float, default=0.0, help="Steering coefficient (>0 suppresses reflection/transition)")
    parser.add_argument("--seal_apply_to", type=str, default="last", choices=["last", "all"], help="Steer only the current token ('last') or all positions ('all')")
    parser.add_argument("--seal_agents", type=str, default="judger", help="Which agent roles to steer: comma-separated subset of planner,critic,refiner,judger (or 'all'). Default: judger.")

    # KV-cache steering (arXiv:2507.08799): one-shot edit of the shared K/V cache
    # the Judger consumes (the latent-agent handoff), across all layers. HF backend.
    parser.add_argument("--kvsteer", action="store_true", help="Enable one-shot KV-cache steering of the LatentMAS handoff before the Judger decodes")
    parser.add_argument("--kvsteer_vector", type=str, default=None, help="Path to the KV steering-vector artifact (.pt) with per-layer keys/values")
    parser.add_argument("--kvsteer_cv", type=float, default=0.0, help="Value steering coefficient c_v (0 reproduces plain LatentMAS)")
    parser.add_argument("--kvsteer_ck", type=float, default=0.0, help="Key steering coefficient c_k (paper default ~0)")
    parser.add_argument("--kvsteer_positions", type=str, default="handoff_last",
                        choices=["handoff_last", "handoff_lastk", "handoff_all", "judger_token"],
                        help="Which cache positions to steer: last handoff column, last-k handoff columns, all handoff columns, or the Judger's own final prompt token (control arm)")
    parser.add_argument("--kvsteer_last_k", type=int, default=40, help="Number of trailing handoff columns to steer when --kvsteer_positions handoff_lastk")

    # In-pipeline activation capture (native-vector program): record each agent's
    # layer-L latent state while it runs inside the full pipeline, then save a cache
    # (activations + final correctness) for correctness-contrastive vector building.
    parser.add_argument("--capture_acts", type=str, default=None, help="If set, capture per-agent layer-L activations during the run and save a cache to this path (.pt).")
    parser.add_argument("--capture_layer", type=int, default=-1, help="Layer to capture activations from; -1 falls back to --seal_layer then 28.")

    # CES / K-budget activation steering (trainable residual vector). Off by default.
    parser.add_argument("--ces", action="store_true", help="Enable CES residual steering during LatentMAS (HF)")
    parser.add_argument("--ces_vector", type=str, default=None, help="Optional path to a saved CES vector (.pt)")
    parser.add_argument("--ces_layer", type=int, default=28, help="Decoder layer for CES intervention")
    parser.add_argument("--ces_coef", type=float, default=1.0, help="Multiplier on the CES vector")
    parser.add_argument("--ces_agents", type=str, default="planner,critic,refiner",
                        help="Roles to steer (default: upstream latent agents only)")
    parser.add_argument("--ces_steer_phase", type=str, default="latent_only",
                        choices=["none", "latent_only", "prefill_and_latent"],
                        help="Primary: latent_only (skip textual prefill). Ablation: prefill_and_latent.")
    parser.add_argument("--ces_init_std", type=float, default=0.0, help="Init std for trainable v when no artifact")

    # vLLM support
    parser.add_argument("--embed_assembly", choices=["original", "repaired"], default="original",
                        help="Judger embedding assembly for the hybrid path. 'original' is the authors' code, which crashes at batch size 1. 'repaired' keeps the batch dimension and drops padding.")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()
    
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
    
    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
        )
    progress.close()
    
    total_time = time.time() - start_time

    acc, correct = evaluate(preds)

    # Persist the in-pipeline activation capture (native-vector program).
    if getattr(args, "capture_acts", None):
        import torch as _torch
        import os as _os
        roles = ["planner", "critic", "refiner", "judger"]
        acts_by_role = {r: [] for r in roles}
        keep_idx = {r: [] for r in roles}
        corr, out_toks, preds_list, golds, raw_preds = [], [], [], [], []
        for i, p in enumerate(preds):
            aa = p.get("agent_acts", {}) or {}
            corr.append(bool(p.get("correct", False)))
            out_toks.append(int(p.get("output_tokens", 0)))
            preds_list.append(p.get("prediction", ""))
            golds.append(p.get("gold", ""))
            raw_preds.append(p.get("raw_prediction", ""))
            for r in roles:
                if r in aa:
                    acts_by_role[r].append(aa[r].float().cpu())
                    keep_idx[r].append(i)
        acts_tensors = {r: (_torch.stack(v, 0) if v else _torch.empty(0)) for r, v in acts_by_role.items()}
        blob = {
            "acts": acts_tensors,               # role -> [n_role, D]
            "acts_row_index": keep_idx,         # role -> list of global run indices
            "correct": _torch.tensor(corr, dtype=_torch.bool),
            "output_tokens": _torch.tensor(out_toks, dtype=_torch.long),
            "prediction": preds_list,
            "gold": golds,
            "raw_prediction": raw_preds,
            "layer_index": int(getattr(model, "act_recorder").layer_index),
            "model_name": args.model_name,
            "task": args.task,
            "split": args.split,
            "seed": args.seed,
            "n": len(preds),
        }
        _os.makedirs(_os.path.dirname(_os.path.abspath(args.capture_acts)), exist_ok=True)
        _torch.save(blob, args.capture_acts)
        print(f"[capture] saved {len(preds)} runs -> {args.capture_acts} "
              f"(layer {blob['layer_index']}, correct={sum(corr)}/{len(corr)})")

    # Output-token usage (Judger text decoding); 0 if a method does not report it.
    out_toks = [int(p.get("output_tokens", 0)) for p in preds]
    total_output_tokens = sum(out_toks)
    n_tok = len(out_toks) if out_toks else 1
    mean_output_tokens = total_output_tokens / n_tok

    # Failure taxonomy: correct / wrong (parsed but incorrect) / no_answer (unparseable/empty).
    n_correct_fx = sum(1 for p in preds if p.get("correct"))
    n_no_answer = sum(1 for p in preds if not str(p.get("prediction", "")).strip())
    n_wrong = len(preds) - n_correct_fx - n_no_answer

    # LatentMAS agent-chain / latent-compute accounting (for ablation studies).
    agents_spec = getattr(args, "agents", None)
    latent_forwards = None
    upstream_roles = None
    if args.method == "latent_mas":
        from methods import agents_from_spec as _afs
        chain = _afs(agents_spec)
        upstream_roles = [a.role for a in chain if a.role != "judger"]
        # Each non-Judger agent runs 1 prefill + K_role continuation forwards.
        latent_forwards = 0
        for role in upstream_roles:
            k = getattr(args, f"{role}_steps", None)
            if k is None:
                k = args.latent_steps
            latent_forwards += int(k) + 1

    # Load results in JSON format
    print(
        json.dumps(
            {
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "accuracy": acc,
                "correct": correct,
                "total_time_sec": round(total_time,4),
                "time_per_sample_sec": round(total_time / args.max_samples, 4),
                "total_output_tokens": total_output_tokens,
                "mean_output_tokens": round(mean_output_tokens, 2),
                "seal": bool(getattr(args, "seal", False)),
                "seal_coef": float(getattr(args, "seal_coef", 0.0)),
                "seal_layer": int(getattr(args, "seal_layer", -1)),
                "seal_agents": getattr(args, "seal_agents", "judger"),
                "kvsteer": bool(getattr(args, "kvsteer", False)),
                "kvsteer_cv": float(getattr(args, "kvsteer_cv", 0.0)),
                "kvsteer_ck": float(getattr(args, "kvsteer_ck", 0.0)),
                "kvsteer_positions": getattr(args, "kvsteer_positions", "handoff_last"),
                "kvsteer_last_k": int(getattr(args, "kvsteer_last_k", 40)),
                "agents": agents_spec or "full",
                "upstream_roles": upstream_roles,
                "n_upstream_agents": (len(upstream_roles) if upstream_roles is not None else None),
                "latent_forwards": latent_forwards,
                "n_correct": n_correct_fx,
                "n_wrong": n_wrong,
                "n_no_answer": n_no_answer,
            },
            ensure_ascii=False,
        )
    )



if __name__ == "__main__":
    main()
