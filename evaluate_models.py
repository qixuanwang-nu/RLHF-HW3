"""
Part 4: Analysis and Evaluation entry point.

Quantitative:
- Win rate vs reference model (GPT-4-as-judge if configured; fallback to reward-model judge)
- Reward model score distributions (base/ref/PPO/GRPO/DPO)
- KL drift from reference policy
- Pareto frontier table (reward vs KL)

Qualitative:
- Adversarial prompt set generation and side-by-side dumps

Outputs:
  outputs/evaluation/run_<timestamp>/
    quantitative.json
    pareto_frontier.json
    winrate.json
    qualitative_adversarial.json
    reward_hist.png
    reward_vs_kl.png
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.reward_model import RewardModel
from src.evaluation.judges import get_default_judge
from src.evaluation.model_eval import (
    GenConfig,
    compute_kl_proxy_vs_reference,
    compute_reward_scores,
    generate_responses,
    load_causal_lm,
    load_tokenizer,
    make_response_mask,
    pareto_frontier,
    summarize_distribution,
)
from src.evaluation.prompt_sets import adversarial_prompts, load_hh_prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO/GRPO/DPO models (Part 4).")
    p.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Optional env file to load (e.g., env.example). Loaded via python-dotenv if installed.",
    )

    # Model paths
    p.add_argument("--reference_model", type=str, default="openai-community/gpt2", help="Reference/base model name/path")
    p.add_argument("--ppo_model_path", type=str, default=None, help="Path to PPO saved model dir (outputs/.../ppo/model)")
    p.add_argument("--grpo_model_path", type=str, default=None, help="Path to GRPO saved model dir (outputs/.../grpo/model)")
    p.add_argument("--dpo_model_path", type=str, default=None, help="Path to DPO saved model dir (outputs/.../dpo/model)")

    # Reward model (for reward distributions and fallback judging)
    p.add_argument("--reward_model_path", type=str, required=True, help="Path to trained reward model dir (best_model)")

    # Prompt selection
    p.add_argument("--prompt_split", type=str, default="test", help="Dataset split to sample prompts from")
    p.add_argument("--num_prompts", type=int, default=200, help="Number of prompts for quantitative eval (>=100 recommended)")
    p.add_argument("--seed", type=int, default=42)

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--gen_batch_size", type=int, default=8, help="Batch size for generation (reduce if OOM)")
    p.add_argument("--score_batch_size", type=int, default=2, help="Batch size for reward/KL scoring (reduce if OOM)")
    p.add_argument(
        "--amp_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Autocast dtype for scoring on CUDA (fp16 usually best for memory)",
    )

    # Win-rate judging
    p.add_argument("--winrate_prompts", type=int, default=120, help="Number of prompts for win-rate eval vs reference")
    p.add_argument("--prefer_openai_judge", action="store_true", help="Use GPT-4-as-judge if OPENAI_API_KEY is set")

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/evaluation", help="Output directory")

    return p.parse_args()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _autocast_dtype(args: argparse.Namespace) -> object:
    if not torch.cuda.is_available() or args.amp_dtype == "fp32":
        return None
    if args.amp_dtype == "bf16":
        return torch.bfloat16
    return torch.float16


def _chunks(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def _plot_reward_hist(all_rewards: Dict[str, np.ndarray], out_path: str):
    plt.figure(figsize=(10, 6))
    for name, vals in all_rewards.items():
        if vals.size == 0:
            continue
        plt.hist(vals, bins=30, alpha=0.35, label=name, density=True)
    plt.title("Reward Model Score Distributions")
    plt.xlabel("Reward score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_reward_vs_kl(points: List[Dict[str, float]], frontier: List[Dict[str, float]], out_path: str):
    plt.figure(figsize=(8, 6))
    for p in points:
        plt.scatter(p["kl_mean"], p["reward_mean"], label=p["name"])
        plt.text(p["kl_mean"], p["reward_mean"], p["name"], fontsize=9)
    if frontier:
        xs = [p["kl_mean"] for p in frontier]
        ys = [p["reward_mean"] for p in frontier]
        plt.plot(xs, ys, color="black", linewidth=2, label="Pareto frontier")
    plt.xlabel("KL proxy vs reference (lower is better)")
    plt.ylabel("Reward-model mean score (higher is better)")
    plt.title("Reward vs KL (Pareto frontier)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    device = _device()
    print(f"Using device: {device}")

    # Load env vars from file if python-dotenv is available
    try:
        from dotenv import load_dotenv  # type: ignore

        if args.env_file and os.path.exists(args.env_file):
            load_dotenv(args.env_file, override=False)
        elif os.path.exists("env.example"):
            # Best-effort fallback (usually has empty API key, so OpenAI judge won't be used)
            load_dotenv("env.example", override=False)
    except Exception:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Load reward model
    if not os.path.isdir(args.reward_model_path):
        raise SystemExit(
            "reward_model_path must be a local directory containing a saved RewardModel "
            f"(got: {args.reward_model_path}). "
            "Tip: pass the exact 'best_model' directory produced by train_reward_model.py."
        )
    print(f"Loading reward model from {args.reward_model_path} ...")
    reward_model = RewardModel.from_pretrained(args.reward_model_path).to(device)
    reward_model.eval()

    # Tokenizer for generation (use reference tokenizer)
    tokenizer = load_tokenizer(args.reference_model)

    # Load models
    ref = load_causal_lm(args.reference_model, device)
    models: Dict[str, object] = {"reference": ref}
    if args.ppo_model_path:
        models["ppo"] = load_causal_lm(args.ppo_model_path, device)
    if args.grpo_model_path:
        models["grpo"] = load_causal_lm(args.grpo_model_path, device)
    if args.dpo_model_path:
        models["dpo"] = load_causal_lm(args.dpo_model_path, device)

    # Prompts
    prompts = load_hh_prompts(split=args.prompt_split, num_prompts=args.num_prompts, seed=args.seed)
    print(f"Loaded {len(prompts)} prompts for quantitative eval")

    gen_cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    amp_dtype = _autocast_dtype(args)

    # Quantitative eval per model
    per_model = {}
    pareto_points: List[Dict[str, float]] = []
    reward_arrays: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(f"Generating for model: {name}")
        # Stream evaluation in batches to avoid GPU OOM.
        all_rewards: List[float] = []
        kl_means: List[float] = []
        kl_maxs: List[float] = []

        for batch_prompts in _chunks(prompts, args.gen_batch_size):
            # generation
            _, out_ids, attn, prompt_lens = generate_responses(model, tokenizer, batch_prompts, device, gen_cfg)
            response_mask = make_response_mask(attn, prompt_lens)

            # reward scoring (reward model usually smaller than policy, but still batch if needed)
            # If gen_batch_size is large, further split for scoring.
            bsz = out_ids.size(0)
            for s in range(0, bsz, args.score_batch_size):
                o = out_ids[s : s + args.score_batch_size]
                m = attn[s : s + args.score_batch_size]
                r = compute_reward_scores(reward_model, o, m).detach().cpu().numpy().tolist()
                all_rewards.extend(r)

            # KL drift scoring (policy forward is expensive; score in small batches)
            kl = {"kl_mean": 0.0, "kl_max": 0.0}
            if name != "reference":
                for s in range(0, bsz, args.score_batch_size):
                    o = out_ids[s : s + args.score_batch_size]
                    m = attn[s : s + args.score_batch_size]
                    rm = response_mask[s : s + args.score_batch_size]
                    kl_b = compute_kl_proxy_vs_reference(
                        model, ref, o, m, rm, autocast_dtype=amp_dtype
                    )
                    kl_means.append(kl_b["kl_mean"])
                    kl_maxs.append(kl_b["kl_max"])

            # Free transient GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rewards_np = np.array(all_rewards, dtype=np.float64)
        reward_arrays[name] = rewards_np

        if name == "reference":
            kl_summary = {"kl_mean": 0.0, "kl_max": 0.0}
        else:
            kl_summary = {
                "kl_mean": float(np.mean(kl_means)) if kl_means else 0.0,
                "kl_max": float(np.max(kl_maxs)) if kl_maxs else 0.0,
            }

        per_model[name] = {"reward": summarize_distribution(rewards_np), "kl": kl_summary}

        pareto_points.append(
            {"name": name, "reward_mean": per_model[name]["reward"]["mean"], "kl_mean": per_model[name]["kl"]["kl_mean"]}
        )

        # Move non-reference models off GPU between evaluations to save memory
        if torch.cuda.is_available() and name != "reference":
            model.to("cpu")
            torch.cuda.empty_cache()

    frontier = pareto_frontier(pareto_points)

    # Win-rate vs reference (judge)
    judge, judge_name = get_default_judge(reward_model, tokenizer, device, prefer_openai=args.prefer_openai_judge)
    print(f"Judge: {judge_name}")

    win_prompts = load_hh_prompts(split=args.prompt_split, num_prompts=args.winrate_prompts, seed=args.seed + 999)
    print(f"Loaded {len(win_prompts)} prompts for win-rate eval")

    # Generate reference once
    # Generate reference in batches
    ref_resps: List[str] = []
    for bp in _chunks(win_prompts, args.gen_batch_size):
        rr, _, _, _ = generate_responses(ref, tokenizer, bp, device, gen_cfg)
        ref_resps.extend(rr)

    winrate = {}
    for name, model in models.items():
        if name == "reference":
            continue
        model_resps: List[str] = []
        for bp in _chunks(win_prompts, args.gen_batch_size):
            rr, _, _, _ = generate_responses(model, tokenizer, bp, device, gen_cfg)
            model_resps.extend(rr)
        wins = 0
        losses = 0
        ties = 0
        samples = []
        for p, a, b in zip(win_prompts, model_resps, ref_resps):
            res = judge.judge(p, a, b, meta={"model": name, "ref": "reference"})
            if res.winner == "A":
                wins += 1
            elif res.winner == "B":
                losses += 1
            else:
                ties += 1
            # Keep a small sample for reporting/debug
            if len(samples) < 25:
                samples.append({"prompt": p, "model": a, "reference": b, "winner": res.winner, "reason": res.reason})
        total = wins + losses + ties
        winrate[name] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / max(total, 1),
            "judge": judge_name,
            "sampled_cases": samples,
        }

    # Qualitative adversarial set
    adv = adversarial_prompts()
    adv_out = {"prompts": adv, "responses": {}}
    for name, model in models.items():
        adv_resps: List[str] = []
        for bp in _chunks(adv, args.gen_batch_size):
            rr, _, _, _ = generate_responses(model, tokenizer, bp, device, gen_cfg)
            adv_resps.extend(rr)
        adv_out["responses"][name] = adv_resps

    # Save outputs
    quantitative = {
        "config": {
            **vars(args),
            "device": device,
            "judge": judge_name,
            "models_loaded": list(models.keys()),
            "gen": asdict(gen_cfg),
        },
        "per_model": per_model,
    }

    with open(os.path.join(run_dir, "quantitative.json"), "w") as f:
        json.dump(quantitative, f, indent=2)
    with open(os.path.join(run_dir, "pareto_frontier.json"), "w") as f:
        json.dump({"points": pareto_points, "frontier": frontier}, f, indent=2)
    with open(os.path.join(run_dir, "winrate.json"), "w") as f:
        json.dump(winrate, f, indent=2)
    with open(os.path.join(run_dir, "qualitative_adversarial.json"), "w") as f:
        json.dump(adv_out, f, indent=2)

    _plot_reward_hist(reward_arrays, os.path.join(run_dir, "reward_hist.png"))
    _plot_reward_vs_kl(pareto_points, frontier, os.path.join(run_dir, "reward_vs_kl.png"))

    print(f"\nSaved evaluation artifacts to: {run_dir}")


if __name__ == "__main__":
    main()


