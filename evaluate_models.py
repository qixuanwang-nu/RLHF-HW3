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
  evaluation/run_<timestamp>/
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
    # Per assignment: RL evaluation artifacts live under ./evaluation/ by default.
    p.add_argument("--output_dir", type=str, default="evaluation", help="Output directory")

    return p.parse_args()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _load_env_file_nonempty(path: str) -> None:
    """
    Load KEY=VALUE pairs from a file into os.environ, but only apply non-empty values.

    This avoids the common pitfall where an env template contains OPENAI_API_KEY= (empty)
    and accidentally disables OpenAI judging.
    """
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Strip optional surrounding quotes
                if len(val) >= 2 and ((val[0] == val[-1]) and val[0] in ("'", '"')):
                    val = val[1:-1]
                if not key or val == "":
                    continue
                os.environ[key] = val
    except OSError:
        return


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


def _try_load_training_stats_from_model_path(model_path: str) -> List[Dict]:
    """
    Given a saved model directory like outputs/policy/run_xxx/ppo/model,
    try to load the corresponding training_stats.json from the parent dir.
    """
    if not model_path:
        return []
    # model_path may be .../<method>/model
    parent = os.path.dirname(model_path.rstrip("/"))
    stats_path = os.path.join(parent, "training_stats.json")
    if not os.path.exists(stats_path):
        return []
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _plot_training_curves(
    ppo_stats: List[Dict],
    grpo_stats: List[Dict],
    dpo_stats: List[Dict],
    out_dir: str,
) -> None:
    """
    Plot training curves (reward/perf, KL, loss) and trade-offs among them.
    Saves:
      - training_curves.png
      - training_reward_vs_kl.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # Extract series
    def series(stats: List[Dict], key: str, default: float = 0.0) -> List[float]:
        return [float(s.get(key, default)) for s in stats]

    ppo_reward = series(ppo_stats, "reward_mean")
    ppo_kl = series(ppo_stats, "kl_divergence")
    ppo_loss = series(ppo_stats, "total_loss")

    grpo_reward = series(grpo_stats, "reward_mean")
    grpo_kl = series(grpo_stats, "kl_divergence")
    grpo_loss = series(grpo_stats, "total_loss")

    # DPO doesn't have reward_mean / kl_divergence; use preference margin and implicit KL proxy
    dpo_perf = series(dpo_stats, "reward_margin")  # >0 means chosen preferred more strongly
    dpo_kl = series(dpo_stats, "implicit_kl")
    dpo_loss = series(dpo_stats, "loss")

    # 1) Curves figure
    plt.figure(figsize=(16, 4.8))

    ax1 = plt.subplot(1, 3, 1)
    if ppo_reward:
        ax1.plot(ppo_reward, label="PPO (reward_mean)")
    if grpo_reward:
        ax1.plot(grpo_reward, label="GRPO (reward_mean)")
    if dpo_perf:
        ax1.plot(dpo_perf, label="DPO (reward_margin)")
    ax1.set_title("Training Performance")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Value")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(1, 3, 2)
    if ppo_kl:
        ax2.plot(ppo_kl, label="PPO (KL)")
    if grpo_kl:
        ax2.plot(grpo_kl, label="GRPO (KL)")
    if dpo_kl:
        ax2.plot(dpo_kl, label="DPO (implicit_kl)")
    ax2.set_title("Drift vs Reference")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("KL / proxy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(1, 3, 3)
    if ppo_loss:
        ax3.plot(ppo_loss, label="PPO (total_loss)")
    if grpo_loss:
        ax3.plot(grpo_loss, label="GRPO (total_loss)")
    if dpo_loss:
        ax3.plot(dpo_loss, label="DPO (loss)")
    ax3.set_title("Training Loss")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()

    # 2) Tradeoff scatter: performance vs KL (per-step)
    plt.figure(figsize=(7.5, 6.0))
    if ppo_reward and ppo_kl:
        plt.scatter(ppo_kl, ppo_reward, s=14, alpha=0.7, label="PPO")
    if grpo_reward and grpo_kl:
        plt.scatter(grpo_kl, grpo_reward, s=14, alpha=0.7, label="GRPO")
    if dpo_perf and dpo_kl:
        plt.scatter(dpo_kl, dpo_perf, s=14, alpha=0.7, label="DPO")
    plt.xlabel("KL / proxy (lower is better)")
    plt.ylabel("Performance (higher is better)")
    plt.title("Training Trade-off: Performance vs KL")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_reward_vs_kl.png"), dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    device = _device()
    print(f"Using device: {device}")

    # Load env vars from file (non-empty values only)
    if args.env_file:
        _load_env_file_nonempty(args.env_file)
    else:
        _load_env_file_nonempty("env.example")

    if args.prefer_openai_judge:
        print(f"OPENAI_API_KEY detected: {bool(os.getenv('OPENAI_API_KEY'))}")
        print(f"OPENAI_MODEL: {os.getenv('OPENAI_MODEL')}")

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
    generations_quant = {"prompts": prompts, "responses": {}, "per_prompt": {}}

    for name, model in models.items():
        print(f"Generating for model: {name}")
        # Stream evaluation in batches to avoid GPU OOM.
        all_rewards: List[float] = []
        all_resps: List[str] = []
        kl_means: List[float] = []
        kl_maxs: List[float] = []
        per_prompt_kl_mean: List[float] = []
        per_prompt_reward: List[float] = []

        for batch_prompts in _chunks(prompts, args.gen_batch_size):
            # generation
            batch_resps, out_ids, attn, prompt_lens, prompt_padded_len = generate_responses(
                model, tokenizer, batch_prompts, device, gen_cfg
            )
            all_resps.extend(batch_resps)
            response_mask = make_response_mask(attn, prompt_padded_len)

            # reward scoring (reward model usually smaller than policy, but still batch if needed)
            # If gen_batch_size is large, further split for scoring.
            bsz = out_ids.size(0)
            for s in range(0, bsz, args.score_batch_size):
                o = out_ids[s : s + args.score_batch_size]
                m = attn[s : s + args.score_batch_size]
                r = compute_reward_scores(reward_model, o, m).detach().cpu().numpy().tolist()
                all_rewards.extend(r)
                per_prompt_reward.extend(r)

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
                    per_prompt_kl_mean.append(kl_b["kl_mean"])

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

        # Save per-prompt generations (used for sample export and analysis)
        generations_quant["responses"][name] = all_resps
        generations_quant["per_prompt"][name] = {
            "reward": per_prompt_reward,
            "kl_mean": ([0.0] * len(per_prompt_reward)) if name == "reference" else per_prompt_kl_mean,
        }

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

    # Ensure all models are on the correct device before win-rate evaluation
    for name, model in models.items():
        model.to(device)

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
    # Ensure all models are on the correct device before adversarial evaluation
    for name, model in models.items():
        model.to(device)
    
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
    with open(os.path.join(run_dir, "generations_quantitative.json"), "w") as f:
        json.dump(generations_quant, f, indent=2)
    with open(os.path.join(run_dir, "pareto_frontier.json"), "w") as f:
        json.dump({"points": pareto_points, "frontier": frontier}, f, indent=2)
    with open(os.path.join(run_dir, "winrate.json"), "w") as f:
        json.dump(winrate, f, indent=2)
    with open(os.path.join(run_dir, "qualitative_adversarial.json"), "w") as f:
        json.dump(adv_out, f, indent=2)

    _plot_reward_hist(reward_arrays, os.path.join(run_dir, "reward_hist.png"))
    _plot_reward_vs_kl(pareto_points, frontier, os.path.join(run_dir, "reward_vs_kl.png"))

    # Training curves (from training_stats.json if available alongside model dirs)
    ppo_stats = _try_load_training_stats_from_model_path(args.ppo_model_path) if args.ppo_model_path else []
    grpo_stats = _try_load_training_stats_from_model_path(args.grpo_model_path) if args.grpo_model_path else []
    dpo_stats = _try_load_training_stats_from_model_path(args.dpo_model_path) if args.dpo_model_path else []
    if ppo_stats or grpo_stats or dpo_stats:
        _plot_training_curves(ppo_stats, grpo_stats, dpo_stats, run_dir)

    print(f"\nSaved evaluation artifacts to: {run_dir}")


if __name__ == "__main__":
    main()


