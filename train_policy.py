"""
Part 2 & 3: Policy Optimization Training Script
Trains language model policies using PPO, GRPO, or DPO.

Usage:
    # Train with PPO
    python train_policy.py --method ppo --steps 100
    
    # Train with GRPO
    python train_policy.py --method grpo --steps 100 --group_size 4
    
    # Train with DPO (no reward model needed)
    python train_policy.py --method dpo --epochs 1
    
    # Compare all methods
    python train_policy.py --method all --steps 100
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.reward_model import RewardModel, create_reward_model
from src.algorithms.ppo import PPOConfig, PPOTrainer, create_ppo_trainer
from src.algorithms.grpo import GRPOConfig, GRPOTrainer, create_grpo_trainer
from src.algorithms.dpo import (
    DPOConfig, DPOTrainer, DPODataset,
    create_dpo_trainer, prepare_dpo_data
)
from src.algorithms.comparison import (
    compare_algorithms,
    visualize_comparison,
    generate_comparison_report
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train policy with PPO, GRPO, or DPO")
    
    # Method selection
    parser.add_argument("--method", type=str, default="ppo",
                        choices=["ppo", "grpo", "dpo", "both", "all"],
                        help="Training method: ppo, grpo, dpo, both (ppo+grpo), all (ppo+grpo+dpo)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2",
                        help="Pretrained model to use as policy")
    parser.add_argument("--reward_model_path", type=str, default=None,
                        help="Path to trained reward model (uses fresh one if None)")
    
    # Training arguments
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of training steps (for PPO/GRPO)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (for DPO)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (prompts per step)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    
    # PPO specific
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                        help="PPO clip ratio")
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="PPO epochs per batch")
    parser.add_argument("--kl_coef", type=float, default=0.5,
                        help="KL penalty coefficient (higher = more stable)")
    parser.add_argument("--entropy_coef", type=float, default=0.1,
                        help="Entropy bonus coefficient (higher = prevents mode collapse)")
    
    # GRPO specific
    parser.add_argument("--group_size", type=int, default=4,
                        help="GRPO group size (responses per prompt)")
    
    # DPO specific
    parser.add_argument("--dpo_beta", type=float, default=0.1,
                        help="DPO temperature parameter")
    parser.add_argument("--dpo_loss_type", type=str, default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo"],
                        help="DPO loss type")
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/policy",
                        help="Output directory")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log every N steps")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="Number of prompts to use (None for all)")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts(num_prompts: int = None):
    """
    Load prompts from the HH-RLHF dataset.
    
    Returns:
        List of prompt strings
    """
    print("Loading prompts from HH-RLHF dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    
    prompts = []
    for example in dataset:
        # Extract prompt (everything before the last Assistant response)
        text = example["chosen"]
        last_assistant = text.rfind("\n\nAssistant:")
        if last_assistant > 0:
            prompt = text[:last_assistant + len("\n\nAssistant:")]
            prompts.append(prompt)
        
        if num_prompts and len(prompts) >= num_prompts:
            break
    
    print(f"Loaded {len(prompts)} prompts")
    return prompts


class PromptDataset(torch.utils.data.Dataset):
    """Simple dataset for prompts."""
    
    def __init__(self, prompts: list):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]


def collate_prompts(batch):
    """Collate prompts into a batch."""
    return list(batch)


def train_ppo(
    args,
    reward_model,
    prompts: list,
    output_dir: str
) -> list:
    """
    Train policy using PPO.
    
    Args:
        args: Command line arguments
        reward_model: Trained reward model
        prompts: List of prompt strings
        output_dir: Output directory
        
    Returns:
        Training statistics
    """
    print("\n" + "="*60)
    print("Training with PPO")
    print("="*60)
    
    # Create config
    config = PPOConfig(
        model_name=args.model_name,
        clip_ratio=args.clip_ratio,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    
    print(f"PPO Config:")
    print(f"  Clip ratio: {config.clip_ratio}")
    print(f"  KL coefficient: {config.kl_coef}")
    print(f"  Entropy coefficient: {config.entropy_coef}")
    print(f"  PPO epochs: {config.ppo_epochs}")
    print(f"  Batch size: {config.batch_size}")
    
    # Create trainer
    trainer = create_ppo_trainer(config, reward_model)
    
    # Create dataloader
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_prompts
    )
    
    # Train
    stats = trainer.train(
        dataloader=dataloader,
        num_steps=args.steps,
        log_every=args.log_every
    )
    
    # Save results
    ppo_dir = os.path.join(output_dir, "ppo")
    os.makedirs(ppo_dir, exist_ok=True)
    
    with open(os.path.join(ppo_dir, "training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    # Save the trained model
    model_dir = os.path.join(ppo_dir, "model")
    trainer.save_model(model_dir)
    
    print(f"\nPPO training complete. Results saved to {ppo_dir}")
    
    return stats


def train_grpo(
    args,
    reward_model,
    prompts: list,
    output_dir: str
) -> list:
    """
    Train policy using GRPO.
    
    Args:
        args: Command line arguments
        reward_model: Trained reward model
        prompts: List of prompt strings
        output_dir: Output directory
        
    Returns:
        Training statistics
    """
    print("\n" + "="*60)
    print("Training with GRPO")
    print("="*60)
    
    # Create config
    config = GRPOConfig(
        model_name=args.model_name,
        group_size=args.group_size,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    
    print(f"GRPO Config:")
    print(f"  Group size: {config.group_size}")
    print(f"  KL coefficient: {config.kl_coef}")
    print(f"  Entropy coefficient: {config.entropy_coef}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Effective responses per step: {config.batch_size * config.group_size}")
    
    # Create trainer
    trainer = create_grpo_trainer(config, reward_model)
    
    # Create dataloader
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_prompts
    )
    
    # Train
    stats = trainer.train(
        dataloader=dataloader,
        num_steps=args.steps,
        log_every=args.log_every
    )
    
    # Save results
    grpo_dir = os.path.join(output_dir, "grpo")
    os.makedirs(grpo_dir, exist_ok=True)
    
    with open(os.path.join(grpo_dir, "training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    # Save efficiency stats
    efficiency_stats = trainer.get_efficiency_stats()
    with open(os.path.join(grpo_dir, "efficiency_stats.json"), "w") as f:
        json.dump(efficiency_stats, f, indent=2)
    
    # Save the trained model
    model_dir = os.path.join(grpo_dir, "model")
    trainer.save_model(model_dir)
    
    print(f"\nGRPO training complete. Results saved to {grpo_dir}")
    
    return stats


def train_dpo(
    args,
    output_dir: str
) -> list:
    """
    Train policy using DPO (Direct Preference Optimization).
    
    Note: DPO doesn't require a reward model - it learns directly from preferences.
    
    Args:
        args: Command line arguments
        output_dir: Output directory
        
    Returns:
        Training statistics
    """
    print("\n" + "="*60)
    print("Training with DPO (Direct Preference Optimization)")
    print("="*60)
    print("Note: DPO bypasses reward modeling - learning directly from preferences")
    
    # Create config
    config = DPOConfig(
        model_name=args.model_name,
        beta=args.dpo_beta,
        loss_type=args.dpo_loss_type,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_length=512,
        seed=args.seed,
    )
    
    print(f"DPO Config:")
    print(f"  Beta (temperature): {config.beta}")
    print(f"  Loss type: {config.loss_type}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    
    # Create trainer
    trainer, tokenizer = create_dpo_trainer(config)
    
    # Load preference data
    print("\nLoading preference data from HH-RLHF...")
    raw_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    
    # Prepare DPO data
    dpo_data = prepare_dpo_data(
        raw_dataset,
        max_samples=args.num_prompts
    )
    print(f"Prepared {len(dpo_data)} preference pairs")
    
    # Create dataset and dataloader
    dpo_dataset = DPODataset(
        dpo_data,
        tokenizer,
        max_length=config.max_length
    )
    
    dataloader = DataLoader(
        dpo_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Train
    stats = trainer.train(
        dataloader=dataloader,
        num_epochs=args.epochs,
        log_every=args.log_every
    )
    
    # Save results
    dpo_dir = os.path.join(output_dir, "dpo")
    os.makedirs(dpo_dir, exist_ok=True)
    
    with open(os.path.join(dpo_dir, "training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    # Save efficiency stats
    efficiency_stats = trainer.get_efficiency_stats()
    with open(os.path.join(dpo_dir, "efficiency_stats.json"), "w") as f:
        json.dump(efficiency_stats, f, indent=2)
    
    # Save the trained model
    model_dir = os.path.join(dpo_dir, "model")
    trainer.save_model(model_dir)
    
    print(f"\nDPO training complete. Results saved to {dpo_dir}")
    
    return stats


def main():
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n" + "="*80)
    print("PART 2: POLICY OPTIMIZATION")
    print("="*80)
    
    # Load or create reward model
    if args.reward_model_path:
        print(f"\nLoading reward model from {args.reward_model_path}...")
        reward_model = RewardModel.from_pretrained(args.reward_model_path)
        reward_model = reward_model.to(device)
    else:
        print("\nCreating fresh reward model (untrained)...")
        reward_model = create_reward_model(args.model_name, device)
    
    # Load prompts
    prompts = load_prompts(args.num_prompts)
    
    # Train based on method
    ppo_stats = None
    grpo_stats = None
    dpo_stats = None
    
    # Methods that require reward model
    if args.method in ["ppo", "both", "all"]:
        ppo_stats = train_ppo(args, reward_model, prompts, output_dir)
    
    if args.method in ["grpo", "both", "all"]:
        grpo_stats = train_grpo(args, reward_model, prompts, output_dir)
    
    # DPO doesn't need reward model
    if args.method in ["dpo", "all"]:
        dpo_stats = train_dpo(args, output_dir)
    
    # Compare if multiple methods were used
    methods_used = sum([ppo_stats is not None, grpo_stats is not None, dpo_stats is not None])
    
    if methods_used >= 2:
        print("\n" + "="*60)
        methods_names = []
        if ppo_stats:
            methods_names.append("PPO")
        if grpo_stats:
            methods_names.append("GRPO")
        if dpo_stats:
            methods_names.append("DPO")
        print(f"Comparing {' vs '.join(methods_names)}")
        print("="*60)
        
        comparison = compare_algorithms(
            ppo_stats=ppo_stats,
            grpo_stats=grpo_stats,
            dpo_stats=dpo_stats
        )
        
        # Generate visualizations
        visualize_comparison(
            ppo_stats=ppo_stats,
            grpo_stats=grpo_stats,
            dpo_stats=dpo_stats,
            output_dir=os.path.join(output_dir, "comparison")
        )
        
        # Generate report
        report = generate_comparison_report(
            comparison,
            output_dir=os.path.join(output_dir, "comparison")
        )
        print(report)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    
    if ppo_stats:
        print(f"  - ppo/: PPO training results")
    if grpo_stats:
        print(f"  - grpo/: GRPO training results")
    if dpo_stats:
        print(f"  - dpo/: DPO training results")
    if methods_used >= 2:
        print(f"  - comparison/: Comparison report and visualizations")


if __name__ == "__main__":
    main()

