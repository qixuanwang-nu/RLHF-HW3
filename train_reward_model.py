"""
Part 1: Complete Training Pipeline for Reward Model
Implements preference data collection, reward modeling, and evaluation.

Usage:
    python train_reward_model.py --epochs 3 --batch_size 8 --lr 1e-5
"""

import argparse
import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.preprocessing import DataPreprocessor, create_dataloaders
from src.models.reward_model import RewardModel, RewardModelTrainer, create_reward_model
from src.evaluation.error_analysis import ErrorAnalyzer, run_error_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Train a reward model on HH-RLHF")
    
    # Data arguments
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--truncation", type=str, default="left",
                        choices=["left", "right"],
                        help="Truncation strategy")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2",
                        help="Pretrained model to use as backbone")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    
    # Evaluation arguments
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--error_analysis_samples", type=int, default=1000,
                        help="Number of samples for error analysis")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--save_model", action="store_true",
                        help="Save the trained model")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Use a subset of data for quick testing")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_curves(history: dict, output_dir: str):
    """Plot training curves and save to file."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reward Model Training Curves", fontsize=14, fontweight='bold')
    
    # Loss curves
    ax1 = axes[0, 0]
    if history["train_losses"]:
        ax1.plot(history["train_losses"], label="Train Loss", color='#3498db', alpha=0.7)
    if history["val_losses"]:
        val_x = np.linspace(0, len(history["train_losses"]), len(history["val_losses"]))
        ax1.plot(val_x, history["val_losses"], label="Val Loss", color='#e74c3c', 
                 marker='o', markersize=4)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2 = axes[0, 1]
    if history["train_accuracies"]:
        ax2.plot(history["train_accuracies"], label="Train Accuracy", color='#3498db', alpha=0.7)
    if history["val_accuracies"]:
        val_x = np.linspace(0, len(history["train_accuracies"]), len(history["val_accuracies"]))
        ax2.plot(val_x, history["val_accuracies"], label="Val Accuracy", color='#e74c3c',
                 marker='o', markersize=4)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 1.0)
    
    # Gradient norms
    ax3 = axes[1, 0]
    if history["gradient_norms"]:
        ax3.plot(history["gradient_norms"], color='#9b59b6', alpha=0.7)
        ax3.axhline(y=np.mean(history["gradient_norms"]), color='red', linestyle='--',
                   label=f'Mean: {np.mean(history["gradient_norms"]):.2f}')
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Gradient Norm")
    ax3.set_title("Gradient Norms During Training")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning rate
    ax4 = axes[1, 1]
    if history["learning_rates"]:
        ax4.plot(history["learning_rates"], color='#2ecc71')
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Learning Rate")
    ax4.set_title("Learning Rate Schedule")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {output_dir}/training_curves.png")


def train_epoch(
    trainer: RewardModelTrainer,
    train_loader,
    epoch: int,
    eval_steps: int,
    val_loader=None
) -> dict:
    """Train for one epoch with periodic evaluation."""
    epoch_losses = []
    epoch_accuracies = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(pbar):
        metrics = trainer.train_step(batch, step)
        
        epoch_losses.append(metrics["loss"])
        epoch_accuracies.append(metrics["accuracy"])
        trainer.train_losses.append(metrics["loss"])
        trainer.train_accuracies.append(metrics["accuracy"])
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{metrics['loss']:.4f}",
            "acc": f"{metrics['accuracy']:.4f}",
            "margin": f"{metrics['reward_margin']:.4f}"
        })
        
        # Periodic evaluation
        global_step = epoch * len(train_loader) + step
        if val_loader and (global_step + 1) % eval_steps == 0:
            val_metrics = trainer.evaluate(val_loader)
            trainer.val_losses.append(val_metrics["loss"])
            trainer.val_accuracies.append(val_metrics["accuracy"])
            
            print(f"\n  [Step {global_step+1}] Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Margin: {val_metrics['reward_margin']:.4f}")
    
    return {
        "loss": np.mean(epoch_losses),
        "accuracy": np.mean(epoch_accuracies)
    }


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
    os.makedirs(os.path.join(output_dir, "evaluation"), exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n" + "="*80)
    print("PART 1: REWARD MODEL TRAINING")
    print("="*80)
    
    # =========================================================================
    # Part 1.1: Dataset Preparation
    # =========================================================================
    print("\n" + "-"*80)
    print("PART 1.1: Dataset Preparation")
    print("-"*80)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        truncation_strategy=args.truncation,
        seed=args.seed
    )
    
    # Prepare data
    prepared_data, metadata = preprocessor.prepare_data(verbose=True)
    
    # Optionally use subset for testing
    if args.subset_size:
        print(f"\nUsing subset of {args.subset_size} samples for quick testing...")
        train_indices = np.random.choice(
            len(prepared_data["train"]), 
            min(args.subset_size, len(prepared_data["train"])),
            replace=False
        )
        val_indices = np.random.choice(
            len(prepared_data["validation"]),
            min(args.subset_size // 10, len(prepared_data["validation"])),
            replace=False
        )
        prepared_data["train"] = prepared_data["train"].select(train_indices)
        prepared_data["validation"] = prepared_data["validation"].select(val_indices)
        print(f"Training samples: {len(prepared_data['train'])}")
        print(f"Validation samples: {len(prepared_data['validation'])}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        prepared_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # =========================================================================
    # Part 1.2A: Reward Model Training
    # =========================================================================
    print("\n" + "-"*80)
    print("PART 1.2A: Reward Model Training")
    print("-"*80)
    
    # Create model
    model = create_reward_model(args.model_name, device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\nTraining configuration:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Create trainer
    trainer = RewardModelTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm
    )
    
    # Initial evaluation
    print("\nInitial evaluation...")
    initial_metrics = trainer.evaluate(val_loader)
    print(f"  Initial Val Loss: {initial_metrics['loss']:.4f}")
    print(f"  Initial Val Accuracy: {initial_metrics['accuracy']:.4f}")
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    best_val_accuracy = 0.0
    
    for epoch in range(args.epochs):
        epoch_metrics = train_epoch(
            trainer, train_loader, epoch,
            eval_steps=args.eval_steps,
            val_loader=val_loader
        )
        
        # End of epoch evaluation
        val_metrics = trainer.evaluate(val_loader)
        trainer.val_losses.append(val_metrics["loss"])
        trainer.val_accuracies.append(val_metrics["accuracy"])
        
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {epoch_metrics['loss']:.4f}, Train Acc: {epoch_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Reward Margin: {val_metrics['reward_margin']:.4f}")
        
        # Save best model
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            if args.save_model:
                model_path = os.path.join(output_dir, "best_model")
                model.save_pretrained(model_path)
                preprocessor.tokenizer.save_pretrained(model_path)
                print(f"  New best model saved to {model_path}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    # Plot training curves
    history = trainer.get_training_history()
    plot_training_curves(history, output_dir)
    
    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    
    # =========================================================================
    # Part 1.2B: Evaluation and Error Analysis
    # =========================================================================
    print("\n" + "-"*80)
    print("PART 1.2B: Evaluation and Error Analysis")
    print("-"*80)
    
    # Final evaluation
    print("\nFinal evaluation on validation set...")
    final_metrics = trainer.evaluate(val_loader)
    
    print(f"\n{'FINAL VALIDATION METRICS':^60}")
    print("-"*60)
    print(f"{'Metric':<30} {'Value':<20}")
    print("-"*60)
    print(f"{'Loss':<30} {final_metrics['loss']:.4f}")
    print(f"{'Accuracy':<30} {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"{'Chosen Reward (mean)':<30} {final_metrics['chosen_reward_mean']:.4f}")
    print(f"{'Rejected Reward (mean)':<30} {final_metrics['rejected_reward_mean']:.4f}")
    print(f"{'Chosen Reward (std)':<30} {final_metrics['chosen_reward_std']:.4f}")
    print(f"{'Rejected Reward (std)':<30} {final_metrics['rejected_reward_std']:.4f}")
    print(f"{'Reward Margin':<30} {final_metrics['reward_margin']:.4f}")
    print("-"*60)
    
    # Load raw validation data for error analysis
    print("\nRunning error analysis...")
    raw_dataset = load_dataset("Anthropic/hh-rlhf")
    
    error_analysis = run_error_analysis(
        model=model,
        tokenizer=preprocessor.tokenizer,
        dataset=raw_dataset["test"],
        num_samples=min(args.error_analysis_samples, len(raw_dataset["test"])),
        num_examples_in_report=20,
        output_dir=os.path.join(output_dir, "evaluation")
    )
    
    # Save final results
    final_results = {
        "config": vars(args),
        "training_time_minutes": training_time / 60,
        "best_val_accuracy": float(best_val_accuracy),
        "final_metrics": {k: float(v) for k, v in final_metrics.items()},
        "error_analysis_summary": {
            "accuracy": error_analysis["overall_accuracy"],
            "num_errors": error_analysis["num_errors"],
            "total_samples": error_analysis["total_samples"],
        }
    }
    
    with open(os.path.join(output_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  - config.json: Training configuration")
    print(f"  - training_history.json: Loss and accuracy curves")
    print(f"  - training_curves.png: Visualization of training")
    print(f"  - final_results.json: Final metrics and summary")
    print(f"  - evaluation/: Error analysis report and visualizations")
    if args.save_model:
        print(f"  - best_model/: Saved model checkpoint")
    
    return model, final_results


if __name__ == "__main__":
    main()

