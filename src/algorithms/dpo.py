"""
Part 3: Direct Preference Optimization (DPO) Implementation
Implements DPO which bypasses explicit reward modeling by directly optimizing
the policy using preference data.

Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/abs/2305.18290
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, get_linear_schedule_with_warmup


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    
    # Model settings
    model_name: str = "openai-community/gpt2"
    
    # DPO hyperparameters
    beta: float = 0.1  # Temperature parameter (controls deviation from reference)
    label_smoothing: float = 0.0  # Optional label smoothing
    
    # Loss variants
    loss_type: str = "sigmoid"  # "sigmoid" (standard), "hinge", "ipo"
    
    # Training settings
    learning_rate: float = 1e-6  # DPO typically uses lower LR
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Sequence settings
    max_length: int = 512
    max_prompt_length: int = 256
    
    # Reference model
    ref_model_update_freq: int = 0  # 0 = never update reference, >0 = update every N steps

    # Learning rate schedule
    warmup_ratio: float = 0.1  # Fraction of training for warmup

    # Other
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DPODataset(Dataset):
    """
    Dataset for DPO training with preference pairs.
    Each item contains a prompt with chosen and rejected completions.
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        max_prompt_length: int = 256
    ):
        """
        Args:
            data: List of dicts with 'prompt', 'chosen', 'rejected' keys
            tokenizer: Tokenizer for encoding
            max_length: Maximum total sequence length
            max_prompt_length: Maximum prompt length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=True,
            return_tensors=None
        )
        prompt_ids = prompt_tokens["input_ids"]
        prompt_len = len(prompt_ids)
        
        # Tokenize chosen (prompt + chosen response)
        chosen_full = prompt + chosen
        chosen_tokens = self.tokenizer(
            chosen_full,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            add_special_tokens=True,
            return_tensors=None
        )
        
        # Tokenize rejected (prompt + rejected response)
        rejected_full = prompt + rejected
        rejected_tokens = self.tokenizer(
            rejected_full,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            add_special_tokens=True,
            return_tensors=None
        )
        
        return {
            "chosen_input_ids": torch.tensor(chosen_tokens["input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_tokens["attention_mask"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_tokens["input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_tokens["attention_mask"], dtype=torch.long),
            "prompt_length": torch.tensor(prompt_len, dtype=torch.long),
        }


class DPOPolicy(nn.Module):
    """
    Policy model for DPO.
    Simple wrapper around causal LM - no value head needed.
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """Forward pass returning logits."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        average_log_prob: bool = False
    ) -> torch.Tensor:
        """
        Compute log probabilities for the given sequence.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask
            average_log_prob: If True, return average log prob per sequence
            
        Returns:
            Log probabilities [batch, seq_len-1] or [batch] if averaged
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab]
        labels = input_ids[:, 1:]  # Shift right for next token prediction
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch, seq_len-1]
        
        if average_log_prob:
            # Compute average log prob per sequence (masked)
            if attention_mask is not None:
                mask = attention_mask[:, 1:]  # Align with token_log_probs
                token_log_probs = token_log_probs * mask
                seq_log_probs = token_log_probs.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                seq_log_probs = token_log_probs.mean(dim=1)
            return seq_log_probs
        
        return token_log_probs
    
    def generate(self, *args, **kwargs):
        """Generate text using the policy."""
        return self.model.generate(*args, **kwargs)


class DPOLoss:
    """
    DPO Loss computation.
    
    The key insight of DPO is that the optimal policy under the RLHF objective
    can be expressed in closed form, leading to a supervised learning loss:
    
    L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
    
    where:
    - y_w = chosen (winning) response
    - y_l = rejected (losing) response  
    - β = temperature controlling deviation from reference
    - π = policy model
    - π_ref = reference model (frozen)
    """
    
    def __init__(self, config: DPOConfig):
        self.config = config
        self.beta = config.beta
        self.label_smoothing = config.label_smoothing
        self.loss_type = config.loss_type
    
    def compute_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log probs of chosen under policy [batch]
            policy_rejected_logps: Log probs of rejected under policy [batch]
            ref_chosen_logps: Log probs of chosen under reference [batch]
            ref_rejected_logps: Log probs of rejected under reference [batch]
            
        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        # Compute log ratios
        # π(y|x) / π_ref(y|x) = exp(log π(y|x) - log π_ref(y|x))
        chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
        rejected_log_ratios = policy_rejected_logps - ref_rejected_logps
        
        # Compute preference logits: β * (log π_w/π_ref_w - log π_l/π_ref_l)
        logits = self.beta * (chosen_log_ratios - rejected_log_ratios)
        
        # Compute loss based on type
        if self.loss_type == "sigmoid":
            # Standard DPO loss: -log σ(β * ...)
            if self.label_smoothing > 0:
                # Soft labels
                losses = (
                    -F.logsigmoid(logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-logits) * self.label_smoothing
                )
            else:
                losses = -F.logsigmoid(logits)
                
        elif self.loss_type == "hinge":
            # Hinge loss variant
            losses = F.relu(1 - logits)
            
        elif self.loss_type == "ipo":
            # IPO (Identity Preference Optimization) loss
            # L = (logits - 1/(2β))^2
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        loss = losses.mean()
        
        # Compute metrics
        with torch.no_grad():
            # Accuracy: how often is chosen preferred over rejected?
            accuracy = (logits > 0).float().mean()
            
            # Reward margins (implicit rewards)
            chosen_rewards = self.beta * chosen_log_ratios
            rejected_rewards = self.beta * rejected_log_ratios
            reward_margin = (chosen_rewards - rejected_rewards).mean()
            
            # Log ratio statistics
            chosen_log_ratio_mean = chosen_log_ratios.mean()
            rejected_log_ratio_mean = rejected_log_ratios.mean()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "chosen_log_ratio": chosen_log_ratio_mean.item(),
            "rejected_log_ratio": rejected_log_ratio_mean.item(),
            "logits_mean": logits.mean().item(),
            "logits_std": logits.std().item(),
        }
        
        return loss, metrics


class DPOTrainer:
    """
    DPO Trainer for direct preference optimization.
    
    Key advantages over PPO/GRPO:
    - No reward model needed
    - No RL optimization (direct supervised learning)
    - More stable training
    - Simpler implementation
    """
    
    def __init__(
        self,
        config: DPOConfig,
        policy_model: DPOPolicy,
        ref_model: DPOPolicy,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Args:
            config: DPO configuration
            policy_model: Policy to be trained
            ref_model: Reference policy (frozen)
            tokenizer: Tokenizer
        """
        self.config = config
        self.policy = policy_model
        self.ref_policy = ref_model
        self.tokenizer = tokenizer
        self.device = config.device
        
        # Freeze reference model
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        self.ref_policy.eval()
        
        # Loss computation
        self.dpo_loss = DPOLoss(config)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler will be initialized in train() when we know total steps
        self.scheduler = None

        # Training metrics
        self.training_stats = []
        self.global_step = 0
        
        # Efficiency tracking
        self.step_times = []
        self.memory_usage = []
    
    def compute_logps_for_batch(
        self,
        model: DPOPolicy,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
        average: bool = True
    ) -> torch.Tensor:
        """
        Compute log probabilities for response tokens only (excluding prompt).
        
        Args:
            model: Policy model
            input_ids: Full sequence (prompt + response) [batch, seq_len]
            attention_mask: Attention mask
            prompt_lengths: Length of prompt for each example [batch]
            average: Whether to average log probs
            
        Returns:
            Log probabilities for response tokens
        """
        # Get full sequence log probs
        token_log_probs = model.get_log_probs(input_ids, attention_mask, average_log_prob=False)
        
        # Create mask for response tokens only (exclude prompt)
        batch_size, seq_len = token_log_probs.shape
        response_mask = torch.zeros_like(token_log_probs)
        
        for i in range(batch_size):
            prompt_len = prompt_lengths[i].item()
            # Response starts after prompt (accounting for shift in log_probs)
            response_start = max(0, prompt_len - 1)
            # Calculate how many response tokens we have in log_probs
            response_len = min(seq_len - response_start, attention_mask[i, prompt_len:].sum().item())
            if response_len > 0:
                response_mask[i, response_start:response_start + int(response_len)] = 1.0
        
        # Mask out prompt tokens
        masked_log_probs = token_log_probs * response_mask
        
        if average:
            # Average over response tokens
            seq_log_probs = masked_log_probs.sum(dim=1) / (response_mask.sum(dim=1) + 1e-8)
            return seq_log_probs
        
        return masked_log_probs
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one DPO training step.
        
        Args:
            batch: Batch containing chosen and rejected examples
            
        Returns:
            metrics: Training metrics
        """
        start_time = time.time()
        
        self.policy.train()
        
        # Track memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Move batch to device
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)
        prompt_lengths = batch["prompt_length"].to(self.device)
        
        # Compute log probs under policy
        policy_chosen_logps = self.compute_logps_for_batch(
            self.policy, chosen_ids, chosen_mask, prompt_lengths
        )
        policy_rejected_logps = self.compute_logps_for_batch(
            self.policy, rejected_ids, rejected_mask, prompt_lengths
        )
        
        # Compute log probs under reference (no grad)
        with torch.no_grad():
            ref_chosen_logps = self.compute_logps_for_batch(
                self.ref_policy, chosen_ids, chosen_mask, prompt_lengths
            )
            ref_rejected_logps = self.compute_logps_for_batch(
                self.ref_policy, rejected_ids, rejected_mask, prompt_lengths
            )
        
        # Compute DPO loss
        loss, metrics = self.dpo_loss.compute_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.max_grad_norm
        )
        metrics["grad_norm"] = grad_norm.item()

        self.optimizer.step()

        # Step scheduler if initialized
        if self.scheduler is not None:
            self.scheduler.step()
            metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
        
        # Track timing and memory
        step_time = time.time() - start_time
        self.step_times.append(step_time)
        metrics["step_time"] = step_time
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            self.memory_usage.append(peak_memory)
            metrics["peak_memory_gb"] = peak_memory
        
        self.global_step += 1
        self.training_stats.append(metrics)
        
        return metrics
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        log_every: int = 10
    ) -> List[Dict[str, float]]:
        """
        Full DPO training loop.

        Args:
            dataloader: DataLoader with preference pairs
            num_epochs: Number of training epochs
            log_every: Logging frequency

        Returns:
            training_stats: List of training metrics
        """
        total_steps = len(dataloader) * num_epochs

        # Initialize learning rate scheduler with warmup
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print(f"LR scheduler: {warmup_steps} warmup steps, {total_steps} total steps")

        pbar = tqdm(total=total_steps, desc="DPO Training")
        
        for epoch in range(num_epochs):
            for step, batch in enumerate(dataloader):
                metrics = self.train_step(batch)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['accuracy']:.4f}",
                    "margin": f"{metrics['reward_margin']:.4f}"
                })
                pbar.update(1)
                
                # Logging
                global_step = epoch * len(dataloader) + step
                if (global_step + 1) % log_every == 0:
                    print(f"\nStep {global_step + 1}: "
                          f"loss={metrics['loss']:.4f}, "
                          f"acc={metrics['accuracy']:.4f}, "
                          f"margin={metrics['reward_margin']:.4f}, "
                          f"chosen_r={metrics['chosen_rewards']:.4f}, "
                          f"rejected_r={metrics['rejected_rewards']:.4f}")
        
        pbar.close()
        
        # Print efficiency summary
        print("\n" + "="*60)
        print("DPO Training Efficiency Summary")
        print("="*60)
        print(f"Average step time: {np.mean(self.step_times):.3f}s")
        if self.memory_usage:
            print(f"Peak memory usage: {max(self.memory_usage):.2f} GB")
        print(f"Total steps: {len(self.training_stats)}")
        
        return self.training_stats
    
    def get_efficiency_stats(self) -> Dict[str, float]:
        """Get computational efficiency statistics."""
        return {
            "avg_step_time": np.mean(self.step_times) if self.step_times else 0,
            "total_time": sum(self.step_times),
            "peak_memory_gb": max(self.memory_usage) if self.memory_usage else 0,
            "avg_memory_gb": np.mean(self.memory_usage) if self.memory_usage else 0,
        }
    
    def save_model(self, output_dir: str):
        """
        Save the trained policy model.
        
        Args:
            output_dir: Directory to save the model
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the policy model
        self.policy.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"DPO policy model saved to {output_dir}")
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Evaluation DataLoader
            
        Returns:
            Evaluation metrics
        """
        self.policy.eval()
        
        all_metrics = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            chosen_ids = batch["chosen_input_ids"].to(self.device)
            chosen_mask = batch["chosen_attention_mask"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_mask = batch["rejected_attention_mask"].to(self.device)
            prompt_lengths = batch["prompt_length"].to(self.device)
            
            # Compute log probs
            policy_chosen_logps = self.compute_logps_for_batch(
                self.policy, chosen_ids, chosen_mask, prompt_lengths
            )
            policy_rejected_logps = self.compute_logps_for_batch(
                self.policy, rejected_ids, rejected_mask, prompt_lengths
            )
            ref_chosen_logps = self.compute_logps_for_batch(
                self.ref_policy, chosen_ids, chosen_mask, prompt_lengths
            )
            ref_rejected_logps = self.compute_logps_for_batch(
                self.ref_policy, rejected_ids, rejected_mask, prompt_lengths
            )
            
            _, metrics = self.dpo_loss.compute_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps
            )
            all_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics


def prepare_dpo_data(
    dataset,
    max_samples: int = None
) -> List[Dict]:
    """
    Prepare DPO training data from HH-RLHF dataset.

    Args:
        dataset: HuggingFace dataset with 'chosen' and 'rejected'
        max_samples: Maximum samples to use

    Returns:
        List of dicts with 'prompt', 'chosen', 'rejected'
    """
    data = []
    
    for i, example in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        
        chosen_text = example["chosen"]
        rejected_text = example["rejected"]
        
        # Extract prompt and response
        # Format: Human: ... Assistant: [response]
        chosen_split = chosen_text.rfind("\n\nAssistant:")
        rejected_split = rejected_text.rfind("\n\nAssistant:")
        
        if chosen_split == -1 or rejected_split == -1:
            continue
        
        # Prompt should be the same for both
        prompt = chosen_text[:chosen_split + len("\n\nAssistant:")]
        chosen_response = chosen_text[chosen_split + len("\n\nAssistant:"):]
        rejected_response = rejected_text[rejected_split + len("\n\nAssistant:"):]
        
        data.append({
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response
        })
    
    return data


def create_dpo_trainer(
    config: DPOConfig,
    model_name: Optional[str] = None
) -> DPOTrainer:
    """
    Factory function to create a DPO trainer.
    
    Args:
        config: DPO configuration
        model_name: Model name (overrides config if provided)
        
    Returns:
        Initialized DPOTrainer
    """
    if model_name:
        config.model_name = model_name
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models
    
    # Create policy model
    print(f"Loading DPO policy model from {config.model_name}...")
    policy = DPOPolicy(config.model_name, config.device)
    
    # Create reference model (frozen copy)
    print("Creating reference model...")
    ref_policy = DPOPolicy(config.model_name, config.device)
    
    # Create trainer
    trainer = DPOTrainer(
        config=config,
        policy_model=policy,
        ref_model=ref_policy,
        tokenizer=tokenizer
    )
    
    return trainer, tokenizer


if __name__ == "__main__":
    print("DPO module loaded successfully.")
    print(f"DPOConfig defaults: beta={DPOConfig.beta}, loss_type={DPOConfig.loss_type}")
    
    # Quick test
    config = DPOConfig()
    loss_fn = DPOLoss(config)
    
    # Test loss computation
    batch_size = 4
    policy_chosen = torch.randn(batch_size)
    policy_rejected = torch.randn(batch_size)
    ref_chosen = torch.randn(batch_size)
    ref_rejected = torch.randn(batch_size)
    
    loss, metrics = loss_fn.compute_loss(
        policy_chosen, policy_rejected,
        ref_chosen, ref_rejected
    )
    
    print(f"\nTest loss: {loss.item():.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test margin: {metrics['reward_margin']:.4f}")

