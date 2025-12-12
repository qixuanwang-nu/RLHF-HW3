"""
Part 2.2: GRPO (Group Relative Policy Optimization) Implementation
Implements simplified policy gradient with group-based advantage estimation.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    
    # Model settings
    model_name: str = "openai-community/gpt2"
    
    # GRPO hyperparameters
    group_size: int = 4  # Number of responses per prompt (4-8 recommended)
    kl_coef: float = 0.1  # KL penalty coefficient
    entropy_coef: float = 0.05  # Entropy bonus coefficient
    min_entropy: float = 2.0  # Minimum entropy threshold (GPT-2 text entropy ~3-5)
    
    # Advantage normalization
    normalize_advantages: bool = True
    advantage_clip: float = 10.0  # Clip extreme advantages
    
    # Training settings
    learning_rate: float = 1e-5
    batch_size: int = 2  # Prompts per batch (effective responses = batch_size * group_size)
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Generation settings
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    
    # Other
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GRPOPolicy(nn.Module):
    """
    Policy model for GRPO.
    Simpler than PPO - no value head needed.
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
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
        pad_token_id: int = None,
        **kwargs
    ):
        """Generate multiple responses."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=pad_token_id,
            **kwargs
        )
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute log probabilities for the given sequence.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask
            
        Returns:
            Log probabilities [batch, seq_len-1]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab]
        labels = input_ids[:, 1:]  # Shift right
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs
    
    def get_sequence_log_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sequence-level log probability (sum over response tokens).
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            response_mask: Mask for response tokens only
            
        Returns:
            Sequence log probabilities [batch]
        """
        token_log_probs = self.get_log_probs(input_ids, attention_mask)
        
        if response_mask is not None:
            # Align mask to log_probs (shifted by 1)
            response_mask = response_mask[:, 1:]
            if response_mask.size(1) > token_log_probs.size(1):
                response_mask = response_mask[:, :token_log_probs.size(1)]
            elif response_mask.size(1) < token_log_probs.size(1):
                token_log_probs = token_log_probs[:, :response_mask.size(1)]
            
            # Sum only response tokens
            seq_log_probs = (token_log_probs * response_mask).sum(dim=1)
        else:
            seq_log_probs = token_log_probs.sum(dim=1)
        
        return seq_log_probs


class GRPOLoss:
    """
    GRPO Loss computation with group-based advantage estimation.
    
    Key difference from PPO:
    - Advantages computed relative to group mean (no value function needed)
    - Simpler policy gradient without clipping
    """
    
    def __init__(self, config: GRPOConfig):
        self.config = config
        self.kl_coef = config.kl_coef
        self.entropy_coef = config.entropy_coef
        self.min_entropy = getattr(config, 'min_entropy', 1.0)
        self.normalize_advantages = config.normalize_advantages
        self.advantage_clip = config.advantage_clip
    
    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute advantages relative to group mean.
        
        For each prompt, we have `group_size` responses.
        Advantage = reward - mean(rewards in group)
        
        Args:
            rewards: Rewards for all responses [batch_size * group_size]
            group_size: Number of responses per prompt
            
        Returns:
            advantages: Advantage estimates [batch_size * group_size]
            stats: Statistics about the advantages
        """
        batch_size = rewards.size(0) // group_size
        
        # Reshape to [batch_size, group_size]
        rewards_grouped = rewards.view(batch_size, group_size)
        
        # Compute group mean (baseline)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        
        # Compute advantages relative to group mean
        advantages_grouped = rewards_grouped - group_mean
        
        # Track within-group statistics before normalization
        within_group_std = rewards_grouped.std(dim=1).mean().item()
        
        # Normalize by GLOBAL std across all advantages (not per-group)
        # This preserves relative magnitude differences between groups
        if self.normalize_advantages:
            global_std = advantages_grouped.std() + 1e-8
            advantages_grouped = advantages_grouped / global_std
        
        # Clip extreme advantages
        advantages_grouped = torch.clamp(
            advantages_grouped,
            -self.advantage_clip,
            self.advantage_clip
        )
        
        # Flatten back
        advantages = advantages_grouped.view(-1)
        
        stats = {
            "within_group_reward_std": within_group_std,
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "advantage_max": advantages.max().item(),
            "advantage_min": advantages.min().item(),
        }
        
        return advantages, stats
    
    def compute_policy_gradient_loss(
        self,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute simplified policy gradient loss.
        
        L = -E[log π(a|s) * A(s,a)]
        
        Unlike PPO, no clipping - relies on KL penalty for stability.
        
        Args:
            log_probs: Log probabilities (per-token or sequence-level)
            advantages: Advantage estimates
            mask: Optional mask
            
        Returns:
            loss: Policy gradient loss
            metrics: Dictionary of metrics
        """
        # For sequence-level advantages, expand to match token-level log_probs
        if advantages.dim() == 1 and log_probs.dim() == 2:
            advantages = advantages.unsqueeze(1).expand_as(log_probs)
        
        # Policy gradient
        pg_loss = -log_probs * advantages.detach()
        
        if mask is not None:
            pg_loss = pg_loss * mask
            loss = pg_loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = pg_loss.mean()
        
        metrics = {
            "policy_loss": loss.item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
        }
        
        return loss, metrics
    
    def compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute KL divergence penalty from reference policy.
        
        Uses a stable, always-nonnegative proxy computed from sampled actions:
        
            kl_proxy(x) = exp(x) - 1 - x,  where x = log π(a|s) - log π_ref(a|s)
        
        Since exp(x) >= 1 + x, this proxy is >= 0 and has well-behaved gradients.
        We clamp x to avoid exp overflow.
        """
        log_ratio = log_probs - ref_log_probs
        log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
        
        # Always-nonnegative proxy: exp(x) - 1 - x
        kl_approx = torch.expm1(log_ratio) - log_ratio
        
        if mask is not None:
            kl_approx = kl_approx * mask
            kl_value = kl_approx.sum() / (mask.sum() + 1e-8)
        else:
            kl_value = kl_approx.mean()
        
        # Clamp to prevent rare explosions dominating optimization
        kl_value = torch.clamp(kl_value, 0.0, 50.0)
        kl_penalty = self.kl_coef * kl_value
        
        return kl_penalty, float(kl_value.item())
    
    def compute_entropy_bonus(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute entropy bonus for exploration with minimum entropy threshold.
        
        Uses a two-part strategy to prevent entropy collapse:
        1. Standard entropy maximization: -entropy_coef * entropy
        2. Minimum entropy penalty: heavily penalizes entropy below threshold
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Entropy per position: H = -sum(p * log(p))
        entropy = -(probs * log_probs).sum(dim=-1)
        
        if mask is not None:
            entropy = entropy * mask
            entropy_value = entropy.sum() / (mask.sum() + 1e-8)
        else:
            entropy_value = entropy.mean()
        
        # Entropy maximization (always active)
        entropy_bonus = -self.entropy_coef * entropy_value
        
        # IMPORTANT: collapse penalty must be differentiable (do NOT use .item())
        deficit = F.relu(torch.tensor(self.min_entropy, device=entropy_value.device, dtype=entropy_value.dtype) - entropy_value)
        collapse_penalty = 10.0 * (deficit ** 2)
        entropy_loss = entropy_bonus + collapse_penalty
        
        return entropy_loss, float(entropy_value.detach().item())
    
    def compute_total_loss(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        logits: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total GRPO loss.
        
        L_total = L_PG + β * KL(π || π_ref) - c_ent * H(π)
        
        Args:
            log_probs: Current policy log probs
            ref_log_probs: Reference policy log probs
            logits: Current policy logits (for entropy)
            advantages: Group-relative advantages
            mask: Token mask
            
        Returns:
            total_loss: Combined loss
            metrics: Dictionary of all metrics
        """
        # Policy gradient loss
        pg_loss, pg_metrics = self.compute_policy_gradient_loss(
            log_probs, advantages, mask
        )
        
        # KL penalty
        kl_penalty, kl_value = self.compute_kl_penalty(
            log_probs, ref_log_probs, mask
        )
        
        # Entropy loss (negative = encourages higher entropy)
        entropy_loss, entropy_value = self.compute_entropy_bonus(logits, mask)
        
        # Total loss: policy gradient + KL penalty + entropy regularization
        total_loss = pg_loss + kl_penalty + entropy_loss
        
        metrics = {
            **pg_metrics,
            "kl_penalty": kl_penalty.item(),
            "kl_divergence": kl_value,
            "entropy_bonus": -entropy_loss.item(),  # Positive value = entropy being encouraged
            "entropy": entropy_value,
            "total_loss": total_loss.item(),
        }
        
        return total_loss, metrics


class GRPOTrainer:
    """
    GRPO Trainer implementing group-based policy optimization.
    
    Key differences from PPO:
    - Samples multiple responses per prompt
    - Computes advantages relative to group mean
    - No value function needed
    - Simpler but potentially less stable
    """
    
    def __init__(
        self,
        config: GRPOConfig,
        policy_model: GRPOPolicy,
        ref_model: GRPOPolicy,
        reward_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Args:
            config: GRPO configuration
            policy_model: Policy to be trained
            ref_model: Reference policy (frozen)
            reward_model: Reward model from Part 1
            tokenizer: Tokenizer
        """
        self.config = config
        self.policy = policy_model
        self.ref_policy = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = config.device
        
        # Freeze reference and reward models
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        self.ref_policy.eval()
        
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()
        
        # Loss computation
        self.grpo_loss = GRPOLoss(config)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # Training metrics
        self.training_stats = []
        self.global_step = 0
        
        # Efficiency tracking
        self.generation_times = []
        self.update_times = []
        self.memory_usage = []
    
    @torch.no_grad()
    def generate_group_responses(
        self,
        prompts: List[str]
    ) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor, List[int]]:
        """
        Generate multiple responses per prompt (group sampling).
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            responses: List of lists of response strings [batch_size, group_size]
            all_input_ids: Token IDs for all responses [batch_size * group_size, seq_len]
            all_attention_masks: Attention masks
            prompt_lengths: Length of each prompt
        """
        self.policy.eval()
        
        group_size = self.config.group_size
        
        # Tokenize prompts
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512 - self.config.max_new_tokens,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        
        all_responses = []
        all_input_ids = []
        all_attention_masks = []
        
        # Generate group_size responses per prompt
        for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
            # Expand prompt for batch generation
            ids_expanded = ids.unsqueeze(0)
            mask_expanded = mask.unsqueeze(0)
            
            # Generate multiple responses
            output_ids = self.policy.generate(
                input_ids=ids_expanded,
                attention_mask=mask_expanded,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_return_sequences=group_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            # Decode responses
            prompt_len = prompt_lengths[i]
            responses = []
            for out in output_ids:
                response_tokens = out[prompt_len:]
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                responses.append(response)
            
            all_responses.append(responses)
            all_input_ids.append(output_ids)
            
            # Create attention masks
            output_mask = (output_ids != self.tokenizer.pad_token_id).long()
            all_attention_masks.append(output_mask)
        
        # Stack all responses
        # Pad to same length
        max_len = max(ids.size(1) for ids in all_input_ids)
        
        padded_ids = []
        padded_masks = []
        
        for ids, masks in zip(all_input_ids, all_attention_masks):
            if ids.size(1) < max_len:
                padding = torch.full(
                    (ids.size(0), max_len - ids.size(1)),
                    self.tokenizer.pad_token_id,
                    dtype=ids.dtype,
                    device=ids.device
                )
                ids = torch.cat([ids, padding], dim=1)
                
                mask_padding = torch.zeros(
                    (masks.size(0), max_len - masks.size(1)),
                    dtype=masks.dtype,
                    device=masks.device
                )
                masks = torch.cat([masks, mask_padding], dim=1)
            
            padded_ids.append(ids)
            padded_masks.append(masks)
        
        all_input_ids = torch.cat(padded_ids, dim=0)
        all_attention_masks = torch.cat(padded_masks, dim=0)
        
        return all_responses, all_input_ids, all_attention_masks, prompt_lengths
    
    @torch.no_grad()
    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards for all responses.
        """
        rewards, _ = self.reward_model.get_reward(input_ids, attention_mask)
        return rewards.squeeze(-1)
    
    def grpo_update(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        advantages: torch.Tensor,
        prompt_lengths: List[int]
    ) -> Dict[str, float]:
        """
        Perform GRPO update.
        
        Args:
            input_ids: Token IDs [batch_size * group_size, seq_len]
            attention_mask: Attention mask
            advantages: Group-relative advantages [batch_size * group_size]
            prompt_lengths: Prompt lengths for creating response masks
            
        Returns:
            metrics: Training metrics
        """
        self.policy.train()
        
        # Create response masks
        batch_size = len(prompt_lengths)
        group_size = self.config.group_size
        seq_len = input_ids.size(1)
        
        response_mask = torch.zeros_like(attention_mask, dtype=torch.float)
        for i, prompt_len in enumerate(prompt_lengths):
            for j in range(group_size):
                idx = i * group_size + j
                response_mask[idx, prompt_len:] = attention_mask[idx, prompt_len:].float()
        
        # Forward pass through current policy
        outputs = self.policy(input_ids, attention_mask)
        logits = outputs.logits
        
        # Compute log probs
        log_probs = self.policy.get_log_probs(input_ids, attention_mask)
        
        # Get reference log probs
        with torch.no_grad():
            ref_log_probs = self.ref_policy.get_log_probs(input_ids, attention_mask)
        
        # Align masks and tensors
        response_mask_aligned = response_mask[:, 1:]
        logits_aligned = logits[:, :-1, :]
        
        min_len = min(log_probs.size(1), response_mask_aligned.size(1), 
                     ref_log_probs.size(1), logits_aligned.size(1))
        
        log_probs = log_probs[:, :min_len]
        ref_log_probs = ref_log_probs[:, :min_len]
        response_mask_aligned = response_mask_aligned[:, :min_len]
        logits_aligned = logits_aligned[:, :min_len, :]
        
        # Compute total loss
        total_loss, metrics = self.grpo_loss.compute_total_loss(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            logits=logits_aligned,
            advantages=advantages,
            mask=response_mask_aligned
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.config.max_grad_norm
        )
        metrics["grad_norm"] = grad_norm.item()
        
        self.optimizer.step()
        
        return metrics
    
    def train_step(
        self,
        prompts: List[str]
    ) -> Dict[str, float]:
        """
        Perform one GRPO training step.
        
        Args:
            prompts: Batch of prompts
            
        Returns:
            metrics: Training metrics
        """
        start_time = time.time()
        
        # Track memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # 1. Generate group responses
        gen_start = time.time()
        responses, input_ids, attention_mask, prompt_lengths = self.generate_group_responses(prompts)
        gen_time = time.time() - gen_start
        self.generation_times.append(gen_time)
        
        # 2. Compute rewards
        rewards = self.compute_rewards(input_ids, attention_mask)
        
        # Store raw reward statistics
        raw_reward_mean = rewards.mean().item()
        raw_reward_std = rewards.std().item()
        raw_reward_max = rewards.max().item()
        raw_reward_min = rewards.min().item()
        
        # 3. Normalize rewards using running statistics for stability
        if not hasattr(self, 'reward_running_mean'):
            self.reward_running_mean = raw_reward_mean
            self.reward_running_std = raw_reward_std + 1e-8
        else:
            # Exponential moving average
            alpha = 0.1
            self.reward_running_mean = (1 - alpha) * self.reward_running_mean + alpha * raw_reward_mean
            self.reward_running_std = (1 - alpha) * self.reward_running_std + alpha * (raw_reward_std + 1e-8)
        
        # Normalize rewards before computing advantages
        rewards_normalized = (rewards - self.reward_running_mean) / self.reward_running_std
        
        # 4. Compute group-relative advantages using normalized rewards
        advantages, adv_stats = self.grpo_loss.compute_group_advantages(
            rewards_normalized, self.config.group_size
        )
        
        # 5. GRPO update
        update_start = time.time()
        metrics = self.grpo_update(
            input_ids, attention_mask, advantages, prompt_lengths
        )
        update_time = time.time() - update_start
        self.update_times.append(update_time)
        
        # 6. Adaptive KL coefficient adjustment
        kl = metrics.get("kl_divergence", 0)
        target_kl = 0.5  # Target KL divergence
        if kl > 2.0 * target_kl:
            self.grpo_loss.kl_coef = min(1.0, self.grpo_loss.kl_coef * 1.5)
        elif kl < 0.5 * target_kl:
            self.grpo_loss.kl_coef = max(0.01, self.grpo_loss.kl_coef * 0.8)
        
        # Track memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            self.memory_usage.append(peak_memory)
            metrics["peak_memory_gb"] = peak_memory
        
        # Add reward statistics (both raw and normalized)
        metrics["reward_mean"] = raw_reward_mean
        metrics["reward_std"] = raw_reward_std
        metrics["reward_max"] = raw_reward_max
        metrics["reward_min"] = raw_reward_min
        metrics["reward_normalized_mean"] = rewards_normalized.mean().item()
        metrics["reward_normalized_std"] = rewards_normalized.std().item()
        
        # Add advantage statistics
        metrics.update(adv_stats)
        
        # Add timing and other info
        metrics["generation_time"] = gen_time
        metrics["update_time"] = update_time
        metrics["step_time"] = time.time() - start_time
        metrics["group_size"] = self.config.group_size
        metrics["num_responses"] = len(prompts) * self.config.group_size
        metrics["kl_coef"] = self.grpo_loss.kl_coef
        
        self.global_step += 1
        self.training_stats.append(metrics)
        
        return metrics
    
    def train(
        self,
        dataloader: DataLoader,
        num_steps: int = 1000,
        log_every: int = 10
    ) -> List[Dict[str, float]]:
        """
        Full GRPO training loop.
        
        Args:
            dataloader: DataLoader yielding prompts
            num_steps: Total training steps
            log_every: Logging frequency
            
        Returns:
            training_stats: List of training metrics
        """
        pbar = tqdm(total=num_steps, desc="GRPO Training")
        
        data_iter = iter(dataloader)
        
        for step in range(num_steps):
            # Get batch of prompts
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            if isinstance(batch, dict):
                prompts = batch.get("prompt", batch.get("text", None))
                if prompts is None:
                    prompts = [self.tokenizer.decode(ids) for ids in batch["input_ids"]]
            elif isinstance(batch, (list, tuple)):
                prompts = batch[0] if isinstance(batch[0], list) else batch
            else:
                prompts = batch
            
            # Training step
            metrics = self.train_step(prompts)
            
            # Update progress bar
            pbar.set_postfix({
                "reward": f"{metrics['reward_mean']:.3f}",
                "kl": f"{metrics['kl_divergence']:.4f}",
                "loss": f"{metrics['total_loss']:.4f}"
            })
            pbar.update(1)
            
            # Logging
            if (step + 1) % log_every == 0:
                print(f"\nStep {step + 1}: "
                      f"reward={metrics['reward_mean']:.3f}±{metrics['reward_std']:.3f}, "
                      f"kl={metrics['kl_divergence']:.4f}, "
                      f"entropy={metrics.get('entropy', 0):.4f}, "
                      f"loss={metrics['total_loss']:.4f}")
        
        pbar.close()
        
        # Print efficiency summary
        print("\n" + "="*60)
        print("GRPO Training Efficiency Summary")
        print("="*60)
        print(f"Average generation time: {np.mean(self.generation_times):.3f}s")
        print(f"Average update time: {np.mean(self.update_times):.3f}s")
        if self.memory_usage:
            print(f"Peak memory usage: {max(self.memory_usage):.2f} GB")
        print(f"Responses per step: {self.config.batch_size * self.config.group_size}")
        
        return self.training_stats
    
    def get_efficiency_stats(self) -> Dict[str, float]:
        """Get computational efficiency statistics."""
        return {
            "avg_generation_time": np.mean(self.generation_times) if self.generation_times else 0,
            "avg_update_time": np.mean(self.update_times) if self.update_times else 0,
            "total_generation_time": sum(self.generation_times),
            "total_update_time": sum(self.update_times),
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
        
        print(f"GRPO policy model saved to {output_dir}")


def create_grpo_trainer(
    config: GRPOConfig,
    reward_model: nn.Module,
    model_name: Optional[str] = None
) -> GRPOTrainer:
    """
    Factory function to create a GRPO trainer.
    
    Args:
        config: GRPO configuration
        reward_model: Trained reward model
        model_name: Model name (overrides config if provided)
        
    Returns:
        Initialized GRPOTrainer
    """
    if model_name:
        config.model_name = model_name
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models during generation
    
    # Create policy model
    print(f"Loading GRPO policy model from {config.model_name}...")
    policy = GRPOPolicy(config.model_name, config.device)
    
    # Create reference model
    print("Creating reference model...")
    ref_policy = GRPOPolicy(config.model_name, config.device)
    
    # Create trainer
    trainer = GRPOTrainer(
        config=config,
        policy_model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        tokenizer=tokenizer
    )
    
    return trainer


if __name__ == "__main__":
    print("GRPO module loaded successfully.")
    print(f"GRPOConfig defaults: group_size={GRPOConfig.group_size}, kl_coef={GRPOConfig.kl_coef}")

