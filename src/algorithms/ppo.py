"""
Part 2.1: PPO-based RLHF Implementation
Implements Proximal Policy Optimization for language model fine-tuning.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    
    # Model settings
    model_name: str = "openai-community/gpt2"
    
    # PPO hyperparameters
    clip_ratio: float = 0.2  # ε in clipped surrogate
    kl_coef: float = 0.2  # β for KL penalty (increased for stability)
    entropy_coef: float = 0.05  # Entropy bonus coefficient (increased to prevent collapse)
    value_coef: float = 0.5  # Value loss coefficient
    
    # Training settings
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    ppo_epochs: int = 4  # Number of PPO update epochs per batch
    max_grad_norm: float = 0.5
    
    # Generation settings
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    
    # KL control
    target_kl: float = 0.5  # Target KL divergence (increased for language models)
    kl_horizon: int = 10000  # Steps for adaptive KL
    adaptive_kl: bool = True  # Whether to use adaptive KL coefficient
    
    # GAE (Generalized Advantage Estimation)
    gamma: float = 1.0  # Discount factor
    lam: float = 0.95  # GAE lambda
    
    # Other
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PolicyModel(nn.Module):
    """
    Policy model wrapper for RLHF training.
    Wraps a causal LM and adds a value head for PPO.
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        
        # Value head for estimating state values
        hidden_size = self.model.config.n_embd
        self.value_head = nn.Linear(hidden_size, 1)
        self.value_head.to(device)
        
        self.device = device
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        """Forward pass returning logits and values."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get last hidden state for value estimation
        hidden_states = outputs.hidden_states[-1]
        
        # Get value for each position (or just the last)
        values = self.value_head(hidden_states).squeeze(-1)
        
        if return_dict:
            return {
                "logits": outputs.logits,
                "values": values,
                "hidden_states": hidden_states
            }
        return outputs.logits, values
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        pad_token_id: int = None,
        **kwargs
    ):
        """Generate text using the policy."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=pad_token_id,
            **kwargs
        )
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute log probabilities for the given input.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask
            labels: Target token IDs (if None, uses input_ids shifted)
            
        Returns:
            Log probabilities [batch, seq_len-1]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits[:, :-1, :]  # [batch, seq_len-1, vocab]
        
        if labels is None:
            labels = input_ids[:, 1:]  # Shift right
        else:
            labels = labels[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs


class PPOLoss:
    """
    PPO Loss computation with all components:
    - Clipped surrogate objective
    - KL divergence penalty
    - Entropy bonus
    - Value function loss
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.clip_ratio = config.clip_ratio
        self.kl_coef = config.kl_coef
        self.entropy_coef = config.entropy_coef
        self.value_coef = config.value_coef
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the clipped surrogate policy loss.
        
        L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
        
        where r_t = π_θ(a|s) / π_θ_old(a|s) = exp(log_prob - old_log_prob)
        
        Args:
            log_probs: Current policy log probabilities [batch, seq_len]
            old_log_probs: Old policy log probabilities [batch, seq_len]
            advantages: Advantage estimates [batch, seq_len] or [batch]
            mask: Mask for valid tokens [batch, seq_len]
            
        Returns:
            policy_loss: Scalar loss
            metrics: Dictionary of metrics
        """
        # Compute probability ratio with numerical stability
        log_ratio = log_probs - old_log_probs
        # Clamp log ratio to prevent exp explosion
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
        ratio = torch.exp(log_ratio)
        
        # Expand advantages if needed
        if advantages.dim() == 1 and log_probs.dim() == 2:
            advantages = advantages.unsqueeze(1).expand_as(log_probs)
        
        # Clip advantages for stability
        advantages = torch.clamp(advantages, -10.0, 10.0)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        
        # Take minimum (pessimistic bound)
        policy_loss = -torch.min(surr1, surr2)
        
        # Apply mask if provided
        if mask is not None:
            policy_loss = policy_loss * mask
            policy_loss = policy_loss.sum() / mask.sum()
        else:
            policy_loss = policy_loss.mean()
        
        # Compute metrics
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
            # Approximate KL using (r-1) - log(r) formula (always positive)
            approx_kl = ((ratio - 1) - torch.log(ratio + 1e-8)).mean().item()
        
        metrics = {
            "policy_loss": policy_loss.item(),
            "clip_fraction": clip_fraction,
            "approx_kl": max(0, approx_kl),  # Ensure non-negative
            "ratio_mean": ratio.mean().item(),
        }
        
        return policy_loss, metrics
    
    def compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute KL divergence penalty from reference policy.
        
        Uses the "k3" estimator which is always non-negative:
        
            KL ≈ 0.5 * E[(r - 1)^2]  where r = π(a|s) / π_ref(a|s)
        
        This approximates the true KL while being numerically stable
        and always positive, preventing the policy from drifting
        arbitrarily from the reference.
        
        Args:
            log_probs: Current policy log probs
            ref_log_probs: Reference policy log probs
            mask: Mask for valid tokens
            
        Returns:
            kl_penalty: Weighted KL penalty
            kl_value: Raw KL divergence value
        """
        log_ratio = log_probs - ref_log_probs
        # Clamp to prevent numerical issues with exp
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
        ratio = torch.exp(log_ratio)
        
        # k3 estimator: 0.5 * (r - 1)^2, always non-negative
        kl_approx = 0.5 * (ratio - 1) ** 2
        
        if mask is not None:
            kl_approx = kl_approx * mask
            kl_value = kl_approx.sum() / (mask.sum() + 1e-8)
        else:
            kl_value = kl_approx.mean()
        
        kl_penalty = self.kl_coef * kl_value
        
        return kl_penalty, float(kl_value.item())
    
    def compute_entropy_bonus(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute entropy bonus for exploration.
        
        We maximize entropy to prevent mode collapse. The entropy term is
        SUBTRACTED from the loss (equivalently, we add -entropy_coef * entropy).
        Higher entropy = lower loss = optimizer encouraged to maintain diversity.
        
        Args:
            logits: Policy logits [batch, seq_len, vocab_size]
            mask: Mask for valid tokens
            
        Returns:
            entropy_loss: Entropy term to ADD to loss (negative of bonus)
            entropy_value: Raw entropy value
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
        
        # Maximize entropy by adding -entropy_coef * entropy to loss
        # When entropy is high: entropy_loss is negative -> lower total loss -> good
        # When entropy is low: entropy_loss is less negative -> higher total loss -> penalized
        entropy_loss = -self.entropy_coef * entropy_value
        
        return entropy_loss, entropy_value.item()
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute value function loss with optional clipping.
        
        Args:
            values: Current value estimates
            returns: Target returns
            old_values: Old value estimates (for clipping)
            mask: Mask for valid positions
            
        Returns:
            value_loss: Weighted value loss
            value_loss_raw: Raw MSE value
        """
        if old_values is not None:
            # Clipped value loss (PPO-style)
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.clip_ratio,
                self.clip_ratio
            )
            value_loss1 = (values - returns) ** 2
            value_loss2 = (values_clipped - returns) ** 2
            value_loss = torch.max(value_loss1, value_loss2)
        else:
            value_loss = (values - returns) ** 2
        
        if mask is not None:
            value_loss = value_loss * mask
            value_loss_raw = value_loss.sum() / mask.sum()
        else:
            value_loss_raw = value_loss.mean()
        
        value_loss_weighted = self.value_coef * value_loss_raw
        
        return value_loss_weighted, value_loss_raw.item()
    
    def compute_total_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total PPO loss combining all components.
        
        L_total = L_CLIP + β * KL(π || π_ref) - c_ent * H(π) + c_vf * L_VF
        
        Args:
            log_probs: Current policy log probs
            old_log_probs: Old policy log probs (for ratio)
            ref_log_probs: Reference policy log probs (for KL)
            logits: Current policy logits (for entropy)
            values: Current value estimates
            returns: Target returns
            advantages: Advantage estimates
            old_values: Old value estimates
            mask: Token mask
            
        Returns:
            total_loss: Combined loss
            metrics: Dictionary of all metrics
        """
        # Policy loss (clipped surrogate)
        policy_loss, policy_metrics = self.compute_policy_loss(
            log_probs, old_log_probs, advantages, mask
        )
        
        # KL penalty from reference
        kl_penalty, kl_value = self.compute_kl_penalty(
            log_probs, ref_log_probs, mask
        )
        
        # Entropy loss (negative = encourages higher entropy)
        entropy_loss, entropy_value = self.compute_entropy_bonus(logits, mask)
        
        # Value loss
        value_loss, value_loss_raw = self.compute_value_loss(
            values, returns, old_values, mask
        )
        
        # Total loss: policy + KL penalty + entropy regularization + value
        total_loss = policy_loss + kl_penalty + entropy_loss + value_loss
        
        metrics = {
            **policy_metrics,
            "kl_penalty": kl_penalty.item(),
            "kl_divergence": kl_value,
            "entropy_bonus": -entropy_loss.item(),  # Positive value = entropy being encouraged
            "entropy": entropy_value,
            "value_loss": value_loss_raw,
            "total_loss": total_loss.item(),
        }
        
        return total_loss, metrics


class PPOTrainer:
    """
    PPO Trainer for RLHF policy optimization.
    
    Implements the full PPO training loop with:
    - Response generation
    - Reward computation
    - Advantage estimation (GAE)
    - Policy updates with clipping
    """
    
    def __init__(
        self,
        config: PPOConfig,
        policy_model: PolicyModel,
        ref_model: PolicyModel,
        reward_model: nn.Module,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Args:
            config: PPO configuration
            policy_model: Policy to be trained
            ref_model: Reference policy (frozen)
            reward_model: Reward model from Part 1
            tokenizer: Tokenizer for text processing
        """
        self.config = config
        self.policy = policy_model
        self.ref_policy = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = config.device
        
        # Freeze reference model
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        self.ref_policy.eval()
        
        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()
        
        # Loss computation
        self.ppo_loss = PPOLoss(config)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # Adaptive KL coefficient
        self.kl_coef = config.kl_coef
        
        # Training metrics
        self.training_stats = []
        self.global_step = 0
    
    @torch.no_grad()
    def generate_responses(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            max_new_tokens: Max tokens to generate
            
        Returns:
            responses: List of generated response strings
            input_ids: Full sequence token IDs
            attention_mask: Attention masks
        """
        self.policy.eval()
        
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        
        # Tokenize prompts
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512 - max_new_tokens,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        prompt_lengths = attention_mask.sum(dim=1)
        
        # Generate
        output_ids = self.policy.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Decode responses
        responses = []
        for i, (out, prompt_len) in enumerate(zip(output_ids, prompt_lengths)):
            response_tokens = out[prompt_len:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response)
        
        # Create attention mask for full sequence
        full_attention_mask = (output_ids != self.tokenizer.pad_token_id).long()
        
        return responses, output_ids, full_attention_mask
    
    @torch.no_grad()
    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rewards using the reward model.
        
        Args:
            input_ids: Full sequence token IDs
            attention_mask: Attention mask
            
        Returns:
            rewards: Reward values [batch_size]
        """
        rewards, _ = self.reward_model.get_reward(input_ids, attention_mask)
        return rewards.squeeze(-1)
    
    @torch.no_grad()
    def compute_advantages_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        response_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        A_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t+1}δ_{T-1}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        
        For language models, we typically assign the reward to the last token.
        
        Args:
            rewards: Rewards [batch_size] (assigned to last response token)
            values: Value estimates [batch_size, seq_len]
            response_mask: Mask indicating response tokens [batch_size, seq_len]
            
        Returns:
            advantages: Advantage estimates
            returns: Target returns for value function
        """
        batch_size, seq_len = values.shape
        gamma = self.config.gamma
        lam = self.config.lam
        
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        
        # For simplicity, assign full reward to last token and compute returns
        for i in range(batch_size):
            # Find last response token
            response_positions = response_mask[i].nonzero(as_tuple=True)[0]
            if len(response_positions) == 0:
                continue
                
            last_pos = response_positions[-1].item()
            
            # Assign reward to last position
            returns[i, last_pos] = rewards[i]
            
            # Backward pass for GAE
            last_gae = 0
            for t in reversed(range(last_pos + 1)):
                if t == last_pos:
                    next_value = 0
                    reward = rewards[i]
                else:
                    next_value = values[i, t + 1]
                    reward = 0
                
                delta = reward + gamma * next_value - values[i, t]
                last_gae = delta + gamma * lam * last_gae
                advantages[i, t] = last_gae
                returns[i, t] = advantages[i, t] + values[i, t]
        
        # Normalize advantages
        adv_mean = advantages[response_mask.bool()].mean()
        adv_std = advantages[response_mask.bool()].std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        
        return advantages, returns
    
    def ppo_update(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        response_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform PPO update for multiple epochs.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            old_log_probs: Log probs from old policy
            ref_log_probs: Log probs from reference policy
            old_values: Values from old policy
            advantages: Advantage estimates
            returns: Target returns
            response_mask: Mask for response tokens
            
        Returns:
            metrics: Training metrics
        """
        self.policy.train()
        
        all_metrics = []
        
        for epoch in range(self.config.ppo_epochs):
            # Forward pass
            outputs = self.policy(input_ids, attention_mask)
            logits = outputs["logits"]
            values = outputs["values"]
            
            # Compute current log probs
            log_probs = self.policy.get_log_probs(input_ids, attention_mask)
            
            # Align dimensions (log_probs is seq_len-1)
            response_mask_aligned = response_mask[:, 1:]
            old_log_probs_aligned = old_log_probs[:, 1:] if old_log_probs.size(1) > log_probs.size(1) else old_log_probs
            ref_log_probs_aligned = ref_log_probs[:, 1:] if ref_log_probs.size(1) > log_probs.size(1) else ref_log_probs
            advantages_aligned = advantages[:, 1:] if advantages.size(1) > log_probs.size(1) else advantages
            returns_aligned = returns[:, 1:] if returns.size(1) > log_probs.size(1) else returns
            values_aligned = values[:, :-1] if values.size(1) > log_probs.size(1) else values
            old_values_aligned = old_values[:, :-1] if old_values.size(1) > log_probs.size(1) else old_values
            logits_aligned = logits[:, :-1, :] if logits.size(1) > log_probs.size(1) + 1 else logits[:, :log_probs.size(1), :]
            
            # Ensure matching sizes
            min_len = min(log_probs.size(1), response_mask_aligned.size(1), 
                         old_log_probs_aligned.size(1), ref_log_probs_aligned.size(1),
                         advantages_aligned.size(1), returns_aligned.size(1),
                         values_aligned.size(1), old_values_aligned.size(1))
            
            log_probs = log_probs[:, :min_len]
            response_mask_aligned = response_mask_aligned[:, :min_len]
            old_log_probs_aligned = old_log_probs_aligned[:, :min_len]
            ref_log_probs_aligned = ref_log_probs_aligned[:, :min_len]
            advantages_aligned = advantages_aligned[:, :min_len]
            returns_aligned = returns_aligned[:, :min_len]
            values_aligned = values_aligned[:, :min_len]
            old_values_aligned = old_values_aligned[:, :min_len]
            logits_aligned = logits_aligned[:, :min_len, :]
            
            # Compute total loss
            total_loss, metrics = self.ppo_loss.compute_total_loss(
                log_probs=log_probs,
                old_log_probs=old_log_probs_aligned,
                ref_log_probs=ref_log_probs_aligned,
                logits=logits_aligned,
                values=values_aligned,
                returns=returns_aligned,
                advantages=advantages_aligned,
                old_values=old_values_aligned,
                mask=response_mask_aligned.float()
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
            
            all_metrics.append(metrics)
            
            # Early stopping if KL too high (use actual KL, not approx)
            # Use a more reasonable threshold for language models
            if metrics["kl_divergence"] > 10.0:  # Stop if KL > 10
                break
        
        # Aggregate metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        avg_metrics["ppo_epochs_actual"] = len(all_metrics)
        
        # Adaptive KL coefficient - more aggressive adjustment
        if self.config.adaptive_kl:
            kl = avg_metrics["kl_divergence"]
            target = max(0.1, self.config.target_kl)  # Minimum target of 0.1
            
            if kl > 4.0 * target:  # KL way too high
                self.ppo_loss.kl_coef *= 2.0
            elif kl > 2.0 * target:  # KL too high
                self.ppo_loss.kl_coef *= 1.5
            elif kl < 0.5 * target:  # KL too low
                self.ppo_loss.kl_coef *= 0.8
            
            # Keep coefficient in reasonable range
            self.ppo_loss.kl_coef = max(0.01, min(2.0, self.ppo_loss.kl_coef))
            avg_metrics["kl_coef"] = self.ppo_loss.kl_coef
        
        return avg_metrics
    
    def train_step(
        self,
        prompts: List[str]
    ) -> Dict[str, float]:
        """
        Perform one PPO training step.
        
        Args:
            prompts: Batch of prompts
            
        Returns:
            metrics: Training metrics
        """
        start_time = time.time()
        
        # 1. Generate responses
        responses, full_ids, attention_mask = self.generate_responses(prompts)
        
        # 2. Compute rewards and normalize
        raw_rewards = self.compute_rewards(full_ids, attention_mask)
        
        # Normalize rewards for stability (running mean/std) on raw rewards
        if not hasattr(self, 'reward_mean'):
            self.reward_mean = raw_rewards.mean().item()
            self.reward_std = raw_rewards.std().item() + 1e-8
        else:
            # Exponential moving average
            alpha = 0.1
            self.reward_mean = (1 - alpha) * self.reward_mean + alpha * raw_rewards.mean().item()
            self.reward_std = (1 - alpha) * self.reward_std + alpha * (raw_rewards.std().item() + 1e-8)
        
        rewards = (raw_rewards - self.reward_mean) / self.reward_std
        
        # 3. Compute log probs and values from current and reference policy
        self.policy.eval()
        with torch.no_grad():
            outputs = self.policy(full_ids, attention_mask)
            old_values = outputs["values"]
            old_log_probs = self.policy.get_log_probs(full_ids, attention_mask)
            
            ref_log_probs = self.ref_policy.get_log_probs(full_ids, attention_mask)
        
        # 4. Create response mask (tokens after prompt)
        # For simplicity, mark all tokens as part of response
        # In practice, you'd track prompt length
        prompt_encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        prompt_lengths = prompt_encoded["attention_mask"].sum(dim=1)
        
        response_mask = torch.zeros_like(attention_mask)
        for i, prompt_len in enumerate(prompt_lengths):
            response_mask[i, prompt_len:] = attention_mask[i, prompt_len:]
        
        # 5. Compute advantages using GAE
        advantages, returns = self.compute_advantages_gae(rewards, old_values, response_mask)
        
        # 6. PPO update
        metrics = self.ppo_update(
            full_ids, attention_mask,
            old_log_probs, ref_log_probs, old_values,
            advantages, returns, response_mask
        )
        
        # Add reward statistics (both raw and normalized for debugging)
        metrics["reward_mean"] = raw_rewards.mean().item()
        metrics["reward_std"] = raw_rewards.std().item()
        metrics["reward_mean_norm"] = rewards.mean().item()
        metrics["reward_std_norm"] = rewards.std().item()
        metrics["response_length_mean"] = response_mask.sum(dim=1).float().mean().item()
        metrics["step_time"] = time.time() - start_time
        
        self.global_step += 1
        self.training_stats.append(metrics)
        
        return metrics
    
    def train(
        self,
        dataloader: DataLoader,
        num_steps: int = 1000,
        eval_every: int = 100,
        log_every: int = 10
    ) -> List[Dict[str, float]]:
        """
        Full PPO training loop.
        
        Args:
            dataloader: DataLoader yielding prompts
            num_steps: Total training steps
            eval_every: Evaluation frequency
            log_every: Logging frequency
            
        Returns:
            training_stats: List of training metrics
        """
        pbar = tqdm(total=num_steps, desc="PPO Training")
        
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
                      f"reward={metrics['reward_mean']:.3f}, "
                      f"kl={metrics['kl_divergence']:.4f}, "
                      f"entropy={metrics['entropy']:.4f}, "
                      f"loss={metrics['total_loss']:.4f}")
        
        pbar.close()
        return self.training_stats
    
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
        
        print(f"PPO policy model saved to {output_dir}")


def create_ppo_trainer(
    config: PPOConfig,
    reward_model: nn.Module,
    model_name: Optional[str] = None
) -> PPOTrainer:
    """
    Factory function to create a PPO trainer.
    
    Args:
        config: PPO configuration
        reward_model: Trained reward model
        model_name: Model name (overrides config if provided)
        
    Returns:
        Initialized PPOTrainer
    """
    if model_name:
        config.model_name = model_name
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models during generation
    
    # Create policy model
    print(f"Loading policy model from {config.model_name}...")
    policy = PolicyModel(config.model_name, config.device)
    
    # Create reference model (copy of initial policy)
    print("Creating reference model...")
    ref_policy = PolicyModel(config.model_name, config.device)
    
    # Create trainer
    trainer = PPOTrainer(
        config=config,
        policy_model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        tokenizer=tokenizer
    )
    
    return trainer


if __name__ == "__main__":
    # Quick test
    print("PPO module loaded successfully.")
    print(f"PPOConfig defaults: clip_ratio={PPOConfig.clip_ratio}, kl_coef={PPOConfig.kl_coef}")

