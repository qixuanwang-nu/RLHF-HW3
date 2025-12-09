"""
Part 1.2A: Reward Model Implementation
Fine-tunes GPT-2 as a reward model using pairwise ranking loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import GPT2Model, GPT2PreTrainedModel, AutoConfig


class RewardModel(GPT2PreTrainedModel):
    """
    Reward Model based on GPT-2.
    
    Architecture:
    - GPT-2 backbone for sequence encoding
    - Linear head to project hidden states to scalar reward
    
    The model is trained with pairwise ranking loss:
    L = -log(σ(r(x, y_chosen) - r(x, y_rejected)))
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.transformer = GPT2Model(config)
        
        # Reward head: projects the last token's hidden state to a scalar
        self.reward_head = nn.Linear(config.n_embd, 1, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute reward for a batch of sequences.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden_states: Whether to return the hidden states
            
        Returns:
            rewards: Scalar rewards [batch_size, 1]
            hidden_states: Optional hidden states if requested
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_hidden_states
        )
        
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Get the hidden state of the last non-padding token
        if attention_mask is not None:
            # Find the position of the last non-padding token
            # attention_mask: 1 for real tokens, 0 for padding
            sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
            
            # Handle edge case where sequence_lengths might be -1
            sequence_lengths = sequence_lengths.clamp(min=0)
            
            # Gather the last token's hidden state for each sequence
            batch_size = hidden_states.size(0)
            last_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
            ]  # [batch_size, hidden_size]
        else:
            # No attention mask: use the last position
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # Project to scalar reward
        rewards = self.reward_head(last_hidden)  # [batch_size, 1]
        
        if return_hidden_states:
            return rewards, outputs.hidden_states
        
        return rewards, None
    
    def forward(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        return_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pairwise ranking loss for preference learning.
        
        Loss = -log(σ(r(x, y_chosen) - r(x, y_rejected)))
        
        Args:
            chosen_input_ids: Token IDs for chosen responses [batch_size, seq_len]
            chosen_attention_mask: Attention mask for chosen [batch_size, seq_len]
            rejected_input_ids: Token IDs for rejected responses [batch_size, seq_len]
            rejected_attention_mask: Attention mask for rejected [batch_size, seq_len]
            return_outputs: Whether to return detailed outputs
            
        Returns:
            Dictionary containing:
            - loss: Scalar loss value
            - chosen_rewards: Rewards for chosen responses
            - rejected_rewards: Rewards for rejected responses
            - accuracy: Batch accuracy (chosen_reward > rejected_reward)
        """
        # Get rewards for chosen and rejected responses
        chosen_rewards, _ = self.get_reward(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )
        
        rejected_rewards, _ = self.get_reward(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        
        # Compute pairwise ranking loss
        # L = -log(σ(r_chosen - r_rejected))
        # This is equivalent to: -log_sigmoid(r_chosen - r_rejected)
        # Or: log(1 + exp(r_rejected - r_chosen))
        reward_diff = chosen_rewards - rejected_rewards  # [batch_size, 1]
        loss = -F.logsigmoid(reward_diff).mean()
        
        # Compute accuracy: how often is chosen_reward > rejected_reward?
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        outputs = {
            "loss": loss,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "accuracy": accuracy,
        }
        
        if return_outputs:
            outputs["reward_diff"] = reward_diff
            outputs["reward_margin"] = reward_diff.mean()
        
        return outputs
    
    @classmethod
    def from_pretrained_gpt2(cls, model_name: str = "openai-community/gpt2", **kwargs):
        """
        Initialize a reward model from a pretrained GPT-2 checkpoint.
        
        Args:
            model_name: HuggingFace model name/path
            **kwargs: Additional arguments for from_pretrained
            
        Returns:
            Initialized RewardModel
        """
        config = AutoConfig.from_pretrained(model_name)
        model = cls.from_pretrained(model_name, config=config, **kwargs)
        return model


class RewardModelTrainer:
    """
    Trainer for the reward model with comprehensive metric tracking.
    """
    
    def __init__(
        self,
        model: RewardModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            model: RewardModel instance
            optimizer: Optimizer
            scheduler: Optional learning rate scheduler
            device: Device to use
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Training metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.gradient_norms = []
        self.learning_rates = []
    
    def compute_gradient_norm(self) -> float:
        """Compute the total gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def train_step(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of preference pairs
            step: Current step number (for gradient accumulation)
            
        Returns:
            Dictionary of metrics for this step
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            chosen_input_ids=batch["chosen_input_ids"],
            chosen_attention_mask=batch["chosen_attention_mask"],
            rejected_input_ids=batch["rejected_input_ids"],
            rejected_attention_mask=batch["rejected_attention_mask"],
            return_outputs=True
        )
        
        loss = outputs["loss"] / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        metrics = {
            "loss": outputs["loss"].item(),
            "accuracy": outputs["accuracy"].item(),
            "reward_margin": outputs["reward_margin"].item(),
            "chosen_reward_mean": outputs["chosen_rewards"].mean().item(),
            "rejected_reward_mean": outputs["rejected_rewards"].mean().item(),
        }
        
        # Gradient step
        if (step + 1) % self.gradient_accumulation_steps == 0:
            # Compute gradient norm before clipping
            grad_norm = self.compute_gradient_norm()
            self.gradient_norms.append(grad_norm)
            metrics["gradient_norm"] = grad_norm
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
                metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
                self.learning_rates.append(metrics["learning_rate"])
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluate the model on a validation set.
        
        Args:
            dataloader: Validation DataLoader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_chosen_reward = 0.0
        total_rejected_reward = 0.0
        num_batches = 0
        
        all_chosen_rewards = []
        all_rejected_rewards = []
        
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(
                chosen_input_ids=batch["chosen_input_ids"],
                chosen_attention_mask=batch["chosen_attention_mask"],
                rejected_input_ids=batch["rejected_input_ids"],
                rejected_attention_mask=batch["rejected_attention_mask"],
                return_outputs=True
            )
            
            total_loss += outputs["loss"].item()
            total_accuracy += outputs["accuracy"].item()
            total_chosen_reward += outputs["chosen_rewards"].mean().item()
            total_rejected_reward += outputs["rejected_rewards"].mean().item()
            num_batches += 1
            
            all_chosen_rewards.extend(outputs["chosen_rewards"].squeeze().cpu().tolist())
            all_rejected_rewards.extend(outputs["rejected_rewards"].squeeze().cpu().tolist())
        
        # Handle case where rewards might be scalars
        if isinstance(all_chosen_rewards[0], float):
            chosen_rewards_tensor = torch.tensor(all_chosen_rewards)
            rejected_rewards_tensor = torch.tensor(all_rejected_rewards)
        else:
            chosen_rewards_tensor = torch.tensor([r for r in all_chosen_rewards])
            rejected_rewards_tensor = torch.tensor([r for r in all_rejected_rewards])
        
        metrics = {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
            "chosen_reward_mean": total_chosen_reward / num_batches,
            "rejected_reward_mean": total_rejected_reward / num_batches,
            "chosen_reward_std": chosen_rewards_tensor.std().item(),
            "rejected_reward_std": rejected_rewards_tensor.std().item(),
            "reward_margin": (chosen_rewards_tensor - rejected_rewards_tensor).mean().item(),
        }
        
        return metrics
    
    def get_training_history(self) -> Dict[str, list]:
        """Get all tracked metrics from training."""
        return {
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "gradient_norms": self.gradient_norms,
            "learning_rates": self.learning_rates,
        }


def create_reward_model(
    model_name: str = "openai-community/gpt2",
    device: str = None
) -> RewardModel:
    """
    Factory function to create a reward model.
    
    Args:
        model_name: HuggingFace model name
        device: Device to use (auto-detected if None)
        
    Returns:
        Initialized RewardModel
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Creating reward model from {model_name}...")
    model = RewardModel.from_pretrained_gpt2(model_name)
    model = model.to(device)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model loaded on {device}")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    return model


if __name__ == "__main__":
    # Test the reward model
    print("Testing RewardModel...")
    
    # Create model
    model = create_reward_model()
    
    # Create dummy input
    batch_size = 2
    seq_len = 128
    
    dummy_batch = {
        "chosen_input_ids": torch.randint(0, 50257, (batch_size, seq_len)),
        "chosen_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "rejected_input_ids": torch.randint(0, 50257, (batch_size, seq_len)),
        "rejected_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }
    
    # Move to device
    device = next(model.parameters()).device
    dummy_batch = {k: v.to(device) for k, v in dummy_batch.items()}
    
    # Forward pass
    outputs = model(**dummy_batch, return_outputs=True)
    
    print(f"\nTest outputs:")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Accuracy: {outputs['accuracy'].item():.4f}")
    print(f"  Chosen rewards shape: {outputs['chosen_rewards'].shape}")
    print(f"  Rejected rewards shape: {outputs['rejected_rewards'].shape}")
    print(f"  Reward margin: {outputs['reward_margin'].item():.4f}")
    
    print("\nRewardModel test passed!")

