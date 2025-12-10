# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLHF training framework implementing reward modeling and policy optimization on the Anthropic HH-RLHF dataset. Uses GPT-2 as the base model.

## Common Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Dataset exploration
python explore_dataset.py

# Train reward model (quick test)
python train_reward_model.py --epochs 1 --batch_size 4 --subset_size 200

# Train reward model (full)
python train_reward_model.py --epochs 3 --batch_size 8 --lr 1e-5 --save_model

# Train policy with PPO
python train_policy.py --method ppo --steps 100 --batch_size 2

# Train policy with GRPO
python train_policy.py --method grpo --steps 100 --group_size 4

# Train policy with DPO (no reward model needed)
python train_policy.py --method dpo --epochs 1 --dpo_beta 0.1

# Compare all methods
python train_policy.py --method all --steps 100
```

## Architecture

### Two-Phase Training Pipeline

**Phase 1: Reward Model** (`train_reward_model.py`)
- Loads Anthropic HH-RLHF preference data
- Trains a GPT-2 + linear head to predict human preferences
- Loss: Bradley-Terry pairwise ranking loss `-log(σ(r_chosen - r_rejected))`
- Outputs to `outputs/run_<timestamp>/`

**Phase 2: Policy Optimization** (`train_policy.py`)
- Three algorithms implemented, all using the reward model from Phase 1:

| Algorithm | File | Key Concept |
|-----------|------|-------------|
| PPO | `src/algorithms/ppo.py` | Clipped surrogate objective + value function + GAE |
| GRPO | `src/algorithms/grpo.py` | Group-relative advantages (no value head needed) |
| DPO | `src/algorithms/dpo.py` | Direct preference optimization (no reward model needed) |

### Key Components

- **RewardModel** (`src/models/reward_model.py`): GPT-2 backbone + reward head, uses last token hidden state
- **PolicyModel/GRPOPolicy/DPOPolicy**: Wrappers around causal LM for generation and log-prob computation
- **Loss Functions**: Each algorithm has its own loss class with `compute_total_loss()` method
- **Trainers**: Handle the full training loop including generation, reward computation, and updates

### Data Flow

```
HH-RLHF Dataset → DataPreprocessor → (chosen_ids, rejected_ids) → RewardModel → rewards
                                                                                    ↓
Prompts → Policy.generate() → responses → RewardModel.get_reward() → advantages → Policy update
```

## Key Configuration Classes

- `PPOConfig`: clip_ratio, kl_coef, entropy_coef, ppo_epochs, GAE params
- `GRPOConfig`: group_size (responses per prompt), kl_coef, entropy_coef
- `DPOConfig`: beta (temperature), loss_type (sigmoid/hinge/ipo)

## Output Structure

```
outputs/
├── run_<timestamp>/           # Reward model training
│   ├── config.json
│   ├── training_curves.png
│   ├── final_results.json
│   ├── evaluation/
│   └── best_model/
└── policy/run_<timestamp>/    # Policy training
    ├── ppo/
    ├── grpo/
    ├── dpo/
    └── comparison/
```
