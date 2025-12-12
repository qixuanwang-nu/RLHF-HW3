## RLHF / GRPO / DPO Assignment

This repository implements a practical RLHF-style alignment assignment using the Anthropic HH-RLHF dataset:

- **Part 1**: Preference data exploration + reward model (pairwise ranking loss)
- **Part 2**: Policy optimization with **PPO** and **GRPO**
- **Part 3**: **DPO** (direct preference optimization)
- **Part 4**: Quantitative + qualitative evaluation (including GPT-4.1‑mini as judge when configured)

---

## Latest results (for graders)

- **RL evaluation artifacts**: `evaluation/run_20251212_081643/`
- **Exported samples (~20/model)**: `samples/run_20251212_081643/`

---

## Setup

### Option A: Local (recommended for development)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: Docker (required by the assignment)

Build:

```bash
docker build -t rlhf-assignment .
```

Run (example):

```bash
docker run --rm -it rlhf-assignment python train_reward_model.py --help
```

If you want to use a GPU, run with your platform’s NVIDIA runtime support (e.g., `--gpus all`) and an appropriate CUDA-enabled base image. The included Dockerfile is CPU-focused for portability.

---

## Compute requirements

- **CPU-only**: Works for smoke tests and short runs (`--subset_size`, small `--steps`).
- **GPU recommended**:
  - Reward model training on HH-RLHF is substantially faster on GPU.
  - Policy optimization (PPO/GRPO) benefits strongly from GPU.
  - Part 4 evaluation with multiple models can be memory-heavy; use:
    - `--gen_batch_size` and `--score_batch_size` in `evaluate_models.py`
    - `--amp_dtype fp16` to reduce VRAM

Typical VRAM guidance:
- **>= 16GB**: workable with small batch sizes
- **>= 24GB**: comfortable for most runs with batching enabled

---

## Part 1: Dataset exploration + Reward model

Dataset exploration (writes plots/stats to `outputs/exploration/`):

```bash
python explore_dataset.py
```

Train reward model (example):

```bash
python train_reward_model.py --epochs 1 --batch_size 8 --lr 1e-5 --save_model
```

The trained reward model is saved under `outputs/run_<timestamp>/best_model/`.

---

## Part 2 & 3: Train policy models (PPO / GRPO / DPO)

### PPO

```bash
python train_policy.py \
  --method ppo \
  --reward_model_path <PATH_TO_REWARD_MODEL_DIR> \
  --steps 500 \
  --batch_size 8 \
  --lr 1e-5
```

### GRPO

```bash
python train_policy.py \
  --method grpo \
  --reward_model_path <PATH_TO_REWARD_MODEL_DIR> \
  --steps 500 \
  --batch_size 4 \
  --group_size 4 \
  --lr 1e-5
```

### DPO

```bash
python train_policy.py \
  --method dpo \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-6
```

Outputs are written under `outputs/policy/run_<timestamp>/` with `*/model/` directories saved.

---

## Part 4: Evaluation (quantitative + qualitative)

**Important folder convention (per assignment):**

- **Reward model evaluation** artifacts live under `outputs/run_<timestamp>/evaluation/`.
- **RL policy evaluation** artifacts live under `evaluation/run_<timestamp>/`.

The RL evaluation script produces:
- win-rate vs reference (**GPT-4.1-as-judge** when OpenAI API is configured)
- reward-model score distributions (reference/PPO/GRPO/DPO)
- KL drift vs reference (proxy)
- Pareto frontier (reward vs KL)
- adversarial qualitative samples
- training curves + reward–KL tradeoff curves (loaded from `training_stats.json`)

### Configure GPT-4.1 Mini judge

This repo uses an env file. The template is `env.example`:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
```

Or edit `env.example` and pass:

```bash
python evaluate_models.py --env_file env.example --prefer_openai_judge ...
```

### Run evaluation

```bash
python evaluate_models.py \
  --env_file env.example \
  --prefer_openai_judge \
  --reward_model_path <PATH_TO_REWARD_MODEL_DIR> \
  --reference_model openai-community/gpt2 \
  --ppo_model_path outputs/policy/<run>/ppo/model \
  --grpo_model_path outputs/policy/<run>/grpo/model \
  --dpo_model_path outputs/policy/<run>/dpo/model \
  --num_prompts 200 \
  --winrate_prompts 120 \
  --gen_batch_size 4 \
  --score_batch_size 1 \
  --amp_dtype fp16
```

Artifacts are written under `evaluation/run_<timestamp>/`.

### Export ~20 samples per model (for submission)

Given an evaluation run directory, export ~20 prompt/response pairs per model:

```bash
python export_samples.py --eval_run_dir evaluation/run_<timestamp> --out_dir samples --per_model 20
```

# RLHF Implementation: Reward Modeling on Anthropic HH-RLHF

This project implements Part 1 of an RLHF (Reinforcement Learning from Human Feedback) assignment, focusing on preference data collection and reward modeling.

## Project Structure

```
RLHF/
├── src/
│   ├── data/
│   │   ├── exploration.py      # Part 1.1A: Dataset analysis
│   │   ├── preprocessing.py    # Part 1.1B: Data preprocessing pipeline
│   │   └── dataset.py          # Dataset utilities
│   ├── models/
│   │   └── reward_model.py     # Part 1.2A: Reward model implementation
│   └── evaluation/
│       └── error_analysis.py   # Part 1.2B: Evaluation and error analysis
├── explore_dataset.py          # Script to run dataset exploration
├── train_reward_model.py       # Main training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Part 1.1: Preference Data Collection

### Task A: Dataset Exploration

Explore the Anthropic HH-RLHF dataset structure and analyze preference pair distributions:

```bash
python explore_dataset.py
```

This will:
- Load the dataset from HuggingFace
- Compute statistics (length distributions, word counts, conversation depth)
- Identify biases and patterns (length bias, verbosity patterns)
- Generate visualizations in `outputs/exploration/`
- Display sample examples for qualitative analysis

### Task B: Data Preprocessing Pipeline

The preprocessing pipeline (`src/data/preprocessing.py`) handles:

- **Tokenization**: Uses GPT-2 tokenizer with configurable max length
- **Balanced splits**: Stratified by sequence length for representative validation
- **Edge cases**:
  - Tie detection (nearly identical responses)
  - Very long sequence truncation (left truncation to keep recent context)
  - Minimum response length filtering
  - Empty response handling

## Part 1.2: Reward Model Training

### Task A: Reward Model Implementation

The reward model (`src/models/reward_model.py`) features:

- **Architecture**: GPT-2 backbone + linear reward head
- **Loss function**: Pairwise ranking loss
  ```
  L = -log(σ(r(x, y_chosen) - r(x, y_rejected)))
  ```
- **Metrics tracked**:
  - Accuracy on preference predictions
  - Loss curves (train/val)
  - Gradient norms
  - Learning rate schedule

### Task B: Evaluation and Error Analysis

Run the complete training and evaluation pipeline:

```bash
# Full training (recommended for GPU)
python train_reward_model.py --epochs 3 --batch_size 8 --lr 1e-5 --save_model

# Quick test run (CPU-friendly)
python train_reward_model.py --epochs 1 --batch_size 4 --subset_size 1000

# With custom settings
python train_reward_model.py \
    --epochs 2 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_length 512 \
    --gradient_accumulation_steps 4 \
    --eval_steps 200 \
    --error_analysis_samples 2000
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 1 | Number of training epochs |
| `--batch_size` | 8 | Training batch size |
| `--lr` | 1e-5 | Learning rate |
| `--max_length` | 512 | Maximum sequence length |
| `--model_name` | openai-community/gpt2 | Pretrained model backbone |
| `--gradient_accumulation_steps` | 1 | Gradient accumulation steps |
| `--max_grad_norm` | 1.0 | Max gradient norm for clipping |
| `--eval_steps` | 500 | Evaluate every N steps |
| `--error_analysis_samples` | 1000 | Samples for error analysis |
| `--subset_size` | None | Use subset for quick testing |
| `--save_model` | False | Save the trained model |

## Output Files

After training, outputs are saved to `outputs/run_<timestamp>/`:

```
outputs/run_<timestamp>/
├── config.json              # Training configuration
├── training_history.json    # Loss, accuracy, gradient norms over time
├── training_curves.png      # Visualization of training metrics
├── final_results.json       # Final metrics and summary
├── evaluation/
│   ├── error_analysis_report.txt   # Detailed error analysis (20+ examples)
│   ├── error_analysis.json         # Error analysis data
│   └── error_analysis_plots.png    # Error visualizations
└── best_model/              # Saved model (if --save_model)
    ├── config.json
    ├── model.safetensors
    └── tokenizer files
```

## Key Implementation Details

### Pairwise Ranking Loss

The reward model is trained with the Bradley-Terry loss:

```python
L = -log(σ(r_chosen - r_rejected))
```

This encourages the model to assign higher rewards to chosen responses.

### Error Analysis

The error analysis module (`src/evaluation/error_analysis.py`) provides:

1. **Overall accuracy** on held-out validation data
2. **Pattern identification**:
   - Length bias (does model prefer longer responses?)
   - Confidence analysis (low vs high confidence errors)
   - Reward margin distribution
3. **Detailed examples** (20+ error cases with full context)

### Identified Biases

Common biases in HH-RLHF and reward models:
- **Length bias**: Longer responses often preferred
- **Verbosity**: More detailed responses ranked higher
- **Refusal patterns**: Certain refusal phrases strongly associated with "chosen"

---

## Part 2: Policy Optimization

### 2.1 PPO-based RLHF (`src/algorithms/ppo.py`)

#### Task A: PPO Loss Function

The implementation includes all required components:

```python
# Clipped Surrogate Objective
L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

# KL Divergence Penalty
L_KL = β * KL(π_θ || π_ref)

# Entropy Bonus
L_entropy = -c_ent * H(π)

# Overall Loss
L_total = L_CLIP + L_KL + L_entropy + c_vf * L_VF
```

**Key hyperparameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_ratio` | 0.2 | ε for clipping |
| `kl_coef` | 0.1 | β for KL penalty |
| `entropy_coef` | 0.01 | Exploration bonus |
| `ppo_epochs` | 4 | Updates per batch |

#### Task B: Policy Training

```bash
# Train with PPO
python train_policy.py --method ppo --steps 100 --batch_size 2

# With custom hyperparameters
python train_policy.py --method ppo \
    --steps 500 \
    --clip_ratio 0.2 \
    --kl_coef 0.05 \
    --lr 1e-5
```

### 2.2 GRPO Implementation (`src/algorithms/grpo.py`)

#### Task A: Group-Based Advantage Estimation

#### Task A: Group-Based Advantage Estimation

GRPO samples multiple responses per prompt and computes advantages relative to the group mean:

```python
# For each prompt, sample group_size responses
responses = [generate(prompt) for _ in range(group_size)]

# Compute rewards
rewards = [reward_model(r) for r in responses]

# Group-relative advantage
advantage[i] = reward[i] - mean(rewards)
```

**Key differences from PPO:**
- No value function needed
- Simpler policy gradient (no clipping)
- Relies on KL penalty for stability
- More memory-efficient (no value head)

#### Task B: Training and Comparison

```bash
# Train with GRPO
python train_policy.py --method grpo --steps 100 --group_size 4

# Compare both methods
python train_policy.py --method both --steps 100

# Full comparison with more steps
python train_policy.py --method both \
    --steps 500 \
    --batch_size 4 \
    --group_size 4
```

---

## Part 3: Direct Preference Optimization (DPO)

### Implementation (`src/algorithms/dpo.py`)

DPO bypasses explicit reward modeling by directly optimizing the policy using preference data.

#### Key Formula

```python
L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

Where:
- `y_w` = chosen (winning) response
- `y_l` = rejected (losing) response
- `β` = temperature controlling deviation from reference
- `π` = policy model
- `π_ref` = reference model (frozen)

#### Advantages over PPO/GRPO

| Feature | PPO | GRPO | DPO |
|---------|-----|------|-----|
| Reward Model | Required | Required | **Not needed** |
| Training Type | RL | RL | **Supervised** |
| Value Function | Yes | No | **No** |
| Stability | High | Medium | **Highest** |
| Complexity | High | Medium | **Low** |

#### Usage

```bash
# Train with DPO only
python train_policy.py --method dpo --epochs 1 --dpo_beta 0.1

# With custom settings
python train_policy.py --method dpo \
    --epochs 3 \
    --batch_size 4 \
    --dpo_beta 0.1 \
    --dpo_loss_type sigmoid \
    --lr 1e-6

# Compare all three methods
python train_policy.py --method all --steps 100 --epochs 1
```

**DPO-specific arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--dpo_beta` | 0.1 | Temperature parameter |
| `--dpo_loss_type` | sigmoid | Loss type: sigmoid, hinge, ipo |

---

### Comparison Metrics

When using `--method both` or `--method all`, the system compares:

1. **Training Stability**
   - Loss variance
   - Reward variance
   - Oscillation in training

2. **Convergence Speed**
   - Steps to reach reward threshold
   - Final reward achieved
   - Convergence rate

3. **Computational Efficiency**
   - Time per training step
   - Peak memory usage
   - Samples processed per step

4. **Sample Quality**
   - KL divergence from reference
   - Entropy (exploration)
   - Reward improvement

### Output Files

```
outputs/policy/run_<timestamp>/
├── config.json
├── ppo/
│   └── training_stats.json
├── grpo/
│   ├── training_stats.json
│   └── efficiency_stats.json
└── comparison/
    ├── ppo_vs_grpo_comparison.png
    ├── comparison_report.txt
    └── comparison_results.json
```

---

## Complete Training Pipeline

### Quick Demo (CPU-friendly)

```bash
# Activate environment
source .venv/bin/activate

# 1. Explore dataset
python explore_dataset.py

# 2. Train reward model (small subset)
python train_reward_model.py --epochs 1 --batch_size 4 --subset_size 200

# 3. Train policy with PPO/GRPO comparison
python train_policy.py --method both --steps 50 --num_prompts 100
```

### Full Training (GPU recommended)

```bash
# 1. Full reward model training
python train_reward_model.py \
    --epochs 3 \
    --batch_size 8 \
    --lr 1e-5 \
    --save_model

# 2. Policy training with trained reward model
python train_policy.py \
    --method both \
    --steps 500 \
    --reward_model_path outputs/run_*/best_model \
    --batch_size 4 \
    --group_size 4
```

---

## References

- [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862)
- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [GPT-2 Model](https://huggingface.co/openai-community/gpt2)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath: GRPO](https://arxiv.org/abs/2402.03300)

