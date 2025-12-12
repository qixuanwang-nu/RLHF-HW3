## Part 4: Analysis and Evaluation (PPO vs GRPO vs DPO)

This write-up uses the evaluation artifacts under `evaluation/run_20251212_063805/` and the reward-model evaluation under `outputs/run_20251210_063751/`.

### 4.1 Quantitative evaluation

#### Setup

- **Reference model**: `openai-community/gpt2`
- **Reward model**: `outputs/run_20251210_063751/best_model` (trained separately; see Part 1 results)
- **Policy checkpoints evaluated** (paths recorded in the run config):
  - PPO: `outputs/policy/run_20251212_041156/ppo/model`
  - GRPO: `outputs/policy/run_20251212_042432/grpo/model`
  - DPO: `outputs/policy/run_20251212_013629/dpo/model`
- **Prompt split**: `test`
- **Prompts for reward/KL scoring**: 200
- **Prompts for win-rate**: 120
- **Judge for win-rate**: OpenAI (configured as GPT‑4.1‑mini)

#### Reward-model score distributions (higher is better)

From `evaluation/run_20251212_063805/quantitative.json` (mean reward-model scores over 200 prompts):

- **Reference**: mean = -1.0913
- **PPO**: mean = -0.4763
- **GRPO**: mean = -0.9190
- **DPO**: mean = -0.9569

Interpretation:
- PPO achieved the strongest improvement in reward-model mean score vs the reference model on this run.
- GRPO and DPO improved only modestly vs reference on reward-model mean score.

#### KL drift vs reference (lower is better)

KL here is a **proxy** computed on sampled tokens (see `src/evaluation/model_eval.py`):  
`kl_proxy(x) = exp(x) - 1 - x`, where `x = logπ - logπ_ref`, clamped to [-5, 5], so the per-token proxy is always non-negative and bounded (max ≈ 148.4).

From `evaluation/run_20251212_063805/quantitative.json`:

- **PPO**: KL mean = 2.9076, KL max = 4.0067
- **GRPO**: KL mean = 2.2215, KL max = 142.4132
- **DPO**: KL mean = 12.6471, KL max = 142.4132

Interpretation:
- GRPO and DPO showed **rare but very large** KL spikes on some tokens (near the clamp limit), while PPO had a much smaller KL max on this run.
- DPO had the largest average KL drift.

#### Win-rate vs reference (GPT-4.1-as-judge)

From `evaluation/run_20251212_063805/winrate.json`:

- **PPO vs reference**: win-rate ≈ 0.6583
- **GRPO vs reference**: win-rate ≈ 0.3833
- **DPO vs reference**: win-rate ≈ 0.4750

Notes:
- The stored `winrate.json` includes a small set of sampled judged cases (used for reporting/debugging).
- Some PPO sampled cases are empty/near-empty responses (see qualitative section).

#### Pareto frontier (reward vs KL)

From `evaluation/run_20251212_063805/pareto_frontier.json`, the computed frontier is:

- **reference** (KL=0, reward=-1.0913)
- **grpo** (KL=2.2215, reward=-0.9190)
- **ppo** (KL=2.9076, reward=-0.4763)

Interpretation:
- PPO dominates GRPO in reward score but uses more KL on average.
- DPO is not on the frontier because it has worse reward at much higher KL in this run.

---

### 4.2 Qualitative analysis (failure modes and examples)

The adversarial prompt set and model responses are saved in:
- `evaluation/run_20251212_063805/qualitative_adversarial.json`

#### Failure mode: “blank” PPO outputs / early termination

PPO sometimes produces an empty response (EOS as the first generated token). This shows up directly in the adversarial dump (PPO responses contain `""` for multiple prompts).

Why this happens in RLHF-style training:
- If the reward model strongly penalizes “bad” outputs (toxicity, incoherence, unsafe content), a weak policy can find a shortcut: **say nothing**.
- Without a minimum length constraint or explicit penalty for empty/very short responses, early-EOS is a stable local optimum.

Mitigation implemented in code:
- The generation code uses left-padding correctly.
- A minimum output length (`min_new_tokens`) is supported for PPO/GRPO training (and for evaluation generation) to reduce empty outputs.

#### Failure mode: Sycophancy / excessive agreement

On prompts explicitly asking the assistant to “agree no matter what” or to confirm false statements, several models show signs of:
- agreeing with incorrect premises
- mirroring user framing without correction

This is expected because:
- HH-RLHF preferences contain “helpfulness/harmlessness” signals, but may not perfectly punish sycophancy.
- The base GPT‑2 policy is small and can degrade under optimization pressure.

#### Failure mode: Unsafe / policy-violating content

On weapon-making or hacking prompts:
- Some models still produce unsafe or policy-violating content.

This is expected in a minimal assignment setup:
- We are not using a strong safety-tuned base model, and GPT‑2 is not instruction-aligned.
- The reward model is trained on HH-RLHF preferences, but that does not guarantee full safety compliance.

---

### Part 1 (Reward model) sanity check for downstream use

Reward model final results (from `final_results_reward_model.json` and `outputs/run_20251210_063751/final_results.json`):

- **Best validation accuracy**: ~0.6476
- **Final validation accuracy**: ~0.6463
- **Reward margin (chosen - rejected)**: ~0.5085

Interpretation:
- The reward model is meaningfully better than random (0.5) on preference prediction and shows a positive reward margin, so it is usable as a training signal for PPO/GRPO.
- However, it is still imperfect (~35% errors). Some downstream instabilities and reward hacking behaviors (e.g., empty responses) can happen.


