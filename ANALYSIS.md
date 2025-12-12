## Part 4: Analysis and Evaluation (PPO vs GRPO vs DPO)

This write-up uses the evaluation artifacts under `evaluation/run_20251212_081643/` and the reward-model evaluation under `outputs/run_20251210_063751/`.

### 4.1 Quantitative evaluation

#### Setup

- **Reference model**: `openai-community/gpt2`
- **Reward model**: `outputs/run_20251210_063751/best_model` (trained separately; see Part 1 results)
- **Policy checkpoints evaluated** (paths recorded in the run config):
  - PPO: `outputs/policy/run_20251212_072959/ppo/model`
  - GRPO: `outputs/policy/run_20251212_074636/grpo/model`
  - DPO: `outputs/policy/run_20251212_013629/dpo/model`
- **Prompt split**: `test`
- **Prompts for reward/KL scoring**: 200
- **Prompts for win-rate**: 120
- **Judge for win-rate**: OpenAI (configured as GPT‑4.1‑mini)

#### Reward-model score distributions (higher is better)

From `evaluation/run_20251212_081643/quantitative.json` (mean reward-model scores over 200 prompts):

- **Reference**: mean = -1.0588
- **PPO**: mean = -1.0811
- **GRPO**: mean = -0.8713
- **DPO**: mean = -1.0285

Interpretation:
- GRPO achieved the strongest improvement in reward-model mean score vs the reference model on this run.
- PPO slightly regressed vs the reference model on reward-model mean score.

#### KL drift vs reference (lower is better)

KL here is a **proxy** computed on sampled tokens (see `src/evaluation/model_eval.py`):  
`kl_proxy(x) = exp(x) - 1 - x`, where `x = logπ - logπ_ref`, clamped to [-5, 5], so the per-token proxy is always non-negative and bounded (max ≈ 148.4).

From `evaluation/run_20251212_081643/quantitative.json`:

- **PPO**: KL mean = 8.5415, KL max = 142.4132
- **GRPO**: KL mean = 2.1972, KL max = 142.4132
- **DPO**: KL mean = 11.3321, KL max = 142.4132

Interpretation:
- All trained policies show rare but very large KL spikes on some tokens (near the clamp limit), and PPO/DPO have high average KL drift in this run.

#### Win-rate vs reference (GPT-4.1-as-judge)

From `evaluation/run_20251212_081643/winrate.json`:

- **PPO vs reference**: win-rate ≈ 0.0333 (wins=4, losses=33, ties=83)
- **GRPO vs reference**: win-rate ≈ 0.4833 (wins=58, losses=4, ties=58)
- **DPO vs reference**: win-rate ≈ 0.4667 (wins=56, losses=1, ties=63)

Notes:
- The stored `winrate.json` includes a small set of sampled judged cases (used for reporting/debugging).
- Some PPO sampled cases are empty/near-empty responses (see qualitative section).

#### Pareto frontier (reward vs KL)

From `evaluation/run_20251212_081643/pareto_frontier.json`, the computed frontier is:

- **reference** (KL=0, reward=-1.0588)
- **grpo** (KL=2.1972, reward=-0.8713)

Interpretation:
- GRPO is the only trained method on the frontier for this run (best reward improvement at relatively low KL).
- PPO and DPO are not on the frontier due to higher KL without better reward.

---

### 4.2 Qualitative analysis (failure modes and examples)

The adversarial prompt set and model responses are saved in:
- `evaluation/run_20251212_081643/qualitative_adversarial.json`

#### Failure mode: PPO degeneration (repetition / low-quality text)

In this run, PPO frequently produces highly repetitive or low-information text (e.g., “happiness … happiness …”), which shows up clearly in:
- `evaluation/run_20251212_081643/qualitative_adversarial.json` (PPO responses)
- `samples/run_20251212_081643/ppo.json`

Why this happens in RLHF-style training:
- A small base model (GPT‑2) can collapse into repetitive token loops under optimization pressure.
- If the reward model is imperfect, the policy can exploit spurious reward correlations (“reward hacking”), even if the text is low quality.

Mitigation implemented in code:
- The generation code uses left-padding correctly.
- A minimum output length (`min_new_tokens`) is supported for PPO/GRPO training (and for evaluation generation) to reduce empty outputs, though it does not by itself prevent repetition.

#### Training curves and reward–KL trade-offs (what to look for)

The evaluation run also saves:
- `evaluation/run_20251212_081643/training_curves.png` (performance, KL/proxy, and loss over steps)
- `evaluation/run_20251212_081643/training_reward_vs_kl.png` (scatter of per-step performance vs KL/proxy)

Key observations from these plots:
- **PPO** shows **high KL drift** (points extend to much larger KL values) while performance remains mostly negative; this matches the very low win-rate and the repetitive text degeneration in samples.
- **GRPO** stays in a **lower-KL band** (tight cluster at low KL) while reaching the best reward-model mean score among the trained methods in this run, consistent with being the only method on the Pareto frontier.
- **DPO** shows relatively **low implicit reward-margin scale** but still non-trivial drift (implicit KL proxy), and its win-rate lands between GRPO and PPO in this run.

#### Alignment achieved (what changed qualitatively)

- **GRPO**: best overall alignment in this run by the assignment metrics (win-rate near 0.48 vs reference, improved reward-model mean, low KL on average). It still shows failure cases on adversarial prompts (sycophancy/unsafe content), reflecting limitations of GPT‑2 and the reward signal.
- **DPO**: improves over reference on some judged prompts (win-rate ~0.47) but does not improve reward-model mean as much as GRPO here, and shows higher KL drift.
- **PPO**: unstable in this run—collapsed into repetitive, low-quality outputs and achieved an extremely low win-rate.

#### Computational efficiency (expected + logged signals)

This codebase tracks efficiency in training:
- **GRPO** logs `generation_time`, `update_time`, and peak memory in `src/algorithms/grpo.py`.
- **DPO** logs `step_time` and peak memory in `src/algorithms/dpo.py`.

Method-level expectations (consistent with implementation):
- **GRPO** is typically more expensive per step than PPO at fixed `batch_size` because it samples `group_size` responses per prompt, but it avoids a value head and can be competitive in stability.
- **DPO** is usually the simplest/most stable optimization loop (supervised), but depends heavily on the preference dataset quality and can drift from the reference unless `beta` and LR are tuned.

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


