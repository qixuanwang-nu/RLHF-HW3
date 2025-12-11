# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `src/`: `data/` (exploration, preprocessing, dataset utilities), `models/` (reward model and trainer), `evaluation/` (error analysis), and `algorithms/` (PPO/GRPO/DPO logic used by policy training).
- Entry points: `explore_dataset.py` (HH-RLHF dataset stats/plots), `train_reward_model.py` (reward model training/eval), and `train_policy.py` (policy optimization comparisons).
- Outputs land under `outputs/` (e.g., `outputs/exploration/` for plots, `outputs/run_<timestamp>/` for reward model artifacts, `outputs/policy/` for policy runs). Avoid committing generated artifacts.

## Setup, Build, Test, and Development Commands
- Create an isolated env: `python -m venv .venv && source .venv/bin/activate`.
- Install deps: `pip install -r requirements.txt` (GPU-accelerated PyTorch recommended).
- Quick data pass: `python explore_dataset.py` (writes plots/stats to `outputs/exploration/`).
- Reward model smoke test: `python train_reward_model.py --epochs 1 --batch_size 4 --subset_size 200` (fast CPU-friendly check; add `--save_model` for artifacts).
- Policy training sample: `python train_policy.py --method ppo --steps 100 --batch_size 2` (use `--method grpo|dpo|both|all` to compare). Point `--reward_model_path` to a saved model when available.
- Use `CUDA_VISIBLE_DEVICES` to target GPUs; set seeds via `--seed` for reproducibility.

## Coding Style & Naming Conventions
- Python, PEP8, 4-space indentation; prefer type hints on public functions and clear docstrings for scripts/classes.
- Snake_case for functions/variables/filenames; PascalCase for classes; align new argparse flags with existing naming patterns.
- Favor small, testable helpers inside `src/` rather than expanding script files; keep plotting/logging behind optional flags where feasible.

## Testing Guidelines
- No formal test suite yet; validate changes with fast runs (`--subset_size` for data, low `--steps` for policy). Capture key metrics (loss/accuracy/reward margin) from stdout and `outputs/`.
- When adding logic, consider lightweight unit tests under `tests/` with `pytest` (e.g., dataset splitting, reward margin computation) and ensure deterministic seeds.

## Data, Outputs, and Configuration
- HH-RLHF downloads cache via `datasets`; do not commit caches or large models. Prune `outputs/` before pushing; keep configs/plots only if illustrative.
- W&B is available; keep API keys out of tracked files and prefer environment variables.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope (e.g., `feat: add dpo hinge loss`, `fix: guard empty preference pairs`). Include user-facing flags or metric shifts in the body when relevant.
- PRs: link issues, describe goal, commands run, and notable metrics (val accuracy/reward margin). Add screenshots of plots when they illustrate behavior changes.
