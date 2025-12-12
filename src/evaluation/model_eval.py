"""
Part 4: Analysis and Evaluation utilities.

This module implements:
- Text generation for a model on a prompt set
- Reward-model scoring of generated responses
- KL drift metrics against a reference policy
- Simple Pareto frontier computation (reward vs KL)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


@dataclass
class GenConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    max_prompt_length: int = 384


@torch.no_grad()
def generate_responses(
    model,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    device: str,
    gen: GenConfig,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate responses and return:
    - responses: list[str]
    - output_ids: [B, T]
    - attention_mask: [B, T]
    - prompt_lengths: [B] number of prompt tokens
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=gen.max_prompt_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_lengths = attention_mask.sum(dim=1)

    model.eval()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen.max_new_tokens,
        do_sample=True,
        temperature=gen.temperature,
        top_p=gen.top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    full_attention_mask = (out != tokenizer.pad_token_id).long()
    responses: List[str] = []
    for seq, p_len in zip(out, prompt_lengths.tolist()):
        resp_tokens = seq[p_len:]
        responses.append(tokenizer.decode(resp_tokens, skip_special_tokens=True))

    return responses, out, full_attention_mask, prompt_lengths


@torch.no_grad()
def compute_reward_scores(
    reward_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    rewards, _ = reward_model.get_reward(input_ids, attention_mask)
    return rewards.squeeze(-1)


@torch.no_grad()
def compute_token_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    autocast_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Returns per-token logp for the *next-token* targets: [B, T-1]
    """
    if input_ids.is_cuda and autocast_dtype is not None:
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            return token_logps

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return token_logps


def make_response_mask(
    attention_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Create a float mask [B, T] with 1 for response tokens and 0 otherwise.
    """
    response_mask = torch.zeros_like(attention_mask, dtype=torch.float)
    for i, p_len in enumerate(prompt_lengths.tolist()):
        response_mask[i, p_len:] = attention_mask[i, p_len:].float()
    return response_mask


@torch.no_grad()
def compute_kl_proxy_vs_reference(
    policy_model,
    ref_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """
    Compute KL drift proxy on sampled tokens:
      kl_proxy(x) = exp(x) - 1 - x, with x = logπ - logπ_ref, always >= 0.

    We compute on the next-token log-probs; align response_mask accordingly.
    """
    policy_logps = compute_token_log_probs(
        policy_model, input_ids, attention_mask, autocast_dtype=autocast_dtype
    )  # [B, T-1]
    ref_logps = compute_token_log_probs(
        ref_model, input_ids, attention_mask, autocast_dtype=autocast_dtype
    )  # [B, T-1]

    x = policy_logps - ref_logps
    x = torch.clamp(x, -5.0, 5.0)
    kl_proxy = torch.expm1(x) - x  # >= 0

    mask = response_mask[:, 1:].float()
    if mask.size(1) > kl_proxy.size(1):
        mask = mask[:, : kl_proxy.size(1)]
    elif mask.size(1) < kl_proxy.size(1):
        kl_proxy = kl_proxy[:, : mask.size(1)]

    denom = mask.sum().item() + 1e-8
    mean_kl = float((kl_proxy * mask).sum().item() / denom)
    max_kl = float((kl_proxy * mask).max().item()) if denom > 0 else 0.0
    return {"kl_mean": mean_kl, "kl_max": max_kl}


def pareto_frontier(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Compute a simple Pareto frontier for (reward_mean higher better, kl_mean lower better).
    """
    # Non-dominated: no other point has reward >= and kl <= with at least one strict.
    frontier: List[Dict[str, float]] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if (q["reward_mean"] >= p["reward_mean"] and q["kl_mean"] <= p["kl_mean"]) and (
                q["reward_mean"] > p["reward_mean"] or q["kl_mean"] < p["kl_mean"]
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    # Sort by KL ascending
    frontier.sort(key=lambda d: (d["kl_mean"], -d["reward_mean"]))
    return frontier


def summarize_distribution(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def load_causal_lm(model_path_or_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_path_or_name)
    model.to(device)
    model.eval()
    return model


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


