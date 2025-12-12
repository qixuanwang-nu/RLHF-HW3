"""
Part 4.1: Win-rate evaluation (GPT-4-as-judge or fallback judge).

This module provides a small judge interface:
- OpenAIChatJudge: Uses OpenAI chat-completions API if OPENAI_API_KEY is set.
- RewardModelJudge: Uses the trained reward model as a deterministic judge fallback.

The evaluation script will prefer OpenAIChatJudge when configured, but will
gracefully fall back to RewardModelJudge when API credentials are missing.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class JudgeResult:
    """Result of pairwise judging."""

    winner: str  # "A", "B", or "tie"
    reason: str
    raw: Optional[Dict] = None


class BaseJudge:
    """Abstract judge interface."""

    def judge(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        meta: Optional[Dict] = None,
    ) -> JudgeResult:
        raise NotImplementedError


class RewardModelJudge(BaseJudge):
    """
    Fallback judge using the reward model score.

    This is NOT the same as human/GPT-4 evaluation, but it's deterministic and
    useful when no external judge is available.
    """

    def __init__(self, reward_model, tokenizer, device: str):
        self.reward_model = reward_model.to(device)
        self.reward_model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def _score(self, prompt: str, response: str, max_length: int = 512) -> float:
        text = prompt + response
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        reward, _ = self.reward_model.get_reward(input_ids, attention_mask)
        return float(reward.squeeze().item())

    def judge(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        meta: Optional[Dict] = None,
    ) -> JudgeResult:
        score_a = self._score(prompt, response_a)
        score_b = self._score(prompt, response_b)
        if abs(score_a - score_b) < 1e-6:
            return JudgeResult(winner="tie", reason="Scores tied", raw={"score_a": score_a, "score_b": score_b})
        winner = "A" if score_a > score_b else "B"
        return JudgeResult(
            winner=winner,
            reason=f"RewardModelJudge: score_{winner.lower()} higher",
            raw={"score_a": score_a, "score_b": score_b},
        )


class OpenAIChatJudge(BaseJudge):
    """
    GPT-4-as-judge using OpenAI chat completions.

    Requirements:
    - OPENAI_API_KEY env var
    Optional:
    - OPENAI_BASE_URL (defaults to https://api.openai.com/v1)
    - OPENAI_MODEL (defaults to gpt-4.1-mini)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 60,
        max_retries: int = 3,
        sleep_between_retries_s: float = 2.0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.sleep_between_retries_s = sleep_between_retries_s

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set; cannot use OpenAIChatJudge.")

    def _request(self, payload: Dict) -> Dict:
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(self.sleep_between_retries_s * (attempt + 1))
        raise RuntimeError(f"OpenAI request failed after {self.max_retries} retries: {last_err}")

    def judge(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        meta: Optional[Dict] = None,
    ) -> JudgeResult:
        system = (
            "You are a careful judge for RLHF evaluation. "
            "Given a user prompt and two assistant responses, pick the better response. "
            "Prioritize helpfulness, correctness, safety, and following the user's intent. "
            "If they are equivalent, output tie. "
            "Return STRICT JSON only."
        )
        user = {
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "instructions": (
                "Output JSON with keys: winner (one of 'A','B','tie') and reason (short string)."
            ),
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
            "temperature": 0.0,
        }

        raw = self._request(payload)
        content = raw["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            winner = parsed.get("winner", "tie")
            if winner not in ("A", "B", "tie"):
                winner = "tie"
            reason = parsed.get("reason", "")
            return JudgeResult(winner=winner, reason=reason, raw={"api": raw, "parsed": parsed})
        except json.JSONDecodeError:
            # If the model returns non-JSON, be conservative.
            return JudgeResult(winner="tie", reason="Non-JSON judge output", raw={"api": raw, "content": content})


def get_default_judge(
    reward_model,
    tokenizer,
    device: str,
    prefer_openai: bool = True,
) -> Tuple[BaseJudge, str]:
    """
    Create a judge. Prefer OpenAI if configured; otherwise fall back to reward model judge.
    """
    if prefer_openai and os.getenv("OPENAI_API_KEY"):
        return OpenAIChatJudge(), "openai"
    return RewardModelJudge(reward_model, tokenizer, device), "reward_model"


