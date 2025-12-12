"""
Export ~20 generated samples per model for submission.

This script is intentionally lightweight and only depends on existing evaluation artifacts:
- evaluation/<run>/winrate.json (contains prompt + model response + reference response)
- evaluation/<run>/qualitative_adversarial.json (contains prompt list + per-model responses)

It writes:
  samples/<run>/
    reference.json
    ppo.json
    grpo.json
    dpo.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def _nonempty(s: str) -> bool:
    return bool(str(s).strip())


def _load_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _take_samples_from_winrate(winrate: Dict, model_name: str, limit: int) -> List[Dict]:
    """
    Each sampled case is:
      {prompt, model, reference, winner, reason}
    """
    out: List[Dict] = []
    cases = (winrate.get(model_name, {}) or {}).get("sampled_cases", []) or []
    for c in cases:
        if len(out) >= limit:
            break
        out.append(
            {
                "source": "winrate_sampled_case",
                "prompt": c.get("prompt", ""),
                "response": c.get("model", ""),
                "reference_response": c.get("reference", ""),
                "judge_winner": c.get("winner", ""),
                "judge_reason": c.get("reason", ""),
            }
        )
    return out


def _take_samples_from_adversarial(adv: Dict, model_name: str, limit: int) -> List[Dict]:
    prompts = adv.get("prompts", []) or []
    responses = (adv.get("responses", {}) or {}).get(model_name, []) or []
    out: List[Dict] = []
    for p, r in zip(prompts, responses):
        if len(out) >= limit:
            break
        out.append(
            {
                "source": "adversarial",
                "prompt": p,
                "response": r,
            }
        )
    return out


def _ensure_20_nonempty(items: List[Dict], fallback: List[Dict], target: int) -> List[Dict]:
    kept: List[Dict] = []
    for x in items:
        if len(kept) >= target:
            break
        kept.append(x)
    if len(kept) >= target:
        return kept

    # supplement with fallback, prefer non-empty
    for x in fallback:
        if len(kept) >= target:
            break
        if _nonempty(x.get("response", "")):
            kept.append(x)
    # if still short, allow empties (but keep deterministic)
    for x in fallback:
        if len(kept) >= target:
            break
        kept.append(x)
    return kept[:target]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export ~20 samples/model from evaluation artifacts.")
    ap.add_argument("--eval_run_dir", type=str, required=True, help="Path like evaluation/run_YYYYMMDD_HHMMSS")
    ap.add_argument("--out_dir", type=str, default="samples", help="Output directory")
    ap.add_argument("--per_model", type=int, default=20, help="Samples per model to export")
    args = ap.parse_args()

    run_dir = args.eval_run_dir.rstrip("/")
    run_name = os.path.basename(run_dir)
    winrate_path = os.path.join(run_dir, "winrate.json")
    adv_path = os.path.join(run_dir, "qualitative_adversarial.json")

    winrate = _load_json(winrate_path)
    adv = _load_json(adv_path)

    models = ["reference", "ppo", "grpo", "dpo"]

    out_run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_run_dir, exist_ok=True)

    # reference: use the reference responses embedded in each method's winrate samples, plus adversarial reference
    ref_items: List[Dict] = []
    for m in ["ppo", "grpo", "dpo"]:
        for c in (winrate.get(m, {}) or {}).get("sampled_cases", []) or []:
            if len(ref_items) >= args.per_model:
                break
            ref_items.append(
                {
                    "source": f"winrate_reference_from_{m}",
                    "prompt": c.get("prompt", ""),
                    "response": c.get("reference", ""),
                }
            )
        if len(ref_items) >= args.per_model:
            break

    ref_fallback = _take_samples_from_adversarial(adv, "reference", args.per_model)
    ref_items = _ensure_20_nonempty(ref_items, ref_fallback, args.per_model)

    with open(os.path.join(out_run_dir, "reference.json"), "w", encoding="utf-8") as f:
        json.dump(ref_items, f, indent=2, ensure_ascii=False)

    for m in ["ppo", "grpo", "dpo"]:
        from_win = _take_samples_from_winrate(winrate, m, args.per_model)
        from_adv = _take_samples_from_adversarial(adv, m, args.per_model)
        items = _ensure_20_nonempty(from_win, from_adv, args.per_model)
        with open(os.path.join(out_run_dir, f"{m}.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Saved samples to: {out_run_dir}")


if __name__ == "__main__":
    main()


