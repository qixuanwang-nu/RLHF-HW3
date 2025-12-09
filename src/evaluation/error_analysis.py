"""
Part 1.2B: Evaluation and Error Analysis
Evaluates the reward model and performs detailed error analysis.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class ErrorAnalyzer:
    """
    Performs detailed error analysis on reward model predictions.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = None
    ):
        """
        Args:
            model: Trained RewardModel
            tokenizer: Tokenizer used for preprocessing
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def extract_text_components(self, text: str) -> Dict[str, str]:
        """
        Extract prompt and response from raw text.
        """
        last_assistant_idx = text.rfind("\n\nAssistant:")
        
        if last_assistant_idx == -1:
            last_assistant_idx = text.rfind("Assistant:")
            if last_assistant_idx == -1:
                return {"prompt": text, "response": ""}
            prompt = text[:last_assistant_idx].strip()
            response = text[last_assistant_idx + len("Assistant:"):].strip()
        else:
            prompt = text[:last_assistant_idx].strip()
            response = text[last_assistant_idx + len("\n\nAssistant:"):].strip()
        
        return {"prompt": prompt, "response": response}
    
    @torch.no_grad()
    def predict_preferences(
        self,
        chosen_texts: List[str],
        rejected_texts: List[str],
        max_length: int = 512,
        batch_size: int = 8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict preferences for a list of (chosen, rejected) pairs.
        
        Returns:
            Tuple of (chosen_rewards, rejected_rewards, predictions)
            where predictions is 1 if chosen > rejected, 0 otherwise
        """
        all_chosen_rewards = []
        all_rejected_rewards = []
        
        for i in range(0, len(chosen_texts), batch_size):
            batch_chosen = chosen_texts[i:i + batch_size]
            batch_rejected = rejected_texts[i:i + batch_size]
            
            # Tokenize
            chosen_encoded = self.tokenizer(
                batch_chosen,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            rejected_encoded = self.tokenizer(
                batch_rejected,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Move to device
            chosen_ids = chosen_encoded["input_ids"].to(self.device)
            chosen_mask = chosen_encoded["attention_mask"].to(self.device)
            rejected_ids = rejected_encoded["input_ids"].to(self.device)
            rejected_mask = rejected_encoded["attention_mask"].to(self.device)
            
            # Get rewards
            chosen_rewards, _ = self.model.get_reward(chosen_ids, chosen_mask)
            rejected_rewards, _ = self.model.get_reward(rejected_ids, rejected_mask)
            
            all_chosen_rewards.extend(chosen_rewards.squeeze().cpu().numpy().tolist())
            all_rejected_rewards.extend(rejected_rewards.squeeze().cpu().numpy().tolist())
        
        chosen_rewards = np.array(all_chosen_rewards)
        rejected_rewards = np.array(all_rejected_rewards)
        predictions = (chosen_rewards > rejected_rewards).astype(int)
        
        return chosen_rewards, rejected_rewards, predictions
    
    def analyze_errors(
        self,
        dataset: Dataset,
        num_samples: int = None,
        seed: int = 42
    ) -> Dict:
        """
        Perform comprehensive error analysis on a dataset.
        
        Args:
            dataset: HuggingFace Dataset with 'chosen' and 'rejected' columns
            num_samples: Number of samples to analyze (None for all)
            seed: Random seed for sampling
            
        Returns:
            Dictionary containing error analysis results
        """
        np.random.seed(seed)
        
        if num_samples and num_samples < len(dataset):
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            samples = [dataset[int(i)] for i in indices]
        else:
            samples = [dataset[i] for i in range(len(dataset))]
        
        chosen_texts = [s["chosen"] for s in samples]
        rejected_texts = [s["rejected"] for s in samples]
        
        print(f"Analyzing {len(samples)} samples...")
        
        # Get predictions
        chosen_rewards, rejected_rewards, predictions = self.predict_preferences(
            chosen_texts, rejected_texts
        )
        
        # Compute accuracy
        accuracy = predictions.mean()
        
        # Identify errors (where prediction != ground truth)
        # Ground truth is always: chosen should have higher reward
        errors = predictions == 0
        error_indices = np.where(errors)[0]
        correct_indices = np.where(~errors)[0]
        
        print(f"Overall accuracy: {accuracy:.4f}")
        print(f"Number of errors: {len(error_indices)} / {len(samples)}")
        
        # Analyze error characteristics
        error_analysis = {
            "overall_accuracy": float(accuracy),
            "num_errors": int(len(error_indices)),
            "num_correct": int(len(correct_indices)),
            "total_samples": len(samples),
            "error_rate": float(1 - accuracy),
        }
        
        # Detailed error examples
        error_examples = []
        for idx in error_indices:
            i = int(idx)
            chosen_comp = self.extract_text_components(chosen_texts[i])
            rejected_comp = self.extract_text_components(rejected_texts[i])
            
            error_examples.append({
                "index": i,
                "prompt": chosen_comp["prompt"][:500],  # Truncate for readability
                "chosen_response": chosen_comp["response"][:500],
                "rejected_response": rejected_comp["response"][:500],
                "chosen_reward": float(chosen_rewards[i]),
                "rejected_reward": float(rejected_rewards[i]),
                "reward_diff": float(chosen_rewards[i] - rejected_rewards[i]),
                "chosen_length": len(chosen_comp["response"]),
                "rejected_length": len(rejected_comp["response"]),
            })
        
        error_analysis["error_examples"] = error_examples[:50]  # Keep top 50 for report
        
        # Analyze patterns in errors
        error_patterns = self._analyze_error_patterns(
            error_examples, chosen_texts, rejected_texts, 
            chosen_rewards, rejected_rewards, error_indices
        )
        error_analysis["patterns"] = error_patterns
        
        # Reward distribution statistics
        error_analysis["reward_stats"] = {
            "chosen_mean": float(np.mean(chosen_rewards)),
            "chosen_std": float(np.std(chosen_rewards)),
            "rejected_mean": float(np.mean(rejected_rewards)),
            "rejected_std": float(np.std(rejected_rewards)),
            "reward_diff_mean": float(np.mean(chosen_rewards - rejected_rewards)),
            "reward_diff_std": float(np.std(chosen_rewards - rejected_rewards)),
            "margin_on_correct": float(np.mean(chosen_rewards[correct_indices] - rejected_rewards[correct_indices])) if len(correct_indices) > 0 else 0,
            "margin_on_errors": float(np.mean(chosen_rewards[error_indices] - rejected_rewards[error_indices])) if len(error_indices) > 0 else 0,
        }
        
        return error_analysis
    
    def _analyze_error_patterns(
        self,
        error_examples: List[Dict],
        chosen_texts: List[str],
        rejected_texts: List[str],
        chosen_rewards: np.ndarray,
        rejected_rewards: np.ndarray,
        error_indices: np.ndarray
    ) -> Dict:
        """
        Analyze patterns in the errors.
        """
        patterns = {
            "length_bias": {},
            "reward_magnitude": {},
            "confidence_analysis": {},
        }
        
        if len(error_examples) == 0:
            return patterns
        
        # Length bias analysis
        error_chosen_longer = 0
        error_rejected_longer = 0
        error_similar_length = 0
        
        for ex in error_examples:
            len_diff = ex["chosen_length"] - ex["rejected_length"]
            if abs(len_diff) < 50:
                error_similar_length += 1
            elif len_diff > 0:
                error_chosen_longer += 1
            else:
                error_rejected_longer += 1
        
        patterns["length_bias"] = {
            "error_chosen_longer": error_chosen_longer,
            "error_rejected_longer": error_rejected_longer,
            "error_similar_length": error_similar_length,
            "rejected_longer_error_rate": error_rejected_longer / len(error_examples) if error_examples else 0,
        }
        
        # Analyze if model prefers longer responses incorrectly
        error_lengths = [(ex["chosen_length"], ex["rejected_length"]) for ex in error_examples]
        avg_chosen_len_errors = np.mean([l[0] for l in error_lengths])
        avg_rejected_len_errors = np.mean([l[1] for l in error_lengths])
        
        patterns["length_bias"]["avg_chosen_length_in_errors"] = float(avg_chosen_len_errors)
        patterns["length_bias"]["avg_rejected_length_in_errors"] = float(avg_rejected_len_errors)
        
        # Reward magnitude analysis
        reward_diffs = [ex["reward_diff"] for ex in error_examples]
        patterns["reward_magnitude"] = {
            "mean_incorrect_margin": float(np.mean(reward_diffs)),
            "median_incorrect_margin": float(np.median(reward_diffs)),
            "max_incorrect_margin": float(np.min(reward_diffs)),  # Most confident wrong prediction
            "close_calls": sum(1 for d in reward_diffs if abs(d) < 0.1),  # Within 0.1 of each other
        }
        
        # Confidence analysis
        all_margins = chosen_rewards - rejected_rewards
        patterns["confidence_analysis"] = {
            "low_confidence_errors": sum(1 for i in error_indices if abs(all_margins[i]) < 0.5),
            "high_confidence_errors": sum(1 for i in error_indices if abs(all_margins[i]) >= 1.0),
        }
        
        return patterns
    
    def generate_error_report(
        self,
        error_analysis: Dict,
        output_dir: str = "outputs/evaluation",
        num_examples_to_show: int = 20
    ) -> str:
        """
        Generate a detailed error report.
        
        Args:
            error_analysis: Results from analyze_errors()
            output_dir: Directory to save visualizations
            num_examples_to_show: Number of error examples to include in report
            
        Returns:
            Formatted report string
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report = []
        report.append("=" * 80)
        report.append("REWARD MODEL ERROR ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Overall metrics
        report.append(f"\n{'OVERALL METRICS':^80}")
        report.append("-" * 80)
        report.append(f"Total samples analyzed: {error_analysis['total_samples']}")
        report.append(f"Overall accuracy: {error_analysis['overall_accuracy']:.4f} ({error_analysis['overall_accuracy']*100:.2f}%)")
        report.append(f"Number of errors: {error_analysis['num_errors']}")
        report.append(f"Error rate: {error_analysis['error_rate']:.4f} ({error_analysis['error_rate']*100:.2f}%)")
        
        # Reward statistics
        report.append(f"\n{'REWARD STATISTICS':^80}")
        report.append("-" * 80)
        stats = error_analysis['reward_stats']
        report.append(f"Chosen reward: mean={stats['chosen_mean']:.4f}, std={stats['chosen_std']:.4f}")
        report.append(f"Rejected reward: mean={stats['rejected_mean']:.4f}, std={stats['rejected_std']:.4f}")
        report.append(f"Reward difference: mean={stats['reward_diff_mean']:.4f}, std={stats['reward_diff_std']:.4f}")
        report.append(f"Average margin on correct predictions: {stats['margin_on_correct']:.4f}")
        report.append(f"Average margin on errors: {stats['margin_on_errors']:.4f}")
        
        # Error patterns
        report.append(f"\n{'ERROR PATTERNS ANALYSIS':^80}")
        report.append("-" * 80)
        
        patterns = error_analysis['patterns']
        
        report.append("\n1. LENGTH BIAS IN ERRORS:")
        lb = patterns['length_bias']
        report.append(f"   - Errors where chosen was longer: {lb.get('error_chosen_longer', 0)}")
        report.append(f"   - Errors where rejected was longer: {lb.get('error_rejected_longer', 0)}")
        report.append(f"   - Errors with similar lengths: {lb.get('error_similar_length', 0)}")
        report.append(f"   - Avg chosen length in errors: {lb.get('avg_chosen_length_in_errors', 0):.1f}")
        report.append(f"   - Avg rejected length in errors: {lb.get('avg_rejected_length_in_errors', 0):.1f}")
        
        if lb.get('error_rejected_longer', 0) > lb.get('error_chosen_longer', 0):
            report.append("   ⚠️  Pattern detected: Model may have length bias (prefers longer responses)")
        
        report.append("\n2. CONFIDENCE ANALYSIS:")
        ca = patterns['confidence_analysis']
        report.append(f"   - Low confidence errors (margin < 0.5): {ca.get('low_confidence_errors', 0)}")
        report.append(f"   - High confidence errors (margin >= 1.0): {ca.get('high_confidence_errors', 0)}")
        
        report.append("\n3. REWARD MAGNITUDE ON ERRORS:")
        rm = patterns['reward_magnitude']
        report.append(f"   - Mean incorrect margin: {rm.get('mean_incorrect_margin', 0):.4f}")
        report.append(f"   - Close calls (|margin| < 0.1): {rm.get('close_calls', 0)}")
        
        # Detailed error examples
        report.append(f"\n{'DETAILED ERROR EXAMPLES':^80}")
        report.append("=" * 80)
        
        examples = error_analysis['error_examples'][:num_examples_to_show]
        
        for i, ex in enumerate(examples):
            report.append(f"\n--- Error Example {i+1} ---")
            report.append(f"Chosen Reward: {ex['chosen_reward']:.4f}")
            report.append(f"Rejected Reward: {ex['rejected_reward']:.4f}")
            report.append(f"Reward Difference: {ex['reward_diff']:.4f}")
            report.append(f"Chosen Length: {ex['chosen_length']} chars")
            report.append(f"Rejected Length: {ex['rejected_length']} chars")
            
            report.append(f"\n[PROMPT]")
            report.append(ex['prompt'][:300] + "..." if len(ex['prompt']) > 300 else ex['prompt'])
            
            report.append(f"\n[CHOSEN RESPONSE] (Ground truth: Better)")
            response = ex['chosen_response']
            report.append(response[:400] + "..." if len(response) > 400 else response)
            
            report.append(f"\n[REJECTED RESPONSE] (Ground truth: Worse)")
            response = ex['rejected_response']
            report.append(response[:400] + "..." if len(response) > 400 else response)
            
            report.append("-" * 40)
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = os.path.join(output_dir, "error_analysis_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        
        print(f"\nError report saved to {report_path}")
        
        # Save full analysis as JSON
        json_path = os.path.join(output_dir, "error_analysis.json")
        with open(json_path, "w") as f:
            # Make a copy without the full examples for JSON
            json_analysis = {k: v for k, v in error_analysis.items() if k != 'error_examples'}
            json_analysis['num_error_examples_saved'] = len(error_analysis.get('error_examples', []))
            json.dump(json_analysis, f, indent=2)
        
        return report_text
    
    def visualize_errors(
        self,
        error_analysis: Dict,
        output_dir: str = "outputs/evaluation"
    ):
        """
        Create visualizations for error analysis.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Reward Model Error Analysis", fontsize=14, fontweight='bold')
        
        stats = error_analysis['reward_stats']
        patterns = error_analysis['patterns']
        
        # 1. Accuracy pie chart
        ax1 = axes[0, 0]
        sizes = [error_analysis['num_correct'], error_analysis['num_errors']]
        labels = [f'Correct\n({error_analysis["num_correct"]})', 
                  f'Errors\n({error_analysis["num_errors"]})']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0, 0.05)
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, 
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
        ax1.set_title(f"Overall Accuracy: {error_analysis['overall_accuracy']*100:.1f}%")
        
        # 2. Length bias in errors
        ax2 = axes[0, 1]
        lb = patterns['length_bias']
        categories = ['Chosen Longer', 'Rejected Longer', 'Similar Length']
        values = [
            lb.get('error_chosen_longer', 0),
            lb.get('error_rejected_longer', 0),
            lb.get('error_similar_length', 0)
        ]
        colors = ['#3498db', '#e74c3c', '#95a5a6']
        bars = ax2.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)
        ax2.set_ylabel("Number of Errors")
        ax2.set_title("Length Distribution in Error Cases")
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(val), ha='center', va='bottom', fontweight='bold')
        
        # 3. Reward margin comparison
        ax3 = axes[1, 0]
        categories = ['Correct Predictions', 'Incorrect Predictions']
        margins = [stats['margin_on_correct'], stats['margin_on_errors']]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax3.bar(categories, margins, color=colors, edgecolor='white', linewidth=1.5)
        ax3.set_ylabel("Average Reward Margin")
        ax3.set_title("Reward Margin: Correct vs Incorrect")
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, margins):
            ax3.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.1,
                    f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')
        
        # 4. Confidence analysis
        ax4 = axes[1, 1]
        ca = patterns['confidence_analysis']
        categories = ['Low Confidence\n(margin < 0.5)', 'High Confidence\n(margin >= 1.0)']
        values = [ca.get('low_confidence_errors', 0), ca.get('high_confidence_errors', 0)]
        colors = ['#f39c12', '#9b59b6']
        bars = ax4.bar(categories, values, color=colors, edgecolor='white', linewidth=1.5)
        ax4.set_ylabel("Number of Errors")
        ax4.set_title("Error Distribution by Confidence Level")
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_analysis_plots.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/error_analysis_plots.png")


def run_error_analysis(
    model,
    tokenizer,
    dataset: Dataset,
    num_samples: int = 1000,
    num_examples_in_report: int = 20,
    output_dir: str = "outputs/evaluation"
) -> Dict:
    """
    Run complete error analysis pipeline.
    
    Args:
        model: Trained RewardModel
        tokenizer: Tokenizer
        dataset: Validation dataset
        num_samples: Number of samples to analyze
        num_examples_in_report: Number of error examples in the report
        output_dir: Output directory
        
    Returns:
        Error analysis results dictionary
    """
    analyzer = ErrorAnalyzer(model, tokenizer)
    
    # Run analysis
    error_analysis = analyzer.analyze_errors(dataset, num_samples=num_samples)
    
    # Generate report
    report = analyzer.generate_error_report(
        error_analysis, 
        output_dir=output_dir,
        num_examples_to_show=num_examples_in_report
    )
    print(report)
    
    # Create visualizations
    analyzer.visualize_errors(error_analysis, output_dir=output_dir)
    
    return error_analysis


if __name__ == "__main__":
    print("Error analysis module loaded. Use run_error_analysis() after training.")

