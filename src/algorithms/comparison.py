"""
Part 2.2B: PPO vs GRPO Comparison Analysis
Compares training stability, convergence speed, computational efficiency, and sample quality.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats


@dataclass
class ComparisonMetrics:
    """Container for comparison metrics between PPO and GRPO."""
    
    # Training stability
    ppo_loss_variance: float = 0.0
    grpo_loss_variance: float = 0.0
    ppo_reward_variance: float = 0.0
    grpo_reward_variance: float = 0.0
    
    # Convergence
    ppo_steps_to_threshold: int = -1
    grpo_steps_to_threshold: int = -1
    ppo_final_reward: float = 0.0
    grpo_final_reward: float = 0.0
    
    # Efficiency
    ppo_time_per_step: float = 0.0
    grpo_time_per_step: float = 0.0
    ppo_memory_peak: float = 0.0
    grpo_memory_peak: float = 0.0
    ppo_samples_per_step: int = 0
    grpo_samples_per_step: int = 0
    
    # Quality
    ppo_kl_final: float = 0.0
    grpo_kl_final: float = 0.0


def compute_training_stability(stats: List[Dict]) -> Dict[str, float]:
    """
    Compute training stability metrics.
    
    Args:
        stats: List of training statistics dictionaries
        
    Returns:
        Dictionary with stability metrics
    """
    if not stats:
        return {"loss_variance": 0, "reward_variance": 0, "loss_trend": 0}
    
    losses = [s.get("total_loss", s.get("loss", 0)) for s in stats]
    rewards = [s.get("reward_mean", 0) for s in stats]
    
    # Variance (lower is more stable)
    loss_variance = np.var(losses) if len(losses) > 1 else 0
    reward_variance = np.var(rewards) if len(rewards) > 1 else 0
    
    # Compute trend (should be decreasing for loss, increasing for reward)
    if len(losses) >= 2:
        slope_result = scipy_stats.linregress(range(len(losses)), losses)
        loss_slope = slope_result.slope
    else:
        loss_slope = 0
    
    # Compute oscillation (number of sign changes in gradient)
    loss_diffs = np.diff(losses)
    sign_changes = np.sum(np.diff(np.sign(loss_diffs)) != 0)
    oscillation = sign_changes / max(len(loss_diffs) - 1, 1)
    
    return {
        "loss_variance": float(loss_variance),
        "reward_variance": float(reward_variance),
        "loss_trend": float(loss_slope),
        "oscillation": float(oscillation),
    }


def compute_convergence_speed(
    stats: List[Dict],
    reward_threshold: float = 0.0,
    window_size: int = 10
) -> Dict[str, float]:
    """
    Compute convergence speed metrics.
    
    Args:
        stats: Training statistics
        reward_threshold: Reward threshold to consider "converged"
        window_size: Window for smoothing
        
    Returns:
        Convergence metrics
    """
    if not stats:
        return {"steps_to_threshold": -1, "final_reward": 0, "convergence_rate": 0}
    
    rewards = [s.get("reward_mean", 0) for s in stats]
    
    # Smooth rewards
    if len(rewards) >= window_size:
        smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    else:
        smoothed = rewards
    
    # Find first step where smoothed reward exceeds threshold
    steps_to_threshold = -1
    for i, r in enumerate(smoothed):
        if r >= reward_threshold:
            steps_to_threshold = i + window_size // 2
            break
    
    # Final reward (average of last 10%)
    final_window = max(len(rewards) // 10, 1)
    final_reward = np.mean(rewards[-final_window:])
    
    # Convergence rate (reward improvement per step)
    if len(rewards) > 1:
        convergence_rate = (rewards[-1] - rewards[0]) / len(rewards)
    else:
        convergence_rate = 0
    
    return {
        "steps_to_threshold": steps_to_threshold,
        "final_reward": float(final_reward),
        "convergence_rate": float(convergence_rate),
        "max_reward": float(max(rewards)),
    }


def compute_efficiency_metrics(stats: List[Dict]) -> Dict[str, float]:
    """
    Compute computational efficiency metrics.
    
    Args:
        stats: Training statistics
        
    Returns:
        Efficiency metrics
    """
    if not stats:
        return {"time_per_step": 0, "memory_peak": 0, "samples_per_step": 0}
    
    # Time per step
    times = [s.get("step_time", 0) for s in stats]
    avg_time = np.mean(times) if times else 0
    
    # Memory usage
    memory = [s.get("peak_memory_gb", 0) for s in stats]
    peak_memory = max(memory) if memory else 0
    
    # Samples per step
    samples = [s.get("num_responses", s.get("batch_size", 1)) for s in stats]
    avg_samples = np.mean(samples) if samples else 0
    
    # Generation vs update time breakdown
    gen_times = [s.get("generation_time", 0) for s in stats]
    update_times = [s.get("update_time", 0) for s in stats]
    
    return {
        "time_per_step": float(avg_time),
        "memory_peak": float(peak_memory),
        "samples_per_step": float(avg_samples),
        "generation_time_ratio": float(np.mean(gen_times) / (avg_time + 1e-8)),
        "update_time_ratio": float(np.mean(update_times) / (avg_time + 1e-8)),
    }


def compute_sample_quality(stats: List[Dict]) -> Dict[str, float]:
    """
    Compute sample quality metrics.
    
    Args:
        stats: Training statistics
        
    Returns:
        Quality metrics
    """
    if not stats:
        return {"final_kl": 0, "final_entropy": 0, "reward_improvement": 0}
    
    # KL divergence from reference
    kl_values = [s.get("kl_divergence", 0) for s in stats]
    final_kl = kl_values[-1] if kl_values else 0
    
    # Entropy
    entropies = [s.get("entropy", 0) for s in stats]
    final_entropy = entropies[-1] if entropies else 0
    
    # Reward improvement
    rewards = [s.get("reward_mean", 0) for s in stats]
    reward_improvement = rewards[-1] - rewards[0] if len(rewards) > 1 else 0
    
    return {
        "final_kl": float(final_kl),
        "final_entropy": float(final_entropy),
        "reward_improvement": float(reward_improvement),
        "avg_kl": float(np.mean(kl_values)) if kl_values else 0,
    }


def compare_algorithms(
    ppo_stats: List[Dict] = None,
    grpo_stats: List[Dict] = None,
    dpo_stats: List[Dict] = None,
    reward_threshold: float = 0.0
) -> Dict:
    """
    Comprehensive comparison between PPO, GRPO, and DPO.
    
    Args:
        ppo_stats: PPO training statistics (optional)
        grpo_stats: GRPO training statistics (optional)
        dpo_stats: DPO training statistics (optional)
        reward_threshold: Threshold for convergence comparison
        
    Returns:
        Comparison results
    """
    comparison = {}
    methods = []
    
    if ppo_stats:
        comparison["ppo"] = {
            "stability": compute_training_stability(ppo_stats),
            "convergence": compute_convergence_speed(ppo_stats, reward_threshold),
            "efficiency": compute_efficiency_metrics(ppo_stats),
            "quality": compute_sample_quality(ppo_stats),
        }
        methods.append("ppo")
    
    if grpo_stats:
        comparison["grpo"] = {
            "stability": compute_training_stability(grpo_stats),
            "convergence": compute_convergence_speed(grpo_stats, reward_threshold),
            "efficiency": compute_efficiency_metrics(grpo_stats),
            "quality": compute_sample_quality(grpo_stats),
        }
        methods.append("grpo")
    
    if dpo_stats:
        comparison["dpo"] = {
            "stability": compute_training_stability(dpo_stats),
            "convergence": compute_convergence_speed(dpo_stats, reward_threshold),
            "efficiency": compute_efficiency_metrics(dpo_stats),
            "quality": compute_sample_quality(dpo_stats),
        }
        methods.append("dpo")
    
    # Compute relative comparisons if at least 2 methods
    if len(methods) >= 2:
        # Find winners for each metric
        loss_variances = {m: comparison[m]["stability"]["loss_variance"] for m in methods}
        final_rewards = {m: comparison[m]["convergence"]["final_reward"] for m in methods}
        step_times = {m: comparison[m]["efficiency"]["time_per_step"] for m in methods}
        memory_peaks = {m: comparison[m]["efficiency"]["memory_peak"] for m in methods}
        
        comparison["relative"] = {
            "stability_winner": min(loss_variances, key=loss_variances.get).upper(),
            "convergence_winner": max(final_rewards, key=final_rewards.get).upper(),
            "efficiency_winner": min(step_times, key=step_times.get).upper(),
            "memory_winner": min(memory_peaks, key=memory_peaks.get).upper(),
            "methods_compared": [m.upper() for m in methods],
        }
        
        # Pairwise ratios
        if "ppo" in methods and "grpo" in methods:
            comparison["relative"]["ppo_vs_grpo_time_ratio"] = (
                comparison["grpo"]["efficiency"]["time_per_step"] / 
                (comparison["ppo"]["efficiency"]["time_per_step"] + 1e-8)
            )
        
        if "ppo" in methods and "dpo" in methods:
            comparison["relative"]["ppo_vs_dpo_time_ratio"] = (
                comparison["dpo"]["efficiency"]["time_per_step"] / 
                (comparison["ppo"]["efficiency"]["time_per_step"] + 1e-8)
            )
        
        if "grpo" in methods and "dpo" in methods:
            comparison["relative"]["grpo_vs_dpo_time_ratio"] = (
                comparison["dpo"]["efficiency"]["time_per_step"] / 
                (comparison["grpo"]["efficiency"]["time_per_step"] + 1e-8)
            )
    
    return comparison


def visualize_comparison(
    ppo_stats: List[Dict] = None,
    grpo_stats: List[Dict] = None,
    dpo_stats: List[Dict] = None,
    output_dir: str = "outputs/comparison"
):
    """
    Create comprehensive comparison visualizations for 2 or 3 methods.
    
    Args:
        ppo_stats: PPO training statistics (optional)
        grpo_stats: GRPO training statistics (optional)
        dpo_stats: DPO training statistics (optional)
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Colors for each method
    colors = {
        'PPO': '#3498db',
        'GRPO': '#e74c3c',
        'DPO': '#2ecc71'
    }
    
    # Collect available methods
    methods_data = {}
    if ppo_stats:
        methods_data['PPO'] = ppo_stats
    if grpo_stats:
        methods_data['GRPO'] = grpo_stats
    if dpo_stats:
        methods_data['DPO'] = dpo_stats
    
    methods = list(methods_data.keys())
    num_methods = len(methods)
    
    if num_methods < 2:
        print("Need at least 2 methods to compare")
        return
    
    title = " vs ".join(methods) + " Comparison"
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Reward/Accuracy curves
    ax1 = axes[0, 0]
    for method, stats in methods_data.items():
        # DPO uses accuracy, PPO/GRPO use reward_mean
        if method == 'DPO':
            values = [s.get("accuracy", s.get("reward_mean", 0)) for s in stats]
            label = f'{method} (Accuracy)'
        else:
            values = [s.get("reward_mean", 0) for s in stats]
            label = f'{method} (Reward)'
        ax1.plot(values, label=label, color=colors[method], linewidth=2)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Value")
    ax1.set_title("Performance During Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss curves
    ax2 = axes[0, 1]
    for method, stats in methods_data.items():
        losses = [s.get("total_loss", s.get("loss", 0)) for s in stats]
        ax2.plot(losses, label=method, color=colors[method], linewidth=2)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. KL divergence / Log ratios
    ax3 = axes[0, 2]
    for method, stats in methods_data.items():
        if method == 'DPO':
            values = [s.get("chosen_log_ratio", s.get("kl_divergence", 0)) for s in stats]
            label = f'{method} (Log Ratio)'
        else:
            values = [s.get("kl_divergence", 0) for s in stats]
            label = f'{method} (KL)'
        ax3.plot(values, label=label, color=colors[method], linewidth=2)
    ax3.set_xlabel("Training Step")
    ax3.set_ylabel("Value")
    ax3.set_title("KL Divergence / Log Ratios")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Time per step
    ax4 = axes[1, 0]
    avg_times = []
    for method, stats in methods_data.items():
        times = [s.get("step_time", 0) for s in stats]
        avg_times.append(np.mean(times) if times else 0)
    
    bars = ax4.bar(methods, avg_times, color=[colors[m] for m in methods], 
                   edgecolor='white', linewidth=2)
    ax4.set_ylabel("Time (seconds)")
    ax4.set_title("Average Time per Step")
    for bar, val in zip(bars, avg_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # 5. Final performance comparison
    ax5 = axes[1, 1]
    final_values = []
    for method, stats in methods_data.items():
        if method == 'DPO':
            values = [s.get("accuracy", 0) for s in stats]
        else:
            values = [s.get("reward_mean", 0) for s in stats]
        final_values.append(values[-1] if values else 0)
    
    bars = ax5.bar(methods, final_values, color=[colors[m] for m in methods],
                   edgecolor='white', linewidth=2)
    ax5.set_ylabel("Final Performance")
    ax5.set_title("Final Performance Comparison")
    for bar, val in zip(bars, final_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Overall comparison (normalized)
    ax6 = axes[1, 2]
    
    metric_names = ['Performance', 'Stability', 'Speed']
    
    # Compute normalized scores
    all_scores = {m: [] for m in methods}
    
    for method, stats in methods_data.items():
        # Performance (normalize by max)
        if method == 'DPO':
            perf = [s.get("accuracy", 0) for s in stats][-1] if stats else 0
        else:
            perf = [s.get("reward_mean", 0) for s in stats][-1] if stats else 0
        all_scores[method].append(perf)
        
        # Stability (1 / (1 + variance))
        losses = [s.get("total_loss", s.get("loss", 0)) for s in stats]
        stability = 1 / (1 + np.var(losses)) if losses else 0
        all_scores[method].append(stability)
        
        # Speed (1 / (1 + time))
        times = [s.get("step_time", 0) for s in stats]
        speed = 1 / (1 + np.mean(times)) if times else 0
        all_scores[method].append(speed)
    
    # Normalize each metric across methods
    for i in range(len(metric_names)):
        max_val = max(all_scores[m][i] for m in methods) + 1e-8
        for m in methods:
            all_scores[m][i] = all_scores[m][i] / max_val
    
    x = np.arange(len(metric_names))
    width = 0.8 / num_methods
    
    for i, method in enumerate(methods):
        offset = (i - num_methods/2 + 0.5) * width
        ax6.bar(x + offset, all_scores[method], width, label=method, 
               color=colors[method], edgecolor='white')
    
    ax6.set_ylabel("Normalized Score")
    ax6.set_title("Overall Comparison (Normalized)")
    ax6.set_xticks(x)
    ax6.set_xticklabels(metric_names)
    ax6.legend()
    ax6.set_ylim(0, 1.2)
    
    plt.tight_layout()
    filename = "_vs_".join([m.lower() for m in methods]) + "_comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}/{filename}")


def generate_comparison_report(
    comparison: Dict,
    output_dir: str = "outputs/comparison"
) -> str:
    """
    Generate a detailed comparison report for 2 or 3 methods.
    
    Args:
        comparison: Comparison results from compare_algorithms()
        output_dir: Output directory
        
    Returns:
        Report text
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get methods being compared
    methods = [k for k in comparison.keys() if k not in ['relative']]
    methods_upper = [m.upper() for m in methods]
    
    report = []
    report.append("="*90)
    report.append(f"{' vs '.join(methods_upper)} COMPARISON REPORT".center(90))
    report.append("="*90)
    
    # Training Stability
    report.append(f"\n{'TRAINING STABILITY':^90}")
    report.append("-"*90)
    
    header = f"{'Metric':<25}"
    for m in methods_upper:
        header += f"{m:<20}"
    header += f"{'Winner':<15}"
    report.append(header)
    report.append("-"*90)
    
    # Loss Variance
    row = f"{'Loss Variance':<25}"
    loss_vars = {}
    for m in methods:
        val = comparison[m]["stability"]["loss_variance"]
        loss_vars[m] = val
        row += f"{val:<20.6f}"
    winner = min(loss_vars, key=loss_vars.get).upper()
    row += f"{winner:<15}"
    report.append(row)
    
    # Reward Variance
    row = f"{'Reward Variance':<25}"
    reward_vars = {}
    for m in methods:
        val = comparison[m]["stability"]["reward_variance"]
        reward_vars[m] = val
        row += f"{val:<20.6f}"
    winner = min(reward_vars, key=reward_vars.get).upper()
    row += f"{winner:<15}"
    report.append(row)
    
    # Oscillation
    row = f"{'Oscillation':<25}"
    oscillations = {}
    for m in methods:
        val = comparison[m]["stability"]["oscillation"]
        oscillations[m] = val
        row += f"{val:<20.4f}"
    winner = min(oscillations, key=oscillations.get).upper()
    row += f"{winner:<15}"
    report.append(row)
    
    # Convergence Speed
    report.append(f"\n{'CONVERGENCE SPEED':^90}")
    report.append("-"*90)
    report.append(header)
    report.append("-"*90)
    
    # Final Reward/Performance
    row = f"{'Final Performance':<25}"
    final_rewards = {}
    for m in methods:
        val = comparison[m]["convergence"]["final_reward"]
        final_rewards[m] = val
        row += f"{val:<20.4f}"
    winner = max(final_rewards, key=final_rewards.get).upper()
    row += f"{winner:<15}"
    report.append(row)
    
    # Max Reward
    row = f"{'Max Performance':<25}"
    max_rewards = {}
    for m in methods:
        val = comparison[m]["convergence"]["max_reward"]
        max_rewards[m] = val
        row += f"{val:<20.4f}"
    winner = max(max_rewards, key=max_rewards.get).upper()
    row += f"{winner:<15}"
    report.append(row)
    
    # Convergence Rate
    row = f"{'Convergence Rate':<25}"
    conv_rates = {}
    for m in methods:
        val = comparison[m]["convergence"]["convergence_rate"]
        conv_rates[m] = val
        row += f"{val:<20.6f}"
    winner = max(conv_rates, key=conv_rates.get).upper()
    row += f"{winner:<15}"
    report.append(row)
    
    # Computational Efficiency
    report.append(f"\n{'COMPUTATIONAL EFFICIENCY':^90}")
    report.append("-"*90)
    report.append(header)
    report.append("-"*90)
    
    # Time per Step
    row = f"{'Time per Step (s)':<25}"
    step_times = {}
    for m in methods:
        val = comparison[m]["efficiency"]["time_per_step"]
        step_times[m] = val
        row += f"{val:<20.4f}"
    winner = min(step_times, key=step_times.get).upper()
    row += f"{winner:<15}"
    report.append(row)
    
    # Peak Memory
    row = f"{'Peak Memory (GB)':<25}"
    memory_peaks = {}
    for m in methods:
        val = comparison[m]["efficiency"]["memory_peak"]
        memory_peaks[m] = val
        row += f"{val:<20.4f}"
    winner = min(memory_peaks, key=memory_peaks.get).upper() if any(v > 0 for v in memory_peaks.values()) else "N/A"
    row += f"{winner:<15}"
    report.append(row)
    
    # Sample Quality
    report.append(f"\n{'SAMPLE QUALITY':^90}")
    report.append("-"*90)
    report.append(header)
    report.append("-"*90)
    
    # Final KL
    row = f"{'Final KL/Log Ratio':<25}"
    final_kls = {}
    for m in methods:
        val = comparison[m]["quality"]["final_kl"]
        final_kls[m] = abs(val)
        row += f"{val:<20.6f}"
    winner = min(final_kls, key=final_kls.get).upper()
    row += f"{winner:<15}"
    report.append(row)
    
    # Reward Improvement
    row = f"{'Reward Improvement':<25}"
    improvements = {}
    for m in methods:
        val = comparison[m]["quality"]["reward_improvement"]
        improvements[m] = val
        row += f"{val:<20.4f}"
    winner = max(improvements, key=improvements.get).upper()
    row += f"{winner:<15}"
    report.append(row)
    
    # Summary
    report.append(f"\n{'SUMMARY':^90}")
    report.append("="*90)
    
    if 'relative' in comparison:
        rel = comparison['relative']
        report.append(f"Methods Compared: {', '.join(rel.get('methods_compared', methods_upper))}")
        report.append(f"Stability Winner: {rel.get('stability_winner', 'N/A')}")
        report.append(f"Convergence Winner: {rel.get('convergence_winner', 'N/A')}")
        report.append(f"Efficiency Winner: {rel.get('efficiency_winner', 'N/A')}")
        report.append(f"Memory Winner: {rel.get('memory_winner', 'N/A')}")
        
        # Pairwise ratios
        report.append("\nPairwise Time Ratios:")
        if 'ppo_vs_grpo_time_ratio' in rel:
            report.append(f"  GRPO/PPO: {rel['ppo_vs_grpo_time_ratio']:.2f}x")
        if 'ppo_vs_dpo_time_ratio' in rel:
            report.append(f"  DPO/PPO: {rel['ppo_vs_dpo_time_ratio']:.2f}x")
        if 'grpo_vs_dpo_time_ratio' in rel:
            report.append(f"  DPO/GRPO: {rel['grpo_vs_dpo_time_ratio']:.2f}x")
    
    # Method characteristics
    report.append(f"\n{'METHOD CHARACTERISTICS':^90}")
    report.append("-"*90)
    
    method_chars = {
        'ppo': "PPO: Value function + clipped surrogate + KL penalty. Most stable, higher compute.",
        'grpo': "GRPO: Group sampling + relative advantages. No value fn, lower memory.",
        'dpo': "DPO: Direct preference optimization. No reward model, supervised learning."
    }
    
    for m in methods:
        if m in method_chars:
            report.append(method_chars[m])
    
    report.append("\n" + "="*90)
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    
    # Save as JSON
    json_path = os.path.join(output_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison report saved to {report_path}")
    
    return report_text


if __name__ == "__main__":
    print("Comparison module loaded successfully.")

