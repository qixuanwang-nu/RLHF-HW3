"""
Part 1.1A: Dataset Exploration and Analysis
Loads and explores the Anthropic HH-RLHF dataset structure.
Analyzes the distribution of preference pairs and identifies biases/patterns.
"""

import os
import json
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from tqdm import tqdm


def load_hh_rlhf_dataset(subset: str = None) -> dict:
    """
    Load the Anthropic HH-RLHF dataset from HuggingFace.
    
    Args:
        subset: Optional subset to load ('harmless-base', 'helpful-base', etc.)
                If None, loads all data.
    
    Returns:
        Dictionary containing train and test splits.
    """
    print("Loading Anthropic HH-RLHF dataset...")
    
    if subset:
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir=subset)
    else:
        dataset = load_dataset("Anthropic/hh-rlhf")
    
    print(f"Dataset loaded successfully!")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    return dataset


def extract_conversation_structure(text: str) -> List[Dict[str, str]]:
    """
    Parse the conversation structure from raw text.
    Format: Human: ... Assistant: ...
    
    Returns:
        List of turns with role and content.
    """
    turns = []
    
    # Split by Human/Assistant markers
    parts = text.split("\n\nHuman: ")
    
    for i, part in enumerate(parts):
        if i == 0 and not part.strip():
            continue
            
        if "Assistant: " in part:
            human_assistant = part.split("\n\nAssistant: ")
            if len(human_assistant) >= 1:
                human_text = human_assistant[0].replace("Human: ", "").strip()
                if human_text:
                    turns.append({"role": "human", "content": human_text})
            if len(human_assistant) >= 2:
                assistant_text = human_assistant[1].strip()
                if assistant_text:
                    turns.append({"role": "assistant", "content": assistant_text})
        else:
            human_text = part.replace("Human: ", "").strip()
            if human_text:
                turns.append({"role": "human", "content": human_text})
    
    return turns


def analyze_dataset_statistics(dataset) -> Dict:
    """
    Compute comprehensive statistics about the dataset.
    
    Returns:
        Dictionary containing various statistics.
    """
    stats = {
        "total_samples": len(dataset),
        "chosen_lengths": [],
        "rejected_lengths": [],
        "chosen_word_counts": [],
        "rejected_word_counts": [],
        "num_turns_chosen": [],
        "num_turns_rejected": [],
        "length_differences": [],
        "chosen_longer_count": 0,
        "rejected_longer_count": 0,
        "equal_length_count": 0,
    }
    
    print("\nAnalyzing dataset statistics...")
    for example in tqdm(dataset, desc="Processing examples"):
        chosen = example["chosen"]
        rejected = example["rejected"]
        
        # Character lengths
        chosen_len = len(chosen)
        rejected_len = len(rejected)
        stats["chosen_lengths"].append(chosen_len)
        stats["rejected_lengths"].append(rejected_len)
        
        # Word counts
        chosen_words = len(chosen.split())
        rejected_words = len(rejected.split())
        stats["chosen_word_counts"].append(chosen_words)
        stats["rejected_word_counts"].append(rejected_words)
        
        # Number of conversation turns
        chosen_turns = extract_conversation_structure(chosen)
        rejected_turns = extract_conversation_structure(rejected)
        stats["num_turns_chosen"].append(len(chosen_turns))
        stats["num_turns_rejected"].append(len(rejected_turns))
        
        # Length differences
        stats["length_differences"].append(chosen_len - rejected_len)
        
        if chosen_len > rejected_len:
            stats["chosen_longer_count"] += 1
        elif rejected_len > chosen_len:
            stats["rejected_longer_count"] += 1
        else:
            stats["equal_length_count"] += 1
    
    return stats


def identify_patterns_and_biases(dataset, stats: Dict) -> Dict:
    """
    Identify patterns and potential biases in the preference data.
    
    Returns:
        Dictionary containing identified patterns and biases.
    """
    patterns = {}
    
    # Length bias analysis
    chosen_longer_pct = stats["chosen_longer_count"] / stats["total_samples"] * 100
    rejected_longer_pct = stats["rejected_longer_count"] / stats["total_samples"] * 100
    
    patterns["length_bias"] = {
        "chosen_longer_percentage": chosen_longer_pct,
        "rejected_longer_percentage": rejected_longer_pct,
        "mean_chosen_length": np.mean(stats["chosen_lengths"]),
        "mean_rejected_length": np.mean(stats["rejected_lengths"]),
        "median_chosen_length": np.median(stats["chosen_lengths"]),
        "median_rejected_length": np.median(stats["rejected_lengths"]),
    }
    
    # Verbosity analysis
    patterns["verbosity"] = {
        "mean_chosen_words": np.mean(stats["chosen_word_counts"]),
        "mean_rejected_words": np.mean(stats["rejected_word_counts"]),
        "word_count_correlation_with_preference": np.corrcoef(
            stats["chosen_word_counts"], stats["rejected_word_counts"]
        )[0, 1]
    }
    
    # Conversation depth analysis
    patterns["conversation_depth"] = {
        "mean_turns_chosen": np.mean(stats["num_turns_chosen"]),
        "mean_turns_rejected": np.mean(stats["num_turns_rejected"]),
    }
    
    # Analyze common starting patterns
    chosen_starts = Counter()
    rejected_starts = Counter()
    
    print("\nAnalyzing response patterns...")
    for example in tqdm(dataset, desc="Analyzing patterns"):
        chosen_turns = extract_conversation_structure(example["chosen"])
        rejected_turns = extract_conversation_structure(example["rejected"])
        
        # Get the last assistant response (the one being compared)
        if chosen_turns and chosen_turns[-1]["role"] == "assistant":
            first_words = " ".join(chosen_turns[-1]["content"].split()[:5])
            chosen_starts[first_words] += 1
            
        if rejected_turns and rejected_turns[-1]["role"] == "assistant":
            first_words = " ".join(rejected_turns[-1]["content"].split()[:5])
            rejected_starts[first_words] += 1
    
    patterns["common_chosen_starts"] = chosen_starts.most_common(10)
    patterns["common_rejected_starts"] = rejected_starts.most_common(10)
    
    return patterns


def visualize_statistics(stats: Dict, output_dir: str = "outputs/exploration"):
    """
    Create visualizations for dataset statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Anthropic HH-RLHF Dataset Analysis", fontsize=14, fontweight='bold')
    
    # 1. Distribution of character lengths
    ax1 = axes[0, 0]
    ax1.hist(stats["chosen_lengths"], bins=50, alpha=0.7, label="Chosen", color='#2ecc71')
    ax1.hist(stats["rejected_lengths"], bins=50, alpha=0.7, label="Rejected", color='#e74c3c')
    ax1.set_xlabel("Character Length")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Response Lengths")
    ax1.legend()
    ax1.set_xlim(0, np.percentile(stats["chosen_lengths"] + stats["rejected_lengths"], 95))
    
    # 2. Distribution of word counts
    ax2 = axes[0, 1]
    ax2.hist(stats["chosen_word_counts"], bins=50, alpha=0.7, label="Chosen", color='#2ecc71')
    ax2.hist(stats["rejected_word_counts"], bins=50, alpha=0.7, label="Rejected", color='#e74c3c')
    ax2.set_xlabel("Word Count")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Word Counts")
    ax2.legend()
    ax2.set_xlim(0, np.percentile(stats["chosen_word_counts"] + stats["rejected_word_counts"], 95))
    
    # 3. Length difference distribution
    ax3 = axes[0, 2]
    ax3.hist(stats["length_differences"], bins=50, alpha=0.7, color='#3498db')
    ax3.axvline(x=0, color='red', linestyle='--', label='Equal length')
    ax3.axvline(x=np.mean(stats["length_differences"]), color='green', linestyle='--', 
                label=f'Mean: {np.mean(stats["length_differences"]):.0f}')
    ax3.set_xlabel("Length Difference (Chosen - Rejected)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Distribution of Length Differences")
    ax3.legend()
    
    # 4. Which is longer pie chart
    ax4 = axes[1, 0]
    sizes = [stats["chosen_longer_count"], stats["rejected_longer_count"], stats["equal_length_count"]]
    labels = ['Chosen Longer', 'Rejected Longer', 'Equal']
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    explode = (0.05, 0.05, 0)
    ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title("Length Comparison Distribution")
    
    # 5. Number of turns distribution
    ax5 = axes[1, 1]
    turn_counts = Counter(stats["num_turns_chosen"])
    turns = sorted(turn_counts.keys())
    counts = [turn_counts[t] for t in turns]
    ax5.bar(turns, counts, color='#9b59b6', alpha=0.7)
    ax5.set_xlabel("Number of Conversation Turns")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Distribution of Conversation Depth")
    ax5.set_xticks(turns[:10])
    
    # 6. Scatter plot of chosen vs rejected lengths
    ax6 = axes[1, 2]
    sample_size = min(5000, len(stats["chosen_lengths"]))
    indices = np.random.choice(len(stats["chosen_lengths"]), sample_size, replace=False)
    chosen_sample = [stats["chosen_lengths"][i] for i in indices]
    rejected_sample = [stats["rejected_lengths"][i] for i in indices]
    ax6.scatter(chosen_sample, rejected_sample, alpha=0.3, s=10, c='#3498db')
    max_len = max(max(chosen_sample), max(rejected_sample))
    ax6.plot([0, max_len], [0, max_len], 'r--', label='Equal length')
    ax6.set_xlabel("Chosen Length")
    ax6.set_ylabel("Rejected Length")
    ax6.set_title("Chosen vs Rejected Lengths")
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {output_dir}/dataset_analysis.png")


def print_analysis_report(stats: Dict, patterns: Dict):
    """
    Print a comprehensive analysis report.
    """
    print("\n" + "="*80)
    print("ANTHROPIC HH-RLHF DATASET ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n{'BASIC STATISTICS':^80}")
    print("-"*80)
    print(f"Total samples: {stats['total_samples']:,}")
    
    print(f"\n{'LENGTH ANALYSIS':^80}")
    print("-"*80)
    print(f"{'Metric':<30} {'Chosen':<20} {'Rejected':<20}")
    print("-"*80)
    print(f"{'Mean character length':<30} {np.mean(stats['chosen_lengths']):<20.1f} {np.mean(stats['rejected_lengths']):<20.1f}")
    print(f"{'Median character length':<30} {np.median(stats['chosen_lengths']):<20.1f} {np.median(stats['rejected_lengths']):<20.1f}")
    print(f"{'Std character length':<30} {np.std(stats['chosen_lengths']):<20.1f} {np.std(stats['rejected_lengths']):<20.1f}")
    print(f"{'Mean word count':<30} {np.mean(stats['chosen_word_counts']):<20.1f} {np.mean(stats['rejected_word_counts']):<20.1f}")
    print(f"{'Mean conversation turns':<30} {np.mean(stats['num_turns_chosen']):<20.1f} {np.mean(stats['num_turns_rejected']):<20.1f}")
    
    print(f"\n{'IDENTIFIED BIASES AND PATTERNS':^80}")
    print("-"*80)
    
    length_bias = patterns["length_bias"]
    print(f"\n1. LENGTH BIAS:")
    print(f"   - Chosen responses are longer in {length_bias['chosen_longer_percentage']:.1f}% of cases")
    print(f"   - Rejected responses are longer in {length_bias['rejected_longer_percentage']:.1f}% of cases")
    print(f"   - Mean length difference: {length_bias['mean_chosen_length'] - length_bias['mean_rejected_length']:.1f} characters")
    
    if length_bias['chosen_longer_percentage'] > 55:
        print(f"   ⚠️  POTENTIAL BIAS: Strong preference for longer responses detected!")
    
    print(f"\n2. VERBOSITY PATTERNS:")
    verbosity = patterns["verbosity"]
    print(f"   - Mean chosen words: {verbosity['mean_chosen_words']:.1f}")
    print(f"   - Mean rejected words: {verbosity['mean_rejected_words']:.1f}")
    
    print(f"\n3. COMMON RESPONSE PATTERNS:")
    print(f"   Top 5 chosen response starts:")
    for start, count in patterns["common_chosen_starts"][:5]:
        print(f"      '{start}...' ({count} occurrences)")
    
    print(f"\n   Top 5 rejected response starts:")
    for start, count in patterns["common_rejected_starts"][:5]:
        print(f"      '{start}...' ({count} occurrences)")
    
    print("\n" + "="*80)


def explore_sample_examples(dataset, num_examples: int = 5):
    """
    Display sample examples from the dataset for qualitative analysis.
    """
    print(f"\n{'SAMPLE EXAMPLES':^80}")
    print("="*80)
    
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    for i, idx in enumerate(indices):
        example = dataset[int(idx)]
        print(f"\n--- Example {i+1} (Index: {idx}) ---")
        
        chosen_turns = extract_conversation_structure(example["chosen"])
        rejected_turns = extract_conversation_structure(example["rejected"])
        
        # Show the shared context (prompt)
        print("\n[SHARED CONTEXT]")
        for turn in chosen_turns[:-1]:
            role = "Human" if turn["role"] == "human" else "Assistant"
            content = turn["content"][:200] + "..." if len(turn["content"]) > 200 else turn["content"]
            print(f"{role}: {content}")
        
        # Show the differing responses
        if chosen_turns and chosen_turns[-1]["role"] == "assistant":
            chosen_response = chosen_turns[-1]["content"]
            print(f"\n[CHOSEN RESPONSE] (length: {len(chosen_response)} chars)")
            print(chosen_response[:500] + "..." if len(chosen_response) > 500 else chosen_response)
        
        if rejected_turns and rejected_turns[-1]["role"] == "assistant":
            rejected_response = rejected_turns[-1]["content"]
            print(f"\n[REJECTED RESPONSE] (length: {len(rejected_response)} chars)")
            print(rejected_response[:500] + "..." if len(rejected_response) > 500 else rejected_response)
        
        print("\n" + "-"*80)


def run_exploration(output_dir: str = "outputs/exploration"):
    """
    Run the complete dataset exploration pipeline.
    """
    # Load dataset
    dataset = load_hh_rlhf_dataset()
    
    # Use training set for analysis
    train_data = dataset["train"]
    
    # Compute statistics
    stats = analyze_dataset_statistics(train_data)
    
    # Identify patterns and biases
    patterns = identify_patterns_and_biases(train_data, stats)
    
    # Print analysis report
    print_analysis_report(stats, patterns)
    
    # Create visualizations
    visualize_statistics(stats, output_dir)
    
    # Show sample examples
    explore_sample_examples(train_data, num_examples=3)
    
    # Save analysis results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "statistics": {
            "total_samples": stats["total_samples"],
            "mean_chosen_length": float(np.mean(stats["chosen_lengths"])),
            "mean_rejected_length": float(np.mean(stats["rejected_lengths"])),
            "std_chosen_length": float(np.std(stats["chosen_lengths"])),
            "std_rejected_length": float(np.std(stats["rejected_lengths"])),
            "chosen_longer_percentage": stats["chosen_longer_count"] / stats["total_samples"] * 100,
            "rejected_longer_percentage": stats["rejected_longer_count"] / stats["total_samples"] * 100,
        },
        "patterns": {
            "length_bias": patterns["length_bias"],
            "verbosity": patterns["verbosity"],
            "conversation_depth": patterns["conversation_depth"],
        }
    }
    
    with open(os.path.join(output_dir, "analysis_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}/analysis_results.json")
    
    return dataset, stats, patterns


if __name__ == "__main__":
    dataset, stats, patterns = run_exploration()

