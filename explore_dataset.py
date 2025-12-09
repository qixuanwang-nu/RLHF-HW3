"""
Part 1.1A: Dataset Exploration Script
Run this to explore and analyze the Anthropic HH-RLHF dataset.

Usage:
    python explore_dataset.py
"""

from src.data.exploration import run_exploration


if __name__ == "__main__":
    print("="*80)
    print("PART 1.1A: Dataset Exploration and Analysis")
    print("="*80)
    
    dataset, stats, patterns = run_exploration(output_dir="outputs/exploration")
    
    print("\n" + "="*80)
    print("Exploration complete! Check outputs/exploration/ for results.")
    print("="*80)

