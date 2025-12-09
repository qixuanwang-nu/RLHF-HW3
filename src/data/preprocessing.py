"""
Part 1.1B: Data Preprocessing Pipeline
Implements tokenization, balanced splits, and edge case handling for preference data.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer


class DataPreprocessor:
    """
    Preprocessor for the Anthropic HH-RLHF dataset.
    
    Handles:
    - Tokenization of prompts and responses
    - Balanced training/validation splits
    - Edge cases (ties, very long sequences)
    """
    
    def __init__(
        self,
        tokenizer_name: str = "openai-community/gpt2",
        max_length: int = 512,
        truncation_strategy: str = "left",  # Keep the most recent context
        min_response_length: int = 10,
        seed: int = 42,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            tokenizer_name: HuggingFace tokenizer to use
            max_length: Maximum sequence length
            truncation_strategy: How to truncate ('left' keeps recent, 'right' keeps beginning)
            min_response_length: Minimum response length to include
            seed: Random seed for reproducibility
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # GPT-2 doesn't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.min_response_length = min_response_length
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def extract_prompt_and_response(self, text: str) -> Tuple[str, str]:
        """
        Extract the prompt (conversation context) and final response from text.
        
        The HH-RLHF format is:
        Human: ... Assistant: ... Human: ... Assistant: [final response]
        
        Returns:
            Tuple of (prompt, response)
        """
        # Find the last "Assistant:" to separate prompt from response
        last_assistant_idx = text.rfind("\n\nAssistant:")
        
        if last_assistant_idx == -1:
            # Fallback: try without double newline
            last_assistant_idx = text.rfind("Assistant:")
            if last_assistant_idx == -1:
                return text, ""
            prompt = text[:last_assistant_idx].strip()
            response = text[last_assistant_idx + len("Assistant:"):].strip()
        else:
            prompt = text[:last_assistant_idx].strip()
            response = text[last_assistant_idx + len("\n\nAssistant:"):].strip()
        
        return prompt, response
    
    def is_tie(self, chosen: str, rejected: str, similarity_threshold: float = 0.95) -> bool:
        """
        Check if chosen and rejected responses are essentially ties.
        
        Uses character-level Jaccard similarity as a simple metric.
        
        Args:
            chosen: Chosen response text
            rejected: Rejected response text
            similarity_threshold: Threshold above which responses are considered ties
            
        Returns:
            Boolean indicating if responses are ties
        """
        # Extract just the final responses
        _, chosen_response = self.extract_prompt_and_response(chosen)
        _, rejected_response = self.extract_prompt_and_response(rejected)
        
        # Exact match
        if chosen_response == rejected_response:
            return True
        
        # Character-level Jaccard similarity
        chosen_chars = set(chosen_response)
        rejected_chars = set(rejected_response)
        
        if len(chosen_chars) == 0 and len(rejected_chars) == 0:
            return True
        
        intersection = len(chosen_chars & rejected_chars)
        union = len(chosen_chars | rejected_chars)
        
        if union == 0:
            return True
        
        similarity = intersection / union
        
        # Also check length similarity
        len_ratio = min(len(chosen_response), len(rejected_response)) / max(len(chosen_response), len(rejected_response) + 1)
        
        return similarity > similarity_threshold and len_ratio > 0.9
    
    def is_valid_example(self, chosen: str, rejected: str) -> Tuple[bool, str]:
        """
        Check if an example is valid for training.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for ties
        if self.is_tie(chosen, rejected):
            return False, "tie"
        
        # Extract responses
        _, chosen_response = self.extract_prompt_and_response(chosen)
        _, rejected_response = self.extract_prompt_and_response(rejected)
        
        # Check minimum response length
        if len(chosen_response) < self.min_response_length:
            return False, "chosen_too_short"
        
        if len(rejected_response) < self.min_response_length:
            return False, "rejected_too_short"
        
        # Check for empty responses
        if not chosen_response.strip() or not rejected_response.strip():
            return False, "empty_response"
        
        return True, "valid"
    
    def tokenize_text(
        self,
        text: str,
        return_attention_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a single text with proper handling for long sequences.
        
        Args:
            text: Text to tokenize
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Dictionary with input_ids and optionally attention_mask
        """
        # Tokenize with truncation
        if self.truncation_strategy == "left":
            # For left truncation, we need to handle it manually
            # First tokenize without truncation to see total length
            full_tokens = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
                return_tensors=None
            )
            
            input_ids = full_tokens["input_ids"]
            
            if len(input_ids) > self.max_length:
                # Keep the rightmost tokens (most recent context)
                input_ids = input_ids[-self.max_length:]
            
            # Pad if necessary
            attention_mask = [1] * len(input_ids)
            padding_length = self.max_length - len(input_ids)
            
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            result = {"input_ids": input_ids}
            if return_attention_mask:
                result["attention_mask"] = attention_mask
                
            return result
        else:
            # Standard right truncation
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                add_special_tokens=True,
                return_tensors=None
            )
            
            result = {"input_ids": tokens["input_ids"]}
            if return_attention_mask:
                result["attention_mask"] = tokens["attention_mask"]
            
            return result
    
    def preprocess_example(self, example: Dict) -> Dict:
        """
        Preprocess a single example from the dataset.
        
        Args:
            example: Dictionary with 'chosen' and 'rejected' keys
            
        Returns:
            Preprocessed example with tokenized inputs
        """
        chosen_text = example["chosen"]
        rejected_text = example["rejected"]
        
        # Tokenize both
        chosen_tokens = self.tokenize_text(chosen_text)
        rejected_tokens = self.tokenize_text(rejected_text)
        
        return {
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }
    
    def filter_and_preprocess_dataset(
        self,
        dataset: Dataset,
        remove_ties: bool = True,
        verbose: bool = True
    ) -> Tuple[Dataset, Dict[str, int]]:
        """
        Filter out invalid examples and preprocess the dataset.
        
        Args:
            dataset: HuggingFace Dataset
            remove_ties: Whether to remove tie examples
            verbose: Whether to print progress
            
        Returns:
            Tuple of (preprocessed_dataset, filter_stats)
        """
        filter_stats = {
            "total": len(dataset),
            "valid": 0,
            "tie": 0,
            "chosen_too_short": 0,
            "rejected_too_short": 0,
            "empty_response": 0,
        }
        
        valid_indices = []
        
        if verbose:
            print("Filtering dataset...")
            iterator = tqdm(enumerate(dataset), total=len(dataset), desc="Filtering")
        else:
            iterator = enumerate(dataset)
        
        for idx, example in iterator:
            is_valid, reason = self.is_valid_example(example["chosen"], example["rejected"])
            
            if is_valid:
                valid_indices.append(idx)
                filter_stats["valid"] += 1
            elif not remove_ties and reason == "tie":
                valid_indices.append(idx)
                filter_stats["valid"] += 1
            else:
                filter_stats[reason] = filter_stats.get(reason, 0) + 1
        
        # Select valid examples
        filtered_dataset = dataset.select(valid_indices)
        
        if verbose:
            print(f"\nFilter statistics:")
            print(f"  Total examples: {filter_stats['total']}")
            print(f"  Valid examples: {filter_stats['valid']} ({filter_stats['valid']/filter_stats['total']*100:.1f}%)")
            print(f"  Removed ties: {filter_stats['tie']}")
            print(f"  Removed (chosen too short): {filter_stats['chosen_too_short']}")
            print(f"  Removed (rejected too short): {filter_stats['rejected_too_short']}")
            print(f"  Removed (empty response): {filter_stats['empty_response']}")
        
        # Tokenize the filtered dataset
        if verbose:
            print("\nTokenizing dataset...")
        
        preprocessed_dataset = filtered_dataset.map(
            self.preprocess_example,
            desc="Tokenizing" if verbose else None,
            remove_columns=filtered_dataset.column_names
        )
        
        return preprocessed_dataset, filter_stats
    
    def create_balanced_splits(
        self,
        dataset: Dataset,
        val_ratio: float = 0.1,
        stratify_by_length: bool = True,
        num_length_bins: int = 5
    ) -> DatasetDict:
        """
        Create balanced training and validation splits.
        
        Args:
            dataset: Preprocessed dataset
            val_ratio: Fraction for validation
            stratify_by_length: Whether to stratify by sequence length
            num_length_bins: Number of length bins for stratification
            
        Returns:
            DatasetDict with 'train' and 'validation' splits
        """
        indices = list(range(len(dataset)))
        
        if stratify_by_length:
            # Compute average sequence lengths for stratification
            lengths = []
            for example in dataset:
                chosen_len = sum(example["chosen_attention_mask"])
                rejected_len = sum(example["rejected_attention_mask"])
                lengths.append((chosen_len + rejected_len) / 2)
            
            # Bin the lengths
            length_bins = np.digitize(
                lengths,
                bins=np.percentile(lengths, np.linspace(0, 100, num_length_bins + 1)[1:-1])
            )
            
            train_indices, val_indices = train_test_split(
                indices,
                test_size=val_ratio,
                stratify=length_bins,
                random_state=self.seed
            )
        else:
            train_indices, val_indices = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=self.seed
            )
        
        return DatasetDict({
            "train": dataset.select(train_indices),
            "validation": dataset.select(val_indices)
        })
    
    def prepare_data(
        self,
        subset: str = None,
        val_ratio: float = 0.1,
        use_test_as_val: bool = True,
        verbose: bool = True
    ) -> Tuple[DatasetDict, Dict]:
        """
        Complete data preparation pipeline.
        
        Args:
            subset: Optional dataset subset ('harmless-base', 'helpful-base', etc.)
            val_ratio: Validation split ratio (if not using test set)
            use_test_as_val: Whether to use the provided test set as validation
            verbose: Whether to print progress
            
        Returns:
            Tuple of (prepared_datasets, metadata)
        """
        if verbose:
            print(f"Loading Anthropic HH-RLHF dataset...")
        
        if subset:
            raw_dataset = load_dataset("Anthropic/hh-rlhf", data_dir=subset)
        else:
            raw_dataset = load_dataset("Anthropic/hh-rlhf")
        
        if verbose:
            print(f"Loaded {len(raw_dataset['train'])} training examples")
            print(f"Loaded {len(raw_dataset['test'])} test examples")
        
        # Process training set
        train_processed, train_stats = self.filter_and_preprocess_dataset(
            raw_dataset["train"],
            verbose=verbose
        )
        
        if use_test_as_val:
            # Process test set as validation
            val_processed, val_stats = self.filter_and_preprocess_dataset(
                raw_dataset["test"],
                verbose=verbose
            )
            
            prepared = DatasetDict({
                "train": train_processed,
                "validation": val_processed
            })
            
            metadata = {
                "train_stats": train_stats,
                "val_stats": val_stats,
                "tokenizer_name": self.tokenizer.name_or_path,
                "max_length": self.max_length,
                "truncation_strategy": self.truncation_strategy,
            }
        else:
            # Create our own validation split
            prepared = self.create_balanced_splits(train_processed, val_ratio=val_ratio)
            
            metadata = {
                "train_stats": train_stats,
                "val_ratio": val_ratio,
                "tokenizer_name": self.tokenizer.name_or_path,
                "max_length": self.max_length,
                "truncation_strategy": self.truncation_strategy,
            }
        
        if verbose:
            print(f"\nFinal dataset sizes:")
            print(f"  Training: {len(prepared['train'])}")
            print(f"  Validation: {len(prepared['validation'])}")
        
        return prepared, metadata


class HHRLHFDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapper for preprocessed HH-RLHF data.
    """
    
    def __init__(self, hf_dataset: Dataset):
        """
        Args:
            hf_dataset: Preprocessed HuggingFace Dataset
        """
        self.dataset = hf_dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        
        return {
            "chosen_input_ids": torch.tensor(example["chosen_input_ids"], dtype=torch.long),
            "chosen_attention_mask": torch.tensor(example["chosen_attention_mask"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(example["rejected_input_ids"], dtype=torch.long),
            "rejected_attention_mask": torch.tensor(example["rejected_attention_mask"], dtype=torch.long),
        }


class PreferenceDataCollator:
    """
    Data collator for preference learning batches.
    """
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of preference examples.
        
        Args:
            features: List of examples from HHRLHFDataset
            
        Returns:
            Batched tensors
        """
        batch = {
            "chosen_input_ids": torch.stack([f["chosen_input_ids"] for f in features]),
            "chosen_attention_mask": torch.stack([f["chosen_attention_mask"] for f in features]),
            "rejected_input_ids": torch.stack([f["rejected_input_ids"] for f in features]),
            "rejected_attention_mask": torch.stack([f["rejected_attention_mask"] for f in features]),
        }
        
        return batch


def create_dataloaders(
    prepared_datasets: DatasetDict,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        prepared_datasets: DatasetDict with 'train' and 'validation'
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = HHRLHFDataset(prepared_datasets["train"])
    val_dataset = HHRLHFDataset(prepared_datasets["validation"])
    
    collator = PreferenceDataCollator()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the preprocessing pipeline
    preprocessor = DataPreprocessor(
        tokenizer_name="openai-community/gpt2",
        max_length=512,
        truncation_strategy="left"
    )
    
    # Prepare data
    prepared_data, metadata = preprocessor.prepare_data(verbose=True)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(prepared_data, batch_size=4)
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

