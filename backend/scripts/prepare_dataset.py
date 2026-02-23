#!/usr/bin/env python3
"""
Dataset Preparation Script for Music Transformer Training

Loads tokenized_dataset.pt and prepares train/validation splits
with proper padding, truncation, and attention masks.

Features:
- Configurable max sequence length
- Train/validation split with stratification by mood
- PyTorch Dataset class for efficient data loading
- Reproducible splits with seed control

Output:
- train_dataset.pt: Training split with Dataset-compatible format
- val_dataset.pt: Validation split with Dataset-compatible format

Usage:
    python scripts/prepare_dataset.py \
        --input data/training/tokenized_dataset.pt \
        --output-dir data/training \
        --max-length 256 \
        --val-ratio 0.15

    # With stratified split by mood
    python scripts/prepare_dataset.py \
        --input data/training/tokenized_dataset.pt \
        --output-dir data/training \
        --stratify
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PrepareConfig:
    """Configuration for dataset preparation."""
    max_length: int = 256
    val_ratio: float = 0.15
    seed: int = 42
    stratify: bool = True
    min_length: int = 10  # Minimum sequence length to keep


# =============================================================================
# PyTorch Dataset
# =============================================================================

class MusicDataset(Dataset):
    """
    PyTorch Dataset for mood-conditioned music sequences.

    Attributes:
        input_ids: Token ID sequences (N, seq_len)
        attention_mask: Attention masks (N, seq_len)
        mood_labels: Mood label indices (N,)
        vocab: Vocabulary information
        config: Dataset configuration
    """

    def __init__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        mood_labels: Tensor,
        vocab: Dict[str, Any],
        config: Dict[str, Any],
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.mood_labels = mood_labels
        self.vocab = vocab
        self.config = config

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "mood_label": self.mood_labels[idx],
        }

    @property
    def vocab_size(self) -> int:
        return self.vocab.get("vocab_size", 0)

    @property
    def pad_token_id(self) -> int:
        return self.vocab.get("pad_token_id", 0)

    @property
    def num_moods(self) -> int:
        return len(self.vocab.get("mood_to_id", {}))

    def get_mood_distribution(self) -> Dict[str, int]:
        """Get distribution of moods in dataset."""
        id_to_mood = {v: k for k, v in self.vocab.get("mood_to_id", {}).items()}
        counter = Counter(self.mood_labels.tolist())
        return {id_to_mood.get(k, f"mood_{k}"): v for k, v in counter.items()}

    def save(self, path: str) -> None:
        """Save dataset to file."""
        data = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "mood_labels": self.mood_labels,
            "vocab": self.vocab,
            "config": self.config,
            "length": len(self),
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: str) -> "MusicDataset":
        """Load dataset from file."""
        data = torch.load(path)
        return cls(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            mood_labels=data["mood_labels"],
            vocab=data["vocab"],
            config=data["config"],
        )


# =============================================================================
# Processing Functions
# =============================================================================

def truncate_and_pad(
    input_ids: Tensor,
    attention_mask: Tensor,
    max_length: int,
    pad_token_id: int,
    eos_token_id: int,
) -> Tuple[Tensor, Tensor]:
    """
    Truncate sequences to max_length and re-pad if needed.

    Ensures:
    - Sequences are exactly max_length
    - EOS token is preserved at the end
    - Attention mask matches actual tokens

    Args:
        input_ids: Original token IDs (N, original_length)
        attention_mask: Original attention masks (N, original_length)
        max_length: Target sequence length
        pad_token_id: PAD token ID
        eos_token_id: EOS token ID

    Returns:
        Tuple of (new_input_ids, new_attention_mask)
    """
    batch_size, original_length = input_ids.shape

    if original_length == max_length:
        return input_ids, attention_mask

    # Initialize new tensors
    new_input_ids = torch.full(
        (batch_size, max_length),
        pad_token_id,
        dtype=input_ids.dtype,
    )
    new_attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=attention_mask.dtype,
    )

    for i in range(batch_size):
        # Get actual sequence length (non-padding)
        seq_mask = attention_mask[i] == 1
        actual_length = seq_mask.sum().item()

        if actual_length <= max_length:
            # Sequence fits - copy as is
            new_input_ids[i, :actual_length] = input_ids[i, :actual_length]
            new_attention_mask[i, :actual_length] = 1
        else:
            # Need to truncate
            # Keep first (max_length - 1) tokens and add EOS
            new_input_ids[i, :max_length - 1] = input_ids[i, :max_length - 1]
            new_input_ids[i, max_length - 1] = eos_token_id
            new_attention_mask[i, :max_length] = 1

    return new_input_ids, new_attention_mask


def filter_by_length(
    input_ids: Tensor,
    attention_mask: Tensor,
    mood_labels: Tensor,
    min_length: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Filter out sequences shorter than min_length.

    Args:
        input_ids: Token IDs
        attention_mask: Attention masks
        mood_labels: Mood labels
        min_length: Minimum sequence length

    Returns:
        Filtered tensors
    """
    # Calculate actual lengths
    lengths = attention_mask.sum(dim=1)

    # Create mask for valid sequences
    valid_mask = lengths >= min_length

    return (
        input_ids[valid_mask],
        attention_mask[valid_mask],
        mood_labels[valid_mask],
    )


def stratified_split(
    labels: Tensor,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    Create stratified train/val split indices.

    Ensures each mood is proportionally represented in both splits.

    Args:
        labels: Mood labels
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        Tuple of (train_indices, val_indices)
    """
    np.random.seed(seed)

    labels_np = labels.numpy()
    unique_labels = np.unique(labels_np)

    train_indices = []
    val_indices = []

    for label in unique_labels:
        # Get indices for this label
        label_indices = np.where(labels_np == label)[0]
        np.random.shuffle(label_indices)

        # Split
        n_val = max(1, int(len(label_indices) * val_ratio))
        val_indices.extend(label_indices[:n_val].tolist())
        train_indices.extend(label_indices[n_val:].tolist())

    # Shuffle final indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return train_indices, val_indices


def random_split(
    n_samples: int,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    Create random train/val split indices.

    Args:
        n_samples: Total number of samples
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        Tuple of (train_indices, val_indices)
    """
    np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_val = int(n_samples * val_ratio)

    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()

    return train_indices, val_indices


# =============================================================================
# Dataset Preparer
# =============================================================================

class DatasetPreparer:
    """
    Prepares tokenized dataset for training.

    Steps:
    1. Load tokenized dataset
    2. Filter by minimum length
    3. Truncate/pad to max length
    4. Create train/val split
    5. Save splits
    """

    def __init__(self, config: PrepareConfig):
        self.config = config
        self.stats = {
            "original_samples": 0,
            "filtered_samples": 0,
            "train_samples": 0,
            "val_samples": 0,
            "train_mood_dist": {},
            "val_mood_dist": {},
        }

    def prepare(self, input_path: str, output_dir: str) -> Tuple[MusicDataset, MusicDataset]:
        """
        Prepare train and validation datasets.

        Args:
            input_path: Path to tokenized_dataset.pt
            output_dir: Directory to save train/val datasets

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        print(f"Loading tokenized dataset from: {input_path}")
        data = torch.load(input_path)

        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        mood_labels = data["mood_labels"]
        vocab = data["vocab"]
        original_config = data["config"]

        self.stats["original_samples"] = len(input_ids)
        print(f"Loaded {self.stats['original_samples']} samples")

        # Get special token IDs
        pad_token_id = vocab.get("pad_token_id", 0)
        eos_token_id = vocab.get("eos_token_id", 2)

        # Filter by minimum length
        print(f"Filtering sequences shorter than {self.config.min_length} tokens...")
        input_ids, attention_mask, mood_labels = filter_by_length(
            input_ids, attention_mask, mood_labels, self.config.min_length
        )
        self.stats["filtered_samples"] = len(input_ids)
        print(f"Kept {self.stats['filtered_samples']} samples after filtering")

        # Truncate and pad
        print(f"Truncating/padding to max length {self.config.max_length}...")
        input_ids, attention_mask = truncate_and_pad(
            input_ids,
            attention_mask,
            self.config.max_length,
            pad_token_id,
            eos_token_id,
        )

        # Create split
        print(f"Creating train/val split (val_ratio={self.config.val_ratio})...")
        if self.config.stratify:
            print("Using stratified split by mood")
            train_idx, val_idx = stratified_split(
                mood_labels, self.config.val_ratio, self.config.seed
            )
        else:
            print("Using random split")
            train_idx, val_idx = random_split(
                len(input_ids), self.config.val_ratio, self.config.seed
            )

        self.stats["train_samples"] = len(train_idx)
        self.stats["val_samples"] = len(val_idx)

        # Build config for saved datasets
        dataset_config = {
            **original_config,
            "max_length": self.config.max_length,
            "min_length": self.config.min_length,
            "val_ratio": self.config.val_ratio,
            "seed": self.config.seed,
            "stratified": self.config.stratify,
        }

        # Create train dataset
        train_dataset = MusicDataset(
            input_ids=input_ids[train_idx],
            attention_mask=attention_mask[train_idx],
            mood_labels=mood_labels[train_idx],
            vocab=vocab,
            config=dataset_config,
        )

        # Create val dataset
        val_dataset = MusicDataset(
            input_ids=input_ids[val_idx],
            attention_mask=attention_mask[val_idx],
            mood_labels=mood_labels[val_idx],
            vocab=vocab,
            config=dataset_config,
        )

        # Get mood distributions
        self.stats["train_mood_dist"] = train_dataset.get_mood_distribution()
        self.stats["val_mood_dist"] = val_dataset.get_mood_distribution()

        # Save datasets
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "train_dataset.pt"
        val_path = output_dir / "val_dataset.pt"

        print(f"Saving train dataset to: {train_path}")
        train_dataset.save(str(train_path))

        print(f"Saving val dataset to: {val_path}")
        val_dataset.save(str(val_path))

        self._print_summary()

        return train_dataset, val_dataset

    def _print_summary(self) -> None:
        """Print preparation summary."""
        print("\n" + "=" * 60)
        print("DATASET PREPARATION SUMMARY")
        print("=" * 60)

        print(f"\nSamples:")
        print(f"  Original:  {self.stats['original_samples']}")
        print(f"  Filtered:  {self.stats['filtered_samples']}")
        print(f"  Train:     {self.stats['train_samples']}")
        print(f"  Val:       {self.stats['val_samples']}")

        print(f"\nConfiguration:")
        print(f"  Max length:    {self.config.max_length}")
        print(f"  Min length:    {self.config.min_length}")
        print(f"  Val ratio:     {self.config.val_ratio}")
        print(f"  Stratified:    {self.config.stratify}")
        print(f"  Seed:          {self.config.seed}")

        print(f"\nTrain Mood Distribution:")
        for mood, count in sorted(
            self.stats["train_mood_dist"].items(),
            key=lambda x: -x[1]
        ):
            pct = count / self.stats["train_samples"] * 100
            print(f"  {mood:20} {count:4} ({pct:5.1f}%)")

        print(f"\nVal Mood Distribution:")
        for mood, count in sorted(
            self.stats["val_mood_dist"].items(),
            key=lambda x: -x[1]
        ):
            pct = count / self.stats["val_samples"] * 100
            print(f"  {mood:20} {count:4} ({pct:5.1f}%)")

        print("=" * 60)


# =============================================================================
# Utility Functions
# =============================================================================

def inspect_dataset(path: str) -> None:
    """Inspect a prepared dataset."""
    print(f"\nLoading dataset from: {path}")
    dataset = MusicDataset.load(path)

    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(f"\nBasic Info:")
    print(f"  Samples:       {len(dataset)}")
    print(f"  Sequence len:  {dataset.input_ids.shape[1]}")
    print(f"  Vocab size:    {dataset.vocab_size}")
    print(f"  Num moods:     {dataset.num_moods}")

    print(f"\nTensor Shapes:")
    print(f"  input_ids:      {dataset.input_ids.shape}")
    print(f"  attention_mask: {dataset.attention_mask.shape}")
    print(f"  mood_labels:    {dataset.mood_labels.shape}")

    print(f"\nMood Distribution:")
    for mood, count in sorted(
        dataset.get_mood_distribution().items(),
        key=lambda x: -x[1]
    ):
        pct = count / len(dataset) * 100
        print(f"  {mood:20} {count:4} ({pct:5.1f}%)")

    # Sequence length statistics
    lengths = dataset.attention_mask.sum(dim=1).float()
    print(f"\nSequence Length Stats:")
    print(f"  Min:    {lengths.min().item():.0f}")
    print(f"  Max:    {lengths.max().item():.0f}")
    print(f"  Mean:   {lengths.mean().item():.1f}")
    print(f"  Median: {lengths.median().item():.0f}")

    # Sample item
    print(f"\nSample Item (idx=0):")
    sample = dataset[0]
    print(f"  input_ids shape:      {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  mood_label:           {sample['mood_label'].item()}")

    # First few tokens
    id_to_token = dataset.vocab.get("id_to_token", {})
    first_tokens = sample["input_ids"][:10].tolist()
    token_strs = [id_to_token.get(str(t), id_to_token.get(t, f"[{t}]")) for t in first_tokens]
    print(f"  First 10 tokens: {token_strs}")

    print("=" * 60)


def create_dataloader(
    dataset: MusicDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader from a MusicDataset.

    Args:
        dataset: The dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare tokenized dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic preparation
    python scripts/prepare_dataset.py \\
        --input data/training/tokenized_dataset.pt \\
        --output-dir data/training

    # Custom configuration
    python scripts/prepare_dataset.py \\
        --input data/training/tokenized_dataset.pt \\
        --output-dir data/training \\
        --max-length 256 \\
        --val-ratio 0.15 \\
        --stratify

    # Inspect prepared dataset
    python scripts/prepare_dataset.py --inspect data/training/train_dataset.pt
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to tokenized_dataset.pt",
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/training",
        help="Output directory for train/val datasets",
    )

    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=256,
        help="Maximum sequence length (default: 256)",
    )

    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum sequence length to keep (default: 10)",
    )

    parser.add_argument(
        "--val-ratio", "-v",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )

    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Use stratified split by mood",
    )

    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Use random split (not stratified)",
    )

    parser.add_argument(
        "--inspect",
        type=str,
        metavar="PATH",
        help="Inspect an existing prepared dataset",
    )

    args = parser.parse_args()

    # Inspect mode
    if args.inspect:
        inspect_dataset(args.inspect)
        return 0

    # Validate required arguments
    if not args.input:
        parser.error("--input is required for dataset preparation")

    if not Path(args.input).is_file():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Create config
    config = PrepareConfig(
        max_length=args.max_length,
        min_length=args.min_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify=not args.no_stratify,  # Default to stratified
    )

    # Prepare datasets
    preparer = DatasetPreparer(config)
    preparer.prepare(args.input, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
