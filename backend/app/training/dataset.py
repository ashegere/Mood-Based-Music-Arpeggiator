"""
Dataset module for mood-conditioned music transformer training.

Loads training data in the format:
- neutral_tokens: Input sequence (structure)
- mood_embedding: Conditioning vector
- target_tokens: Output sequence (expressive or neutral)

Supports two training phases:
- Phase 1: Reconstruction (neutral → neutral)
- Phase 2: Conditioning (neutral + mood → expressive)

Data Sources:
- pairs.jsonl: Generated training pairs from generate_training_data.py
- augmentation_metadata.json: Augmented MIDI metadata
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset

# Import mood embedding (lazy to avoid loading model at import time)
_mood_embedder = None


def get_mood_embedder():
    """Lazy load mood embedder."""
    global _mood_embedder
    if _mood_embedder is None:
        from app.mood.embeddings import get_embedder
        _mood_embedder = get_embedder()
    return _mood_embedder


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrainingSample:
    """
    A single training sample.

    Attributes:
        neutral_tokens: Source token sequence (list of ints)
        target_tokens: Target token sequence (list of ints)
        mood_text: Mood description for embedding
        mood: Mood category name
        metadata: Additional info (key, scale, tempo, etc.)
    """
    neutral_tokens: List[int]
    target_tokens: List[int]
    mood_text: str
    mood: str
    metadata: Dict


@dataclass
class BatchedSample:
    """
    A batched training sample ready for model input.

    All tensors are padded to the same length within the batch.

    Attributes:
        src: Source tokens (batch, src_len)
        tgt_input: Target input tokens (batch, tgt_len) - shifted right
        tgt_output: Target output tokens (batch, tgt_len) - for loss
        mood_embedding: Mood vectors (batch, embed_dim)
        src_padding_mask: Source padding mask (batch, src_len)
        tgt_padding_mask: Target padding mask (batch, tgt_len)
    """
    src: Tensor
    tgt_input: Tensor
    tgt_output: Tensor
    mood_embedding: Tensor
    src_padding_mask: Tensor
    tgt_padding_mask: Tensor


# =============================================================================
# Dataset Classes
# =============================================================================

class MoodArpeggioDataset(Dataset):
    """
    PyTorch Dataset for mood-conditioned arpeggio training.

    Loads pairs from JSONL file and generates mood embeddings
    on-the-fly or from cache.

    Args:
        data_path: Path to pairs.jsonl file or directory containing it.
        phase: Training phase (1 = reconstruction, 2 = conditioning).
        max_seq_len: Maximum sequence length (truncate longer).
        cache_embeddings: Whether to cache mood embeddings.
        embedding_dim: Dimension of mood embeddings.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        phase: int = 2,
        max_seq_len: int = 512,
        cache_embeddings: bool = True,
        embedding_dim: int = 384,
    ):
        self.data_path = Path(data_path)
        self.phase = phase
        self.max_seq_len = max_seq_len
        self.cache_embeddings = cache_embeddings
        self.embedding_dim = embedding_dim

        # Load data
        self.samples = self._load_samples()

        # Embedding cache
        self._embedding_cache: Dict[str, Tensor] = {}

        # Precompute embeddings if caching
        if cache_embeddings:
            self._precompute_embeddings()

    def _load_samples(self) -> List[TrainingSample]:
        """Load samples from JSONL file."""
        # Find pairs file
        if self.data_path.is_file():
            pairs_path = self.data_path
        else:
            pairs_path = self.data_path / "pairs.jsonl"

        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

        samples = []
        with open(pairs_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)

                sample = TrainingSample(
                    neutral_tokens=data["neutral_tokens"],
                    target_tokens=data["expressive_tokens"],
                    mood_text=data["mood_text"],
                    mood=data["mood"],
                    metadata={
                        "sample_id": data.get("sample_id"),
                        "key": data.get("key"),
                        "scale": data.get("scale"),
                        "tempo": data.get("tempo"),
                        "note_count": data.get("note_count"),
                    },
                )
                samples.append(sample)

        print(f"Loaded {len(samples)} training samples from {pairs_path}")
        return samples

    def _precompute_embeddings(self) -> None:
        """Precompute and cache all unique mood embeddings."""
        unique_moods = set(s.mood_text for s in self.samples)
        print(f"Precomputing {len(unique_moods)} unique mood embeddings...")

        embedder = get_mood_embedder()

        for mood_text in unique_moods:
            embedding = embedder.embed(mood_text, use_cache=True)
            self._embedding_cache[mood_text] = embedding.cpu()

    def _get_mood_embedding(self, mood_text: str) -> Tensor:
        """Get mood embedding from cache or compute."""
        if mood_text in self._embedding_cache:
            return self._embedding_cache[mood_text]

        embedder = get_mood_embedder()
        embedding = embedder.embed(mood_text, use_cache=True)
        embedding = embedding.cpu()

        if self.cache_embeddings:
            self._embedding_cache[mood_text] = embedding

        return embedding

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Get a single sample.

        Returns dict with:
        - src_tokens: Source token tensor
        - tgt_tokens: Target token tensor
        - mood_embedding: Mood embedding tensor
        - mood: Mood name (string)
        """
        sample = self.samples[idx]

        # Truncate if necessary
        src_tokens = sample.neutral_tokens[:self.max_seq_len]

        # Phase 1: target = neutral (reconstruction)
        # Phase 2: target = expressive (conditioning)
        if self.phase == 1:
            tgt_tokens = sample.neutral_tokens[:self.max_seq_len]
        else:
            tgt_tokens = sample.target_tokens[:self.max_seq_len]

        # Get mood embedding
        mood_embedding = self._get_mood_embedding(sample.mood_text)

        return {
            "src_tokens": torch.tensor(src_tokens, dtype=torch.long),
            "tgt_tokens": torch.tensor(tgt_tokens, dtype=torch.long),
            "mood_embedding": mood_embedding,
            "mood": sample.mood,
        }


class ReconstructionDataset(Dataset):
    """
    Dataset for Phase 1 reconstruction training.

    Uses only neutral sequences (input = output = neutral).
    Mood embedding is a zero vector (no conditioning).

    Args:
        data_path: Path to data directory or pairs.jsonl.
        max_seq_len: Maximum sequence length.
        embedding_dim: Dimension of mood embeddings.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        max_seq_len: int = 512,
        embedding_dim: int = 384,
    ):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # Load unique neutral sequences
        self.sequences = self._load_unique_sequences()

    def _load_unique_sequences(self) -> List[List[int]]:
        """Load unique neutral token sequences."""
        if self.data_path.is_file():
            pairs_path = self.data_path
        else:
            pairs_path = self.data_path / "pairs.jsonl"

        seen_ids = set()
        sequences = []

        with open(pairs_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                sample_id = data.get("sample_id")

                # Only keep one copy per unique neutral sequence
                if sample_id not in seen_ids:
                    seen_ids.add(sample_id)
                    sequences.append(data["neutral_tokens"])

        print(f"Loaded {len(sequences)} unique neutral sequences")
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get a reconstruction sample (neutral → neutral)."""
        tokens = self.sequences[idx][:self.max_seq_len]

        return {
            "src_tokens": torch.tensor(tokens, dtype=torch.long),
            "tgt_tokens": torch.tensor(tokens, dtype=torch.long),
            "mood_embedding": torch.zeros(self.embedding_dim),
            "mood": "neutral",
        }


# =============================================================================
# Collate Function
# =============================================================================

def collate_fn(
    batch: List[Dict[str, Tensor]],
    pad_token_id: int = 0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
) -> BatchedSample:
    """
    Collate function for DataLoader.

    Pads sequences to same length and creates attention masks.
    Also prepares target input (shifted right) and output.

    Args:
        batch: List of sample dicts from dataset.
        pad_token_id: Token ID for padding.
        bos_token_id: Token ID for beginning of sequence.
        eos_token_id: Token ID for end of sequence.

    Returns:
        BatchedSample with padded tensors.
    """
    # Get max lengths
    src_lens = [len(s["src_tokens"]) for s in batch]
    tgt_lens = [len(s["tgt_tokens"]) for s in batch]
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)

    batch_size = len(batch)

    # Initialize tensors
    src = torch.full((batch_size, max_src_len), pad_token_id, dtype=torch.long)
    tgt_input = torch.full((batch_size, max_tgt_len), pad_token_id, dtype=torch.long)
    tgt_output = torch.full((batch_size, max_tgt_len), pad_token_id, dtype=torch.long)

    # Stack mood embeddings
    mood_embeddings = torch.stack([s["mood_embedding"] for s in batch])

    # Fill tensors
    for i, sample in enumerate(batch):
        src_len = len(sample["src_tokens"])
        tgt_len = len(sample["tgt_tokens"])

        # Source
        src[i, :src_len] = sample["src_tokens"]

        # Target input: [BOS, tok1, tok2, ..., tokN-1]
        # Target output: [tok1, tok2, ..., tokN, EOS] (shifted)
        tgt_tokens = sample["tgt_tokens"]

        # Input is original sequence (already has BOS)
        tgt_input[i, :tgt_len] = tgt_tokens

        # Output is shifted left by 1 (for next-token prediction)
        if tgt_len > 1:
            tgt_output[i, :tgt_len - 1] = tgt_tokens[1:]
            tgt_output[i, tgt_len - 1] = eos_token_id

    # Create padding masks (True = padded, False = real token)
    src_padding_mask = src == pad_token_id
    tgt_padding_mask = tgt_input == pad_token_id

    return BatchedSample(
        src=src,
        tgt_input=tgt_input,
        tgt_output=tgt_output,
        mood_embedding=mood_embeddings,
        src_padding_mask=src_padding_mask,
        tgt_padding_mask=tgt_padding_mask,
    )


# =============================================================================
# Data Loaders
# =============================================================================

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_token_id: int = 0,
) -> DataLoader:
    """
    Create a DataLoader with proper collation.

    Args:
        dataset: The dataset to load from.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of worker processes.
        pad_token_id: Padding token ID.

    Returns:
        Configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token_id=pad_token_id),
        pin_memory=True,
        drop_last=False,
    )


def create_phase1_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 32,
    max_seq_len: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create DataLoader for Phase 1 (reconstruction) training.

    Args:
        data_path: Path to training data.
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        shuffle: Whether to shuffle.
        num_workers: Number of workers.

    Returns:
        DataLoader for Phase 1.
    """
    dataset = ReconstructionDataset(
        data_path=data_path,
        max_seq_len=max_seq_len,
    )
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def create_phase2_dataloader(
    data_path: Union[str, Path],
    batch_size: int = 32,
    max_seq_len: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    cache_embeddings: bool = True,
) -> DataLoader:
    """
    Create DataLoader for Phase 2 (conditioning) training.

    Args:
        data_path: Path to training data.
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        shuffle: Whether to shuffle.
        num_workers: Number of workers.
        cache_embeddings: Whether to cache mood embeddings.

    Returns:
        DataLoader for Phase 2.
    """
    dataset = MoodArpeggioDataset(
        data_path=data_path,
        phase=2,
        max_seq_len=max_seq_len,
        cache_embeddings=cache_embeddings,
    )
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# =============================================================================
# Train/Validation Split
# =============================================================================

def split_dataset(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Dataset to split.
        val_ratio: Fraction for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    from torch.utils.data import Subset

    # Get indices
    n = len(dataset)
    indices = list(range(n))

    # Shuffle with seed
    rng = random.Random(seed)
    rng.shuffle(indices)

    # Split
    val_size = int(n * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Split: {len(train_indices)} train, {len(val_indices)} validation")

    return train_dataset, val_dataset
