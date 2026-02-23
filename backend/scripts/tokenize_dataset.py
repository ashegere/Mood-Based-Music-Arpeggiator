#!/usr/bin/env python3
"""
MIDI Tokenizer for Music Transformer Training

Converts MIDI files into event-based token sequences for training
a mood-conditioned music transformer.

Token Vocabulary:
- Special tokens: PAD, BOS, EOS, UNK
- Mood tokens: MOOD_melancholic, MOOD_dreamy, MOOD_energetic, MOOD_tense, etc.
- Note events: NOTE_ON_0 to NOTE_ON_127, NOTE_OFF_0 to NOTE_OFF_127
- Time shifts: TIME_SHIFT_1 to TIME_SHIFT_100 (10ms resolution, max 1 second)
- Velocity: VELOCITY_1 to VELOCITY_32 (quantized to 32 bins)

Output Format:
- tokenized_dataset.pt containing:
  - input_ids: Token sequences (padded)
  - attention_mask: Attention masks
  - mood_labels: Mood label indices
  - vocab: Vocabulary mapping
  - config: Tokenizer configuration

Usage:
    python scripts/tokenize_dataset.py \
        --midi-dir data/training/augmented \
        --metadata data/training/arpeggio_mood_data_updated.json \
        --output data/training/tokenized_dataset.pt

    # With custom max length
    python scripts/tokenize_dataset.py \
        --midi-dir data/training/augmented \
        --metadata data/training/arpeggio_mood_data_updated.json \
        --output data/training/tokenized_dataset.pt \
        --max-length 1024
"""

import argparse
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pretty_midi
import torch
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TokenizerConfig:
    """Configuration for the MIDI tokenizer."""
    # Time resolution
    time_resolution_ms: int = 10  # 10ms per time shift unit
    max_time_shift: int = 100  # Max time shift tokens (1 second)

    # Velocity quantization
    num_velocity_bins: int = 32  # Quantize velocity to 32 bins

    # Pitch range
    min_pitch: int = 0
    max_pitch: int = 127

    # Sequence length
    max_sequence_length: int = 512

    # Valid moods
    valid_moods: Tuple[str, ...] = (
        "melancholic", "dreamy", "energetic", "tense",
        "happy", "sad", "calm", "dark", "joyful", "uplifting",
        "intense", "peaceful", "dramatic", "epic", "mysterious",
        "romantic", "neutral", "flowing", "ominous"
    )


# =============================================================================
# Vocabulary Builder
# =============================================================================

class MIDIVocabulary:
    """
    Vocabulary for MIDI event tokens.

    Token structure:
    - 0: PAD
    - 1: BOS (beginning of sequence)
    - 2: EOS (end of sequence)
    - 3: UNK (unknown)
    - 4-N: MOOD_* tokens
    - N+1 to N+128: NOTE_ON_0 to NOTE_ON_127
    - N+129 to N+256: NOTE_OFF_0 to NOTE_OFF_127
    - N+257 to N+356: TIME_SHIFT_1 to TIME_SHIFT_100
    - N+357 to N+388: VELOCITY_1 to VELOCITY_32
    """

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.token_to_id: Dict[str, int] = OrderedDict()
        self.id_to_token: Dict[int, str] = OrderedDict()
        self.mood_to_id: Dict[str, int] = {}

        self._build_vocabulary()

    def _build_vocabulary(self) -> None:
        """Build the complete vocabulary."""
        idx = 0

        # Special tokens
        for token in ["PAD", "BOS", "EOS", "UNK"]:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Mood tokens
        for mood in self.config.valid_moods:
            token = f"MOOD_{mood}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            self.mood_to_id[mood] = idx
            idx += 1

        # NOTE_ON tokens (0-127)
        for pitch in range(self.config.min_pitch, self.config.max_pitch + 1):
            token = f"NOTE_ON_{pitch}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # NOTE_OFF tokens (0-127)
        for pitch in range(self.config.min_pitch, self.config.max_pitch + 1):
            token = f"NOTE_OFF_{pitch}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # TIME_SHIFT tokens (1-100)
        for t in range(1, self.config.max_time_shift + 1):
            token = f"TIME_SHIFT_{t}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # VELOCITY tokens (1-32)
        for v in range(1, self.config.num_velocity_bins + 1):
            token = f"VELOCITY_{v}"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["PAD"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["BOS"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["EOS"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["UNK"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode_token(self, token: str) -> int:
        """Convert token string to ID."""
        return self.token_to_id.get(token, self.unk_token_id)

    def decode_token(self, token_id: int) -> str:
        """Convert token ID to string."""
        return self.id_to_token.get(token_id, "UNK")

    def encode_mood(self, mood: str) -> int:
        """Get token ID for a mood."""
        mood_lower = mood.lower().strip()
        return self.mood_to_id.get(mood_lower, self.unk_token_id)

    def get_mood_token(self, mood: str) -> str:
        """Get mood token string."""
        return f"MOOD_{mood.lower().strip()}"


# =============================================================================
# MIDI Tokenizer
# =============================================================================

class MIDITokenizer:
    """
    Tokenizes MIDI files into event sequences.

    Event sequence format:
    [BOS] [MOOD_*] [VELOCITY_*] [NOTE_ON_*] [TIME_SHIFT_*]... [EOS]
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.vocab = MIDIVocabulary(self.config)

    def quantize_velocity(self, velocity: int) -> int:
        """Quantize MIDI velocity (0-127) to bins (1-32)."""
        # Clamp to valid range
        velocity = max(1, min(127, velocity))
        # Map to 1-32 range
        bin_idx = int(velocity / 128 * self.config.num_velocity_bins) + 1
        return min(bin_idx, self.config.num_velocity_bins)

    def quantize_time(self, time_seconds: float) -> List[int]:
        """
        Quantize time delta to TIME_SHIFT tokens.

        Returns list of time shift values (may need multiple tokens
        for times > 1 second).
        """
        time_ms = int(time_seconds * 1000)
        time_units = time_ms // self.config.time_resolution_ms

        if time_units <= 0:
            return []

        shifts = []
        while time_units > 0:
            shift = min(time_units, self.config.max_time_shift)
            shifts.append(shift)
            time_units -= shift

        return shifts

    def tokenize_midi(self, midi_path: str, mood: Optional[str] = None) -> List[int]:
        """
        Tokenize a MIDI file into event tokens.

        Args:
            midi_path: Path to MIDI file.
            mood: Optional mood label for conditioning.

        Returns:
            List of token IDs.
        """
        # Load MIDI
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            raise ValueError(f"Failed to load MIDI {midi_path}: {e}")

        # Collect all note events across instruments
        events = []
        for instrument in pm.instruments:
            for note in instrument.notes:
                # Note on event
                events.append({
                    "type": "note_on",
                    "time": note.start,
                    "pitch": note.pitch,
                    "velocity": note.velocity,
                })
                # Note off event
                events.append({
                    "type": "note_off",
                    "time": note.end,
                    "pitch": note.pitch,
                })

        # Sort by time
        events.sort(key=lambda x: (x["time"], x["type"] == "note_on"))

        # Convert to tokens
        tokens = []

        # Add BOS
        tokens.append(self.vocab.bos_token_id)

        # Add mood token if provided
        if mood:
            mood_token = self.vocab.get_mood_token(mood)
            mood_id = self.vocab.encode_token(mood_token)
            if mood_id != self.vocab.unk_token_id:
                tokens.append(mood_id)

        # Process events
        current_time = 0.0
        current_velocity = None

        for event in events:
            # Add time shift if needed
            time_delta = event["time"] - current_time
            if time_delta > 0:
                time_shifts = self.quantize_time(time_delta)
                for shift in time_shifts:
                    token = f"TIME_SHIFT_{shift}"
                    tokens.append(self.vocab.encode_token(token))
                current_time = event["time"]

            if event["type"] == "note_on":
                # Add velocity if changed
                velocity_bin = self.quantize_velocity(event["velocity"])
                if velocity_bin != current_velocity:
                    token = f"VELOCITY_{velocity_bin}"
                    tokens.append(self.vocab.encode_token(token))
                    current_velocity = velocity_bin

                # Add note on
                token = f"NOTE_ON_{event['pitch']}"
                tokens.append(self.vocab.encode_token(token))

            elif event["type"] == "note_off":
                # Add note off
                token = f"NOTE_OFF_{event['pitch']}"
                tokens.append(self.vocab.encode_token(token))

        # Add EOS
        tokens.append(self.vocab.eos_token_id)

        return tokens

    def pad_sequence(
        self,
        tokens: List[int],
        max_length: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Pad or truncate sequence to max_length.

        Returns:
            Tuple of (padded_tokens, attention_mask)
        """
        max_len = max_length or self.config.max_sequence_length

        if len(tokens) > max_len:
            # Truncate (keep BOS at start, add EOS at end)
            tokens = tokens[:max_len - 1] + [self.vocab.eos_token_id]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)

        # Pad if needed
        padding_length = max_len - len(tokens)
        if padding_length > 0:
            tokens = tokens + [self.vocab.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        return tokens, attention_mask

    def decode(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs back to token strings."""
        return [self.vocab.decode_token(tid) for tid in token_ids]


# =============================================================================
# Dataset Builder
# =============================================================================

class DatasetBuilder:
    """
    Builds a tokenized dataset from MIDI files and metadata.
    """

    def __init__(
        self,
        midi_dir: str,
        metadata_path: str,
        config: Optional[TokenizerConfig] = None,
    ):
        self.midi_dir = Path(midi_dir)
        self.metadata_path = Path(metadata_path)
        self.config = config or TokenizerConfig()
        self.tokenizer = MIDITokenizer(self.config)

        # Load metadata
        self.metadata = self._load_metadata()

        # Track statistics
        self.stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "total_tokens": 0,
            "avg_sequence_length": 0,
            "mood_distribution": {},
        }

    def _load_metadata(self) -> Dict[str, str]:
        """Load metadata mapping filename -> mood."""
        with open(self.metadata_path, "r") as f:
            data = json.load(f)

        # Handle array format
        if isinstance(data, list):
            metadata = {}
            for entry in data:
                filename = entry.get("Filename") or entry.get("filename")
                mood = entry.get("Mood Description") or entry.get("mood")
                if filename and mood:
                    metadata[filename] = mood.lower().strip()
            return metadata

        # Handle dict format
        return {k: v.lower().strip() for k, v in data.items()}

    def build(self) -> Dict[str, torch.Tensor]:
        """
        Build the tokenized dataset.

        Returns:
            Dictionary containing:
            - input_ids: (N, max_length) tensor of token IDs
            - attention_mask: (N, max_length) tensor of attention masks
            - mood_labels: (N,) tensor of mood label indices
        """
        all_input_ids = []
        all_attention_masks = []
        all_mood_labels = []

        # Get mood label mapping
        mood_to_label = {mood: idx for idx, mood in enumerate(self.config.valid_moods)}

        # Process each file
        filenames = list(self.metadata.keys())
        self.stats["total_files"] = len(filenames)

        print(f"Processing {len(filenames)} MIDI files...")

        for filename in tqdm(filenames, desc="Tokenizing"):
            midi_path = self.midi_dir / filename

            if not midi_path.exists():
                self.stats["failed"] += 1
                continue

            mood = self.metadata.get(filename, "neutral")

            try:
                # Tokenize
                tokens = self.tokenizer.tokenize_midi(str(midi_path), mood)

                # Pad/truncate
                padded_tokens, attention_mask = self.tokenizer.pad_sequence(tokens)

                # Get mood label
                mood_label = mood_to_label.get(mood, mood_to_label.get("neutral", 0))

                # Append
                all_input_ids.append(padded_tokens)
                all_attention_masks.append(attention_mask)
                all_mood_labels.append(mood_label)

                # Update stats
                self.stats["successful"] += 1
                self.stats["total_tokens"] += len(tokens)
                self.stats["mood_distribution"][mood] = \
                    self.stats["mood_distribution"].get(mood, 0) + 1

            except Exception as e:
                print(f"\nWarning: Failed to tokenize {filename}: {e}")
                self.stats["failed"] += 1

        # Calculate average sequence length
        if self.stats["successful"] > 0:
            self.stats["avg_sequence_length"] = \
                self.stats["total_tokens"] / self.stats["successful"]

        # Convert to tensors
        input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(all_attention_masks, dtype=torch.long)
        mood_labels = torch.tensor(all_mood_labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mood_labels": mood_labels,
        }

    def save(self, output_path: str) -> None:
        """Build and save the dataset."""
        # Build dataset
        dataset = self.build()

        # Add vocabulary and config
        dataset["vocab"] = {
            "token_to_id": dict(self.tokenizer.vocab.token_to_id),
            "id_to_token": {int(k): v for k, v in self.tokenizer.vocab.id_to_token.items()},
            "mood_to_id": self.tokenizer.vocab.mood_to_id,
            "vocab_size": self.tokenizer.vocab.vocab_size,
            "pad_token_id": self.tokenizer.vocab.pad_token_id,
            "bos_token_id": self.tokenizer.vocab.bos_token_id,
            "eos_token_id": self.tokenizer.vocab.eos_token_id,
            "unk_token_id": self.tokenizer.vocab.unk_token_id,
        }

        dataset["config"] = asdict(self.config)
        dataset["stats"] = self.stats

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, output_path)

        print(f"\nDataset saved to: {output_path}")
        self._print_summary()

    def _print_summary(self) -> None:
        """Print dataset summary."""
        print("\n" + "=" * 60)
        print("TOKENIZATION SUMMARY")
        print("=" * 60)
        print(f"Total files:           {self.stats['total_files']}")
        print(f"Successfully tokenized: {self.stats['successful']}")
        print(f"Failed:                {self.stats['failed']}")
        print(f"Vocabulary size:       {self.tokenizer.vocab.vocab_size}")
        print(f"Max sequence length:   {self.config.max_sequence_length}")
        print(f"Avg sequence length:   {self.stats['avg_sequence_length']:.1f}")

        print("\nMood Distribution:")
        for mood, count in sorted(
            self.stats["mood_distribution"].items(),
            key=lambda x: -x[1]
        ):
            pct = count / self.stats["successful"] * 100 if self.stats["successful"] else 0
            print(f"  {mood:20} {count:4} ({pct:5.1f}%)")

        print("=" * 60)


# =============================================================================
# Utility Functions
# =============================================================================

def load_tokenized_dataset(path: str) -> Dict:
    """
    Load a tokenized dataset.

    Returns dictionary with:
    - input_ids: Token ID tensor
    - attention_mask: Attention mask tensor
    - mood_labels: Mood label tensor
    - vocab: Vocabulary info
    - config: Tokenizer config
    - stats: Dataset statistics
    """
    return torch.load(path)


def inspect_dataset(path: str) -> None:
    """Print information about a tokenized dataset."""
    data = load_tokenized_dataset(path)

    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(f"\nTensors:")
    print(f"  input_ids:      {data['input_ids'].shape}")
    print(f"  attention_mask: {data['attention_mask'].shape}")
    print(f"  mood_labels:    {data['mood_labels'].shape}")

    print(f"\nVocabulary:")
    print(f"  Size: {data['vocab']['vocab_size']}")
    print(f"  PAD: {data['vocab']['pad_token_id']}")
    print(f"  BOS: {data['vocab']['bos_token_id']}")
    print(f"  EOS: {data['vocab']['eos_token_id']}")

    print(f"\nConfig:")
    for key, value in data['config'].items():
        if key != "valid_moods":
            print(f"  {key}: {value}")

    print(f"\nStatistics:")
    for key, value in data['stats'].items():
        if key != "mood_distribution":
            print(f"  {key}: {value}")

    # Sample sequence
    print(f"\nSample sequence (first 20 tokens):")
    sample_ids = data['input_ids'][0][:20].tolist()
    id_to_token = data['vocab']['id_to_token']
    tokens = [id_to_token.get(str(tid), id_to_token.get(tid, "?")) for tid in sample_ids]
    print(f"  IDs:    {sample_ids}")
    print(f"  Tokens: {tokens}")

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize MIDI dataset for music transformer training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic tokenization
    python scripts/tokenize_dataset.py \\
        --midi-dir data/training/augmented \\
        --metadata data/training/arpeggio_mood_data_updated.json \\
        --output data/training/tokenized_dataset.pt

    # Custom configuration
    python scripts/tokenize_dataset.py \\
        --midi-dir data/training/augmented \\
        --metadata data/training/arpeggio_mood_data_updated.json \\
        --output data/training/tokenized_dataset.pt \\
        --max-length 1024 \\
        --time-resolution 10 \\
        --velocity-bins 32

    # Inspect existing dataset
    python scripts/tokenize_dataset.py --inspect data/training/tokenized_dataset.pt
        """,
    )

    parser.add_argument(
        "--midi-dir", "-m",
        type=str,
        help="Directory containing MIDI files",
    )

    parser.add_argument(
        "--metadata", "-d",
        type=str,
        help="Path to metadata JSON file",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="tokenized_dataset.pt",
        help="Output path for tokenized dataset",
    )

    parser.add_argument(
        "--max-length", "-l",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )

    parser.add_argument(
        "--time-resolution",
        type=int,
        default=10,
        help="Time resolution in ms (default: 10)",
    )

    parser.add_argument(
        "--velocity-bins",
        type=int,
        default=32,
        help="Number of velocity bins (default: 32)",
    )

    parser.add_argument(
        "--inspect", "-i",
        type=str,
        metavar="PATH",
        help="Inspect an existing tokenized dataset",
    )

    args = parser.parse_args()

    # Inspect mode
    if args.inspect:
        inspect_dataset(args.inspect)
        return 0

    # Validate required arguments
    if not args.midi_dir or not args.metadata:
        parser.error("--midi-dir and --metadata are required for tokenization")

    if not Path(args.midi_dir).is_dir():
        print(f"Error: MIDI directory not found: {args.midi_dir}")
        return 1

    if not Path(args.metadata).is_file():
        print(f"Error: Metadata file not found: {args.metadata}")
        return 1

    # Create config
    config = TokenizerConfig(
        max_sequence_length=args.max_length,
        time_resolution_ms=args.time_resolution,
        num_velocity_bins=args.velocity_bins,
    )

    # Build and save dataset
    builder = DatasetBuilder(
        midi_dir=args.midi_dir,
        metadata_path=args.metadata,
        config=config,
    )

    builder.save(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
