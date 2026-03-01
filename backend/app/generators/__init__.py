"""
Arpeggio generation backends.

Consumers should import from here rather than from the sub-modules so that
the public surface area stays stable even if internal file names change.

Available backends
------------------
CustomTransformerGenerator
    Mood-conditioned decoder-only transformer trained in-house on the
    411-token MIDI event vocabulary.  Uses teacher-forcing inference.

PretrainedMusicTransformerGenerator
    Decoder-only GPT-style transformer trained on the REMI-style bar/
    position/pitch/duration/velocity vocabulary (app.music.tokenization).
    Uses true autoregressive decoding with temperature + top-k sampling
    and scale-constrained PITCH token selection.
    Supports optional mood conditioning via a ``MoodConditioningModule``
    adapter loaded from a separate checkpoint.

Mood conditioning
-----------------
MoodAdapterConfig
    Dataclass holding adapter hyperparameters (stored with the checkpoint).
MoodConditioningModule
    Lightweight nn.Embedding adapter that maps mood labels → d_model vectors.
MoodFineTuner
    Fine-tuning harness: freezes backbone, trains only the adapter.
MoodSequenceDataset
    Dataset of (mood_label, token_sequence) pairs.
FineTuningConfig
    Hyperparameter dataclass for the fine-tuning run.
"""

from .base import BaseGenerator, GenerationRequest, GenerationResult, NoteResult
from .transformer import CustomTransformerGenerator
from .pretrained_transformer import (
    MoodAdapterConfig,
    MoodConditioningModule,
    PretrainedMusicTransformerGenerator,
)
from .mood_finetuner import FineTuningConfig, MoodFineTuner, MoodSequenceDataset
from .mood_classifier import ClassifierConfig, MoodClassifierHead, MoodAlignmentScorer

__all__ = [
    # Base interface
    "BaseGenerator",
    "GenerationRequest",
    "GenerationResult",
    "NoteResult",
    # Backends
    "CustomTransformerGenerator",
    "PretrainedMusicTransformerGenerator",
    # Mood conditioning
    "MoodAdapterConfig",
    "MoodConditioningModule",
    "FineTuningConfig",
    "MoodFineTuner",
    "MoodSequenceDataset",
    # Alignment classifier
    "ClassifierConfig",
    "MoodClassifierHead",
    "MoodAlignmentScorer",
]
