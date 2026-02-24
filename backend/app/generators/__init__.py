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
"""

from .base import BaseGenerator, GenerationRequest, GenerationResult, NoteResult
from .transformer import CustomTransformerGenerator
from .pretrained_transformer import PretrainedMusicTransformerGenerator

__all__ = [
    "BaseGenerator",
    "GenerationRequest",
    "GenerationResult",
    "NoteResult",
    "CustomTransformerGenerator",
    "PretrainedMusicTransformerGenerator",
]
