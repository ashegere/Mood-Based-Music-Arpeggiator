"""
Arpeggio generation backends.

Consumers should import from here rather than from the sub-modules so that
the public surface area stays stable even if internal file names change.
"""

from .base import BaseGenerator, GenerationRequest, GenerationResult, NoteResult
from .transformer import CustomTransformerGenerator

__all__ = [
    "BaseGenerator",
    "GenerationRequest",
    "GenerationResult",
    "NoteResult",
    "CustomTransformerGenerator",
]
