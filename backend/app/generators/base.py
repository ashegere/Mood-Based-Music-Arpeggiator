"""
Abstract base class for all arpeggio generation backends.

New backends (e.g. a diffusion model, a rule-only engine) need only:
  1. Subclass ``BaseGenerator``.
  2. Implement ``load()``, ``generate()``, ``is_ready``, and ``name``.
  3. Pass an instance to ``app.api.dependencies.set_generator()`` at startup.

The rest of the application depends only on this interface, so the active
backend can be swapped without touching routes, schemas, or main.py logic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transfer objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRequest:
    """Validated, backend-agnostic request for arpeggio generation."""

    key: str
    scale: str
    tempo: int
    note_count: int
    mood: str
    octave: int
    seed: Optional[int] = field(default=None)
    pattern: Optional[str] = field(default=None)
    bars: int = field(default=1)

    # ---- Per-request sampling overrides ---------------------------------
    # ``None`` means "use the generator's instance-level default".
    # When set, these take precedence over the values passed to __init__.
    temperature: Optional[float] = field(default=None)
    """Sampling temperature (0.01–2.0). Lower → more conservative."""

    top_k: Optional[int] = field(default=None)
    """Top-k filter width. 0 disables; keep only the k most likely tokens."""

    top_p: Optional[float] = field(default=None)
    """Nucleus sampling threshold (0 < top_p ≤ 1.0). 1.0 disables."""

    repetition_penalty: Optional[float] = field(default=None)
    """Pitch repetition penalty (≥ 1.0). 1.0 disables."""

    max_length: Optional[int] = field(default=None)
    """Maximum number of new tokens to generate (overrides instance default)."""


@dataclass(frozen=True)
class NoteResult:
    """A single MIDI note event in the generation result."""

    pitch: int        # MIDI pitch 0-127
    velocity: int     # MIDI velocity 1-127
    position: float   # Start time in beats from the beginning
    duration: float   # Duration in beats


@dataclass(frozen=True)
class GenerationResult:
    """Complete result returned by any ``BaseGenerator.generate()`` call."""

    midi_bytes: bytes
    notes: List[NoteResult]
    note_count: int
    duration_seconds: float
    key: str
    scale: str
    tempo: int
    mood: str
    pattern_used: str
    # Actual sampling hyperparameters used for this generation call.
    # Populated by PretrainedMusicTransformerGenerator; defaults to {} for
    # backends that don't expose per-request sampling (backward-compatible).
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    # Mood alignment score in [0, 1] from MoodAlignmentScorer, or None when
    # the classifier checkpoint is not loaded.  Reflects the highest score
    # achieved across all regeneration attempts.
    alignment_score: Optional[float] = field(default=None)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseGenerator(ABC):
    """
    Interface all arpeggio generation backends must implement.

    Lifecycle
    ---------
    1. Instantiate the concrete subclass (``__init__`` may accept config).
    2. Call ``load()`` once — allocates heavy resources (model weights, etc.).
    3. Call ``generate()`` for each request — must be thread-safe.

    Properties
    ----------
    is_ready : bool
        ``True`` once ``load()`` has completed successfully.
    name : str
        Human-readable identifier, e.g. ``"custom-transformer"``.
    """

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """
        Allocate any heavy resources (model weights, embeddings, etc.).

        Called once at application startup.  Implementations should be
        idempotent — calling ``load()`` a second time should either be a
        no-op or reload cleanly.

        Raises:
            FileNotFoundError: If required model artefacts are missing.
            RuntimeError: If resource allocation fails.
        """

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate a mood-conditioned arpeggio.

        Must be thread-safe — FastAPI may call this from multiple
        concurrent request handler coroutines.

        Args:
            request: Validated, backend-agnostic generation parameters.

        Returns:
            ``GenerationResult`` containing MIDI bytes, note events,
            and metadata suitable for building an API response.

        Raises:
            RuntimeError: If ``load()`` has not been called or failed.
        """

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """``True`` after a successful ``load()`` call."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend, used in logs and health checks."""

    # ------------------------------------------------------------------
    # Optional hooks (concrete implementations may override)
    # ------------------------------------------------------------------

    def _log_generation(
        self,
        request: GenerationRequest,
        result: GenerationResult,
    ) -> None:
        """
        Emit a structured INFO log line after each successful generation.

        Override to add backend-specific fields (e.g. latency, token count).
        """
        logger.info(
            "[%s] key=%s scale=%s tempo=%d mood=%r notes=%d dur=%.2fs pattern=%s",
            self.name,
            result.key,
            result.scale,
            result.tempo,
            result.mood,
            result.note_count,
            result.duration_seconds,
            result.pattern_used,
        )
