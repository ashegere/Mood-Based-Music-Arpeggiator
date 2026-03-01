"""
Request and response schemas for the arpeggio generation API.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Validation sets (kept in sync with arpeggio_generator.py)
# ---------------------------------------------------------------------------

_VALID_KEYS = {
    "C", "C#", "Db", "D", "D#", "Eb", "E",
    "F", "F#", "Gb", "G", "G#", "Ab",
    "A", "A#", "Bb", "B",
}

_VALID_SCALES = {
    "aeolian", "blues", "chromatic", "dorian", "harmonic_minor",
    "ionian", "locrian", "lydian", "major", "melodic_minor",
    "minor", "mixolydian", "natural_minor",
    "pentatonic_major", "pentatonic_minor", "phrygian",
}

_VALID_PATTERNS = {"ascending", "descending", "up_down", "down_up"}


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class GenerateArpeggioRequest(BaseModel):
    """Input payload for POST /generate-arpeggio."""

    key: str = Field(
        "C",
        description="Musical key (C, C#, Db, D, D#, Eb, E, F, F#, Gb, G, G#, Ab, A, A#, Bb, B)",
    )
    scale: str = Field(
        "major",
        description=(
            "Scale type: major, minor, dorian, phrygian, lydian, mixolydian, "
            "aeolian, locrian, harmonic_minor, melodic_minor, natural_minor, "
            "ionian, pentatonic_major, pentatonic_minor, blues, chromatic"
        ),
    )
    tempo: int = Field(
        120,
        ge=20,
        le=400,
        description="Tempo in beats per minute (20–400)",
    )
    note_count: int = Field(
        16,
        ge=1,
        le=128,
        description="Number of notes to generate (1–128)",
    )
    mood: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Mood description text. Exact vocabulary matches are used directly; "
            "other text is mapped to the nearest known mood."
        ),
    )
    octave: int = Field(
        4,
        ge=0,
        le=8,
        description="Starting octave (0–8; 4 = middle C octave)",
    )
    pattern: Optional[str] = Field(
        None,
        description=(
            "Arpeggio pattern: ascending, descending, up_down, down_up. "
            "Omit to let the model choose based on mood."
        ),
    )
    bars: int = Field(
        1,
        ge=1,
        le=8,
        description=(
            "Number of times to repeat the generated pattern (1–8). "
            "The note sequence is tiled end-to-end so the MIDI contains "
            "bars × note_count notes total."
        ),
    )
    seed: Optional[int] = Field(
        None,
        description="Optional random seed for reproducible output",
    )

    # ---- Per-request sampling overrides ---------------------------------
    # All optional: omit to use the server-configured defaults.

    temperature: Optional[float] = Field(
        None,
        ge=0.01,
        le=2.0,
        description=(
            "Sampling temperature (0.01–2.0). "
            "Lower values produce more conservative, repetitive output; "
            "higher values produce more varied and surprising sequences. "
            "Omit to use the server default (0.95)."
        ),
    )
    top_k: Optional[int] = Field(
        None,
        ge=0,
        le=200,
        description=(
            "Top-k filtering: at each step keep only the k most likely tokens "
            "before sampling (0 = disabled). "
            "Lower values increase focus; 0 gives pure temperature sampling. "
            "Omit to use the server default (50)."
        ),
    )
    top_p: Optional[float] = Field(
        None,
        gt=0.0,
        le=1.0,
        description=(
            "Nucleus (top-p) sampling threshold (0 < top_p ≤ 1.0). "
            "Keeps the smallest set of tokens whose cumulative probability "
            "reaches top_p. 1.0 disables nucleus filtering. "
            "Can be combined with top_k. "
            "Omit to use the server default (1.0 = disabled)."
        ),
    )
    repetition_penalty: Optional[float] = Field(
        None,
        ge=1.0,
        le=5.0,
        description=(
            "Pitch repetition penalty (≥ 1.0). Applied only to PITCH tokens "
            "that have appeared in the recent context window. "
            "1.0 disables; 1.1–1.3 is a mild nudge toward variety; "
            ">2.0 is very aggressive. "
            "Omit to use the server default (1.0 = off)."
        ),
    )
    max_length: Optional[int] = Field(
        None,
        ge=16,
        le=4096,
        description=(
            "Maximum number of new tokens the model may generate for this "
            "request (16–4096). The effective budget is at least "
            "note_count × 8 regardless of this value. "
            "Omit to use the server default (1024)."
        ),
    )

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        if v not in _VALID_KEYS:
            raise ValueError(
                f"Invalid key '{v}'. Valid keys: {sorted(_VALID_KEYS)}"
            )
        return v

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v: str) -> str:
        v = v.lower()
        if v not in _VALID_SCALES:
            raise ValueError(
                f"Invalid scale '{v}'. Valid scales: {sorted(_VALID_SCALES)}"
            )
        return v

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.lower()
        if v not in _VALID_PATTERNS:
            raise ValueError(
                f"Invalid pattern '{v}'. Valid patterns: {sorted(_VALID_PATTERNS)}"
            )
        return v


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class NoteEvent(BaseModel):
    """A single MIDI note event in the generated arpeggio."""

    pitch: int = Field(..., ge=0, le=127, description="MIDI pitch (0–127; middle C = 60)")
    velocity: int = Field(..., ge=1, le=127, description="MIDI velocity (1–127)")
    position: float = Field(..., ge=0.0, description="Start time in beats from the beginning")
    duration: float = Field(..., gt=0.0, description="Duration in beats")


class SamplingParams(BaseModel):
    """
    Sampling hyperparameters actually used for a generation call.

    Echoed back in every response so clients can see the resolved values
    when they omitted fields (and the server default was applied) and so
    results can be exactly reproduced by passing these values back.
    """

    temperature: float = Field(..., description="Temperature used")
    top_k: int = Field(..., description="Top-k filter width used (0 = disabled)")
    top_p: float = Field(..., description="Nucleus threshold used (1.0 = disabled)")
    repetition_penalty: float = Field(..., description="Pitch repetition penalty used")
    max_length: int = Field(..., description="Token budget used for this request")


class GenerateArpeggioResponse(BaseModel):
    """Response payload from POST /generate-arpeggio."""

    midi_base64: str = Field(
        ...,
        description="Complete MIDI file encoded as a base64 string",
    )
    notes: List[NoteEvent] = Field(
        ...,
        description="Individual note events for client-side playback",
    )
    key: str = Field(..., description="Musical key used")
    scale: str = Field(..., description="Scale type used")
    tempo: int = Field(..., description="Tempo in BPM")
    mood: str = Field(..., description="Mood text as supplied in the request")
    note_count: int = Field(..., description="Number of notes in the output")
    duration_seconds: float = Field(..., description="Total playback duration in seconds")
    sampling: SamplingParams = Field(
        ...,
        description="Resolved sampling hyperparameters used for this generation",
    )
    alignment_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Mood-alignment probability [0–1] from the classifier head, or null "
            "when the classifier checkpoint is not loaded.  Reflects the best "
            "score achieved across all regeneration attempts."
        ),
    )
