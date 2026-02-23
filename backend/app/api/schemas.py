"""
Request and response schemas for the arpeggio generation API.
"""

from typing import List, Optional

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
    pattern: str = Field(
        "ascending",
        description="Arpeggio pattern: ascending, descending, up_down, down_up",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional random seed for reproducible output",
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
    def validate_pattern(cls, v: str) -> str:
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
