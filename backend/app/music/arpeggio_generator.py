"""
Rule-based arpeggio generator for symbolic music generation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List
import random


# =============================================================================
# Constants
# =============================================================================

MIN_PITCH = 36
MAX_PITCH = 96
DEFAULT_OCTAVE = 4


# =============================================================================
# Scale Definitions
# =============================================================================

SCALE_PATTERNS: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "pentatonic_major": (0, 2, 4, 7, 9),
    "pentatonic_minor": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "chromatic": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
}

KEY_OFFSETS: dict[str, int] = {
    "C": 0, "c": 0,
    "C#": 1, "c#": 1, "Db": 1, "db": 1,
    "D": 2, "d": 2,
    "D#": 3, "d#": 3, "Eb": 3, "eb": 3,
    "E": 4, "e": 4,
    "F": 5, "f": 5,
    "F#": 6, "f#": 6, "Gb": 6, "gb": 6,
    "G": 7, "g": 7,
    "G#": 8, "g#": 8, "Ab": 8, "ab": 8,
    "A": 9, "a": 9,
    "A#": 10, "a#": 10, "Bb": 10, "bb": 10,
    "B": 11, "b": 11,
}


# =============================================================================
# Arpeggio Patterns
# =============================================================================

class ArpeggioPattern(Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"
    UP_DOWN = "up_down"
    DOWN_UP = "down_up"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class Note:
    """A single musical note."""
    pitch: int
    duration: float
    velocity: int
    position: float

    def __post_init__(self):
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"Pitch must be 0-127, got {self.pitch}")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"Velocity must be 0-127, got {self.velocity}")
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")
        if self.position < 0:
            raise ValueError(f"Position must be non-negative, got {self.position}")


# =============================================================================
# Scale and Pitch Utilities
# =============================================================================

def get_root_pitch(key: str, octave: int) -> int:
    semitone_offset = KEY_OFFSETS[key]
    return (octave + 1) * 12 + semitone_offset


def build_scale_pitches(
    key: str,
    scale: str,
    octave: int,
    num_octaves: int = 2,
) -> List[int]:
    root = get_root_pitch(key, octave)
    pattern = SCALE_PATTERNS[scale.lower()]

    pitches = []
    for oct_offset in range(num_octaves):
        for interval in pattern:
            pitch = root + interval + (oct_offset * 12)
            if MIN_PITCH <= pitch <= MAX_PITCH:
                pitches.append(pitch)

    final_pitch = root + (num_octaves * 12)
    if MIN_PITCH <= final_pitch <= MAX_PITCH:
        pitches.append(final_pitch)

    return pitches


# =============================================================================
# Mood Pitch Walk
# =============================================================================

def generate_mood_pitch_sequence(
    scale_pitches: List[int],
    note_count: int,
    rng: random.Random,
    step_weights: List[float],
    octave_jump_prob: float,
    contour_bias: float,
    start_region: str,
) -> List[int]:
    """
    Generate a melodic pitch sequence using a mood-driven random walk.

    At each step the walk randomly moves −2, −1, +1, or +2 scale degrees,
    with weights tuned per mood.  A contour bias nudges the walk toward
    ascending or descending motion.  Occasional octave displacements add
    register variety without leaving the key.
    """
    n = len(scale_pitches)
    if n == 0:
        return []

    if start_region == "low":
        idx = rng.randint(0, max(0, n // 3))
    elif start_region == "high":
        idx = rng.randint(min(2 * n // 3, n - 1), n - 1)
    else:  # "mid"
        idx = rng.randint(n // 4, min(3 * n // 4, n - 1))

    steps = [-2, -1, +1, +2]
    result: List[int] = []

    for _ in range(note_count):
        idx = max(0, min(n - 1, idx))
        pitch = scale_pitches[idx]

        if rng.random() < octave_jump_prob:
            shift = rng.choice([-12, 12])
            shifted = pitch + shift
            if MIN_PITCH <= shifted <= MAX_PITCH:
                pitch = shifted

        result.append(pitch)

        w = list(step_weights)
        if contour_bias > 0:
            w[2] *= 1.0 + contour_bias
            w[3] *= 1.0 + contour_bias * 0.5
        elif contour_bias < 0:
            w[0] *= 1.0 - contour_bias
            w[1] *= 1.0 - contour_bias * 0.5
        step = rng.choices(steps, weights=w)[0]
        idx += step

    return result
