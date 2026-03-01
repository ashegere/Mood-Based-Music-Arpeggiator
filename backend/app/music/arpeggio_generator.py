"""
Rule-based arpeggio generator for symbolic music generation.

This module provides deterministic, scale-correct arpeggio generation
without mood conditioning. It serves as the foundation for mood-aware
generation in later stages.

Musical Assumptions:
- MIDI pitch range: 0-127 (we constrain to 36-96 for musical relevance)
- Middle C (C4) = MIDI pitch 60
- Default octave for generation starts at octave 4
- Velocity range: 0-127 (we use fixed velocity for neutral output)
- Time positions are in beats (quarter notes)
- Duration is in beats
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import random


# =============================================================================
# Constants
# =============================================================================

# MIDI pitch constraints
MIN_PITCH = 36  # C2 - low but still musical
MAX_PITCH = 96  # C7 - high but still musical
DEFAULT_OCTAVE = 4  # Starting octave for generation

# Default musical parameters
DEFAULT_VELOCITY = 80  # Mezzo-forte, neutral dynamic
DEFAULT_NOTE_DURATION = 0.5  # Eighth note in beats

# Rhythmic grid: notes are placed at fixed intervals
# Using 8th note grid (0.5 beats) for standard arpeggio feel
RHYTHMIC_GRID_UNIT = 0.5  # beats


# =============================================================================
# Scale Definitions
# =============================================================================

# Scale patterns as semitone intervals from root
# Each pattern defines the intervals that make up the scale
SCALE_PATTERNS: dict[str, tuple[int, ...]] = {
    # Major modes
    "major": (0, 2, 4, 5, 7, 9, 11),
    "ionian": (0, 2, 4, 5, 7, 9, 11),  # Same as major

    # Minor modes
    "minor": (0, 2, 3, 5, 7, 8, 10),  # Natural minor
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),  # Same as natural minor
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),  # Ascending form

    # Other modes
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),

    # Pentatonic scales
    "pentatonic_major": (0, 2, 4, 7, 9),
    "pentatonic_minor": (0, 3, 5, 7, 10),

    # Blues scale
    "blues": (0, 3, 5, 6, 7, 10),

    # Chromatic (all 12 semitones)
    "chromatic": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
}

# Key name to semitone offset from C
# C = 0, C# = 1, D = 2, etc.
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
    """
    Defines the order in which scale degrees are played.

    These patterns determine note selection from the available scale pitches.
    For a 7-note scale, indices cycle through the pattern.
    """
    ASCENDING = "ascending"        # 1-2-3-4-5-6-7-8...
    DESCENDING = "descending"      # 8-7-6-5-4-3-2-1...
    UP_DOWN = "up_down"            # 1-2-3-4-5-6-7-8-7-6-5-4-3-2-1...
    DOWN_UP = "down_up"            # 8-7-6-5-4-3-2-1-2-3-4-5-6-7-8...


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class Note:
    """
    Represents a single musical note in the arpeggio.

    Attributes:
        pitch: MIDI pitch number (0-127). Middle C = 60.
        duration: Note length in beats. 1.0 = quarter note.
        velocity: MIDI velocity (0-127). Controls loudness/attack.
        position: Start time in beats from the beginning. 0.0 = first beat.

    All attributes are immutable to ensure deterministic behavior.
    """
    pitch: int
    duration: float
    velocity: int
    position: float

    def __post_init__(self):
        """Validate note parameters are within MIDI spec."""
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"Pitch must be 0-127, got {self.pitch}")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"Velocity must be 0-127, got {self.velocity}")
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")
        if self.position < 0:
            raise ValueError(f"Position must be non-negative, got {self.position}")


@dataclass
class Arpeggio:
    """
    Container for a sequence of notes forming an arpeggio.

    Attributes:
        notes: Ordered list of Note objects.
        key: The musical key (e.g., "C", "F#", "Bb").
        scale: The scale type (e.g., "major", "minor", "dorian").
        tempo: Beats per minute for playback reference.
        seed: Random seed used for generation (ensures reproducibility).
    """
    notes: List[Note]
    key: str
    scale: str
    tempo: int
    seed: int

    @property
    def duration_beats(self) -> float:
        """Total duration of the arpeggio in beats."""
        if not self.notes:
            return 0.0
        last_note = self.notes[-1]
        return last_note.position + last_note.duration

    @property
    def duration_seconds(self) -> float:
        """Total duration of the arpeggio in seconds."""
        if self.tempo <= 0:
            return 0.0
        return self.duration_beats * (60.0 / self.tempo)

    @property
    def note_count(self) -> int:
        """Number of notes in the arpeggio."""
        return len(self.notes)


@dataclass
class GeneratorConfig:
    """
    Configuration for arpeggio generation.

    Attributes:
        key: Musical key (e.g., "C", "F#", "Bb").
        scale: Scale type (e.g., "major", "minor").
        tempo: Beats per minute (40-300 typical range).
        note_count: Number of notes to generate.
        seed: Random seed for deterministic output.
        pattern: Arpeggio pattern (ascending, descending, etc.).
        octave: Starting octave (0-8, where 4 = middle C octave).
        velocity: Fixed velocity for all notes (0-127).
        note_duration: Duration of each note in beats.
        grid_unit: Rhythmic grid spacing in beats.
    """
    key: str
    scale: str
    tempo: int
    note_count: int
    seed: int = 42
    pattern: ArpeggioPattern = ArpeggioPattern.ASCENDING
    octave: int = DEFAULT_OCTAVE
    velocity: int = DEFAULT_VELOCITY
    note_duration: float = DEFAULT_NOTE_DURATION
    grid_unit: float = RHYTHMIC_GRID_UNIT


# =============================================================================
# Validation
# =============================================================================

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_key(key: str) -> str:
    """
    Validate and normalize a musical key.

    Args:
        key: Key name (e.g., "C", "f#", "Bb").

    Returns:
        Normalized key name.

    Raises:
        ValidationError: If key is not recognized.
    """
    if key not in KEY_OFFSETS:
        valid_keys = sorted(set(k for k in KEY_OFFSETS.keys() if len(k) <= 2 and k[0].isupper()))
        raise ValidationError(
            f"Invalid key '{key}'. Valid keys: {', '.join(valid_keys)}"
        )
    return key


def validate_scale(scale: str) -> str:
    """
    Validate and normalize a scale name.

    Args:
        scale: Scale name (e.g., "major", "minor", "dorian").

    Returns:
        Normalized scale name (lowercase).

    Raises:
        ValidationError: If scale is not recognized.
    """
    scale_lower = scale.lower()
    if scale_lower not in SCALE_PATTERNS:
        valid_scales = sorted(SCALE_PATTERNS.keys())
        raise ValidationError(
            f"Invalid scale '{scale}'. Valid scales: {', '.join(valid_scales)}"
        )
    return scale_lower


def validate_tempo(tempo: int) -> int:
    """
    Validate tempo is within reasonable bounds.

    Args:
        tempo: Beats per minute.

    Returns:
        Validated tempo.

    Raises:
        ValidationError: If tempo is out of range.
    """
    if not isinstance(tempo, int):
        raise ValidationError(f"Tempo must be an integer, got {type(tempo).__name__}")
    if not 20 <= tempo <= 400:
        raise ValidationError(f"Tempo must be 20-400 BPM, got {tempo}")
    return tempo


def validate_note_count(note_count: int) -> int:
    """
    Validate note count is positive and reasonable.

    Args:
        note_count: Number of notes to generate.

    Returns:
        Validated note count.

    Raises:
        ValidationError: If note_count is invalid.
    """
    if not isinstance(note_count, int):
        raise ValidationError(f"Note count must be an integer, got {type(note_count).__name__}")
    if note_count < 1:
        raise ValidationError(f"Note count must be at least 1, got {note_count}")
    if note_count > 1000:
        raise ValidationError(f"Note count must be at most 1000, got {note_count}")
    return note_count


def validate_config(config: GeneratorConfig) -> GeneratorConfig:
    """
    Validate all configuration parameters.

    Args:
        config: Generator configuration to validate.

    Returns:
        Validated configuration (may have normalized values).

    Raises:
        ValidationError: If any parameter is invalid.
    """
    validated_key = validate_key(config.key)
    validated_scale = validate_scale(config.scale)
    validated_tempo = validate_tempo(config.tempo)
    validated_note_count = validate_note_count(config.note_count)

    # Validate octave
    if not 0 <= config.octave <= 8:
        raise ValidationError(f"Octave must be 0-8, got {config.octave}")

    # Validate velocity
    if not 0 <= config.velocity <= 127:
        raise ValidationError(f"Velocity must be 0-127, got {config.velocity}")

    # Validate note duration
    if config.note_duration <= 0:
        raise ValidationError(f"Note duration must be positive, got {config.note_duration}")

    # Validate grid unit
    if config.grid_unit <= 0:
        raise ValidationError(f"Grid unit must be positive, got {config.grid_unit}")

    return GeneratorConfig(
        key=validated_key,
        scale=validated_scale,
        tempo=validated_tempo,
        note_count=validated_note_count,
        seed=config.seed,
        pattern=config.pattern,
        octave=config.octave,
        velocity=config.velocity,
        note_duration=config.note_duration,
        grid_unit=config.grid_unit,
    )


# =============================================================================
# Scale and Pitch Utilities
# =============================================================================

def get_root_pitch(key: str, octave: int) -> int:
    """
    Calculate the MIDI pitch for a key at a given octave.

    MIDI pitch formula: pitch = (octave + 1) * 12 + semitone_offset
    This gives C4 (middle C) = 60.

    Args:
        key: Musical key (e.g., "C", "F#").
        octave: Octave number (4 = middle C octave).

    Returns:
        MIDI pitch number.
    """
    semitone_offset = KEY_OFFSETS[key]
    return (octave + 1) * 12 + semitone_offset


def build_scale_pitches(
    key: str,
    scale: str,
    octave: int,
    num_octaves: int = 2
) -> List[int]:
    """
    Build a list of MIDI pitches for a scale across multiple octaves.

    Args:
        key: Musical key.
        scale: Scale type.
        octave: Starting octave.
        num_octaves: Number of octaves to span.

    Returns:
        List of MIDI pitches in the scale, constrained to valid range.
    """
    root = get_root_pitch(key, octave)
    pattern = SCALE_PATTERNS[scale.lower()]

    pitches = []
    for oct_offset in range(num_octaves):
        for interval in pattern:
            pitch = root + interval + (oct_offset * 12)
            if MIN_PITCH <= pitch <= MAX_PITCH:
                pitches.append(pitch)

    # Add the octave above the last note for complete arpeggios
    final_pitch = root + (num_octaves * 12)
    if MIN_PITCH <= final_pitch <= MAX_PITCH:
        pitches.append(final_pitch)

    return pitches


# =============================================================================
# Pattern Generation
# =============================================================================

def generate_pitch_sequence(
    scale_pitches: List[int],
    note_count: int,
    pattern: ArpeggioPattern,
    rng: random.Random
) -> List[int]:
    """
    Generate a sequence of pitches following the specified pattern.

    The seeded RNG is used for a random starting position within the scale
    and for occasional octave displacements, so different seeds always
    produce different sequences even for the same key/scale/pattern.

    Args:
        scale_pitches: Available pitches from the scale.
        note_count: Number of pitches to generate.
        pattern: The arpeggio pattern to follow.
        rng: Seeded random number generator.

    Returns:
        List of MIDI pitches in pattern order.
    """
    if not scale_pitches:
        return []

    n = len(scale_pitches)
    # Random starting position so each seed produces a distinct sequence.
    start = rng.randint(0, n - 1)
    pitches = []

    if pattern == ArpeggioPattern.ASCENDING:
        for i in range(note_count):
            pitches.append(scale_pitches[(start + i) % n])

    elif pattern == ArpeggioPattern.DESCENDING:
        reversed_pitches = list(reversed(scale_pitches))
        for i in range(note_count):
            pitches.append(reversed_pitches[(start + i) % n])

    elif pattern == ArpeggioPattern.UP_DOWN:
        cycle_length = max(1, 2 * n - 2) if n > 1 else 1
        for i in range(note_count):
            pos = (start + i) % cycle_length
            if pos < n:
                pitches.append(scale_pitches[pos])
            else:
                desc_pos = cycle_length - pos
                pitches.append(scale_pitches[max(0, min(n - 1, desc_pos))])

    elif pattern == ArpeggioPattern.DOWN_UP:
        reversed_pitches = list(reversed(scale_pitches))
        cycle_length = max(1, 2 * n - 2) if n > 1 else 1
        for i in range(note_count):
            pos = (start + i) % cycle_length
            if pos < n:
                pitches.append(reversed_pitches[pos])
            else:
                desc_pos = cycle_length - pos
                pitches.append(reversed_pitches[max(0, min(n - 1, desc_pos))])

    # Occasional octave displacement: 15% chance per note to shift ±12 semitones.
    # Stays within valid MIDI range and preserves scale identity.
    result = []
    for p in pitches:
        if rng.random() < 0.15:
            shift = rng.choice([-12, 12])
            new_p = p + shift
            if MIN_PITCH <= new_p <= MAX_PITCH:
                p = new_p
        result.append(p)
    return result


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

    Args:
        scale_pitches:    Available pitches sorted ascending.
        note_count:       Number of pitches to generate.
        rng:              Seeded RNG — same seed always reproduces same output.
        step_weights:     Weights for steps [−2, −1, +1, +2] through the
                          scale-degree index.
        octave_jump_prob: Per-note probability [0, 1] of an octave shift.
        contour_bias:     Directional bias in [−1, 1].  −1 = fully descending,
                          +1 = fully ascending, 0 = neutral.
        start_region:     "low", "mid", or "high" — starting register.

    Returns:
        List of MIDI pitch values within [MIN_PITCH, MAX_PITCH].
    """
    n = len(scale_pitches)
    if n == 0:
        return []

    # Starting index based on the requested register.
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

        # Occasional octave displacement — same scale class, different register.
        if rng.random() < octave_jump_prob:
            shift = rng.choice([-12, 12])
            shifted = pitch + shift
            if MIN_PITCH <= shifted <= MAX_PITCH:
                pitch = shifted

        result.append(pitch)

        # Weighted random step with optional contour bias.
        w = list(step_weights)
        if contour_bias > 0:           # nudge ascending
            w[2] *= 1.0 + contour_bias
            w[3] *= 1.0 + contour_bias * 0.5
        elif contour_bias < 0:         # nudge descending
            w[0] *= 1.0 - contour_bias   # bias is negative → increases weight
            w[1] *= 1.0 - contour_bias * 0.5
        step = rng.choices(steps, weights=w)[0]
        idx += step

    return result


# =============================================================================
# Main Generator
# =============================================================================

class ArpeggioGenerator:
    """
    Rule-based arpeggio generator for neutral (non-mood-conditioned) output.

    This generator produces deterministic, scale-correct arpeggios with
    fixed rhythmic grid and velocity. It serves as the foundation for
    mood-aware generation in later stages.

    Usage:
        generator = ArpeggioGenerator()
        arpeggio = generator.generate(
            key="C",
            scale="major",
            tempo=120,
            note_count=8
        )

    All outputs are fully deterministic given the same seed.
    """

    def __init__(self, default_seed: int = 42):
        """
        Initialize the generator.

        Args:
            default_seed: Default random seed for reproducibility.
        """
        self.default_seed = default_seed

    def generate(
        self,
        key: str,
        scale: str,
        tempo: int,
        note_count: int,
        seed: Optional[int] = None,
        pattern: ArpeggioPattern = ArpeggioPattern.ASCENDING,
        octave: int = DEFAULT_OCTAVE,
        velocity: int = DEFAULT_VELOCITY,
        note_duration: float = DEFAULT_NOTE_DURATION,
        grid_unit: float = RHYTHMIC_GRID_UNIT,
    ) -> Arpeggio:
        """
        Generate a deterministic arpeggio.

        Args:
            key: Musical key (e.g., "C", "F#", "Bb").
            scale: Scale type (e.g., "major", "minor", "dorian").
            tempo: Beats per minute.
            note_count: Number of notes to generate.
            seed: Random seed for determinism. Uses default if None.
            pattern: Arpeggio pattern (default: ascending).
            octave: Starting octave (default: 4, middle C octave).
            velocity: Fixed velocity for all notes (default: 80).
            note_duration: Duration of each note in beats (default: 0.5).
            grid_unit: Rhythmic grid spacing in beats (default: 0.5).

        Returns:
            Arpeggio object containing the generated notes.

        Raises:
            ValidationError: If any input parameter is invalid.
        """
        # Build and validate config
        config = GeneratorConfig(
            key=key,
            scale=scale,
            tempo=tempo,
            note_count=note_count,
            seed=seed if seed is not None else self.default_seed,
            pattern=pattern,
            octave=octave,
            velocity=velocity,
            note_duration=note_duration,
            grid_unit=grid_unit,
        )
        config = validate_config(config)

        # Initialize seeded RNG for deterministic output
        rng = random.Random(config.seed)

        # Build scale pitches
        scale_pitches = build_scale_pitches(
            key=config.key,
            scale=config.scale,
            octave=config.octave,
            num_octaves=2,  # Span 2 octaves for musical range
        )

        if not scale_pitches:
            raise ValidationError(
                f"No valid pitches for {config.key} {config.scale} at octave {config.octave}"
            )

        # Generate pitch sequence based on pattern
        pitch_sequence = generate_pitch_sequence(
            scale_pitches=scale_pitches,
            note_count=config.note_count,
            pattern=config.pattern,
            rng=rng,
        )

        # Create notes on fixed rhythmic grid
        notes = []
        for i, pitch in enumerate(pitch_sequence):
            position = i * config.grid_unit
            note = Note(
                pitch=pitch,
                duration=config.note_duration,
                velocity=config.velocity,
                position=position,
            )
            notes.append(note)

        return Arpeggio(
            notes=notes,
            key=config.key,
            scale=config.scale,
            tempo=config.tempo,
            seed=config.seed,
        )

    def generate_from_config(self, config: GeneratorConfig) -> Arpeggio:
        """
        Generate an arpeggio from a configuration object.

        Args:
            config: Generator configuration.

        Returns:
            Arpeggio object containing the generated notes.
        """
        return self.generate(
            key=config.key,
            scale=config.scale,
            tempo=config.tempo,
            note_count=config.note_count,
            seed=config.seed,
            pattern=config.pattern,
            octave=config.octave,
            velocity=config.velocity,
            note_duration=config.note_duration,
            grid_unit=config.grid_unit,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_arpeggio(
    key: str,
    scale: str,
    tempo: int,
    note_count: int,
    seed: int = 42,
) -> Arpeggio:
    """
    Convenience function to generate a simple arpeggio.

    Uses default pattern (ascending), octave (4), velocity (80),
    and rhythmic settings.

    Args:
        key: Musical key (e.g., "C", "F#", "Bb").
        scale: Scale type (e.g., "major", "minor").
        tempo: Beats per minute.
        note_count: Number of notes to generate.
        seed: Random seed for determinism.

    Returns:
        Arpeggio object containing the generated notes.
    """
    generator = ArpeggioGenerator(default_seed=seed)
    return generator.generate(
        key=key,
        scale=scale,
        tempo=tempo,
        note_count=note_count,
        seed=seed,
    )


def get_available_scales() -> List[str]:
    """Return list of all available scale names."""
    return sorted(SCALE_PATTERNS.keys())


def get_available_keys() -> List[str]:
    """Return list of all available key names (normalized)."""
    return sorted(set(k for k in KEY_OFFSETS.keys() if len(k) <= 2 and k[0].isupper()))
