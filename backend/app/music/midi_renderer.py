"""
MIDI renderer for converting token sequences back to MIDI files.

This module handles the final conversion from internal representation
to standard MIDI format. It enforces musical constraints and ensures
all values are safely clamped to valid MIDI ranges.

MIDI Specifications Enforced:
- Pitch: 0-127 (we constrain to 36-96 for musical range)
- Velocity: 0-127
- Channel: 0-15 (we use channel 0 by default)
- Timing: Absolute ticks from start

Dependencies:
- midiutil: For MIDI file creation
- Standard library only for core logic
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, BinaryIO
import io

# midiutil for MIDI file generation
from midiutil import MIDIFile

from app.music.arpeggio_generator import (
    Note,
    Arpeggio,
    SCALE_PATTERNS,
    KEY_OFFSETS,
    MIN_PITCH,
    MAX_PITCH,
)
from app.music.tokenization import (
    TokenSequence,
    Tokenizer,
    get_tokenizer,
)


# =============================================================================
# Constants
# =============================================================================

# MIDI constraints
MIDI_PITCH_MIN: int = 0
MIDI_PITCH_MAX: int = 127
MIDI_VELOCITY_MIN: int = 0
MIDI_VELOCITY_MAX: int = 127
MIDI_CHANNEL_DEFAULT: int = 0

# Default MIDI settings
DEFAULT_TICKS_PER_BEAT: int = 480  # Standard resolution
DEFAULT_TRACK_NAME: str = "Arpeggio"
DEFAULT_INSTRUMENT: int = 0  # Acoustic Grand Piano

# Time signature (4/4 by default)
DEFAULT_NUMERATOR: int = 4
DEFAULT_DENOMINATOR: int = 4


# =============================================================================
# Scale Constraint Enforcement
# =============================================================================

def get_scale_pitches(key: str, scale: str) -> set:
    """
    Get all valid MIDI pitches for a key/scale combination.

    Generates pitches across the full MIDI range that belong
    to the specified scale.

    Args:
        key: Musical key (e.g., "C", "F#").
        scale: Scale type (e.g., "major", "minor").

    Returns:
        Set of valid MIDI pitch numbers.
    """
    if key not in KEY_OFFSETS:
        key = "C"  # Fallback to C
    if scale.lower() not in SCALE_PATTERNS:
        scale = "major"  # Fallback to major

    root_offset = KEY_OFFSETS[key]
    pattern = SCALE_PATTERNS[scale.lower()]

    valid_pitches = set()

    # Generate all valid pitches across MIDI range
    for octave in range(-1, 11):  # Cover full MIDI range
        base_pitch = (octave + 1) * 12 + root_offset
        for interval in pattern:
            pitch = base_pitch + interval
            if MIDI_PITCH_MIN <= pitch <= MIDI_PITCH_MAX:
                valid_pitches.add(pitch)

    return valid_pitches


def snap_to_scale(pitch: int, valid_pitches: set) -> int:
    """
    Snap a pitch to the nearest valid scale pitch.

    If the pitch is already in the scale, return it unchanged.
    Otherwise, find the closest pitch that is in the scale.

    Args:
        pitch: Original MIDI pitch.
        valid_pitches: Set of valid pitches in the scale.

    Returns:
        Nearest valid pitch.
    """
    if pitch in valid_pitches:
        return pitch

    if not valid_pitches:
        return pitch  # No valid pitches, return original

    # Find nearest valid pitch
    sorted_pitches = sorted(valid_pitches)

    # Binary search for closest
    closest = min(sorted_pitches, key=lambda p: abs(p - pitch))
    return closest


def enforce_scale_constraint(
    pitch: int,
    key: str,
    scale: str,
    valid_pitches: Optional[set] = None,
) -> int:
    """
    Ensure a pitch belongs to the specified scale.

    Args:
        pitch: Original MIDI pitch.
        key: Musical key.
        scale: Scale type.
        valid_pitches: Pre-computed valid pitches (optional, for efficiency).

    Returns:
        Pitch snapped to scale if necessary.
    """
    if valid_pitches is None:
        valid_pitches = get_scale_pitches(key, scale)

    return snap_to_scale(pitch, valid_pitches)


# =============================================================================
# Value Clamping
# =============================================================================

def clamp_pitch(pitch: int) -> int:
    """
    Clamp pitch to valid MIDI and musical range.

    Args:
        pitch: Input pitch.

    Returns:
        Clamped pitch within MIN_PITCH to MAX_PITCH.
    """
    return max(MIN_PITCH, min(MAX_PITCH, pitch))


def clamp_velocity(velocity: int) -> int:
    """
    Clamp velocity to valid MIDI range.

    Args:
        velocity: Input velocity.

    Returns:
        Clamped velocity within 1-127 (0 = note off).
    """
    # Minimum velocity of 1 to ensure note is audible
    return max(1, min(MIDI_VELOCITY_MAX, velocity))


def clamp_duration(duration: float, min_duration: float = 0.0625) -> float:
    """
    Clamp duration to reasonable range.

    Args:
        duration: Input duration in beats.
        min_duration: Minimum allowed duration (default: 64th note).

    Returns:
        Clamped duration.
    """
    return max(min_duration, duration)


def clamp_position(position: float) -> float:
    """
    Ensure position is non-negative.

    Args:
        position: Input position in beats.

    Returns:
        Non-negative position.
    """
    return max(0.0, position)


# =============================================================================
# MIDI Event Data Structure
# =============================================================================

@dataclass
class MIDIEvent:
    """
    Represents a single MIDI note event.

    All values are validated and clamped to valid ranges.

    Attributes:
        pitch: MIDI pitch (0-127).
        velocity: MIDI velocity (1-127).
        start_time: Start time in beats.
        duration: Duration in beats.
        channel: MIDI channel (0-15).
    """
    pitch: int
    velocity: int
    start_time: float
    duration: float
    channel: int = MIDI_CHANNEL_DEFAULT

    def __post_init__(self):
        """Validate and clamp all values."""
        self.pitch = clamp_pitch(self.pitch)
        self.velocity = clamp_velocity(self.velocity)
        self.start_time = clamp_position(self.start_time)
        self.duration = clamp_duration(self.duration)
        self.channel = max(0, min(15, self.channel))


# =============================================================================
# MIDI Renderer
# =============================================================================

class MIDIRenderer:
    """
    Renders musical data to MIDI format.

    Handles conversion from:
    - Note objects
    - Arpeggio objects
    - Token sequences

    All rendering enforces scale constraints and value clamping.
    """

    def __init__(
        self,
        enforce_scale: bool = True,
        default_tempo: int = 120,
        default_instrument: int = DEFAULT_INSTRUMENT,
    ):
        """
        Initialize the MIDI renderer.

        Args:
            enforce_scale: Whether to snap pitches to scale.
            default_tempo: Default tempo if not specified.
            default_instrument: MIDI instrument number (0-127).
        """
        self.enforce_scale = enforce_scale
        self.default_tempo = default_tempo
        self.default_instrument = default_instrument
        self._tokenizer = get_tokenizer()

    def notes_to_events(
        self,
        notes: List[Note],
        key: str = "C",
        scale: str = "major",
    ) -> List[MIDIEvent]:
        """
        Convert Note objects to MIDIEvents with scale enforcement.

        Args:
            notes: List of Note objects.
            key: Musical key for scale constraint.
            scale: Scale type for constraint.

        Returns:
            List of validated MIDIEvent objects.
        """
        events = []

        # Pre-compute valid pitches for efficiency
        valid_pitches = get_scale_pitches(key, scale) if self.enforce_scale else None

        for note in notes:
            pitch = note.pitch

            # Enforce scale constraint if enabled
            if self.enforce_scale and valid_pitches:
                pitch = snap_to_scale(pitch, valid_pitches)

            event = MIDIEvent(
                pitch=pitch,
                velocity=note.velocity,
                start_time=note.position,
                duration=note.duration,
            )
            events.append(event)

        return events

    def render_to_midi(
        self,
        events: List[MIDIEvent],
        tempo: int,
        track_name: str = DEFAULT_TRACK_NAME,
    ) -> MIDIFile:
        """
        Create a MIDIFile from events.

        Args:
            events: List of MIDIEvent objects.
            tempo: Tempo in BPM.
            track_name: Name for the MIDI track.

        Returns:
            MIDIFile object ready for export.
        """
        # Create MIDI file with one track
        midi = MIDIFile(
            numTracks=1,
            ticks_per_quarternote=DEFAULT_TICKS_PER_BEAT,
        )

        track = 0
        time = 0  # Start at beat 0

        # Set track name and tempo
        midi.addTrackName(track, time, track_name)
        midi.addTempo(track, time, tempo)

        # Set instrument (program change)
        midi.addProgramChange(track, MIDI_CHANNEL_DEFAULT, time, self.default_instrument)

        # Add time signature
        midi.addTimeSignature(
            track,
            time,
            DEFAULT_NUMERATOR,
            int(DEFAULT_DENOMINATOR).bit_length() - 1,  # Denominator as power of 2
            24,  # MIDI clocks per metronome tick
            8,   # 32nd notes per MIDI quarter note
        )

        # Add all note events
        for event in events:
            midi.addNote(
                track=track,
                channel=event.channel,
                pitch=event.pitch,
                time=event.start_time,
                duration=event.duration,
                volume=event.velocity,
            )

        return midi

    def render_notes(
        self,
        notes: List[Note],
        tempo: int,
        key: str = "C",
        scale: str = "major",
        track_name: str = DEFAULT_TRACK_NAME,
    ) -> MIDIFile:
        """
        Render a list of notes to MIDI.

        Args:
            notes: List of Note objects.
            tempo: Tempo in BPM.
            key: Musical key.
            scale: Scale type.
            track_name: MIDI track name.

        Returns:
            MIDIFile object.
        """
        events = self.notes_to_events(notes, key, scale)
        return self.render_to_midi(events, tempo, track_name)

    def render_arpeggio(
        self,
        arpeggio: Arpeggio,
        track_name: str = DEFAULT_TRACK_NAME,
    ) -> MIDIFile:
        """
        Render an Arpeggio to MIDI.

        Args:
            arpeggio: Arpeggio object to render.
            track_name: MIDI track name.

        Returns:
            MIDIFile object.
        """
        return self.render_notes(
            notes=arpeggio.notes,
            tempo=arpeggio.tempo,
            key=arpeggio.key,
            scale=arpeggio.scale,
            track_name=track_name,
        )

    def render_token_sequence(
        self,
        token_sequence: TokenSequence,
        track_name: str = DEFAULT_TRACK_NAME,
    ) -> MIDIFile:
        """
        Render a TokenSequence to MIDI.

        Args:
            token_sequence: Token sequence to render.
            track_name: MIDI track name.

        Returns:
            MIDIFile object.
        """
        notes, key, scale, tempo = self._tokenizer.detokenize(token_sequence)
        return self.render_notes(
            notes=notes,
            tempo=tempo,
            key=key,
            scale=scale,
            track_name=track_name,
        )

    def render_token_ids(
        self,
        token_ids: List[int],
        track_name: str = DEFAULT_TRACK_NAME,
    ) -> MIDIFile:
        """
        Render token IDs to MIDI.

        Args:
            token_ids: List of token IDs.
            track_name: MIDI track name.

        Returns:
            MIDIFile object.
        """
        notes, key, scale, tempo = self._tokenizer.detokenize_ids(token_ids)
        return self.render_notes(
            notes=notes,
            tempo=tempo,
            key=key,
            scale=scale,
            track_name=track_name,
        )


# =============================================================================
# Export Functions
# =============================================================================

def midi_to_bytes(midi_file: MIDIFile) -> bytes:
    """
    Convert MIDIFile to bytes for transmission/storage.

    Args:
        midi_file: MIDIFile object.

    Returns:
        MIDI file as bytes.
    """
    buffer = io.BytesIO()
    midi_file.writeFile(buffer)
    buffer.seek(0)
    return buffer.read()


def save_midi(midi_file: MIDIFile, filepath: str) -> None:
    """
    Save MIDIFile to disk.

    Args:
        midi_file: MIDIFile object.
        filepath: Path to save the file.
    """
    with open(filepath, "wb") as f:
        midi_file.writeFile(f)


def render_arpeggio_to_bytes(
    arpeggio: Arpeggio,
    enforce_scale: bool = True,
) -> bytes:
    """
    Convenience function to render arpeggio directly to bytes.

    Args:
        arpeggio: Arpeggio to render.
        enforce_scale: Whether to enforce scale constraints.

    Returns:
        MIDI file as bytes.
    """
    renderer = MIDIRenderer(enforce_scale=enforce_scale)
    midi_file = renderer.render_arpeggio(arpeggio)
    return midi_to_bytes(midi_file)


def render_notes_to_bytes(
    notes: List[Note],
    tempo: int,
    key: str = "C",
    scale: str = "major",
    enforce_scale: bool = True,
) -> bytes:
    """
    Convenience function to render notes directly to bytes.

    Args:
        notes: List of notes to render.
        tempo: Tempo in BPM.
        key: Musical key.
        scale: Scale type.
        enforce_scale: Whether to enforce scale constraints.

    Returns:
        MIDI file as bytes.
    """
    renderer = MIDIRenderer(enforce_scale=enforce_scale)
    midi_file = renderer.render_notes(notes, tempo, key, scale)
    return midi_to_bytes(midi_file)


def render_tokens_to_bytes(
    token_ids: List[int],
    enforce_scale: bool = True,
) -> bytes:
    """
    Convenience function to render tokens directly to bytes.

    Args:
        token_ids: List of token IDs.
        enforce_scale: Whether to enforce scale constraints.

    Returns:
        MIDI file as bytes.
    """
    renderer = MIDIRenderer(enforce_scale=enforce_scale)
    midi_file = renderer.render_token_ids(token_ids)
    return midi_to_bytes(midi_file)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_midi_output(midi_bytes: bytes) -> bool:
    """
    Validate that bytes represent a valid MIDI file.

    Checks for MIDI file header signature.

    Args:
        midi_bytes: Bytes to validate.

    Returns:
        True if valid MIDI signature, False otherwise.
    """
    if len(midi_bytes) < 4:
        return False
    # MIDI files start with "MThd" header
    return midi_bytes[:4] == b"MThd"


def get_midi_info(midi_bytes: bytes) -> dict:
    """
    Extract basic info from MIDI bytes.

    Args:
        midi_bytes: MIDI file bytes.

    Returns:
        Dictionary with basic MIDI info.
    """
    info = {
        "valid": validate_midi_output(midi_bytes),
        "size_bytes": len(midi_bytes),
    }

    if info["valid"] and len(midi_bytes) >= 14:
        # Parse header for basic info
        # Format type at bytes 8-9
        format_type = int.from_bytes(midi_bytes[8:10], byteorder="big")
        # Number of tracks at bytes 10-11
        num_tracks = int.from_bytes(midi_bytes[10:12], byteorder="big")
        # Ticks per beat at bytes 12-13
        ticks_per_beat = int.from_bytes(midi_bytes[12:14], byteorder="big")

        info["format"] = format_type
        info["tracks"] = num_tracks
        info["ticks_per_beat"] = ticks_per_beat

    return info
