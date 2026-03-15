"""
MIDI renderer — converts Note objects to MIDI files.
"""

from dataclasses import dataclass
from typing import List, Optional
import io

from midiutil import MIDIFile

from app.music.arpeggio_generator import (
    Note,
    SCALE_PATTERNS,
    KEY_OFFSETS,
    MIN_PITCH,
    MAX_PITCH,
)


# =============================================================================
# Constants
# =============================================================================

MIDI_PITCH_MIN: int = 0
MIDI_PITCH_MAX: int = 127
MIDI_VELOCITY_MIN: int = 0
MIDI_VELOCITY_MAX: int = 127
MIDI_CHANNEL_DEFAULT: int = 0

DEFAULT_TICKS_PER_BEAT: int = 480
DEFAULT_TRACK_NAME: str = "Arpeggio"
DEFAULT_INSTRUMENT: int = 0

DEFAULT_NUMERATOR: int = 4
DEFAULT_DENOMINATOR: int = 4


# =============================================================================
# Scale Constraint Enforcement
# =============================================================================

def get_scale_pitches(key: str, scale: str) -> set:
    if key not in KEY_OFFSETS:
        key = "C"
    if scale.lower() not in SCALE_PATTERNS:
        scale = "major"

    root_offset = KEY_OFFSETS[key]
    pattern = SCALE_PATTERNS[scale.lower()]

    valid_pitches = set()
    for octave in range(-1, 11):
        base_pitch = (octave + 1) * 12 + root_offset
        for interval in pattern:
            pitch = base_pitch + interval
            if MIDI_PITCH_MIN <= pitch <= MIDI_PITCH_MAX:
                valid_pitches.add(pitch)

    return valid_pitches


def snap_to_scale(pitch: int, valid_pitches: set) -> int:
    if pitch in valid_pitches:
        return pitch
    if not valid_pitches:
        return pitch
    return min(sorted(valid_pitches), key=lambda p: abs(p - pitch))


def enforce_scale_constraint(
    pitch: int,
    key: str,
    scale: str,
    valid_pitches: Optional[set] = None,
) -> int:
    if valid_pitches is None:
        valid_pitches = get_scale_pitches(key, scale)
    return snap_to_scale(pitch, valid_pitches)


# =============================================================================
# Value Clamping
# =============================================================================

def clamp_pitch(pitch: int) -> int:
    return max(MIN_PITCH, min(MAX_PITCH, pitch))


def clamp_velocity(velocity: int) -> int:
    return max(1, min(MIDI_VELOCITY_MAX, velocity))


def clamp_duration(duration: float, min_duration: float = 0.0625) -> float:
    return max(min_duration, duration)


def clamp_position(position: float) -> float:
    return max(0.0, position)


# =============================================================================
# MIDI Event
# =============================================================================

@dataclass
class MIDIEvent:
    pitch: int
    velocity: int
    start_time: float
    duration: float
    channel: int = MIDI_CHANNEL_DEFAULT

    def __post_init__(self):
        self.pitch = clamp_pitch(self.pitch)
        self.velocity = clamp_velocity(self.velocity)
        self.start_time = clamp_position(self.start_time)
        self.duration = clamp_duration(self.duration)
        self.channel = max(0, min(15, self.channel))


# =============================================================================
# MIDI Renderer
# =============================================================================

class MIDIRenderer:
    def __init__(
        self,
        enforce_scale: bool = True,
        default_tempo: int = 120,
        default_instrument: int = DEFAULT_INSTRUMENT,
    ):
        self.enforce_scale = enforce_scale
        self.default_tempo = default_tempo
        self.default_instrument = default_instrument

    def notes_to_events(
        self,
        notes: List[Note],
        key: str = "C",
        scale: str = "major",
    ) -> List[MIDIEvent]:
        valid_pitches = get_scale_pitches(key, scale) if self.enforce_scale else None
        events = []
        for note in notes:
            pitch = note.pitch
            if self.enforce_scale and valid_pitches:
                pitch = snap_to_scale(pitch, valid_pitches)
            events.append(MIDIEvent(
                pitch=pitch,
                velocity=note.velocity,
                start_time=note.position,
                duration=note.duration,
            ))
        return events

    def render_to_midi(
        self,
        events: List[MIDIEvent],
        tempo: int,
        track_name: str = DEFAULT_TRACK_NAME,
    ) -> MIDIFile:
        midi = MIDIFile(numTracks=1, ticks_per_quarternote=DEFAULT_TICKS_PER_BEAT)
        track = 0
        midi.addTrackName(track, 0, track_name)
        midi.addTempo(track, 0, tempo)
        midi.addProgramChange(track, MIDI_CHANNEL_DEFAULT, 0, self.default_instrument)
        midi.addTimeSignature(
            track, 0,
            DEFAULT_NUMERATOR,
            int(DEFAULT_DENOMINATOR).bit_length() - 1,
            24, 8,
        )
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
        events = self.notes_to_events(notes, key, scale)
        return self.render_to_midi(events, tempo, track_name)


# =============================================================================
# Export
# =============================================================================

def midi_to_bytes(midi_file: MIDIFile) -> bytes:
    buffer = io.BytesIO()
    midi_file.writeFile(buffer)
    buffer.seek(0)
    return buffer.read()
