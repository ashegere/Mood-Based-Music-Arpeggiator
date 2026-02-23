"""
Tokenization module for symbolic music representation.

Converts musical events (notes, bars, metadata) into discrete tokens
suitable for sequence processing. This is a rule-based tokenization
scheme, not learned.

Token Design Philosophy:
- Each musical attribute gets its own token type for clarity
- Quantization reduces vocabulary size while preserving musical meaning
- Position tokens enable absolute timing (bar-relative)
- Global tokens (KEY, SCALE, TEMPO) appear at sequence start

Vocabulary Structure:
- Special tokens: PAD, BOS, EOS, SEP
- Global tokens: KEY_*, SCALE_*, TEMPO_*
- Structural tokens: BAR, BAR_END
- Note tokens: PITCH_*, DUR_*, VEL_*, POS_*
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Union
import re

from app.music.arpeggio_generator import (
    Note,
    Arpeggio,
    KEY_OFFSETS,
    SCALE_PATTERNS,
    get_available_keys,
    get_available_scales,
)


# =============================================================================
# Quantization Settings
# =============================================================================

# Duration quantization: map continuous durations to discrete bins
# Values in beats (quarter notes)
DURATION_BINS: Tuple[float, ...] = (
    0.125,  # 32nd note
    0.25,   # 16th note
    0.375,  # dotted 16th
    0.5,    # 8th note
    0.75,   # dotted 8th
    1.0,    # quarter note
    1.5,    # dotted quarter
    2.0,    # half note
    3.0,    # dotted half
    4.0,    # whole note
)

# Velocity quantization: 8 levels from ppp to fff
# Maps 0-127 to 8 bins for manageable vocabulary
VELOCITY_BINS: Tuple[int, ...] = (
    16,   # ppp
    32,   # pp
    48,   # p
    64,   # mp
    80,   # mf
    96,   # f
    112,  # ff
    127,  # fff
)

# Position quantization: subdivisions within a bar
# Using 16th note grid (16 positions per 4/4 bar)
POSITIONS_PER_BAR: int = 16
POSITION_GRID: float = 0.25  # 16th note = 0.25 beats

# Tempo range for tokenization
TEMPO_MIN: int = 40
TEMPO_MAX: int = 240
TEMPO_STEP: int = 10  # Quantize to nearest 10 BPM

# Pitch range (same as generator)
PITCH_MIN: int = 36
PITCH_MAX: int = 96


# =============================================================================
# Token Types
# =============================================================================

class TokenType(Enum):
    """Enumeration of all token types in the vocabulary."""
    # Special tokens
    PAD = auto()      # Padding for batch processing
    BOS = auto()      # Beginning of sequence
    EOS = auto()      # End of sequence
    SEP = auto()      # Separator between sections
    UNK = auto()      # Unknown token

    # Global tokens (sequence metadata)
    KEY = auto()      # Musical key (e.g., KEY_C, KEY_F#)
    SCALE = auto()    # Scale type (e.g., SCALE_major)
    TEMPO = auto()    # Tempo in BPM (e.g., TEMPO_120)

    # Structural tokens
    BAR = auto()      # Bar marker (e.g., BAR_0, BAR_1)
    BAR_END = auto()  # End of bar marker

    # Note attribute tokens
    PITCH = auto()    # MIDI pitch (e.g., PITCH_60)
    DURATION = auto() # Note duration (e.g., DUR_0.5)
    VELOCITY = auto() # Note velocity (e.g., VEL_80)
    POSITION = auto() # Position in bar (e.g., POS_0, POS_4)


# =============================================================================
# Token Data Structure
# =============================================================================

@dataclass(frozen=True)
class Token:
    """
    A single token in the musical sequence.

    Attributes:
        type: The category of token (PITCH, DURATION, etc.)
        value: The token's value (depends on type)
        token_id: Unique integer ID in vocabulary
    """
    type: TokenType
    value: Union[str, int, float]
    token_id: int

    def __str__(self) -> str:
        """String representation for debugging."""
        if self.type in (TokenType.PAD, TokenType.BOS, TokenType.EOS,
                         TokenType.SEP, TokenType.UNK, TokenType.BAR_END):
            return f"<{self.type.name}>"
        return f"{self.type.name}_{self.value}"


@dataclass
class TokenSequence:
    """
    A sequence of tokens representing a musical piece.

    Attributes:
        tokens: List of Token objects
        key: Original key (for reference)
        scale: Original scale (for reference)
        tempo: Original tempo (for reference)
    """
    tokens: List[Token]
    key: str
    scale: str
    tempo: int

    def to_ids(self) -> List[int]:
        """Convert sequence to list of token IDs."""
        return [t.token_id for t in self.tokens]

    def to_strings(self) -> List[str]:
        """Convert sequence to list of token strings."""
        return [str(t) for t in self.tokens]

    def __len__(self) -> int:
        return len(self.tokens)


# =============================================================================
# Vocabulary Builder
# =============================================================================

class Vocabulary:
    """
    Manages the token vocabulary and ID mappings.

    The vocabulary is built deterministically from the defined
    musical parameters. Token IDs are assigned in a fixed order
    to ensure reproducibility.
    """

    def __init__(self):
        """Initialize vocabulary with all possible tokens."""
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._token_types: Dict[str, TokenType] = {}
        self._build_vocabulary()

    def _build_vocabulary(self) -> None:
        """Build the complete vocabulary in deterministic order."""
        token_id = 0

        # Special tokens (always first)
        for special in [TokenType.PAD, TokenType.BOS, TokenType.EOS,
                        TokenType.SEP, TokenType.UNK]:
            token_str = f"<{special.name}>"
            self._add_token(token_str, token_id, special)
            token_id += 1

        # Key tokens (sorted for determinism)
        for key in sorted(get_available_keys()):
            token_str = f"KEY_{key}"
            self._add_token(token_str, token_id, TokenType.KEY)
            token_id += 1

        # Scale tokens (sorted for determinism)
        for scale in sorted(get_available_scales()):
            token_str = f"SCALE_{scale}"
            self._add_token(token_str, token_id, TokenType.SCALE)
            token_id += 1

        # Tempo tokens (quantized range)
        for tempo in range(TEMPO_MIN, TEMPO_MAX + 1, TEMPO_STEP):
            token_str = f"TEMPO_{tempo}"
            self._add_token(token_str, token_id, TokenType.TEMPO)
            token_id += 1

        # Bar tokens (support up to 64 bars)
        for bar_num in range(64):
            token_str = f"BAR_{bar_num}"
            self._add_token(token_str, token_id, TokenType.BAR)
            token_id += 1

        # Bar end token
        self._add_token("<BAR_END>", token_id, TokenType.BAR_END)
        token_id += 1

        # Pitch tokens (full MIDI range we support)
        for pitch in range(PITCH_MIN, PITCH_MAX + 1):
            token_str = f"PITCH_{pitch}"
            self._add_token(token_str, token_id, TokenType.PITCH)
            token_id += 1

        # Duration tokens (quantized bins)
        for dur in DURATION_BINS:
            token_str = f"DUR_{dur}"
            self._add_token(token_str, token_id, TokenType.DURATION)
            token_id += 1

        # Velocity tokens (quantized bins)
        for vel in VELOCITY_BINS:
            token_str = f"VEL_{vel}"
            self._add_token(token_str, token_id, TokenType.VELOCITY)
            token_id += 1

        # Position tokens (positions within bar)
        for pos in range(POSITIONS_PER_BAR):
            token_str = f"POS_{pos}"
            self._add_token(token_str, token_id, TokenType.POSITION)
            token_id += 1

    def _add_token(self, token_str: str, token_id: int, token_type: TokenType) -> None:
        """Add a token to the vocabulary."""
        self._token_to_id[token_str] = token_id
        self._id_to_token[token_id] = token_str
        self._token_types[token_str] = token_type

    def encode(self, token_str: str) -> int:
        """Convert token string to ID."""
        return self._token_to_id.get(token_str, self._token_to_id["<UNK>"])

    def decode(self, token_id: int) -> str:
        """Convert token ID to string."""
        return self._id_to_token.get(token_id, "<UNK>")

    def get_token(self, token_str: str) -> Token:
        """Get a Token object from a token string."""
        token_id = self.encode(token_str)
        token_type = self._token_types.get(token_str, TokenType.UNK)

        # Extract value from token string
        value: Union[str, int, float] = token_str
        if "_" in token_str and not token_str.startswith("<"):
            parts = token_str.split("_", 1)
            raw_value = parts[1]
            # Try to parse as number
            try:
                if "." in raw_value:
                    value = float(raw_value)
                else:
                    value = int(raw_value)
            except ValueError:
                value = raw_value

        return Token(type=token_type, value=value, token_id=token_id)

    def get_token_by_id(self, token_id: int) -> Token:
        """Get a Token object from a token ID."""
        token_str = self.decode(token_id)
        return self.get_token(token_str)

    @property
    def size(self) -> int:
        """Total vocabulary size."""
        return len(self._token_to_id)

    @property
    def pad_id(self) -> int:
        """ID of PAD token."""
        return self._token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        """ID of BOS token."""
        return self._token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        """ID of EOS token."""
        return self._token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        """ID of UNK token."""
        return self._token_to_id["<UNK>"]


# =============================================================================
# Quantization Functions
# =============================================================================

def quantize_duration(duration: float) -> float:
    """
    Quantize duration to nearest bin.

    Args:
        duration: Duration in beats.

    Returns:
        Nearest duration bin value.
    """
    if duration <= 0:
        return DURATION_BINS[0]

    # Find closest bin
    closest = min(DURATION_BINS, key=lambda x: abs(x - duration))
    return closest


def quantize_velocity(velocity: int) -> int:
    """
    Quantize velocity to nearest bin.

    Args:
        velocity: MIDI velocity (0-127).

    Returns:
        Nearest velocity bin value.
    """
    velocity = max(0, min(127, velocity))

    # Find closest bin
    closest = min(VELOCITY_BINS, key=lambda x: abs(x - velocity))
    return closest


def quantize_tempo(tempo: int) -> int:
    """
    Quantize tempo to nearest step.

    Args:
        tempo: Tempo in BPM.

    Returns:
        Quantized tempo (multiple of TEMPO_STEP).
    """
    tempo = max(TEMPO_MIN, min(TEMPO_MAX, tempo))
    return round(tempo / TEMPO_STEP) * TEMPO_STEP


def quantize_position(position: float, beats_per_bar: float = 4.0) -> Tuple[int, int]:
    """
    Quantize position to bar number and position within bar.

    Args:
        position: Position in beats from start.
        beats_per_bar: Beats per bar (default 4.0 for 4/4 time).

    Returns:
        Tuple of (bar_number, position_in_bar).
    """
    bar_num = int(position // beats_per_bar)
    beat_in_bar = position % beats_per_bar

    # Quantize to 16th note grid
    pos_in_bar = int(round(beat_in_bar / POSITION_GRID))
    pos_in_bar = min(pos_in_bar, POSITIONS_PER_BAR - 1)

    return bar_num, pos_in_bar


# =============================================================================
# Tokenizer
# =============================================================================

class Tokenizer:
    """
    Converts musical structures to token sequences and back.

    The tokenizer handles:
    - Arpeggio to token sequence (encoding)
    - Token sequence to note list (decoding)
    - Quantization of continuous values
    """

    def __init__(self, vocab: Optional[Vocabulary] = None):
        """
        Initialize tokenizer with vocabulary.

        Args:
            vocab: Vocabulary instance. Creates new one if None.
        """
        self.vocab = vocab or Vocabulary()

    def tokenize_arpeggio(
        self,
        arpeggio: Arpeggio,
        add_special_tokens: bool = True,
        beats_per_bar: float = 4.0,
    ) -> TokenSequence:
        """
        Convert an Arpeggio to a TokenSequence.

        Token order per note: POSITION, PITCH, DURATION, VELOCITY
        This order reflects the temporal nature of music: when, what, how long, how loud.

        Args:
            arpeggio: The arpeggio to tokenize.
            add_special_tokens: Whether to add BOS/EOS tokens.
            beats_per_bar: Beats per bar for position calculation.

        Returns:
            TokenSequence with all tokens.
        """
        tokens: List[Token] = []

        # Add BOS token
        if add_special_tokens:
            tokens.append(self.vocab.get_token("<BOS>"))

        # Add global tokens (metadata)
        tokens.append(self.vocab.get_token(f"KEY_{arpeggio.key}"))
        tokens.append(self.vocab.get_token(f"SCALE_{arpeggio.scale}"))

        q_tempo = quantize_tempo(arpeggio.tempo)
        tokens.append(self.vocab.get_token(f"TEMPO_{q_tempo}"))

        # Add separator after metadata
        tokens.append(self.vocab.get_token("<SEP>"))

        # Track current bar for BAR tokens
        current_bar = -1

        # Tokenize each note
        for note in arpeggio.notes:
            bar_num, pos_in_bar = quantize_position(note.position, beats_per_bar)

            # Add bar token if entering new bar
            if bar_num != current_bar:
                # Close previous bar if exists
                if current_bar >= 0:
                    tokens.append(self.vocab.get_token("<BAR_END>"))
                # Open new bar
                bar_token = f"BAR_{min(bar_num, 63)}"  # Cap at 63
                tokens.append(self.vocab.get_token(bar_token))
                current_bar = bar_num

            # Add note tokens in order: POS, PITCH, DUR, VEL
            tokens.append(self.vocab.get_token(f"POS_{pos_in_bar}"))

            # Clamp pitch to valid range
            pitch = max(PITCH_MIN, min(PITCH_MAX, note.pitch))
            tokens.append(self.vocab.get_token(f"PITCH_{pitch}"))

            q_dur = quantize_duration(note.duration)
            tokens.append(self.vocab.get_token(f"DUR_{q_dur}"))

            q_vel = quantize_velocity(note.velocity)
            tokens.append(self.vocab.get_token(f"VEL_{q_vel}"))

        # Close final bar
        if current_bar >= 0:
            tokens.append(self.vocab.get_token("<BAR_END>"))

        # Add EOS token
        if add_special_tokens:
            tokens.append(self.vocab.get_token("<EOS>"))

        return TokenSequence(
            tokens=tokens,
            key=arpeggio.key,
            scale=arpeggio.scale,
            tempo=arpeggio.tempo,
        )

    def tokenize_notes(
        self,
        notes: List[Note],
        key: str,
        scale: str,
        tempo: int,
        add_special_tokens: bool = True,
        beats_per_bar: float = 4.0,
    ) -> TokenSequence:
        """
        Tokenize a list of notes with metadata.

        Convenience method that creates a temporary Arpeggio.

        Args:
            notes: List of Note objects.
            key: Musical key.
            scale: Scale type.
            tempo: Tempo in BPM.
            add_special_tokens: Whether to add BOS/EOS.
            beats_per_bar: Beats per bar.

        Returns:
            TokenSequence.
        """
        arpeggio = Arpeggio(
            notes=notes,
            key=key,
            scale=scale,
            tempo=tempo,
            seed=0,  # Not used for tokenization
        )
        return self.tokenize_arpeggio(arpeggio, add_special_tokens, beats_per_bar)

    def detokenize(
        self,
        token_sequence: TokenSequence,
        beats_per_bar: float = 4.0,
    ) -> Tuple[List[Note], str, str, int]:
        """
        Convert a TokenSequence back to notes and metadata.

        Args:
            token_sequence: The token sequence to decode.
            beats_per_bar: Beats per bar for position calculation.

        Returns:
            Tuple of (notes, key, scale, tempo).
        """
        return self.detokenize_ids(
            token_sequence.to_ids(),
            beats_per_bar,
        )

    def detokenize_ids(
        self,
        token_ids: List[int],
        beats_per_bar: float = 4.0,
    ) -> Tuple[List[Note], str, str, int]:
        """
        Convert token IDs back to notes and metadata.

        Args:
            token_ids: List of token IDs.
            beats_per_bar: Beats per bar.

        Returns:
            Tuple of (notes, key, scale, tempo).
        """
        # Parse metadata
        key = "C"
        scale = "major"
        tempo = 120

        notes: List[Note] = []
        current_bar = 0

        # State for building notes
        current_pos: Optional[int] = None
        current_pitch: Optional[int] = None
        current_dur: Optional[float] = None
        current_vel: Optional[int] = None

        for token_id in token_ids:
            token = self.vocab.get_token_by_id(token_id)
            token_str = str(token)

            # Skip special tokens
            if token.type in (TokenType.PAD, TokenType.BOS, TokenType.EOS,
                              TokenType.SEP, TokenType.UNK):
                continue

            # Parse metadata tokens
            if token.type == TokenType.KEY:
                key = str(token.value)
            elif token.type == TokenType.SCALE:
                scale = str(token.value)
            elif token.type == TokenType.TEMPO:
                tempo = int(token.value)

            # Parse structural tokens
            elif token.type == TokenType.BAR:
                current_bar = int(token.value)
            elif token.type == TokenType.BAR_END:
                pass  # Just a marker

            # Parse note tokens
            elif token.type == TokenType.POSITION:
                # If we have a complete note, save it
                if all(v is not None for v in [current_pos, current_pitch,
                                                current_dur, current_vel]):
                    position = current_bar * beats_per_bar + current_pos * POSITION_GRID
                    notes.append(Note(
                        pitch=current_pitch,
                        duration=current_dur,
                        velocity=current_vel,
                        position=position,
                    ))
                # Start new note
                current_pos = int(token.value)
                current_pitch = None
                current_dur = None
                current_vel = None

            elif token.type == TokenType.PITCH:
                current_pitch = int(token.value)
            elif token.type == TokenType.DURATION:
                current_dur = float(token.value)
            elif token.type == TokenType.VELOCITY:
                current_vel = int(token.value)

        # Don't forget the last note
        if all(v is not None for v in [current_pos, current_pitch,
                                        current_dur, current_vel]):
            position = current_bar * beats_per_bar + current_pos * POSITION_GRID
            notes.append(Note(
                pitch=current_pitch,
                duration=current_dur,
                velocity=current_vel,
                position=position,
            ))

        return notes, key, scale, tempo


# =============================================================================
# Global Vocabulary Instance
# =============================================================================

# Singleton vocabulary for consistent token IDs
_GLOBAL_VOCAB: Optional[Vocabulary] = None


def get_vocabulary() -> Vocabulary:
    """Get the global vocabulary instance."""
    global _GLOBAL_VOCAB
    if _GLOBAL_VOCAB is None:
        _GLOBAL_VOCAB = Vocabulary()
    return _GLOBAL_VOCAB


def get_tokenizer() -> Tokenizer:
    """Get a tokenizer with the global vocabulary."""
    return Tokenizer(get_vocabulary())
