"""
CustomTransformerGenerator — mood-conditioned arpeggio backend.

Encapsulates the complete generation pipeline:
  1. Free-text mood → 0-based label index (vocabulary-matched or
     nearest-neighbour via sentence-transformer embeddings).
  2. Mood-driven pitch walk (key/scale/register/contour parameters per mood).
  3. Mood-matched rhythm and velocity applied per note.
  4. MIDI file rendering.

The FastAPI layer sees only the ``BaseGenerator`` interface.
"""

from __future__ import annotations

import logging
import random as _random
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, Union

from app.generators.base import (
    BaseGenerator,
    GenerationRequest,
    GenerationResult,
    NoteResult,
)
from app.music.arpeggio_generator import (
    ArpeggioPattern,
    Note,
    build_scale_pitches,
    generate_mood_pitch_sequence,
)
from app.music.midi_renderer import MIDIRenderer, midi_to_bytes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mood vocabulary
# Must match the order used in scripts/tokenize_dataset.py exactly so that
# mood_label integers align with the model's nn.Embedding(19, d_model).
# ---------------------------------------------------------------------------

_VALID_MOODS: Tuple[str, ...] = (
    "melancholic", "dreamy",    "energetic", "tense",   "happy",
    "sad",         "calm",      "dark",      "joyful",  "uplifting",
    "intense",     "peaceful",  "dramatic",  "epic",    "mysterious",
    "romantic",    "neutral",   "flowing",   "ominous",
)

_MOOD_TO_LABEL: Dict[str, int] = {m: i for i, m in enumerate(_VALID_MOODS)}


# ---------------------------------------------------------------------------
# Mood → arpeggio pattern mapping
# ---------------------------------------------------------------------------

# Multiple pattern candidates per mood label give variety across generations.
# The per-request seed selects deterministically so identical seeds reproduce.
_MOOD_PATTERNS: Dict[int, List[ArpeggioPattern]] = {
    0:  [ArpeggioPattern.UP_DOWN,    ArpeggioPattern.DESCENDING,  ArpeggioPattern.DOWN_UP],    # melancholic
    1:  [ArpeggioPattern.UP_DOWN,    ArpeggioPattern.ASCENDING,   ArpeggioPattern.DOWN_UP],    # dreamy
    2:  [ArpeggioPattern.ASCENDING,  ArpeggioPattern.UP_DOWN,     ArpeggioPattern.DOWN_UP],    # energetic
    3:  [ArpeggioPattern.DOWN_UP,    ArpeggioPattern.DESCENDING,  ArpeggioPattern.UP_DOWN],    # tense
    4:  [ArpeggioPattern.ASCENDING,  ArpeggioPattern.UP_DOWN,     ArpeggioPattern.DOWN_UP],    # happy
    5:  [ArpeggioPattern.DESCENDING, ArpeggioPattern.DOWN_UP,     ArpeggioPattern.UP_DOWN],    # sad
    6:  [ArpeggioPattern.UP_DOWN,    ArpeggioPattern.ASCENDING,   ArpeggioPattern.DOWN_UP],    # calm
    7:  [ArpeggioPattern.DESCENDING, ArpeggioPattern.DOWN_UP,     ArpeggioPattern.UP_DOWN],    # dark
    8:  [ArpeggioPattern.ASCENDING,  ArpeggioPattern.UP_DOWN,     ArpeggioPattern.DOWN_UP],    # joyful
    9:  [ArpeggioPattern.ASCENDING,  ArpeggioPattern.UP_DOWN,     ArpeggioPattern.DOWN_UP],    # uplifting
    10: [ArpeggioPattern.DOWN_UP,    ArpeggioPattern.ASCENDING,   ArpeggioPattern.DESCENDING], # intense
    11: [ArpeggioPattern.UP_DOWN,    ArpeggioPattern.ASCENDING,   ArpeggioPattern.DOWN_UP],    # peaceful
    12: [ArpeggioPattern.DOWN_UP,    ArpeggioPattern.DESCENDING,  ArpeggioPattern.ASCENDING],  # dramatic
    13: [ArpeggioPattern.ASCENDING,  ArpeggioPattern.DOWN_UP,     ArpeggioPattern.UP_DOWN],    # epic
    14: [ArpeggioPattern.DOWN_UP,    ArpeggioPattern.UP_DOWN,     ArpeggioPattern.DESCENDING], # mysterious
    15: [ArpeggioPattern.UP_DOWN,    ArpeggioPattern.ASCENDING,   ArpeggioPattern.DOWN_UP],    # romantic
    16: [ArpeggioPattern.ASCENDING,  ArpeggioPattern.UP_DOWN,     ArpeggioPattern.DOWN_UP],    # neutral
    17: [ArpeggioPattern.UP_DOWN,    ArpeggioPattern.ASCENDING,   ArpeggioPattern.DOWN_UP],    # flowing
    18: [ArpeggioPattern.DESCENDING, ArpeggioPattern.DOWN_UP,     ArpeggioPattern.UP_DOWN],    # ominous
}

_PATTERN_MAP: Dict[str, ArpeggioPattern] = {
    "ascending":  ArpeggioPattern.ASCENDING,
    "descending": ArpeggioPattern.DESCENDING,
    "up_down":    ArpeggioPattern.UP_DOWN,
    "down_up":    ArpeggioPattern.DOWN_UP,
}

_PATTERN_NAMES: Dict[ArpeggioPattern, str] = {v: k for k, v in _PATTERN_MAP.items()}


# ---------------------------------------------------------------------------
# Mood-driven generation parameters (Option C)
# ---------------------------------------------------------------------------

class _MoodGenParams(NamedTuple):
    """Per-mood parameters for the pitch-and-rhythm generator."""
    step_weights: Tuple[float, float, float, float]  # weights for [-2,-1,+1,+2]
    octave_jump_prob: float    # per-note probability of ±octave displacement
    contour_bias: float        # -1 (descending) … +1 (ascending)
    start_region: str          # "low" | "mid" | "high"
    rhythm_pattern: Tuple[float, ...]  # cycling note durations in beats
    num_octaves: int           # scale span passed to build_scale_pitches


# Indices match _VALID_MOODS (0 = melancholic … 18 = ominous).
# step_weights = [w_-2, w_-1, w_+1, w_+2]
_MOOD_GEN_PARAMS: Dict[int, _MoodGenParams] = {
    0:  _MoodGenParams((2, 4, 3, 1), 0.06, -0.30, "mid",  (1.0, 0.5, 0.5, 1.0, 0.5),          2),  # melancholic
    1:  _MoodGenParams((1, 2, 3, 2), 0.12,  0.10, "mid",  (0.5, 0.25, 0.25, 0.5, 1.0),         2),  # dreamy
    2:  _MoodGenParams((1, 2, 3, 4), 0.22,  0.40, "high", (0.25, 0.25, 0.5, 0.25, 0.25, 0.5),  2),  # energetic
    3:  _MoodGenParams((2, 3, 3, 2), 0.18, -0.10, "mid",  (0.25, 0.5, 0.25, 0.75, 0.25),       2),  # tense
    4:  _MoodGenParams((1, 2, 4, 3), 0.15,  0.30, "mid",  (0.5, 0.25, 0.25, 0.5, 0.5),         2),  # happy
    5:  _MoodGenParams((3, 4, 2, 1), 0.05, -0.40, "low",  (1.0, 1.0, 0.5, 0.5, 1.5),           2),  # sad
    6:  _MoodGenParams((1, 3, 3, 1), 0.05,  0.00, "mid",  (1.0, 0.5, 1.0, 0.5, 1.0),           2),  # calm
    7:  _MoodGenParams((2, 3, 2, 1), 0.10, -0.20, "low",  (0.5, 0.5, 1.0, 0.5, 0.5),           2),  # dark
    8:  _MoodGenParams((1, 2, 3, 4), 0.20,  0.30, "mid",  (0.25, 0.25, 0.25, 0.5, 0.25),       2),  # joyful
    9:  _MoodGenParams((1, 2, 4, 3), 0.15,  0.50, "mid",  (0.5, 0.5, 0.25, 0.25, 1.0),         2),  # uplifting
    10: _MoodGenParams((2, 2, 3, 3), 0.28,  0.20, "high", (0.25, 0.25, 0.25, 0.25, 0.5),       3),  # intense
    11: _MoodGenParams((1, 3, 3, 1), 0.05,  0.00, "mid",  (1.0, 0.5, 1.0, 1.0, 0.5),           2),  # peaceful
    12: _MoodGenParams((1, 2, 3, 4), 0.22, -0.10, "low",  (0.5, 0.25, 0.75, 1.0, 0.5),         3),  # dramatic
    13: _MoodGenParams((1, 2, 3, 4), 0.28,  0.40, "mid",  (1.0, 0.5, 0.5, 2.0, 1.0),           3),  # epic
    14: _MoodGenParams((3, 3, 2, 2), 0.15, -0.10, "mid",  (0.5, 0.75, 0.25, 0.5, 0.75),        2),  # mysterious
    15: _MoodGenParams((1, 2, 3, 2), 0.10,  0.10, "mid",  (0.5, 1.0, 0.5, 0.5, 1.0),           2),  # romantic
    16: _MoodGenParams((2, 2, 2, 2), 0.10,  0.00, "mid",  (0.5, 0.5, 0.5, 0.5, 0.5),           2),  # neutral
    17: _MoodGenParams((1, 2, 3, 2), 0.10,  0.10, "mid",  (0.5, 0.25, 0.25, 0.5, 0.25, 0.25),  2),  # flowing
    18: _MoodGenParams((3, 3, 2, 2), 0.15, -0.30, "low",  (0.5, 0.5, 1.0, 0.75, 0.5),          2),  # ominous
}


# ---------------------------------------------------------------------------
# Mood → velocity profile  (base_velocity, variance)
# ---------------------------------------------------------------------------

# Used when the model returns fewer notes than requested; gives audibly
# distinct dynamics for each mood even when falling back to the base arpeggio.
_MOOD_VELOCITY: Dict[int, Tuple[int, int]] = {
    0:  (62, 12),  # melancholic  — soft, gentle
    1:  (68, 10),  # dreamy       — quiet, smooth
    2:  (98, 18),  # energetic    — loud, punchy
    3:  (88, 22),  # tense        — mid-high, erratic
    4:  (88, 14),  # happy        — bright, consistent
    5:  (58, 10),  # sad          — very soft
    6:  (64,  8),  # calm         — soft, even
    7:  (78, 20),  # dark         — mid, uneven
    8:  (92, 14),  # joyful       — bright, bouncy
    9:  (90, 14),  # uplifting    — high, hopeful
    10: (100, 20), # intense      — very loud, wide range
    11: (62,  8),  # peaceful     — soft, very even
    12: (94, 22),  # dramatic     — loud, very wide
    13: (98, 18),  # epic         — very loud
    14: (70, 20),  # mysterious   — mid, erratic
    15: (76, 12),  # romantic     — mid-soft, smooth
    16: (75, 14),  # neutral      — default mid
    17: (72, 10),  # flowing      — soft-mid, smooth
    18: (82, 24),  # ominous      — mid, very erratic
}


# ---------------------------------------------------------------------------
# CustomTransformerGenerator
# ---------------------------------------------------------------------------


class CustomTransformerGenerator(BaseGenerator):
    """
    Mood-conditioned arpeggio generator using a rule-based pitch walk.

    Args:
        checkpoint_path: Path to the model checkpoint (kept for API compatibility).
    """

    def __init__(self, checkpoint_path: Union[str, Path]) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._loaded = False

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "custom-transformer"

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Mark the generator as ready."""
        self._loaded = True
        logger.info("CustomTransformerGenerator ready | checkpoint=%s", self._checkpoint_path)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Mood-driven generation pipeline:

        1. Resolve mood text → 0-based label.
        2. Look up per-mood generation parameters (step weights, rhythm,
           contour bias, octave span, register).
        3. Build the full scale-pitch pool for the requested key/scale.
        4. Run a seeded random walk through the scale pool, shaped by the
           mood parameters, to produce ``note_count`` pitches.
        5. Apply mood-matched rhythm patterns and velocity dynamics.
        6. Render to MIDI and return ``GenerationResult``.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Generator '{self.name}' is not loaded. Call load() first."
            )

        # 1. Mood resolution.
        mood_label = self._resolve_mood_label(request.mood)

        # Per-request seed: caller's value for reproducibility, else random.
        seed: int = (
            request.seed
            if request.seed is not None
            else _random.randint(0, 2_147_483_647)
        )

        # Pattern kept for metadata; actual pitches come from the mood walk.
        if request.pattern is not None:
            chosen_pattern = _PATTERN_MAP[request.pattern]
        else:
            chosen_pattern = self._resolve_pattern(mood_label, seed)

        # 2. Per-mood generation parameters.
        params = _MOOD_GEN_PARAMS.get(mood_label, _MOOD_GEN_PARAMS[16])

        # 3. Scale pitch pool.
        scale_pitches = build_scale_pitches(
            request.key, request.scale, request.octave,
            num_octaves=params.num_octaves,
        )
        if not scale_pitches:
            # Degenerate fallback: widen to 2 octaves.
            scale_pitches = build_scale_pitches(
                request.key, request.scale, request.octave, num_octaves=2
            )

        # 4. Mood-driven pitch walk.
        pitches = generate_mood_pitch_sequence(
            scale_pitches=scale_pitches,
            note_count=request.note_count,
            rng=_random.Random(seed),
            step_weights=list(params.step_weights),
            octave_jump_prob=params.octave_jump_prob,
            contour_bias=params.contour_bias,
            start_region=params.start_region,
        )

        # 5. Build notes with mood rhythm + velocity.
        out_notes = self._build_mood_notes(
            pitches, params.rhythm_pattern, mood_label, seed
        )

        # 5b. Tile the pattern across the requested number of bars.
        if request.bars > 1 and out_notes:
            pattern_duration = max(n.position + n.duration for n in out_notes)
            tiled: List[Note] = list(out_notes)
            for b in range(1, request.bars):
                offset = pattern_duration * b
                for n in out_notes:
                    tiled.append(Note(
                        pitch=n.pitch,
                        velocity=n.velocity,
                        position=n.position + offset,
                        duration=n.duration,
                    ))
            out_notes = tiled

        # 6. MIDI render.
        midi_file = MIDIRenderer(enforce_scale=True).render_notes(
            notes=out_notes,
            tempo=request.tempo,
            key=request.key,
            scale=request.scale,
        )
        midi_bytes = midi_to_bytes(midi_file)

        note_results = [
            NoteResult(
                pitch=n.pitch,
                velocity=n.velocity,
                position=n.position,
                duration=n.duration,
            )
            for n in out_notes
        ]

        seconds_per_beat = 60.0 / max(request.tempo, 1)
        last = max(out_notes, key=lambda n: n.position + n.duration)
        duration_seconds = round((last.position + last.duration) * seconds_per_beat, 3)
        pattern_used = _PATTERN_NAMES.get(chosen_pattern, "unknown")

        result = GenerationResult(
            midi_bytes=midi_bytes,
            notes=note_results,
            note_count=len(note_results),
            duration_seconds=duration_seconds,
            key=request.key,
            scale=request.scale,
            tempo=request.tempo,
            mood=request.mood,
            pattern_used=pattern_used,
        )
        self._log_generation(request, result)
        return result

    # ------------------------------------------------------------------
    # Mood note builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mood_notes(
        pitches: List[int],
        rhythm_pattern: Tuple[float, ...],
        mood_label: int,
        seed: int,
    ) -> List[Note]:
        """
        Combine a pitch sequence with mood-driven rhythm and velocity.

        Args:
            pitches:        MIDI pitch values (one per note).
            rhythm_pattern: Cycling note durations in beats.
            mood_label:     0-based mood index for velocity profile.
            seed:           RNG seed (XOR-mixed to decouple from pitch seed).

        Returns:
            Fully populated Note objects in ascending time order.
        """
        base_vel, variance = _MOOD_VELOCITY.get(mood_label, (75, 15))
        vel_rng = _random.Random(seed ^ 0xC0FFEE)
        notes: List[Note] = []
        pos = 0.0
        for i, pitch in enumerate(pitches):
            dur = rhythm_pattern[i % len(rhythm_pattern)]
            vel = max(40, min(115, base_vel + vel_rng.randint(-variance, variance)))
            notes.append(Note(pitch=pitch, duration=dur, velocity=vel, position=pos))
            pos += dur
        return notes

    # ------------------------------------------------------------------
    # Mood resolution
    # ------------------------------------------------------------------

    def _resolve_mood_label(self, mood_text: str) -> int:
        """
        Map free-text mood to a 0-based label index (0–18).

        Resolution order:
          1. Exact case-insensitive match against the known vocabulary.
          2. Nearest-neighbour via sentence-transformer cosine similarity
             (requires ``app.mood.embeddings``; skipped gracefully if absent).
          3. Hard fallback to "neutral" (index 16).
        """
        key = mood_text.lower().strip()
        if key in _MOOD_TO_LABEL:
            return _MOOD_TO_LABEL[key]

        try:
            import torch
            import torch.nn.functional as F
            from app.mood.embeddings import get_mood_embeddings  # type: ignore

            query = get_mood_embeddings([key])[0]               # (384,)
            refs  = get_mood_embeddings(list(_VALID_MOODS))     # (19, 384)
            q = F.normalize(query.unsqueeze(0), p=2, dim=1)
            r = F.normalize(refs, p=2, dim=1)
            best = int((r @ q.T).squeeze(-1).argmax().item())
            logger.debug(
                "mood %r → %r (label %d)", mood_text, _VALID_MOODS[best], best
            )
            return best
        except Exception:
            pass

        return _MOOD_TO_LABEL["neutral"]

    # ------------------------------------------------------------------
    # Pattern selection
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_pattern(mood_label: int, seed: int) -> ArpeggioPattern:
        """Pick an arpeggio pattern for the given mood, seeded for reproducibility."""
        candidates = _MOOD_PATTERNS.get(mood_label, [ArpeggioPattern.ASCENDING])
        return _random.Random(seed).choice(candidates)

