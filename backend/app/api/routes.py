"""
FastAPI router for arpeggio generation.

Pipeline per request:
  1. Resolve free-text mood → 0-based label index (0–18)
  2. Generate base arpeggio (rule-based, deterministic)
  3. Tokenise arpeggio → MIDI event token IDs (411-token vocabulary)
  4. Run mood-conditioned transformer inference
  5. Decode output token IDs → Note objects
  6. Render notes → MIDI bytes
  7. Return base64-encoded MIDI + note metadata
"""

import base64
import logging
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException

from app.api.schemas import GenerateArpeggioRequest, GenerateArpeggioResponse, NoteEvent
from app.model.inference import get_engine
from app.music.arpeggio_generator import Arpeggio, ArpeggioGenerator, ArpeggioPattern, Note
from app.music.midi_renderer import MIDIRenderer, midi_to_bytes

logger = logging.getLogger(__name__)
router = APIRouter()


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
# MIDI event vocabulary (411 tokens)
# Token order must mirror scripts/tokenize_dataset.py::MIDIVocabulary exactly.
# ---------------------------------------------------------------------------

_TIME_RESOLUTION_MS: int = 10   # milliseconds per TIME_SHIFT unit
_MAX_TIME_SHIFT: int = 100       # max value of a single TIME_SHIFT token
_NUM_VELOCITY_BINS: int = 32     # velocity quantisation levels


def _build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build the deterministic 411-token vocabulary used during training.

    Layout:
      0-3   : PAD, BOS, EOS, UNK
      4-22  : MOOD_<name>           (19 tokens)
      23-150: NOTE_ON_<pitch>       (128 tokens, pitch 0-127)
      151-278: NOTE_OFF_<pitch>     (128 tokens, pitch 0-127)
      279-378: TIME_SHIFT_<t>       (100 tokens, t 1-100)
      379-410: VELOCITY_<v>         (32 tokens, v 1-32)
    """
    t2i: Dict[str, int] = {}
    i2t: Dict[int, str] = {}
    idx = 0

    for tok in ("PAD", "BOS", "EOS", "UNK"):
        t2i[tok] = idx
        i2t[idx] = tok
        idx += 1

    for mood in _VALID_MOODS:
        tok = f"MOOD_{mood}"
        t2i[tok] = idx
        i2t[idx] = tok
        idx += 1

    for p in range(128):
        tok = f"NOTE_ON_{p}"
        t2i[tok] = idx
        i2t[idx] = tok
        idx += 1

    for p in range(128):
        tok = f"NOTE_OFF_{p}"
        t2i[tok] = idx
        i2t[idx] = tok
        idx += 1

    for ts in range(1, _MAX_TIME_SHIFT + 1):
        tok = f"TIME_SHIFT_{ts}"
        t2i[tok] = idx
        i2t[idx] = tok
        idx += 1

    for v in range(1, _NUM_VELOCITY_BINS + 1):
        tok = f"VELOCITY_{v}"
        t2i[tok] = idx
        i2t[idx] = tok
        idx += 1

    return t2i, i2t


_TOKEN_TO_ID, _ID_TO_TOKEN = _build_vocab()
_BOS_ID = _TOKEN_TO_ID["BOS"]
_EOS_ID = _TOKEN_TO_ID["EOS"]


# ---------------------------------------------------------------------------
# Pattern mapping
# ---------------------------------------------------------------------------

_PATTERN_MAP: Dict[str, ArpeggioPattern] = {
    "ascending":  ArpeggioPattern.ASCENDING,
    "descending": ArpeggioPattern.DESCENDING,
    "up_down":    ArpeggioPattern.UP_DOWN,
    "down_up":    ArpeggioPattern.DOWN_UP,
}


# ---------------------------------------------------------------------------
# Mood resolution
# ---------------------------------------------------------------------------

def _resolve_mood_label(mood_text: str) -> int:
    """
    Map free-text mood to a 0-based label index (0–18).

    Resolution order:
      1. Exact case-insensitive match against the known vocabulary.
      2. Nearest-neighbour via sentence-transformer cosine similarity
         (requires ``app.mood.embeddings``; skipped gracefully if unavailable).
      3. Hard fallback to "neutral" (index 16).

    Args:
        mood_text: Mood string from the API request.

    Returns:
        Integer label in [0, 18].
    """
    key = mood_text.lower().strip()
    if key in _MOOD_TO_LABEL:
        return _MOOD_TO_LABEL[key]

    try:
        import torch
        import torch.nn.functional as F
        from app.mood.embeddings import get_mood_embeddings

        query = get_mood_embeddings([key])[0]                   # (384,)
        refs  = get_mood_embeddings(list(_VALID_MOODS))         # (19, 384)
        q = F.normalize(query.unsqueeze(0), p=2, dim=1)
        r = F.normalize(refs, p=2, dim=1)
        best = int((r @ q.T).squeeze(-1).argmax().item())
        logger.debug("mood '%s' → '%s' (label %d)", mood_text, _VALID_MOODS[best], best)
        return best
    except Exception:
        pass

    return _MOOD_TO_LABEL["neutral"]


# ---------------------------------------------------------------------------
# Arpeggio → MIDI event tokens
# ---------------------------------------------------------------------------

def _arpeggio_to_tokens(arpeggio: Arpeggio, mood_label: int) -> List[int]:
    """
    Serialise an Arpeggio into the 411-token MIDI event vocabulary.

    Sequence format:
      [BOS] [MOOD_<name>]
      ([VELOCITY_<v>] [NOTE_ON_<p>] [TIME_SHIFT_<t>]... [NOTE_OFF_<p>])+
      [EOS]

    Args:
        arpeggio:    Rule-generated base arpeggio.
        mood_label:  0-based mood index used to select the MOOD token.

    Returns:
        List of integer token IDs.
    """
    seconds_per_beat = 60.0 / max(arpeggio.tempo, 1)

    # Build flat event list [(type, time_s, pitch, velocity)]
    events: List[Tuple] = []
    for note in arpeggio.notes:
        t_on  = note.position * seconds_per_beat
        t_off = (note.position + note.duration) * seconds_per_beat
        events.append(("note_on",  t_on,  note.pitch, note.velocity))
        events.append(("note_off", t_off, note.pitch, 0))

    # Sort: by time; note_off before note_on at identical times
    events.sort(key=lambda e: (e[1], e[0] == "note_on"))

    def _time_tokens(delta_s: float) -> List[int]:
        units = int(delta_s * 1000) // _TIME_RESOLUTION_MS
        ids: List[int] = []
        while units > 0:
            shift = min(units, _MAX_TIME_SHIFT)
            ids.append(_TOKEN_TO_ID[f"TIME_SHIFT_{shift}"])
            units -= shift
        return ids

    def _vel_bin(v: int) -> int:
        b = int(max(1, min(127, v)) / 128 * _NUM_VELOCITY_BINS) + 1
        return min(b, _NUM_VELOCITY_BINS)

    tokens: List[int] = [_BOS_ID, _TOKEN_TO_ID[f"MOOD_{_VALID_MOODS[mood_label]}"]]
    current_time = 0.0
    current_vel_bin: Optional[int] = None

    for etype, t, pitch, velocity in events:
        tokens.extend(_time_tokens(t - current_time))
        current_time = t

        if etype == "note_on":
            vb = _vel_bin(velocity)
            if vb != current_vel_bin:
                tokens.append(_TOKEN_TO_ID[f"VELOCITY_{vb}"])
                current_vel_bin = vb
            tokens.append(_TOKEN_TO_ID[f"NOTE_ON_{pitch}"])
        else:
            tokens.append(_TOKEN_TO_ID[f"NOTE_OFF_{pitch}"])

    tokens.append(_EOS_ID)
    return tokens


# ---------------------------------------------------------------------------
# MIDI event tokens → Note objects
# ---------------------------------------------------------------------------

def _tokens_to_notes(token_ids: List[int], tempo: int) -> List[Note]:
    """
    Reconstruct Note objects from an output MIDI event token sequence.

    Args:
        token_ids: Token IDs produced by the inference engine (BOS/EOS stripped).
        tempo:     Original request tempo (used for seconds→beats conversion).

    Returns:
        List of Note objects sorted by start position.
    """
    beats_per_second = tempo / 60.0
    current_time_s = 0.0
    current_vel_bin = 16                          # default: mid-range (~mf)
    active: Dict[int, Tuple[float, int]] = {}     # pitch → (start_s, velocity)
    notes: List[Note] = []

    for tid in token_ids:
        tok = _ID_TO_TOKEN.get(tid, "")

        if tok.startswith("TIME_SHIFT_"):
            current_time_s += int(tok[11:]) * _TIME_RESOLUTION_MS / 1000.0

        elif tok.startswith("VELOCITY_"):
            current_vel_bin = int(tok[9:])

        elif tok.startswith("NOTE_ON_"):
            pitch = int(tok[8:])
            velocity = max(1, min(127, int((current_vel_bin - 1) / _NUM_VELOCITY_BINS * 127) + 1))
            active[pitch] = (current_time_s, velocity)

        elif tok.startswith("NOTE_OFF_"):
            pitch = int(tok[9:])
            if pitch in active:
                start_s, vel = active.pop(pitch)
                duration_s = max(0.01, current_time_s - start_s)
                notes.append(Note(
                    pitch=pitch,
                    duration=max(0.0625, duration_s * beats_per_second),
                    velocity=vel,
                    position=start_s * beats_per_second,
                ))

    # Close any notes that never received a NOTE_OFF
    for pitch, (start_s, vel) in active.items():
        notes.append(Note(
            pitch=pitch,
            duration=0.5,
            velocity=vel,
            position=start_s * beats_per_second,
        ))

    notes.sort(key=lambda n: n.position)
    return notes


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/generate-arpeggio",
    response_model=GenerateArpeggioResponse,
    summary="Generate a mood-conditioned arpeggio",
    tags=["Generation"],
)
async def generate_arpeggio(request: GenerateArpeggioRequest) -> GenerateArpeggioResponse:
    """
    Generate a MIDI arpeggio conditioned on a mood description.

    **Pipeline**:
    1. Resolve `mood` text → nearest model mood label.
    2. Generate a neutral base arpeggio (deterministic, rule-based).
    3. Tokenise the arpeggio using the 411-token MIDI event vocabulary.
    4. Run the mood-conditioned transformer to produce modified tokens.
    5. Decode tokens back to notes; fall back to the base arpeggio if empty.
    6. Render notes to a MIDI file and return it as base64.

    **Returns** a JSON body containing:
    - `midi_base64`: The MIDI file, base64-encoded.
    - `notes`: Note events for direct client-side playback.
    - Metadata: key, scale, tempo, mood, note count, duration.
    """
    engine = get_engine()
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded — try again shortly")

    try:
        # 1. Mood resolution
        mood_label = _resolve_mood_label(request.mood)

        # 2. Base arpeggio
        arpeggio = ArpeggioGenerator().generate(
            key=request.key,
            scale=request.scale,
            tempo=request.tempo,
            note_count=request.note_count,
            seed=request.seed,
            pattern=_PATTERN_MAP[request.pattern],
            octave=request.octave,
        )

        # 3. Tokenise
        tokens = _arpeggio_to_tokens(arpeggio, mood_label)

        # 4. Inference — pass mood as int to use the 0-based label directly
        modified_ids = engine.run(tokens, mood_label)

        # 5. Decode; fall back to base arpeggio when output is too sparse.
        # The model is mood-conditioned but may not always produce coherent
        # note sequences — treat anything below 30 % of the requested count
        # as a failed inference and use the deterministic base arpeggio instead.
        out_notes: List[Note] = []
        if modified_ids:
            out_notes = _tokens_to_notes(modified_ids, request.tempo)

        min_acceptable = max(1, int(request.note_count * 0.30))
        if len(out_notes) < min_acceptable:
            logger.warning(
                "Inference returned only %d note(s) for mood='%s' "
                "(requested %d, threshold %d); falling back to base arpeggio",
                len(out_notes), request.mood, request.note_count, min_acceptable,
            )
            out_notes = list(arpeggio.notes)

        # 6. Render → MIDI bytes
        midi_file = MIDIRenderer(enforce_scale=True).render_notes(
            notes=out_notes,
            tempo=request.tempo,
            key=request.key,
            scale=request.scale,
        )
        midi_bytes = midi_to_bytes(midi_file)
        midi_b64 = base64.b64encode(midi_bytes).decode("utf-8")

        # 7. Build response
        note_events = [
            NoteEvent(
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

        logger.info(
            "Generated | key=%s scale=%s tempo=%d mood=%s(label=%d) notes=%d dur=%.2fs",
            request.key, request.scale, request.tempo,
            request.mood, mood_label, len(note_events), duration_seconds,
        )

        return GenerateArpeggioResponse(
            midi_base64=midi_b64,
            notes=note_events,
            key=request.key,
            scale=request.scale,
            tempo=request.tempo,
            mood=request.mood,
            note_count=len(note_events),
            duration_seconds=duration_seconds,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
