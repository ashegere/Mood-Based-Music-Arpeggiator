#!/usr/bin/env python3
"""
Mood Adapter Fine-Tuning Script
================================

Fine-tunes a lightweight MoodConditioningModule on top of a frozen pretrained
backbone transformer.  Only the mood embedding (and optionally the output
projection head) are trained; all backbone weights stay frozen throughout.

Input
-----
A JSON file mapping MIDI file paths → mood labels.  Three formats are
accepted:

    # Format A — simple dict (paths relative to JSON file or --midi-dir)
    {
        "happy_piece.mid":    "happy",
        "dark_ambient.mid":   "dark"
    }

    # Format B — list of dicts with "file" / "mood" keys
    [{"file": "happy_piece.mid", "mood": "happy"}, ...]

    # Format C — existing metadata format (Filename / Mood Description)
    [{"Filename": "happy_piece.mid", "Mood Description": "happy"}, ...]

Pipeline
--------
    MIDI file → pretty_midi → Note events
              → key/tempo inference
              → app.music.tokenization.Tokenizer (REMI format)
              → MoodSequenceDataset
              → MoodFineTuner (backbone frozen)
              → mood_adapter.pt

Usage
-----
    # Basic
    python scripts/finetune_mood_adapter.py \\
        --mood-map  data/midi_mood_map.json \\
        --checkpoint checkpoints/pretrained_music_transformer.pt \\
        --output     checkpoints/mood_adapter.pt

    # Also fine-tune the output projection head
    python scripts/finetune_mood_adapter.py \\
        --mood-map   data/midi_mood_map.json \\
        --checkpoint checkpoints/pretrained_music_transformer.pt \\
        --output     checkpoints/mood_adapter.pt \\
        --finetune-projection

    # Bias injection (additive mood embedding) instead of prepend token
    python scripts/finetune_mood_adapter.py \\
        --mood-map   data/midi_mood_map.json \\
        --checkpoint checkpoints/pretrained_music_transformer.pt \\
        --output     checkpoints/mood_adapter.pt \\
        --injection  bias

    # Dry-run — tokenise and validate everything, then exit without training
    python scripts/finetune_mood_adapter.py \\
        --mood-map   data/midi_mood_map.json \\
        --checkpoint checkpoints/pretrained_music_transformer.pt \\
        --output     checkpoints/mood_adapter.pt \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap: make sure `backend/` is on the Python path regardless of where
# the script is invoked from.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent        # backend/scripts/
_BACKEND_DIR = _SCRIPT_DIR.parent                    # backend/
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ---------------------------------------------------------------------------
# Third-party / project imports — deferred so we can give a clean error if a
# dependency is missing.
# ---------------------------------------------------------------------------
try:
    import pretty_midi
except ImportError:
    print(
        "ERROR: 'pretty_midi' is required.\n"
        "Install it with:  pip install pretty-midi",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Minimal tqdm shim — prints nothing but doesn't crash.
    def tqdm(iterable, **_):  # type: ignore[misc]
        return iterable

import torch

from app.generators.mood_finetuner import (
    FineTuningConfig,
    MoodFineTuner,
    MoodSequenceDataset,
)
from app.music.arpeggio_generator import Note
from app.music.tokenization import (
    PITCH_MAX,
    PITCH_MIN,
    Tokenizer,
    get_vocabulary,
    quantize_tempo,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid mood vocabulary (must match _VALID_MOODS in pretrained_transformer.py)
# ---------------------------------------------------------------------------

_VALID_MOODS: Tuple[str, ...] = (
    "melancholic", "dreamy",    "energetic", "tense",   "happy",
    "sad",         "calm",      "dark",      "joyful",  "uplifting",
    "intense",     "peaceful",  "dramatic",  "epic",    "mysterious",
    "romantic",    "neutral",   "flowing",   "ominous",
)

_MOOD_TO_IDX: Dict[str, int] = {m: i for i, m in enumerate(_VALID_MOODS)}


# ---------------------------------------------------------------------------
# Key inference — Krumhansl-Schmuckler pitch-class correlation
# ---------------------------------------------------------------------------

# Krumhansl (1990) major and minor tonal hierarchy profiles
_KS_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KS_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

_NOTE_NAMES: Tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
)


def _correlate(histogram: List[float], profile: Tuple[float, ...]) -> float:
    """Pearson correlation between a 12-element histogram and a KS profile."""
    import statistics
    if len(histogram) != 12 or len(profile) != 12:
        return 0.0
    mean_h = statistics.mean(histogram)
    mean_p = statistics.mean(profile)
    num = sum((h - mean_h) * (p - mean_p) for h, p in zip(histogram, profile))
    den_h = sum((h - mean_h) ** 2 for h in histogram) ** 0.5
    den_p = sum((p - mean_p) ** 2 for p in profile) ** 0.5
    if den_h < 1e-9 or den_p < 1e-9:
        return 0.0
    return num / (den_h * den_p)


def infer_key(pitches: List[int]) -> Tuple[str, str]:
    """
    Estimate the musical key and mode from a list of MIDI pitch numbers.

    Uses the Krumhansl-Schmuckler key-finding algorithm: builds a 12-bin
    pitch-class histogram and finds the major/minor key whose tonal profile
    correlates most strongly with it.

    Args:
        pitches: MIDI pitch values (any octave).

    Returns:
        ``(key, scale)`` e.g. ``("C", "major")`` or ``("A", "minor")``.
        Falls back to ``("C", "major")`` when the pitch list is empty.
    """
    if not pitches:
        return "C", "major"

    histogram = [0.0] * 12
    for p in pitches:
        histogram[p % 12] += 1.0

    best_key   = "C"
    best_scale = "major"
    best_score = float("-inf")

    for root in range(12):
        # Rotate histogram so that `root` is position 0
        rotated = histogram[root:] + histogram[:root]

        for profile, scale in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            score = _correlate(rotated, profile)
            if score > best_score:
                best_score = score
                best_key   = _NOTE_NAMES[root]
                best_scale = scale

    # Map inferred minor → a scale name the vocabulary knows
    if best_scale == "minor":
        best_scale = "natural_minor"

    return best_key, best_scale


# ---------------------------------------------------------------------------
# MIDI → REMI token conversion
# ---------------------------------------------------------------------------

def midi_to_remi(
    midi_path: Path,
    beats_per_bar: float = 4.0,
    max_notes: int = 256,
) -> Tuple[List[int], str, str, int]:
    """
    Convert a MIDI file to a REMI token ID sequence.

    Extracts all notes from all non-drum instruments, infers the key/mode
    via Krumhansl-Schmuckler, reads the tempo from the MIDI file, then
    delegates to ``app.music.tokenization.Tokenizer.tokenize_notes()``.

    Args:
        midi_path:     Path to the ``.mid`` file.
        beats_per_bar: Time-signature beats per bar (default 4/4).
        max_notes:     Maximum number of notes to include (oldest first,
                       truncated to keep sequence length manageable).

    Returns:
        ``(token_ids, key, scale, tempo)``

    Raises:
        ValueError: If the MIDI file cannot be parsed or yields no notes.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        raise ValueError(f"pretty_midi failed to load '{midi_path}': {exc}") from exc

    # ---- Tempo -------------------------------------------------------
    tempo_times, tempos = pm.get_tempo_changes()
    if len(tempos) > 0:
        # Use the most common tempo (weighted by duration)
        raw_tempo = float(tempos[0])
    else:
        raw_tempo = pm.estimate_tempo()
    tempo = int(quantize_tempo(int(raw_tempo)))

    # ---- Notes -------------------------------------------------------
    all_notes: List[Note] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for n in instrument.notes:
            pitch    = max(PITCH_MIN, min(PITCH_MAX, n.pitch))
            duration = max(0.0625, n.end - n.start)      # at least a 64th note
            position = n.start                             # seconds → beats below
            velocity = max(1, min(127, n.velocity))
            all_notes.append(Note(
                pitch=pitch,
                duration=duration,
                velocity=velocity,
                position=position,
            ))

    if not all_notes:
        raise ValueError(f"No (non-drum) notes found in '{midi_path}'.")

    # Sort by onset, then truncate
    all_notes.sort(key=lambda n: n.position)
    all_notes = all_notes[:max_notes]

    # Convert absolute seconds → beats using the median tempo
    spb = 60.0 / max(raw_tempo, 1.0)   # seconds per beat
    all_notes = [
        Note(
            pitch=n.pitch,
            duration=n.duration / spb,
            velocity=n.velocity,
            position=n.position / spb,
        )
        for n in all_notes
    ]

    # ---- Key inference -----------------------------------------------
    key, scale = infer_key([n.pitch for n in all_notes])

    # ---- Tokenise ----------------------------------------------------
    vocab     = get_vocabulary()
    tokenizer = Tokenizer(vocab)
    seq       = tokenizer.tokenize_notes(
        notes=all_notes,
        key=key,
        scale=scale,
        tempo=tempo,
        add_special_tokens=True,
        beats_per_bar=beats_per_bar,
    )

    return seq.to_ids(), key, scale, tempo


# ---------------------------------------------------------------------------
# Mood-map JSON loader
# ---------------------------------------------------------------------------

def load_mood_map(
    json_path: Path,
    midi_dir: Optional[Path] = None,
) -> Dict[Path, str]:
    """
    Load a ``{midi_path: mood}`` mapping from a JSON file.

    Supported JSON formats:

    * **Dict** ``{"file.mid": "happy", ...}`` — paths relative to the JSON
      file's parent directory (or ``midi_dir`` if provided).
    * **List of dicts** with ``"file"``/``"mood"`` keys.
    * **List of dicts** with ``"Filename"``/``"Mood Description"`` keys
      (existing metadata format).

    Args:
        json_path: Path to the JSON mapping file.
        midi_dir:  Optional base directory for resolving relative file paths.
                   Defaults to the JSON file's parent directory.

    Returns:
        ``{resolved_path: normalised_mood}`` dict.

    Raises:
        FileNotFoundError: JSON file does not exist.
        ValueError:        JSON format is unrecognised.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Mood-map JSON not found: {json_path}")

    base_dir = midi_dir or json_path.parent

    with json_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    pairs: List[Tuple[str, str]] = []

    if isinstance(raw, dict):
        # Format A: {"file.mid": "mood", ...}
        pairs = [(k, str(v)) for k, v in raw.items()]

    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            # Format B
            filename = item.get("file") or item.get("path") or item.get("midi_file")
            mood     = item.get("mood") or item.get("label")
            # Format C (existing metadata)
            if filename is None:
                filename = item.get("Filename") or item.get("filename")
            if mood is None:
                mood = item.get("Mood Description") or item.get("mood_description")
            if filename and mood:
                pairs.append((str(filename), str(mood)))

    else:
        raise ValueError(
            f"Unrecognised JSON format in '{json_path}'.\n"
            "Expected a dict or a list of dicts."
        )

    result: Dict[Path, str] = {}
    for filename, mood in pairs:
        mood_norm = mood.lower().strip()
        p = Path(filename)
        resolved = p if p.is_absolute() else base_dir / p
        result[resolved] = mood_norm

    return result


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

@dataclass
class DatasetStats:
    """Statistics collected while building the dataset."""
    total:        int = 0
    succeeded:    int = 0
    failed:       int = 0
    unknown_mood: int = 0
    mood_counts:  Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mood_counts is None:
            self.mood_counts = {}


def build_dataset(
    mood_map:  Dict[Path, str],
    max_seq_len: int = 512,
    beats_per_bar: float = 4.0,
    max_notes: int = 256,
) -> Tuple[MoodSequenceDataset, DatasetStats]:
    """
    Convert a ``{midi_path: mood}`` mapping into a ``MoodSequenceDataset``.

    Files that cannot be parsed are skipped with a warning.  Moods not in
    the known vocabulary are mapped to ``"neutral"`` (index preserved).

    Args:
        mood_map:     Mapping from resolved MIDI paths to mood labels.
        max_seq_len:  Maximum token sequence length (longer sequences are
                      truncated at construction time).
        beats_per_bar: Time-signature denominator for REMI tokenisation.
        max_notes:    Maximum notes extracted per MIDI file.

    Returns:
        ``(dataset, stats)``
    """
    stats = DatasetStats(total=len(mood_map))

    mood_labels: List[int] = []
    sequences:   List[List[int]] = []

    neutral_idx = _MOOD_TO_IDX.get("neutral", 0)

    items = list(mood_map.items())
    for midi_path, mood_str in tqdm(items, desc="Tokenising MIDI", unit="file"):
        # ---- Mood label --------------------------------------------------
        mood_norm = mood_str.lower().strip()
        if mood_norm not in _MOOD_TO_IDX:
            logger.warning(
                "Unknown mood '%s' for '%s' — mapped to 'neutral'.",
                mood_str, midi_path.name,
            )
            stats.unknown_mood += 1
            mood_norm = "neutral"
        mood_idx = _MOOD_TO_IDX[mood_norm]

        # ---- MIDI → REMI tokens ------------------------------------------
        if not midi_path.exists():
            logger.warning("MIDI file not found, skipping: %s", midi_path)
            stats.failed += 1
            continue

        try:
            token_ids, key, scale, tempo = midi_to_remi(
                midi_path,
                beats_per_bar=beats_per_bar,
                max_notes=max_notes,
            )
        except ValueError as exc:
            logger.warning("Skipping '%s': %s", midi_path.name, exc)
            stats.failed += 1
            continue

        if len(token_ids) < 5:
            logger.warning(
                "Skipping '%s': sequence too short (%d tokens).",
                midi_path.name, len(token_ids),
            )
            stats.failed += 1
            continue

        logger.debug(
            "  %-40s  mood=%-12s  key=%-4s scale=%-14s  tempo=%3d  tokens=%d",
            midi_path.name, mood_norm, key, scale, tempo, len(token_ids),
        )

        mood_labels.append(mood_idx)
        sequences.append(token_ids)
        stats.succeeded += 1
        stats.mood_counts[mood_norm] = stats.mood_counts.get(mood_norm, 0) + 1

    stats.failed += stats.total - stats.succeeded - stats.failed

    return (
        MoodSequenceDataset(mood_labels, sequences, max_seq_len=max_seq_len),
        stats,
    )


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def _banner(text: str, width: int = 60) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_dataset_summary(stats: DatasetStats) -> None:
    """Print a human-readable dataset summary."""
    _banner("DATASET SUMMARY")
    print(f"  Total MIDI files   : {stats.total}")
    print(f"  Tokenised OK       : {stats.succeeded}")
    print(f"  Failed / skipped   : {stats.failed}")
    print(f"  Unknown mood remapped: {stats.unknown_mood}")
    if stats.mood_counts:
        print("\n  Mood distribution:")
        total = stats.succeeded or 1
        for mood, count in sorted(stats.mood_counts.items(), key=lambda x: -x[1]):
            bar = "█" * int(count / total * 30)
            print(f"    {mood:<16} {count:4d} ({count/total*100:5.1f}%)  {bar}")
    print()


def print_training_config(cfg: FineTuningConfig) -> None:
    """Print training hyperparameters."""
    _banner("TRAINING CONFIG")
    for k, v in asdict(cfg).items():
        if k == "mood_names":
            print(f"  {k:<22}: {len(v)} moods")
        else:
            print(f"  {k:<22}: {v}")
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="finetune_mood_adapter",
        description="Fine-tune mood conditioning adapter on MIDI data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Required ---------------------------------------------------------
    p.add_argument(
        "--mood-map", "-m",
        required=True,
        metavar="JSON",
        help="JSON file mapping MIDI paths → mood labels.",
    )
    p.add_argument(
        "--checkpoint", "-c",
        required=True,
        metavar="PT",
        help="Path to the pretrained backbone checkpoint (.pt).",
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        metavar="PT",
        help="Output path for the saved mood adapter checkpoint.",
    )

    # ---- Data options -----------------------------------------------------
    p.add_argument(
        "--midi-dir", "-d",
        default=None,
        metavar="DIR",
        help=(
            "Base directory for resolving relative MIDI paths in the JSON. "
            "Defaults to the JSON file's parent directory."
        ),
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        metavar="N",
        help="Maximum REMI token sequence length (default: 512).",
    )
    p.add_argument(
        "--max-notes",
        type=int,
        default=256,
        metavar="N",
        help="Maximum notes extracted from each MIDI file (default: 256).",
    )
    p.add_argument(
        "--beats-per-bar",
        type=float,
        default=4.0,
        metavar="F",
        help="Time-signature beats per bar used during tokenisation (default: 4.0).",
    )

    # ---- Model options ----------------------------------------------------
    p.add_argument(
        "--injection",
        choices=["prepend", "bias"],
        default="prepend",
        help=(
            "Mood injection strategy (default: prepend).\n"
            "  prepend — mood as a virtual first token.\n"
            "  bias    — additive shift on all token embeddings."
        ),
    )
    p.add_argument(
        "--finetune-projection",
        action="store_true",
        default=False,
        help=(
            "Also fine-tune an untied copy of the output projection head "
            "(in addition to the mood embedding)."
        ),
    )

    # ---- Optimisation hyperparameters ------------------------------------
    p.add_argument("--epochs",      type=int,   default=50,   metavar="N",  help="Training epochs (default: 50).")
    p.add_argument("--batch-size",  type=int,   default=16,   metavar="N",  help="Batch size (default: 16).")
    p.add_argument("--lr",          type=float, default=3e-4, metavar="F",  help="Peak learning rate (default: 3e-4).")
    p.add_argument("--weight-decay",type=float, default=1e-2, metavar="F",  help="AdamW weight decay (default: 1e-2).")
    p.add_argument("--grad-clip",   type=float, default=1.0,  metavar="F",  help="Gradient clip norm (0 = disabled; default: 1.0).")
    p.add_argument("--label-smoothing", type=float, default=0.1, metavar="F", help="Label smoothing (default: 0.1).")
    p.add_argument("--warmup-steps",type=int,   default=100,  metavar="N",  help="LR warm-up steps (default: 100).")
    p.add_argument("--val-split",   type=float, default=0.1,  metavar="F",  help="Validation fraction (default: 0.1).")
    p.add_argument("--eval-interval",type=int,  default=5,    metavar="N",  help="Validate every N epochs (default: 5).")
    p.add_argument("--patience",    type=int,   default=10,   metavar="N",  help="Early-stopping patience in epochs (default: 10).")

    # ---- Misc ------------------------------------------------------------
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Tokenise and validate data, print summary, then exit without training.",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # ---- Logging ----------------------------------------------------------
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- Path resolution --------------------------------------------------
    mood_map_path  = Path(args.mood_map).resolve()
    checkpoint     = Path(args.checkpoint).resolve()
    output_path    = Path(args.output).resolve()
    midi_dir       = Path(args.midi_dir).resolve() if args.midi_dir else None

    # ---- Validate inputs --------------------------------------------------
    if not mood_map_path.exists():
        logger.error("Mood-map JSON not found: %s", mood_map_path)
        return 1

    if not checkpoint.exists():
        logger.error("Backbone checkpoint not found: %s", checkpoint)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load mood map ----------------------------------------------------
    logger.info("Loading mood map from '%s' …", mood_map_path)
    try:
        mood_map = load_mood_map(mood_map_path, midi_dir=midi_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load mood map: %s", exc)
        return 1

    if not mood_map:
        logger.error("Mood map is empty — nothing to train on.")
        return 1

    logger.info("  %d entries loaded.", len(mood_map))

    # ---- Tokenise MIDI files ----------------------------------------------
    _banner("TOKENISING MIDI FILES")
    t0 = time.perf_counter()

    dataset, stats = build_dataset(
        mood_map,
        max_seq_len=args.max_seq_len,
        beats_per_bar=args.beats_per_bar,
        max_notes=args.max_notes,
    )

    elapsed = time.perf_counter() - t0
    logger.info("Tokenisation complete in %.1f s.", elapsed)
    print_dataset_summary(stats)

    if stats.succeeded == 0:
        logger.error("No usable MIDI files — cannot train.")
        return 1

    if stats.succeeded < 2:
        logger.error(
            "Only %d sequence(s) — need at least 2 for a train/val split.",
            stats.succeeded,
        )
        return 1

    if args.dry_run:
        print("  --dry-run specified: skipping training.")
        return 0

    # ---- Build FineTuningConfig -------------------------------------------
    cfg = FineTuningConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        warmup_steps=args.warmup_steps,
        injection_method=args.injection,
        mood_names=list(_VALID_MOODS),
        checkpoint_path=str(checkpoint),
        adapter_save_path=str(output_path),
        val_split=args.val_split,
        eval_interval=args.eval_interval,
        patience=args.patience,
        finetune_projection=args.finetune_projection,
    )

    print_training_config(cfg)

    # ---- Trainable parameters summary ------------------------------------
    _banner("TRAINABLE PARAMETERS")
    print(f"  Mood embedding  : {len(_VALID_MOODS)} moods × d_model")
    if args.finetune_projection:
        print("  Projection head : YES (untied copy of backbone head)")
    else:
        print("  Projection head : NO  (backbone head frozen)")
    print(f"  Backbone        : FULLY FROZEN")
    print()

    # ---- Train ------------------------------------------------------------
    _banner("TRAINING")
    tuner = MoodFineTuner(config=cfg)

    try:
        adapter = tuner.train(dataset)
    except FileNotFoundError as exc:
        logger.error("Training failed — backbone checkpoint missing: %s", exc)
        return 1
    except RuntimeError as exc:
        logger.error("Training failed: %s", exc)
        return 1

    # ---- Final summary ----------------------------------------------------
    _banner("DONE")
    print(f"  Adapter saved to : {output_path}")
    print(f"  Injection method : {args.injection}")
    print(f"  Projection head  : {'yes' if args.finetune_projection else 'no'}")
    print()
    print("  Load for inference:")
    print("    from app.generators.mood_finetuner import MoodFineTuner")
    print(f"    adapter, cfg, proj = MoodFineTuner.load_adapter('{output_path}')")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
