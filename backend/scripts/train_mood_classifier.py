"""
Train a lightweight mood classifier on top of frozen backbone embeddings.

The classifier is a two-layer MLP (``MoodClassifierHead``) that maps a
mean-pooled backbone hidden state → mood class probabilities.  Because the
backbone is frozen, embeddings are extracted once and cached in RAM, making
training extremely fast (seconds to minutes on CPU).

Input
-----
- ``--mood-map``    JSON file mapping MIDI paths → mood labels.
- ``--checkpoint``  Pretrained backbone checkpoint (``.pt``).

Output
------
- ``--output``      Classifier checkpoint accepted by ``MoodAlignmentScorer``.

JSON formats accepted
---------------------
Any of these three shapes::

    {"data/a.mid": "melancholic", "data/b.mid": "happy"}

    [{"file": "a.mid", "mood": "happy"}, ...]

    [{"Filename": "a.mid", "Mood Description": "happy"}, ...]

Usage
-----
::

    python scripts/train_mood_classifier.py \\
        --mood-map  data/midi_mood_map.json \\
        --checkpoint checkpoints/pretrained_music_transformer.pt \\
        --output     checkpoints/mood_classifier.pt

    # With validation split and early stopping:
    python scripts/train_mood_classifier.py \\
        --mood-map data/midi_mood_map.json \\
        --checkpoint checkpoints/pretrained_music_transformer.pt \\
        --output checkpoints/mood_classifier.pt \\
        --val-split 0.15 --patience 10 --epochs 200

    # Dry-run (check data loading without training):
    python scripts/train_mood_classifier.py \\
        --mood-map data/midi_mood_map.json \\
        --checkpoint checkpoints/pretrained_music_transformer.pt \\
        --output checkpoints/mood_classifier.pt \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Project path setup — allows running from the repo root or backend/
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent       # …/backend/scripts/
_BACKEND    = _SCRIPT_DIR.parent                    # …/backend/
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.generators.pretrained_transformer import (
    _SymbolicMusicTransformer,
    _auto_detect_arch,
)
from app.generators.mood_classifier import (
    ClassifierConfig,
    MoodClassifierHead,
    MoodAlignmentScorer,
    _VALID_MOODS,
)
from app.music.tokenization import (
    PITCH_MAX,
    PITCH_MIN,
    Tokenizer,
    get_vocabulary,
    quantize_duration,
    quantize_tempo,
    quantize_velocity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MIDI → REMI token IDs   (same approach as finetune_mood_adapter.py)
# ---------------------------------------------------------------------------

_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

_NOTE_TO_SEMITONE: Dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F":  5, "F#": 6, "Gb": 6, "G":  7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

_SEMITONE_TO_NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def _infer_key(pitches: List[int]) -> Tuple[str, str]:
    """Krumhansl-Schmuckler key-finding algorithm."""
    import numpy as np
    if not pitches:
        return "C", "major"
    hist = [0.0] * 12
    for p in pitches:
        hist[p % 12] += 1.0
    hist_arr = np.array(hist)
    if hist_arr.sum() > 0:
        hist_arr = hist_arr / hist_arr.sum()
    major_p = np.array(_KS_MAJOR); minor_p = np.array(_KS_MINOR)
    major_p /= major_p.sum(); minor_p /= minor_p.sum()

    best_corr = -9e9; best_key = "C"; best_mode = "major"
    for root in range(12):
        rot = np.roll(hist_arr, -root)
        maj_corr = float(np.corrcoef(rot, major_p)[0, 1])
        min_corr = float(np.corrcoef(rot, minor_p)[0, 1])
        if not math.isfinite(maj_corr): maj_corr = 0.0
        if not math.isfinite(min_corr): min_corr = 0.0
        if maj_corr > best_corr:
            best_corr = maj_corr; best_key = _SEMITONE_TO_NOTE[root]; best_mode = "major"
        if min_corr > best_corr:
            best_corr = min_corr; best_key = _SEMITONE_TO_NOTE[root]; best_mode = "natural_minor"
    return best_key, best_mode


def midi_to_remi(
    midi_path: Path,
    beats_per_bar: int = 4,
    max_notes: int = 256,
) -> Optional[List[int]]:
    """
    Load a MIDI file and convert it to REMI token IDs.

    Returns ``None`` if the file cannot be parsed or yields no notes.
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi is required: pip install pretty_midi")

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        logger.warning("Cannot load MIDI %s: %s", midi_path, exc)
        return None

    # Collect all notes across instruments
    all_notes: List[pretty_midi.Note] = []
    for inst in pm.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)
    if not all_notes:
        return None

    all_notes.sort(key=lambda n: n.start)
    all_notes = all_notes[:max_notes]

    pitches = [n.pitch for n in all_notes]
    key_str, scale_str = _infer_key(pitches)

    tempo_bpm = 120
    if pm.estimate_tempo() is not None:
        tempos, _ = pm.get_tempo_changes()
        if len(tempos) > 0:
            tempo_bpm = int(tempos[0])

    vocab     = get_vocabulary()
    tokenizer = Tokenizer(vocab)

    # Build Note-like objects the tokenizer expects
    from app.music.arpeggio_generator import Note
    notes_obj: List[Note] = []
    spb = 60.0 / max(tempo_bpm, 1)
    for n in all_notes:
        pos  = n.start / spb
        dur  = max(0.125, (n.end - n.start) / spb)
        vel  = n.velocity
        notes_obj.append(Note(
            pitch=max(PITCH_MIN, min(PITCH_MAX, n.pitch)),
            velocity=max(1, min(127, vel)),
            position=pos,
            duration=dur,
        ))

    try:
        token_ids = tokenizer.tokenize(
            notes=notes_obj,
            key=key_str,
            scale=scale_str,
            tempo=tempo_bpm,
        )
    except Exception as exc:
        logger.warning("Tokenization failed for %s: %s", midi_path, exc)
        return None

    return token_ids


# ---------------------------------------------------------------------------
# JSON mood-map loading
# ---------------------------------------------------------------------------

def load_mood_map(
    json_path: Path,
    midi_dir: Optional[Path] = None,
) -> Dict[Path, str]:
    """
    Load ``{midi_path: mood_label}`` from a JSON file.

    Supports three shapes:
    - ``{"file.mid": "mood", ...}``
    - ``[{"file": "file.mid", "mood": "happy"}, ...]``
    - ``[{"Filename": "file.mid", "Mood Description": "happy"}, ...]``
    """
    raw = json.loads(json_path.read_text())
    result: Dict[Path, str] = {}
    base = midi_dir or json_path.parent

    if isinstance(raw, dict):
        for fp, mood in raw.items():
            p = Path(fp)
            if not p.is_absolute():
                p = base / p
            result[p] = mood.strip().lower()
    elif isinstance(raw, list):
        for item in raw:
            fp   = item.get("file") or item.get("Filename", "")
            mood = item.get("mood") or item.get("Mood Description", "")
            if not fp or not mood:
                continue
            p = Path(fp)
            if not p.is_absolute():
                p = base / p
            result[p] = mood.strip().lower()
    else:
        raise ValueError("Unsupported JSON mood-map format.")

    return result


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def load_backbone(
    checkpoint_path: Path,
    device: torch.device,
) -> _SymbolicMusicTransformer:
    """Load backbone from checkpoint (auto-detect arch)."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw["model_state_dict"] if "model_state_dict" in raw else raw
    arch  = _auto_detect_arch(state)
    model = _SymbolicMusicTransformer(
        vocab_size=arch["vocab_size"],
        d_model=arch["d_model"],
        nhead=arch["nhead"],
        num_layers=arch["num_layers"],
        d_ff=arch["d_ff"],
        max_seq_len=arch["max_seq_len"],
        dropout=0.0,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)   # Backbone stays frozen throughout
    return model


# ---------------------------------------------------------------------------
# Embedding extraction  (done once; cached in RAM for all training epochs)
# ---------------------------------------------------------------------------

def extract_embeddings(
    backbone:  _SymbolicMusicTransformer,
    token_seqs: List[List[int]],
    device:    torch.device,
) -> Tensor:
    """
    Mean-pool backbone hidden states for each sequence.

    Args:
        backbone:   Frozen backbone model (eval mode, on ``device``).
        token_seqs: List of token ID lists (one per training example).
        device:     Computation device.

    Returns:
        Float tensor ``(N, d_model)`` of sequence embeddings.
    """
    embeddings: List[Tensor] = []
    max_seq = backbone.max_seq_len

    with torch.no_grad():
        for ids in token_seqs:
            if not ids:
                embeddings.append(torch.zeros(backbone.d_model, device=device))
                continue
            t = torch.tensor([ids[-max_seq:]], dtype=torch.long, device=device)
            h = backbone.forward_hidden(t)   # (1, L, d_model)
            embeddings.append(h.mean(dim=1).squeeze(0))  # (d_model,)

    return torch.stack(embeddings)   # (N, d_model)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    head:            MoodClassifierHead,
    embeddings:      Tensor,           # (N, d_model) — float, on device
    labels:          Tensor,           # (N,)          — long, on device
    val_embeddings:  Optional[Tensor],
    val_labels:      Optional[Tensor],
    device:          torch.device,
    num_epochs:      int   = 100,
    batch_size:      int   = 32,
    lr:              float = 1e-3,
    weight_decay:    float = 1e-4,
    patience:        int   = 15,
    label_smoothing: float = 0.05,
    embedding_noise: float = 0.0,
) -> float:
    """
    Train ``head`` on pre-extracted embeddings.

    Args:
        head:            Classifier head (will be modified in place).
        embeddings:      Pre-extracted training embeddings ``(N, d_model)``.
        labels:          Mood class labels ``(N,)``.
        val_embeddings:  Validation embeddings (or ``None`` to skip).
        val_labels:      Validation labels (or ``None`` to skip).
        device:          Computation device.
        num_epochs:      Maximum training epochs.
        batch_size:      Mini-batch size.
        lr:              AdamW learning rate.
        weight_decay:    AdamW weight decay.
        patience:        Early-stopping patience (epochs without val improvement).
        label_smoothing: Label smoothing for cross-entropy loss.
        embedding_noise: Std of Gaussian noise added to embeddings per step
                         for data augmentation (0 = off).

    Returns:
        Best validation loss (or final training loss if no validation set).
    """
    head.to(device)
    head.train()

    optimizer = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    n         = len(embeddings)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = {k: v.clone() for k, v in head.state_dict().items()}

    for epoch in range(1, num_epochs + 1):
        head.train()
        perm  = torch.randperm(n, device=device)
        total_loss = 0.0; total_steps = 0

        for start in range(0, n, batch_size):
            idx     = perm[start: start + batch_size]
            emb_b   = embeddings[idx]
            lbl_b   = labels[idx]

            if embedding_noise > 0.0:
                emb_b = emb_b + torch.randn_like(emb_b) * embedding_noise

            logits = head(emb_b)
            loss   = F.cross_entropy(logits, lbl_b, label_smoothing=label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            total_steps += 1

        avg_train = total_loss / max(total_steps, 1)

        # Validation
        if val_embeddings is not None and val_labels is not None:
            head.eval()
            with torch.no_grad():
                val_logits = head(val_embeddings)
                val_loss   = F.cross_entropy(
                    val_logits, val_labels,
                    label_smoothing=label_smoothing,
                ).item()
                val_acc = (val_logits.argmax(dim=-1) == val_labels).float().mean().item()

            logger.info(
                "Epoch %3d | train_loss=%.4f  val_loss=%.4f  val_acc=%.1f%%",
                epoch, avg_train, val_loss, val_acc * 100,
            )

            if val_loss < best_val_loss - 1e-5:
                best_val_loss      = val_loss
                epochs_no_improve  = 0
                best_state         = {k: v.clone() for k, v in head.state_dict().items()}
            else:
                epochs_no_improve += 1
                if patience > 0 and epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d, best_val=%.4f)",
                        epoch, patience, best_val_loss,
                    )
                    break
        else:
            if epoch % 10 == 0 or epoch == 1:
                logger.info("Epoch %3d | train_loss=%.4f", epoch, avg_train)
            best_val_loss = avg_train
            best_state    = {k: v.clone() for k, v in head.state_dict().items()}

    # Restore best weights
    head.load_state_dict(best_state)
    head.eval()
    return best_val_loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a mood classifier head on backbone embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--mood-map",    required=True, type=Path,
                   help="JSON file mapping MIDI paths → mood labels.")
    p.add_argument("--checkpoint",  required=True, type=Path,
                   help="Pretrained backbone checkpoint (.pt).")
    p.add_argument("--output",      required=True, type=Path,
                   help="Where to save the trained classifier (.pt).")
    # Data
    p.add_argument("--midi-dir",    type=Path, default=None,
                   help="Base directory for relative MIDI paths in mood-map.")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="Maximum tokens per MIDI file (longer sequences truncated).")
    # Architecture
    p.add_argument("--hidden-dim",  type=int, default=0,
                   help="MLP hidden dimension (0 = auto: d_model // 2).")
    p.add_argument("--dropout",     type=float, default=0.1,
                   help="Dropout rate in MLP head.")
    # Training
    p.add_argument("--epochs",      type=int, default=150)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--embedding-noise", type=float, default=0.0,
                   help="Std of Gaussian noise added to embeddings per step "
                        "(data augmentation). 0 = off.")
    p.add_argument("--val-split",   type=float, default=0.1,
                   help="Fraction of data to use for validation (0 = none).")
    p.add_argument("--patience",    type=int, default=20,
                   help="Early stopping patience in epochs (0 = off).")
    p.add_argument("--seed",        type=int, default=42)
    # Misc
    p.add_argument("--dry-run",     action="store_true",
                   help="Load and tokenise data only; do not train.")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Device selection ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ---- Validate paths ----
    if not args.mood_map.exists():
        sys.exit(f"Mood map not found: {args.mood_map}")
    if not args.checkpoint.exists():
        sys.exit(f"Backbone checkpoint not found: {args.checkpoint}")

    # ---- Load mood map ----
    mood_map = load_mood_map(args.mood_map, args.midi_dir)
    logger.info("Mood map: %d entries", len(mood_map))

    # ---- Collect unique moods and build label encoding ----
    raw_moods = sorted(set(mood_map.values()))
    # Keep canonical order where possible; unknown moods appended.
    ordered_moods: List[str] = [m for m in _VALID_MOODS if m in raw_moods]
    for m in raw_moods:
        if m not in ordered_moods:
            ordered_moods.append(m)
    mood_to_label = {m: i for i, m in enumerate(ordered_moods)}
    num_moods     = len(ordered_moods)
    logger.info("Mood classes (%d): %s", num_moods, ordered_moods)

    # ---- Tokenise MIDI files ----
    token_seqs: List[List[int]] = []
    labels:     List[int]       = []
    skipped = 0

    for midi_path, mood in mood_map.items():
        ids = midi_to_remi(midi_path, max_notes=args.max_seq_len)
        if ids is None:
            skipped += 1
            continue
        if mood not in mood_to_label:
            skipped += 1
            continue
        # Apply max_seq_len truncation
        token_seqs.append(ids[-args.max_seq_len:])
        labels.append(mood_to_label[mood])

    n_samples = len(token_seqs)
    logger.info(
        "Tokenised: %d samples  skipped: %d",
        n_samples, skipped,
    )

    if n_samples == 0:
        sys.exit("No usable training samples found. Check your mood map and MIDI paths.")

    # Per-class counts
    from collections import Counter
    counts = Counter(ordered_moods[l] for l in labels)
    for mood, cnt in sorted(counts.items()):
        logger.info("  %-15s %d samples", mood, cnt)

    if args.dry_run:
        logger.info("Dry-run complete — not training.")
        return

    # ---- Load backbone ----
    logger.info("Loading backbone from %s …", args.checkpoint)
    backbone = load_backbone(args.checkpoint, device)
    d_model  = backbone.d_model
    logger.info("Backbone loaded | d_model=%d", d_model)

    # ---- Pre-extract embeddings (once — cached for all epochs) ----
    logger.info("Extracting backbone embeddings for %d sequences …", n_samples)
    all_embeddings = extract_embeddings(backbone, token_seqs, device).cpu()
    all_labels     = torch.tensor(labels, dtype=torch.long)
    logger.info("Embeddings shape: %s", tuple(all_embeddings.shape))

    # ---- Train / validation split ----
    if args.val_split > 0.0 and n_samples >= 4:
        n_val    = max(1, int(n_samples * args.val_split))
        n_train  = n_samples - n_val
        idx_perm = torch.randperm(n_samples).tolist()
        train_idx = idx_perm[:n_train]
        val_idx   = idx_perm[n_train:]
        train_emb = all_embeddings[train_idx].to(device)
        train_lbl = all_labels[train_idx].to(device)
        val_emb   = all_embeddings[val_idx].to(device)
        val_lbl   = all_labels[val_idx].to(device)
        logger.info("Train: %d  Val: %d", n_train, n_val)
    else:
        train_emb = all_embeddings.to(device)
        train_lbl = all_labels.to(device)
        val_emb = val_lbl = None
        logger.info("No validation split (train on all %d samples).", n_samples)

    # ---- Build classifier head ----
    hidden_dim = args.hidden_dim if args.hidden_dim > 0 else d_model // 2
    config = ClassifierConfig(
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_moods=num_moods,
        mood_names=ordered_moods,
        dropout=args.dropout,
    )
    head = MoodClassifierHead(config)
    n_params = sum(p.numel() for p in head.parameters())
    logger.info(
        "Classifier | d_model=%d  hidden=%d  classes=%d  params=%d",
        d_model, hidden_dim, num_moods, n_params,
    )

    # ---- Train ----
    best_loss = train(
        head=head,
        embeddings=train_emb,
        labels=train_lbl,
        val_embeddings=val_emb,
        val_labels=val_lbl,
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        embedding_noise=args.embedding_noise,
    )
    logger.info("Training complete | best_loss=%.4f", best_loss)

    # ---- Evaluate on full training set ----
    head.eval()
    with torch.no_grad():
        all_logits = head(all_embeddings.to(device))
        preds      = all_logits.argmax(dim=-1).cpu()
    accuracy = (preds == all_labels).float().mean().item()
    logger.info("Train-set accuracy: %.1f%%", accuracy * 100)

    # ---- Save ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scorer = MoodAlignmentScorer(
        backbone=backbone,
        head=head,
        config=config,
        device=device,
    )
    scorer.save(args.output)
    logger.info("Classifier saved → %s", args.output)


if __name__ == "__main__":
    main()
