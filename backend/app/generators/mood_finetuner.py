"""
Fine-tuning harness for the mood conditioning adapter.

Only ``MoodConditioningModule`` parameters are trained; the backbone
``_SymbolicMusicTransformer`` weights remain frozen throughout.

Quick-start
-----------
::

    from app.generators.mood_finetuner import FineTuningConfig, MoodFineTuner
    from app.generators.mood_finetuner import MoodSequenceDataset

    # 1.  Build (or load) a dataset of (mood_label, token_sequence) pairs.
    dataset = MoodSequenceDataset.from_dataset_file(
        "data/training/train_dataset.pt",
        mood_names=list(FineTuningConfig().mood_names),
    )

    # 2.  Run fine-tuning.
    cfg = FineTuningConfig(
        checkpoint_path="checkpoints/pretrained_music_transformer.pt",
        adapter_save_path="checkpoints/mood_adapter.pt",
        num_epochs=50,
        injection_method="prepend",
    )
    tuner   = MoodFineTuner(cfg)
    adapter = tuner.train(dataset)

    # 3.  Load the saved adapter back for inference.
    adapter, config = MoodFineTuner.load_adapter("checkpoints/mood_adapter.pt")

Checkpoint format
-----------------
::

    {
        "adapter_state_dict": {<module weights>},
        "config": {
            "num_moods":         int,
            "d_model":           int,
            "injection_method":  str,   # "prepend" | "bias"
            "mood_names":        List[str],
        },
        "epoch":         int | None,
        "best_val_loss": float | None,
    }
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from app.generators.pretrained_transformer import (
    MoodAdapterConfig,
    MoodConditioningModule,
    _SymbolicMusicTransformer,
    _auto_detect_arch,
)
from app.music.tokenization import get_vocabulary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default mood vocabulary (must match _VALID_MOODS in pretrained_transformer)
# ---------------------------------------------------------------------------

_DEFAULT_MOODS: Tuple[str, ...] = (
    "melancholic", "dreamy",    "energetic", "tense",   "happy",
    "sad",         "calm",      "dark",      "joyful",  "uplifting",
    "intense",     "peaceful",  "dramatic",  "epic",    "mysterious",
    "romantic",    "neutral",   "flowing",   "ominous",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FineTuningConfig:
    """
    Hyperparameters and paths for the mood adapter fine-tuning run.

    Attributes:
        num_epochs:           Total training epochs.
        batch_size:           Samples per batch.
        learning_rate:        Peak AdamW learning rate.
        weight_decay:         AdamW L2 penalty.
        grad_clip:            Maximum gradient norm (0 = disabled).
        label_smoothing:      Cross-entropy label smoothing (0–1).
        warmup_steps:         Linear warm-up steps before cosine decay.
        injection_method:     ``"prepend"`` or ``"bias"``.
        mood_names:           Ordered mood labels; index == label ID.
        checkpoint_path:      Path to the pretrained backbone checkpoint.
        adapter_save_path:    Where to write the best adapter checkpoint.
        dataset_path:         Path to the training dataset ``.pt`` file.
        val_split:            Fraction of the dataset held out for validation.
        eval_interval:        Run validation every N epochs.
        patience:             Early-stopping patience in epochs.
        finetune_projection:  When ``True``, also fine-tune a standalone
                              copy of the final projection head (untied from
                              the backbone's weight-tied embedding).  The
                              projection weights are saved inside the adapter
                              checkpoint and loaded automatically at inference.
    """
    num_epochs:           int       = 50
    batch_size:           int       = 16
    learning_rate:        float     = 3e-4
    weight_decay:         float     = 1e-2
    grad_clip:            float     = 1.0
    label_smoothing:      float     = 0.1
    warmup_steps:         int       = 100
    injection_method:     str       = "prepend"
    mood_names:           List[str] = field(
        default_factory=lambda: list(_DEFAULT_MOODS)
    )
    checkpoint_path:      str       = "checkpoints/pretrained_music_transformer.pt"
    adapter_save_path:    str       = "checkpoints/mood_adapter.pt"
    dataset_path:         str       = "data/training/train_dataset.pt"
    val_split:            float     = 0.1
    eval_interval:        int       = 5
    patience:             int       = 10
    finetune_projection:  bool      = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MoodSequenceDataset(Dataset):
    """
    Dataset of ``(mood_label, token_sequence)`` pairs.

    Each item yields ``(mood_index: int, token_ids: List[int])``.
    Sequences are truncated to ``max_seq_len`` at construction time.

    Supported ``.pt`` file formats
    --------------------------------
    **Format 1** — list of dicts::

        [
            {"mood": "happy",  "token_ids": [3, 12, 47, ...]},
            {"mood": "calm",   "token_ids": [3, 9, 22,  ...]},
            ...
        ]

    **Format 2** — dict of lists (compatible with existing training data)::

        {
            "input_ids":    Tensor of shape (N, L),    # or list of lists
            "mood_labels":  Tensor of shape (N,),      # integer labels
            # optionally: "moods": List[str]
        }
    """

    def __init__(
        self,
        mood_labels: List[int],
        sequences:   List[List[int]],
        max_seq_len: int = 512,
    ) -> None:
        if len(mood_labels) != len(sequences):
            raise ValueError(
                f"mood_labels ({len(mood_labels)}) and sequences "
                f"({len(sequences)}) must have the same length."
            )
        self.mood_labels = mood_labels
        self.sequences   = [s[:max_seq_len] for s in sequences]
        self.max_seq_len = max_seq_len

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.mood_labels)

    def __getitem__(self, idx: int) -> Tuple[int, List[int]]:
        return self.mood_labels[idx], self.sequences[idx]

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(
        batch: List[Tuple[int, List[int]]],
    ) -> Tuple[Tensor, Tensor]:
        """
        Pad sequences to the same length within a batch.

        Args:
            batch: List of ``(mood_label, token_ids)`` pairs.

        Returns:
            ``(mood_labels, padded_sequences)`` where:
                - ``mood_labels`` is ``(B,)`` long.
                - ``padded_sequences`` is ``(B, max_len)`` long (zero-padded).
        """
        mood_labels, sequences = zip(*batch)
        max_len = max(len(s) for s in sequences)
        padded  = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return (
            torch.tensor(mood_labels, dtype=torch.long),
            padded,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset_file(
        cls,
        path:        Union[str, Path],
        mood_names:  List[str],
        max_seq_len: int = 512,
    ) -> "MoodSequenceDataset":
        """
        Load a dataset from a ``.pt`` file.

        Supports both Format 1 (list of dicts) and Format 2 (dict of
        tensors/lists) — see class docstring.

        Args:
            path:        Path to the ``.pt`` file.
            mood_names:  Ordered mood labels matching the adapter config.
            max_seq_len: Sequences longer than this are truncated.

        Returns:
            ``MoodSequenceDataset`` ready for use with ``DataLoader``.

        Raises:
            ValueError: If the file format is unrecognised.
        """
        path = Path(path)
        data = torch.load(path, map_location="cpu", weights_only=False)

        mood_to_idx: Dict[str, int] = {m: i for i, m in enumerate(mood_names)}
        neutral_idx: int            = mood_to_idx.get("neutral", 0)

        if isinstance(data, list):
            # Format 1: [{"mood": str, "token_ids": list}, ...]
            mood_labels, sequences = [], []
            for item in data:
                mood_str = item.get("mood", "neutral")
                label    = mood_to_idx.get(mood_str.lower(), neutral_idx)
                mood_labels.append(label)
                sequences.append(item["token_ids"])

        elif isinstance(data, dict) and (
            "input_ids" in data or "sequences" in data
        ):
            # Format 2: {"input_ids": Tensor, "mood_labels": Tensor, ...}
            seq_key   = "input_ids" if "input_ids" in data else "sequences"
            raw_seqs  = data[seq_key]

            # Convert tensor → list of lists
            if isinstance(raw_seqs, Tensor):
                sequences = raw_seqs.tolist()
            else:
                sequences = [list(s) for s in raw_seqs]

            if "mood_labels" in data:
                raw_labels  = data["mood_labels"]
                mood_labels = (
                    raw_labels.tolist()
                    if isinstance(raw_labels, Tensor)
                    else list(raw_labels)
                )
            elif "moods" in data:
                mood_labels = [
                    mood_to_idx.get(str(m).lower(), neutral_idx)
                    for m in data["moods"]
                ]
            else:
                logger.warning(
                    "Dataset has no mood labels — assigning 'neutral' to all %d sequences.",
                    len(sequences),
                )
                mood_labels = [neutral_idx] * len(sequences)
        else:
            raise ValueError(
                f"Unrecognised dataset format in '{path}'.\n"
                "Expected a list of dicts or a dict with 'input_ids'/'sequences'."
            )

        logger.info(
            "Loaded %d sequences from '%s' (max_seq_len=%d).",
            len(sequences), path, max_seq_len,
        )
        return cls(mood_labels, sequences, max_seq_len=max_seq_len)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _cosine_schedule_with_warmup(
    optimizer:            AdamW,
    num_warmup_steps:     int,
    num_training_steps:   int,
) -> LambdaLR:
    """
    Linear warm-up then cosine decay to zero.

    Args:
        optimizer:           Wrapped optimizer (typically AdamW).
        num_warmup_steps:    Ramp-up duration in optimiser steps.
        num_training_steps:  Total steps (warm-up + decay).

    Returns:
        ``LambdaLR`` scheduler; call ``scheduler.step()`` after each
        ``optimizer.step()``.
    """
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Fine-tuner
# ---------------------------------------------------------------------------

class MoodFineTuner:
    """
    Fine-tunes a ``MoodConditioningModule`` on (mood_label, token_sequence)
    pairs while keeping the backbone ``_SymbolicMusicTransformer`` frozen.

    Training objective
    ------------------
    Cross-entropy language-model loss with label smoothing.

    **"prepend" injection**::

        Input : [mood_tok] + tokens[:-1]  →  length L
        Target:              tokens        →  length L
        Loss  : cross-entropy over all L positions
                (position 0 predicts tokens[0] using only the mood token).

    **"bias" injection**::

        Input : tokens[:-1]  →  length L-1
        Target: tokens[1:]   →  length L-1
        Loss  : standard LM cross-entropy (mood vector added as bias).

    PAD tokens are excluded from the loss via ``ignore_index``.

    Args:
        config: Fine-tuning hyperparameters.
        device: Torch device.  Auto-selected (CUDA → MPS → CPU) if ``None``.
    """

    def __init__(
        self,
        config: FineTuningConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config  = config
        self._device = device or self._auto_device()

        # Populated during train()
        self._backbone:       Optional[_SymbolicMusicTransformer] = None
        self._adapter:        Optional[MoodConditioningModule]     = None
        self._adapter_config: Optional[MoodAdapterConfig]         = None

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Backbone loading + freezing
    # ------------------------------------------------------------------

    def _load_backbone(self) -> Tuple[_SymbolicMusicTransformer, int]:
        """
        Load and return the frozen backbone and its d_model.

        Returns:
            ``(backbone, d_model)``

        Raises:
            FileNotFoundError: Backbone checkpoint missing.
        """
        path = Path(self.config.checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Backbone checkpoint not found: {path}\n"
                "Train the backbone first (scripts/train_transformer.py) or "
                "download a compatible checkpoint."
            )

        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            state    = raw["model_state_dict"]
            ckpt_cfg = raw.get("config", {})
        else:
            state    = raw
            ckpt_cfg = {}

        detected = _auto_detect_arch(state)
        arch     = {**detected, **{k: ckpt_cfg[k] for k in ckpt_cfg if k in detected}}

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
        model.to(self._device)
        return model, arch["d_model"]

    def freeze_backbone(self, model: nn.Module) -> None:
        """
        Disable gradients for all backbone parameters.

        Calling this guarantees that only the adapter is updated during
        back-propagation.

        Args:
            model: The ``_SymbolicMusicTransformer`` instance.
        """
        model.requires_grad_(False)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Backbone frozen: %d parameters (no gradients).", n_params)

    def _build_adapter(self, d_model: int) -> MoodConditioningModule:
        """Construct and initialise a new mood adapter."""
        adapter = MoodConditioningModule(
            num_moods=len(self.config.mood_names),
            d_model=d_model,
        )
        adapter.to(self._device)
        n_params = sum(p.numel() for p in adapter.parameters())
        logger.info("Mood adapter: %d trainable parameters.", n_params)
        return adapter

    def _build_projection_head(
        self,
        backbone: _SymbolicMusicTransformer,
        d_model: int,
        vocab_size: int,
    ) -> nn.Linear:
        """
        Create an untied projection head initialised from the backbone's weights.

        The backbone ties ``output_projection.weight`` to ``token_embedding.weight``
        so both are frozen together.  This method copies those weights into a
        standalone ``nn.Linear`` that can be trained independently while the
        backbone remains completely frozen.

        Args:
            backbone:   Frozen backbone (source of initial weights).
            d_model:    Input feature dimension.
            vocab_size: Output vocabulary size (= backbone vocab size).

        Returns:
            Initialised ``nn.Linear(d_model, vocab_size, bias=True)`` on the
            training device with gradients enabled.
        """
        proj = nn.Linear(d_model, vocab_size, bias=True)
        with torch.no_grad():
            proj.weight.copy_(backbone.output_projection.weight.detach())
            proj.bias.copy_(backbone.output_projection.bias.detach())
        proj.to(self._device)
        n_params = sum(p.numel() for p in proj.parameters())
        logger.info(
            "Projection head (untied): %d trainable parameters.", n_params
        )
        return proj

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        backbone:        _SymbolicMusicTransformer,
        adapter:         MoodConditioningModule,
        loader:          DataLoader,
        optimizer:       AdamW,
        scheduler:       LambdaLR,
        projection_head: Optional[nn.Linear] = None,
    ) -> float:
        """
        One full pass over the training data.

        Args:
            backbone:        Frozen backbone (kept in eval mode throughout).
            adapter:         Trainable mood module (set to train mode here).
            loader:          Training ``DataLoader``.
            optimizer:       AdamW instance.
            scheduler:       LR scheduler; stepped after every batch.
            projection_head: Optional untied projection head to fine-tune
                             alongside the mood embedding.  When provided,
                             ``backbone.forward_hidden()`` is used instead
                             of ``backbone()`` and the logits are produced
                             by this head.

        Returns:
            Token-normalised mean cross-entropy loss for the epoch.
        """
        backbone.eval()
        adapter.train()
        if projection_head is not None:
            projection_head.train()

        vocab     = get_vocabulary()
        pad_id    = vocab.pad_id
        injection = self.config.injection_method

        total_loss   = 0.0
        total_tokens = 0

        for mood_labels, token_seqs in loader:
            mood_labels = mood_labels.to(self._device)    # (B,)
            token_seqs  = token_seqs.to(self._device)     # (B, L)

            mood_vec = adapter(mood_labels)               # (B, d_model)

            if injection == "prepend":
                # input_ids = tokens[:-1]  → (B, L-1)
                # with prepend, logits  → (B, L, V)
                # targets   = tokens    → (B, L)
                # Position 0 of logits predicts tokens[:, 0] from mood only.
                input_ids = token_seqs[:, :-1]
                targets   = token_seqs
            else:  # "bias"
                # Standard LM shift — mood is a bias on embeddings.
                input_ids = token_seqs[:, :-1]   # (B, L-1)
                targets   = token_seqs[:, 1:]    # (B, L-1)

            # Use the standalone projection head when available so the
            # backbone's weight-tied projection is never modified.
            if projection_head is not None:
                hidden = backbone.forward_hidden(input_ids, mood_vec, injection)
                logits = projection_head(hidden)          # (B, *, V)
            else:
                logits = backbone(input_ids, mood_vec, injection)   # (B, *, V)

            B, L, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * L, V),
                targets.reshape(B * L),
                ignore_index=pad_id,
                label_smoothing=self.config.label_smoothing,
            )

            n_tokens = int((targets != pad_id).sum().item())

            optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip > 0:
                # Clip gradients for all trainable modules together.
                params_to_clip = list(adapter.parameters())
                if projection_head is not None:
                    params_to_clip += list(projection_head.parameters())
                nn.utils.clip_grad_norm_(params_to_clip, self.config.grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

        return total_loss / max(total_tokens, 1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(
        self,
        backbone:        _SymbolicMusicTransformer,
        adapter:         MoodConditioningModule,
        loader:          DataLoader,
        projection_head: Optional[nn.Linear] = None,
    ) -> float:
        """
        Compute validation loss.

        Args:
            backbone:        Frozen backbone.
            adapter:         Current adapter weights.
            loader:          Validation ``DataLoader``.
            projection_head: Optional untied projection head (mirrors the
                             argument in ``train_epoch``).

        Returns:
            Token-normalised mean cross-entropy loss.
        """
        backbone.eval()
        adapter.eval()
        if projection_head is not None:
            projection_head.eval()

        vocab     = get_vocabulary()
        pad_id    = vocab.pad_id
        injection = self.config.injection_method

        total_loss   = 0.0
        total_tokens = 0

        for mood_labels, token_seqs in loader:
            mood_labels = mood_labels.to(self._device)
            token_seqs  = token_seqs.to(self._device)

            mood_vec = adapter(mood_labels)

            if injection == "prepend":
                input_ids = token_seqs[:, :-1]
                targets   = token_seqs
            else:
                input_ids = token_seqs[:, :-1]
                targets   = token_seqs[:, 1:]

            if projection_head is not None:
                hidden = backbone.forward_hidden(input_ids, mood_vec, injection)
                logits = projection_head(hidden)
            else:
                logits = backbone(input_ids, mood_vec, injection)

            B, L, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * L, V),
                targets.reshape(B * L),
                ignore_index=pad_id,
                label_smoothing=self.config.label_smoothing,
            )
            n_tokens = int((targets != pad_id).sum().item())

            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

        return total_loss / max(total_tokens, 1)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, dataset: MoodSequenceDataset) -> MoodConditioningModule:
        """
        Full fine-tuning loop with early stopping.

        Loads the backbone, freezes it, builds a fresh adapter, trains for
        up to ``config.num_epochs`` epochs, and saves the best adapter
        checkpoint to ``config.adapter_save_path``.

        Args:
            dataset: Pre-built ``MoodSequenceDataset`` (train + val pooled).

        Returns:
            The best ``MoodConditioningModule`` (also saved to disk).
        """
        cfg = self.config

        # ---- Backbone ----
        logger.info("Loading backbone from '%s'.", cfg.checkpoint_path)
        backbone, d_model = self._load_backbone()
        self.freeze_backbone(backbone)
        backbone.eval()

        # ---- Adapter ----
        adapter = self._build_adapter(d_model)

        # ---- Optional projection head (untied copy of backbone's head) ----
        projection_head: Optional[nn.Linear] = None
        if cfg.finetune_projection:
            vocab_size      = backbone.output_projection.weight.shape[0]
            projection_head = self._build_projection_head(backbone, d_model, vocab_size)

        # ---- Train / val split ----
        val_size   = max(1, int(len(dataset) * cfg.val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        logger.info(
            "Dataset split: %d train / %d val.", train_size, val_size
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=MoodSequenceDataset.collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=MoodSequenceDataset.collate_fn,
            num_workers=0,
        )

        # ---- Optimiser + schedule ----
        # Collect all trainable parameters: always the adapter, optionally
        # the standalone projection head.
        trainable_params = list(adapter.parameters())
        if projection_head is not None:
            trainable_params += list(projection_head.parameters())

        optimizer = AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        steps_per_epoch = max(1, len(train_loader))
        total_steps     = cfg.num_epochs * steps_per_epoch
        scheduler       = _cosine_schedule_with_warmup(
            optimizer, cfg.warmup_steps, total_steps
        )

        # ---- Epoch loop ----
        best_val_loss    = float("inf")
        epochs_no_improv = 0

        for epoch in range(1, cfg.num_epochs + 1):
            train_loss = self.train_epoch(
                backbone, adapter, train_loader, optimizer, scheduler,
                projection_head,
            )

            if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
                val_loss = self.validate(backbone, adapter, val_loader, projection_head)
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch, cfg.num_epochs, train_loss, val_loss,
                )

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    epochs_no_improv = 0
                    self.save_adapter(adapter, d_model, epoch, best_val_loss, projection_head)
                    logger.info("  → New best — adapter saved.")
                else:
                    epochs_no_improv += cfg.eval_interval
                    if epochs_no_improv >= cfg.patience:
                        logger.info(
                            "Early stopping at epoch %d (no improvement for %d epochs).",
                            epoch, epochs_no_improv,
                        )
                        break
            else:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f",
                    epoch, cfg.num_epochs, train_loss,
                )

        return adapter

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_adapter(
        self,
        adapter:         MoodConditioningModule,
        d_model:         int,
        epoch:           Optional[int]    = None,
        best_val_loss:   Optional[float]  = None,
        projection_head: Optional[nn.Linear] = None,
    ) -> None:
        """
        Save adapter weights (and optional projection head) to disk.

        The adapter (a single ``nn.Embedding``) is always saved.  When a
        ``projection_head`` is provided its state-dict and vocab size are
        also embedded in the checkpoint so ``load_adapter`` can restore it
        without any extra arguments.

        Args:
            adapter:         Trained ``MoodConditioningModule``.
            d_model:         Backbone hidden dimension (stored in config).
            epoch:           Current epoch number (metadata only).
            best_val_loss:   Best validation loss seen so far (metadata only).
            projection_head: Optional fine-tuned projection head to save.
        """
        save_path = Path(self.config.adapter_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "adapter_state_dict": adapter.state_dict(),
            "config": {
                "num_moods":          len(self.config.mood_names),
                "d_model":            d_model,
                "injection_method":   self.config.injection_method,
                "mood_names":         list(self.config.mood_names),
                "finetune_projection": projection_head is not None,
            },
            "epoch":         epoch,
            "best_val_loss": best_val_loss,
            # Projection head — None when finetune_projection=False.
            "projection_head_state_dict": (
                projection_head.state_dict() if projection_head is not None else None
            ),
            "projection_head_vocab_size": (
                projection_head.out_features if projection_head is not None else None
            ),
        }
        torch.save(checkpoint, save_path)
        logger.info(
            "Adapter checkpoint → '%s' (projection_head=%s).",
            save_path,
            projection_head is not None,
        )

    @staticmethod
    def load_adapter(
        path:   Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> Tuple[MoodConditioningModule, MoodAdapterConfig, Optional[nn.Linear]]:
        """
        Load a saved adapter (and optional projection head) from disk.

        Args:
            path:   Path to the adapter ``.pt`` checkpoint.
            device: Target device.  Auto-selected if ``None``.

        Returns:
            ``(adapter, config, projection_head)`` where ``projection_head``
            is an ``nn.Linear`` ready for inference, or ``None`` when the
            checkpoint was saved without ``finetune_projection=True``.

        Raises:
            FileNotFoundError: File does not exist.
            RuntimeError:      Unrecognised checkpoint format.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter checkpoint not found: {path}")

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        raw = torch.load(path, map_location="cpu", weights_only=False)

        if "adapter_state_dict" not in raw or "config" not in raw:
            raise RuntimeError(
                f"Unrecognised adapter format in '{path}'. "
                "Expected {'adapter_state_dict': ..., 'config': {...}}."
            )

        cfg_dict = raw["config"]
        config   = MoodAdapterConfig(
            num_moods=cfg_dict["num_moods"],
            d_model=cfg_dict["d_model"],
            injection_method=cfg_dict["injection_method"],
            mood_names=list(cfg_dict["mood_names"]),
            finetune_projection=cfg_dict.get("finetune_projection", False),
        )

        adapter = MoodConditioningModule(config.num_moods, config.d_model)
        adapter.load_state_dict(raw["adapter_state_dict"])
        adapter.to(device)
        adapter.eval()

        # Restore the projection head when it was saved with the adapter.
        projection_head: Optional[nn.Linear] = None
        if raw.get("projection_head_state_dict") is not None:
            vocab_size      = raw["projection_head_vocab_size"]
            projection_head = nn.Linear(config.d_model, vocab_size, bias=True)
            projection_head.load_state_dict(raw["projection_head_state_dict"])
            projection_head.to(device)
            projection_head.eval()

        logger.info(
            "Mood adapter loaded from '%s' | moods=%d d_model=%d "
            "injection=%s projection_head=%s",
            path,
            config.num_moods,
            config.d_model,
            config.injection_method,
            projection_head is not None,
        )
        return adapter, config, projection_head
