"""
Lightweight mood classifier for generated music alignment scoring.

Architecture
------------
``MoodClassifierHead`` is a two-layer MLP trained on top of frozen backbone
embeddings.  Given a generated token sequence it:

1. Runs the sequence through the backbone ``forward_hidden()`` (no grad).
2. Mean-pools the hidden states to a single ``(d_model,)`` vector.
3. Classifies with the MLP → softmax probabilities over mood classes.

Only the MLP weights are ever trained — the backbone remains frozen, so the
classifier needs ≪1 % of the backbone parameter count.

``MoodAlignmentScorer`` wraps the classifier for runtime scoring and exposes:
- ``score(token_ids, target_mood) → float``  — probability for target mood.
- ``top_moods(token_ids, k)``                — top-k predicted moods.
- ``save(path)`` / ``MoodAlignmentScorer.load(path, backbone, device)``

Checkpoint format
-----------------
::

    {
        "classifier_state_dict": {...},
        "config": {
            "d_model":    int,
            "hidden_dim": int,
            "num_moods":  int,
            "mood_names": List[str],
            "dropout":    float,
        },
    }

The backbone weights are NOT stored here — they live in the separate pretrained
backbone checkpoint.  Only the small MLP head is saved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    # Avoid circular import at runtime; only used for type annotations.
    from app.generators.pretrained_transformer import _SymbolicMusicTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical mood list — must stay in sync with pretrained_transformer._VALID_MOODS
# ---------------------------------------------------------------------------

_VALID_MOODS: Tuple[str, ...] = (
    "melancholic", "dreamy",    "energetic", "tense",   "happy",
    "sad",         "calm",      "dark",      "joyful",  "uplifting",
    "intense",     "peaceful",  "dramatic",  "epic",    "mysterious",
    "romantic",    "neutral",   "flowing",   "ominous",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClassifierConfig:
    """
    Hyperparameters for ``MoodClassifierHead``.

    Stored alongside the model weights so the head can be reconstructed
    without any external config file.

    Attributes:
        d_model:    Input dimension — must match the backbone's hidden dim.
        hidden_dim: Intermediate MLP dimension (typically ``d_model // 2``).
        num_moods:  Number of output mood classes.
        mood_names: Ordered list of mood labels; index == class ID.
        dropout:    Dropout probability between MLP layers (0.0 = off).
    """
    d_model:    int
    hidden_dim: int
    num_moods:  int
    mood_names: List[str]
    dropout:    float = 0.1


# ---------------------------------------------------------------------------
# Classifier head
# ---------------------------------------------------------------------------

class MoodClassifierHead(nn.Module):
    """
    Two-layer MLP classifier on backbone sequence embeddings.

    Input:  mean-pooled hidden states ``(batch, d_model)``.
    Output: unnormalised logits ``(batch, num_moods)``.

    The architecture is intentionally minimal so it trains in seconds even
    on CPU with a dataset of a few hundred examples.

    Args:
        config: ``ClassifierConfig`` instance defining all hyperparameters.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_moods),
        )
        # Initialise with small weights — prevents early saturation.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: ``(batch, d_model)`` pooled sequence embeddings.

        Returns:
            Logits ``(batch, num_moods)`` (unnormalised).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Alignment scorer
# ---------------------------------------------------------------------------

class MoodAlignmentScorer:
    """
    Scores a generated token sequence for mood alignment.

    Pipeline::

        token_ids
            → backbone.forward_hidden()   # (1, L, d_model)  [no grad]
            → mean-pool                   # (1, d_model)
            → MoodClassifierHead          # (1, num_moods) logits
            → softmax                     # probabilities
            → prob[target_label]          # alignment score ∈ [0, 1]

    The backbone is never modified — ``torch.no_grad()`` is used for all
    embedding extraction.  Thread safety is the caller's responsibility
    (the generator's ``RLock`` covers all scoring calls).

    Args:
        backbone: Loaded ``_SymbolicMusicTransformer`` in eval mode.
        head:     Trained ``MoodClassifierHead`` in eval mode.
        config:   Classifier configuration (for mood resolution).
        device:   Torch device (must match backbone and head).
    """

    def __init__(
        self,
        backbone: "_SymbolicMusicTransformer",
        head:     MoodClassifierHead,
        config:   ClassifierConfig,
        device:   torch.device,
    ) -> None:
        self._backbone = backbone
        self._head     = head
        self._config   = config
        self._device   = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, token_ids: List[int], target_mood: str) -> float:
        """
        Compute mood-alignment probability for a generated sequence.

        The backbone is run **without** any mood conditioning vector so the
        score reflects the sequence's intrinsic mood signal rather than
        the conditioning input's influence.

        Args:
            token_ids:   Generated token IDs, BOS already stripped (as
                         returned by ``_generate_autoregressive()``).
            target_mood: Free-text mood label (e.g. ``"melancholic"``).

        Returns:
            Softmax probability in ``[0, 1]`` assigned to the target class.
            Returns ``0.0`` for empty sequences or unknown moods.
        """
        if not token_ids:
            return 0.0

        target_label = self._resolve_mood_label(target_mood)

        with torch.no_grad():
            ids = torch.tensor(
                [token_ids], dtype=torch.long, device=self._device
            )
            # Respect backbone context window.
            ids = ids[:, -self._backbone.max_seq_len:]
            # Extract hidden states — unconditional (no mood vector) so the
            # score measures the sequence itself, not the conditioning prompt.
            hidden = self._backbone.forward_hidden(ids)  # (1, L, d_model)
            emb    = hidden.mean(dim=1)                   # (1, d_model)
            logits = self._head(emb)                      # (1, num_moods)
            probs  = F.softmax(logits, dim=-1)            # (1, num_moods)
            return float(probs[0, target_label].item())

    def top_moods(
        self,
        token_ids: List[int],
        k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k predicted moods with their probabilities.

        Useful for debugging alignment failures or auditing generation quality.

        Args:
            token_ids: Generated token IDs (BOS stripped).
            k:         Number of top moods to return.

        Returns:
            List of ``(mood_name, probability)`` tuples, best first.
        """
        if not token_ids:
            return []

        with torch.no_grad():
            ids    = torch.tensor([token_ids], dtype=torch.long, device=self._device)
            ids    = ids[:, -self._backbone.max_seq_len:]
            hidden = self._backbone.forward_hidden(ids)
            emb    = hidden.mean(dim=1)
            logits = self._head(emb)
            probs  = F.softmax(logits, dim=-1)[0]  # (num_moods,)

        k = min(k, len(self._config.mood_names))
        top_vals, top_idx = torch.topk(probs, k)
        return [
            (self._config.mood_names[i], float(p))
            for i, p in zip(top_idx.tolist(), top_vals.tolist())
        ]

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Persist the classifier head and config to ``path``.

        Only the small MLP head is saved, not the backbone.

        Checkpoint format::

            {
                "classifier_state_dict": {...},
                "config": {
                    "d_model": ..., "hidden_dim": ...,
                    "num_moods": ..., "mood_names": [...],
                    "dropout": ...,
                },
            }
        """
        torch.save(
            {
                "classifier_state_dict": self._head.state_dict(),
                "config": {
                    "d_model":    self._config.d_model,
                    "hidden_dim": self._config.hidden_dim,
                    "num_moods":  self._config.num_moods,
                    "mood_names": list(self._config.mood_names),
                    "dropout":    self._config.dropout,
                },
            },
            path,
        )
        logger.info("MoodAlignmentScorer saved → %s", path)

    @classmethod
    def load(
        cls,
        path:     Union[str, Path],
        backbone: "_SymbolicMusicTransformer",
        device:   torch.device,
    ) -> "MoodAlignmentScorer":
        """
        Load a saved classifier checkpoint and build a scorer.

        Args:
            path:     Path to a ``.pt`` file written by ``save()``.
            backbone: Loaded backbone model (eval mode, correct device).
            device:   Target device for the classifier head.

        Raises:
            RuntimeError: Checkpoint format is unrecognised.
        """
        raw = torch.load(path, map_location="cpu", weights_only=False)

        if "classifier_state_dict" not in raw or "config" not in raw:
            raise RuntimeError(
                f"Unrecognised classifier checkpoint: {path}. "
                "Expected {'classifier_state_dict': ..., 'config': {...}}."
            )

        cd = raw["config"]
        config = ClassifierConfig(
            d_model=cd["d_model"],
            hidden_dim=cd["hidden_dim"],
            num_moods=cd["num_moods"],
            mood_names=list(cd["mood_names"]),
            dropout=cd.get("dropout", 0.1),
        )

        head = MoodClassifierHead(config)
        head.load_state_dict(raw["classifier_state_dict"])
        head.to(device)
        head.eval()

        logger.info(
            "MoodAlignmentScorer loaded | moods=%d d_model=%d hidden=%d from %s",
            config.num_moods, config.d_model, config.hidden_dim, path,
        )
        return cls(backbone=backbone, head=head, config=config, device=device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_mood_label(self, mood: str) -> int:
        """
        Map a free-text mood string to a class label index.

        Resolution order:
        1. Exact match (case-insensitive).
        2. Substring/partial match.
        3. Fallback to ``"neutral"`` (or 0 if not found).
        """
        mood_names = self._config.mood_names
        mood_lower = mood.lower().strip()

        # Exact match
        try:
            return mood_names.index(mood_lower)
        except ValueError:
            pass

        # Substring match
        for i, name in enumerate(mood_names):
            if name in mood_lower or mood_lower in name:
                return i

        # Fallback
        try:
            return mood_names.index("neutral")
        except ValueError:
            return 0
