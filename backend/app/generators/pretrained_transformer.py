"""
PretrainedMusicTransformerGenerator — pretrained symbolic music continuation backend.

Loads a decoder-only transformer checkpoint trained on the REMI-style tokenization
vocabulary defined in ``app.music.tokenization``:

    [BOS] [KEY_X] [SCALE_Y] [TEMPO_Z] [SEP]
    [BAR_N] [POS_P] [PITCH_K] [DUR_D] [VEL_V] ... [BAR_END]
    [EOS]

Any checkpoint whose state-dict uses the same key layout as
``_SymbolicMusicTransformer`` can be loaded.  Hyperparameters
(vocab size, model depth, d_ff, max sequence length) are auto-detected
from the state-dict shapes so no external config file is required.

Unlike ``CustomTransformerGenerator`` (which uses teacher-forcing),
this generator uses **true autoregressive decoding**: it builds a
structured prompt from the request parameters and samples one token at
a time until the requested note count is satisfied.

Key features
------------
- Auto-detects architecture from checkpoint state-dict shapes.
- GPU (CUDA) / Apple Silicon (MPS) / CPU device selection.
- Per-request ``torch.no_grad()`` context; fully thread-safe (RLock).
- Temperature + top-k sampling; both configurable at instantiation.
- Scale-constrained PITCH sampling: out-of-scale pitches are suppressed
  with an additive -inf logit penalty before sampling.
- Optional mood conditioning via a lightweight ``MoodConditioningModule``
  adapter that can be loaded from a separate checkpoint.  Two injection
  strategies are supported: ``"prepend"`` (mood as virtual first token)
  and ``"bias"`` (mood vector added to all token embeddings).  The
  backbone weights remain frozen; only the adapter is fine-tuned.
"""

from __future__ import annotations

import logging
import random as _random
import threading
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Imported lazily at first use (avoids circular import; mood_classifier
# imports _SymbolicMusicTransformer only under TYPE_CHECKING).
from app.generators.mood_classifier import MoodAlignmentScorer, ClassifierConfig

# ---------------------------------------------------------------------------
# Mood vocabulary — must match CustomTransformerGenerator ordering so that
# label indices are consistent across both backends.
# ---------------------------------------------------------------------------

_VALID_MOODS: Tuple[str, ...] = (
    "melancholic", "dreamy",    "energetic", "tense",   "happy",
    "sad",         "calm",      "dark",      "joyful",  "uplifting",
    "intense",     "peaceful",  "dramatic",  "epic",    "mysterious",
    "romantic",    "neutral",   "flowing",   "ominous",
)

_MOOD_TO_LABEL: Dict[str, int] = {m: i for i, m in enumerate(_VALID_MOODS)}


# ---------------------------------------------------------------------------
# Mood adapter — lightweight module loaded separately from backbone
# ---------------------------------------------------------------------------

@dataclass
class MoodAdapterConfig:
    """
    Configuration for a trained ``MoodConditioningModule``.

    Stored alongside the adapter weights so the generator can reconstruct
    the module without any external config file.

    Attributes:
        num_moods:           Number of mood categories (rows in the embedding).
        d_model:             Must match the backbone's hidden dimension.
        injection_method:    ``"prepend"`` or ``"bias"`` — injection strategy.
        mood_names:          Ordered list of mood labels; index == label ID.
        finetune_projection: Whether a fine-tuned projection head was saved
                             alongside the adapter.  When ``True``, the
                             checkpoint also contains
                             ``projection_head_state_dict`` and
                             ``projection_head_vocab_size``.
    """
    num_moods:           int
    d_model:             int
    injection_method:    str        # "prepend" | "bias"
    mood_names:          List[str]  # index == label ID
    finetune_projection: bool = False


class MoodConditioningModule(nn.Module):
    """
    Learned mood embedding table.

    Maps a batch of integer mood labels → d_model–dimensional vectors
    that are injected into the backbone transformer at generation time.

    Args:
        num_moods: Number of mood categories.
        d_model:   Hidden dimension; must match the backbone.
    """

    def __init__(self, num_moods: int, d_model: int) -> None:
        super().__init__()
        self.mood_embedding = nn.Embedding(num_moods, d_model)
        nn.init.normal_(self.mood_embedding.weight, mean=0.0, std=0.02)

    def forward(self, mood_indices: Tensor) -> Tensor:
        """
        Args:
            mood_indices: ``(batch,)`` long tensor of label IDs.

        Returns:
            ``(batch, d_model)`` float tensor of mood vectors.
        """
        return self.mood_embedding(mood_indices)

from app.generators.base import (
    BaseGenerator,
    GenerationRequest,
    GenerationResult,
    NoteResult,
)
from app.music.arpeggio_generator import (
    Arpeggio,
    ArpeggioGenerator,
    KEY_OFFSETS,
    Note,
    SCALE_PATTERNS,
)
from app.music.midi_renderer import MIDIRenderer, midi_to_bytes
from app.music.tokenization import (
    PITCH_MAX,
    PITCH_MIN,
    Tokenizer,
    Vocabulary,
    get_vocabulary,
    quantize_duration,
    quantize_tempo,
    quantize_velocity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comprehensive key → semitone mapping (includes flat equivalents)
# ---------------------------------------------------------------------------

_KEY_TO_SEMITONE: Dict[str, int] = {
    "C": 0,  "C#": 1, "Db": 1,  "D": 2,  "D#": 3, "Eb": 3,
    "E": 4,  "F":  5, "F#": 6,  "Gb": 6, "G":  7, "G#": 8,
    "Ab": 8, "A":  9, "A#": 10, "Bb": 10, "B": 11,
}


# ---------------------------------------------------------------------------
# Decoder-only (GPT-style) model
# ---------------------------------------------------------------------------

class _SymbolicMusicTransformer(nn.Module):
    """
    Decoder-only symbolic music transformer.

    Uses ``nn.TransformerEncoder`` with a causal mask — mathematically
    equivalent to a decoder-only language model (GPT architecture).
    No cross-attention: the model attends only to its own previous tokens.

    State-dict key layout (used by ``_auto_detect_arch``):

        token_embedding.weight                           (vocab_size, d_model)
        pos_embedding.weight                             (max_seq_len, d_model)
        transformer_layers.layers.{i}.linear1.weight    (d_ff, d_model)
        transformer_layers.layers.{i}.norm1.weight      (d_model,)
        output_norm.weight                               (d_model,)
        output_projection.weight                         (vocab_size, d_model)
        output_projection.bias                           (vocab_size,)

    Args:
        vocab_size:  Number of tokens in the vocabulary.
        d_model:     Hidden dimension (embedding size).
        nhead:       Number of self-attention heads.
        num_layers:  Number of transformer layers.
        d_ff:        Feed-forward inner dimension (typically 4 × d_model).
        max_seq_len: Maximum sequence length (for learned positional embeddings).
        dropout:     Dropout probability (set to 0.0 during inference).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model    = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding   = nn.Embedding(max_seq_len, d_model)

        # Pre-LayerNorm (norm_first=True) transformer encoder stack used
        # as a causal decoder — standard GPT-style architecture.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # Pre-norm: more stable training
            activation="gelu",
        )

        # Wrap in TransformerEncoder; enable_nested_tensor disabled to
        # avoid shape surprises in eval mode with variable-length inputs.
        try:
            self.transformer_layers = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            # PyTorch < 1.13 doesn't have enable_nested_tensor
            self.transformer_layers = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )

        self.output_norm       = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=True)
        self.dropout           = nn.Dropout(dropout)

        # Weight tying: embedding and output projection share the same matrix.
        # This halves parameter count and often improves language model quality.
        self.output_projection.weight = self.token_embedding.weight

    def forward_hidden(
        self,
        input_ids: Tensor,
        mood_vector: Optional[Tensor] = None,
        injection_method: str = "prepend",
    ) -> Tensor:
        """
        Forward pass returning pre-projection hidden states.

        Identical to ``forward()`` but stops before ``output_projection``.
        Used by the fine-tuner when a standalone (untied) projection head
        replaces the backbone's weight-tied head, and by ``forward()``
        itself to avoid code duplication.

        Args:
            input_ids:        ``(batch, seq_len)`` long tensor of token IDs.
            mood_vector:      ``(batch, d_model)`` mood embedding, or ``None``.
            injection_method: ``"prepend"`` or ``"bias"``.

        Returns:
            Hidden states ``(batch, out_len, d_model)`` after the final
            LayerNorm but before the linear projection.
        """
        _, seq_len = input_ids.shape

        # When prepending, reserve one slot so total length ≤ max_seq_len.
        reserve = 1 if (mood_vector is not None and injection_method == "prepend") else 0
        seq_len = min(seq_len, self.max_seq_len - reserve)
        ids     = input_ids[:, :seq_len]

        positions = torch.arange(seq_len, device=ids.device).unsqueeze(0)  # (1, L)
        x = self.token_embedding(ids) + self.pos_embedding(positions)      # (B, L, d_model)

        # Bias injection: shift all token embeddings by the mood vector.
        if mood_vector is not None and injection_method == "bias":
            x = x + mood_vector.unsqueeze(1)   # (B, 1, d_model) → broadcasts to (B, L, d_model)

        x = self.dropout(x)

        # Prepend injection: insert mood as a virtual first "token" with no
        # positional embedding (mood is position-invariant context).
        if mood_vector is not None and injection_method == "prepend":
            mood_tok = mood_vector.unsqueeze(1)          # (B, 1, d_model)
            x        = torch.cat([mood_tok, x], dim=1)  # (B, L+1, d_model)

        total_len = x.size(1)

        # Causal mask: upper triangle = -inf so each position attends only
        # to itself and previous positions.
        causal_mask = torch.triu(
            torch.full((total_len, total_len), float("-inf"), device=ids.device),
            diagonal=1,
        )

        x = self.transformer_layers(x, mask=causal_mask)
        return self.output_norm(x)    # (B, out_len, d_model)

    def forward(
        self,
        input_ids: Tensor,
        mood_vector: Optional[Tensor] = None,
        injection_method: str = "prepend",
    ) -> Tensor:
        """
        Full sequence forward pass with optional mood conditioning.

        Args:
            input_ids:        ``(batch, seq_len)`` long tensor of token IDs.
            mood_vector:      ``(batch, d_model)`` mood embedding from
                              ``MoodConditioningModule``, or ``None`` for
                              unconditional generation.
            injection_method: How to inject the mood vector:

                              ``"prepend"`` — prepend mood as a virtual
                              token at position 0.  The output has length
                              ``seq_len + 1``; position 0 predicts the
                              first real token.

                              ``"bias"`` — add mood vector to all token
                              embeddings.  Output length equals ``seq_len``.

        Returns:
            Logits ``(batch, out_len, vocab_size)`` where ``out_len`` is
            ``seq_len + 1`` for ``"prepend"`` and ``seq_len`` for ``"bias"``.
        """
        return self.output_projection(
            self.forward_hidden(input_ids, mood_vector, injection_method)
        )


# ---------------------------------------------------------------------------
# Architecture auto-detection
# ---------------------------------------------------------------------------

def _detect_nhead(d_model: int) -> int:
    """
    Infer a valid number of attention heads from d_model.

    Prefers head dimension 64 (common in production models); falls back
    through standard divisors.

    Args:
        d_model: Model hidden dimension.

    Returns:
        Number of attention heads.
    """
    # Target head dimension of 64
    if d_model % 64 == 0 and d_model // 64 >= 1:
        return d_model // 64
    for n in (8, 12, 6, 4, 16, 2, 1):
        if d_model % n == 0:
            return n
    return 1


def _auto_detect_arch(state: Dict[str, Tensor]) -> Dict:
    """
    Infer ``_SymbolicMusicTransformer`` hyperparameters from state-dict shapes.

    Reads the following keys:
        ``token_embedding.weight``                     → (vocab_size, d_model)
        ``pos_embedding.weight``                       → (max_seq_len, d_model)
        ``transformer_layers.layers.0.linear1.weight`` → (d_ff, d_model)
        ``transformer_layers.layers.{i}.*``            → counts num_layers

    Args:
        state: Model state dictionary (CPU tensors).

    Returns:
        Dict with keys: vocab_size, d_model, nhead, num_layers, d_ff, max_seq_len.

    Raises:
        KeyError: If ``token_embedding.weight`` is missing.
    """
    vocab_size = state["token_embedding.weight"].shape[0]
    d_model    = state["token_embedding.weight"].shape[1]

    # Max sequence length from positional embedding table
    max_seq_len = 1024
    if "pos_embedding.weight" in state:
        max_seq_len = state["pos_embedding.weight"].shape[0]

    # Count transformer layers from unique layer indices
    layer_indices: Set[int] = set()
    for k in state:
        if k.startswith("transformer_layers.layers."):
            parts = k.split(".")
            # Key format: transformer_layers . layers . {i} . ...
            #             [0]                 [1]      [2]   [3+]
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))
    num_layers = max(layer_indices) + 1 if layer_indices else 4

    # Feed-forward dimension from first layer's linear1
    d_ff    = d_model * 4
    ff_key  = "transformer_layers.layers.0.linear1.weight"
    if ff_key in state:
        d_ff = state[ff_key].shape[0]

    # Attention heads: heuristic (prefer head_dim = 64)
    nhead = _detect_nhead(d_model)

    return {
        "vocab_size": vocab_size,
        "d_model":    d_model,
        "nhead":      nhead,
        "num_layers": num_layers,
        "d_ff":       d_ff,
        "max_seq_len": max_seq_len,
    }


# ---------------------------------------------------------------------------
# Scale-constrained PITCH token mask
# ---------------------------------------------------------------------------

def _build_scale_pitch_mask(
    vocab: Vocabulary,
    key: str,
    scale: str,
    device: torch.device,
) -> Tensor:
    """
    Build an additive logit mask that suppresses out-of-scale PITCH tokens.

    The returned tensor has:
        - ``-inf`` for PITCH tokens whose pitch is NOT in the requested scale.
        - ``0.0``  for everything else (in-scale pitches + all non-PITCH tokens).

    Adding this to raw logits before softmax zeroes out-of-scale probabilities
    without altering the relative ordering of in-scale and non-PITCH tokens.

    Args:
        vocab:  Vocabulary instance for token ID lookup.
        key:    Musical key (e.g. ``"C"``, ``"F#"``).
        scale:  Scale name (e.g. ``"major"``, ``"dorian"``).
        device: Target tensor device.

    Returns:
        Float tensor of shape ``(vocab_size,)``.
    """
    root_semitone   = _KEY_TO_SEMITONE.get(key, 0)
    intervals       = SCALE_PATTERNS.get(scale, SCALE_PATTERNS["major"])
    valid_semitones: Set[int] = {(root_semitone + iv) % 12 for iv in intervals}

    mask = torch.zeros(vocab.size, device=device)

    for pitch in range(PITCH_MIN, PITCH_MAX + 1):
        if pitch % 12 in valid_semitones:
            continue                              # In scale — no penalty
        token_str = f"PITCH_{pitch}"
        tid = vocab.encode(token_str)
        # Only block if this is a real PITCH token (not UNK collision)
        if vocab.decode(tid) == token_str:
            mask[tid] = float("-inf")

    return mask


# ---------------------------------------------------------------------------
# Token-stream generation state machine
# ---------------------------------------------------------------------------

class _NextExpected(Enum):
    """Expected next token type during autoregressive generation."""
    ANY      = auto()   # Header region — BOS, KEY, SCALE, TEMPO, SEP
    BAR_OPEN = auto()   # After SEP: expecting BAR_N or (if done) EOS
    IN_BAR   = auto()   # After BAR_N opened: expecting POS or BAR_END
    PITCH    = auto()   # After POS: expecting PITCH
    DURATION = auto()   # After PITCH: expecting DUR
    VELOCITY = auto()   # After DUR: expecting VEL


@dataclass
class _GenState:
    """
    Mutable state for the generation token-stream parser.

    Tracks how many complete notes have been decoded and what token
    type the model should produce next.  Used to:
        1. Apply the scale-pitch mask at the right moment.
        2. Count completed notes to know when to stop generating.
    """
    notes_decoded: int   = 0
    current_bar:   int   = 0
    next_expected: _NextExpected = _NextExpected.ANY
    pending_pos:   Optional[int]   = None   # 0-15: 16th-note grid position
    pending_pitch: Optional[int]   = None   # MIDI pitch 36-96
    pending_dur:   Optional[float] = None   # duration in beats
    saw_sep:       bool  = False            # True once <SEP> seen


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class PretrainedMusicTransformerGenerator(BaseGenerator):
    """
    Arpeggio generator backed by a pretrained decoder-only symbolic music
    transformer.

    Autoregressive generation pipeline
    -----------------------------------
    1. Build a structured prompt::

           [BOS][KEY_X][SCALE_Y][TEMPO_Z][SEP][BAR_0]
           [POS_0][PITCH_root][DUR_0.5][VEL_80]

       The single seed note anchors the key and octave so the model
       starts in the right musical context.

    2. Generate tokens one at a time (temperature + top-k sampling).
       When expecting a PITCH token, apply the scale-pitch mask to
       suppress out-of-scale logits before sampling.

    3. Stop when ``note_count`` complete notes have been decoded, EOS is
       generated, or ``max_gen_tokens`` is exhausted.

    4. Decode the full token stream to ``Note`` objects via
       ``Tokenizer.detokenize_ids()``.

    5. Pad with rule-based arpeggio notes if fewer than ``note_count``
       notes were decoded.

    6. Render to a MIDI file via ``MIDIRenderer``.

    Args:
        checkpoint_path: Path to ``pretrained_music_transformer.pt``.
                         Expected format::

                             {"model_state_dict": {...}, "config": {...}}

                         A bare state-dict is also accepted.
        temperature:     Sampling temperature (0.5–1.5 recommended).
                         Lower = more conservative.  Must be > 0.
        top_k:           Top-k filtering — keep only the k most likely
                         tokens before sampling.  ``0`` disables filtering
                         (pure temperature sampling).
        max_gen_tokens:  Hard upper bound on generated tokens per call,
                         independent of ``note_count``.  The effective
                         budget is ``max(max_gen_tokens, note_count * 8)``
                         to accommodate longer sequences automatically.
        mood_adapter_path: Optional path to a ``mood_adapter.pt`` checkpoint
                           produced by ``MoodFineTuner``.  When provided and
                           the file exists, mood conditioning is active.
                           If the file is absent, generation falls back to
                           unconditional mode with a warning.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        temperature: float = 0.95,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_gen_tokens: int = 1024,
        mood_adapter_path: Optional[Union[str, Path]] = None,
        classifier_path: Optional[Union[str, Path]] = None,
        alignment_threshold: float = 0.0,
        alignment_max_attempts: int = 3,
    ) -> None:
        self._checkpoint_path  = Path(checkpoint_path)
        self.temperature       = float(temperature)
        self.top_k             = int(top_k)
        self.top_p             = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.max_gen_tokens    = int(max_gen_tokens)
        self._mood_adapter_path = (
            Path(mood_adapter_path) if mood_adapter_path is not None else None
        )
        # Alignment scorer config
        self._classifier_path       = Path(classifier_path) if classifier_path else None
        self._alignment_threshold   = float(alignment_threshold)
        self._alignment_max_attempts = max(1, int(alignment_max_attempts))

        self._model:  Optional[_SymbolicMusicTransformer]  = None
        self._device: Optional[torch.device]               = None
        self._vocab:  Vocabulary                           = get_vocabulary()
        self._tokenizer: Tokenizer                         = Tokenizer(self._vocab)
        self._lock    = threading.RLock()
        self._loaded: bool = False

        # Mood adapter — populated by _load_mood_adapter() if path is given.
        self._mood_adapter:        Optional[MoodConditioningModule] = None
        self._mood_adapter_config: Optional[MoodAdapterConfig]      = None
        # Optional fine-tuned projection head (untied from backbone embedding).
        # Loaded from the adapter checkpoint when finetune_projection=True.
        self._projection_head:     Optional[nn.Linear]              = None
        # Alignment scorer — populated by _load_mood_classifier() if path given.
        self._alignment_scorer:    Optional[MoodAlignmentScorer]    = None

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "pretrained-music-transformer"

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """
        Load the pretrained checkpoint.

        Auto-detects architecture from state-dict shapes.  If the
        checkpoint stores a ``"config"`` dict, its values take precedence
        over the auto-detected ones.

        Raises:
            FileNotFoundError: Checkpoint file does not exist.
            RuntimeError:      State-dict is incompatible with the
                               auto-detected architecture.
            KeyError:          State-dict is missing required keys.
        """
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {self._checkpoint_path}\n"
                "Train a model with scripts/train_transformer.py and save it "
                "to that path, or download a compatible checkpoint."
            )

        with self._lock:
            # ---- Device selection: CUDA → MPS → CPU ----
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

            # ---- Load raw checkpoint ----
            raw = torch.load(
                self._checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

            if isinstance(raw, dict) and "model_state_dict" in raw:
                state: Dict[str, Tensor] = raw["model_state_dict"]
                ckpt_cfg: Dict           = raw.get("config", {})
            elif isinstance(raw, dict) and all(
                isinstance(v, Tensor) for v in raw.values()
            ):
                # Bare state-dict (no wrapper dict)
                state    = raw
                ckpt_cfg = {}
            else:
                raise RuntimeError(
                    "Unrecognised checkpoint format. Expected a state-dict or "
                    "{'model_state_dict': ..., 'config': {...}}."
                )

            # ---- Architecture resolution ----
            detected = _auto_detect_arch(state)
            # Stored config overrides auto-detected values where keys overlap
            arch = {**detected, **{k: ckpt_cfg[k] for k in ckpt_cfg if k in detected}}

            our_vocab = self._vocab.size
            if arch["vocab_size"] != our_vocab:
                logger.warning(
                    "Checkpoint vocab_size=%d ≠ tokenizer vocab_size=%d. "
                    "Token IDs will not align — generation quality may be poor. "
                    "Ensure the checkpoint was trained with the same tokenizer.",
                    arch["vocab_size"],
                    our_vocab,
                )

            # ---- Build model and load weights ----
            self._model = _SymbolicMusicTransformer(
                vocab_size=arch["vocab_size"],
                d_model=arch["d_model"],
                nhead=arch["nhead"],
                num_layers=arch["num_layers"],
                d_ff=arch["d_ff"],
                max_seq_len=arch["max_seq_len"],
                dropout=0.0,   # Eval mode — no stochastic dropout
            )
            self._model.load_state_dict(state, strict=True)
            self._model.to(self._device)
            self._model.eval()

            self._loaded = True
            logger.info(
                "PretrainedMusicTransformerGenerator loaded | "
                "vocab=%d d_model=%d nhead=%d layers=%d d_ff=%d "
                "max_len=%d device=%s temperature=%.2f top_k=%d",
                arch["vocab_size"],
                arch["d_model"],
                arch["nhead"],
                arch["num_layers"],
                arch["d_ff"],
                arch["max_seq_len"],
                self._device,
                self.temperature,
                self.top_k,
            )

            # ---- Mood adapter (optional) ----
            if self._mood_adapter_path is not None:
                self._load_mood_adapter()

            # ---- Alignment classifier (optional) ----
            if self._classifier_path is not None:
                self._load_mood_classifier()

    # ------------------------------------------------------------------
    # Mood adapter loading
    # ------------------------------------------------------------------

    def _load_mood_adapter(self) -> None:
        """
        Load a fine-tuned ``MoodConditioningModule`` from disk.

        Called automatically by ``load()`` when ``mood_adapter_path`` is set.
        Silently falls back to unconditional generation if the file is absent.

        Raises:
            RuntimeError: If the checkpoint format is unrecognised.
        """
        assert self._mood_adapter_path is not None
        assert self._device is not None

        if not self._mood_adapter_path.exists():
            logger.warning(
                "Mood adapter checkpoint not found: %s — "
                "mood conditioning disabled (unconditional generation active).",
                self._mood_adapter_path,
            )
            return

        raw = torch.load(
            self._mood_adapter_path,
            map_location="cpu",
            weights_only=False,
        )

        if "adapter_state_dict" not in raw or "config" not in raw:
            raise RuntimeError(
                f"Unrecognised mood adapter format in {self._mood_adapter_path}. "
                "Expected {'adapter_state_dict': ..., 'config': {...}}."
            )

        cfg_dict = raw["config"]
        config = MoodAdapterConfig(
            num_moods=cfg_dict["num_moods"],
            d_model=cfg_dict["d_model"],
            injection_method=cfg_dict["injection_method"],
            mood_names=list(cfg_dict["mood_names"]),
            finetune_projection=cfg_dict.get("finetune_projection", False),
        )

        adapter = MoodConditioningModule(config.num_moods, config.d_model)
        adapter.load_state_dict(raw["adapter_state_dict"])
        adapter.to(self._device)
        adapter.eval()

        self._mood_adapter        = adapter
        self._mood_adapter_config = config

        # Restore fine-tuned projection head when the checkpoint includes one.
        if raw.get("projection_head_state_dict") is not None:
            vocab_size = raw["projection_head_vocab_size"]
            proj = nn.Linear(config.d_model, vocab_size, bias=True)
            proj.load_state_dict(raw["projection_head_state_dict"])
            proj.to(self._device)
            proj.eval()
            self._projection_head = proj
            logger.info(
                "Fine-tuned projection head loaded | vocab=%d", vocab_size
            )

        logger.info(
            "Mood adapter loaded | moods=%d d_model=%d injection=%s projection=%s",
            config.num_moods,
            config.d_model,
            config.injection_method,
            config.finetune_projection,
        )

    # ------------------------------------------------------------------
    # Alignment classifier loading
    # ------------------------------------------------------------------

    def _load_mood_classifier(self) -> None:
        """
        Load a ``MoodAlignmentScorer`` from ``self._classifier_path``.

        Called automatically by ``load()`` when ``classifier_path`` is set.
        Falls back to disabled scoring (``self._alignment_scorer = None``)
        if the file is absent, rather than raising an error, to allow the
        service to start even while a classifier is being trained.

        Raises:
            RuntimeError: Checkpoint format is unrecognised.
        """
        assert self._classifier_path is not None
        assert self._model is not None
        assert self._device is not None

        if not self._classifier_path.exists():
            logger.warning(
                "Mood classifier checkpoint not found: %s — "
                "alignment scoring disabled.",
                self._classifier_path,
            )
            return

        self._alignment_scorer = MoodAlignmentScorer.load(
            self._classifier_path,
            backbone=self._model,
            device=self._device,
        )
        logger.info(
            "MoodAlignmentScorer ready | threshold=%.3f  max_attempts=%d",
            self._alignment_threshold,
            self._alignment_max_attempts,
        )

    # ------------------------------------------------------------------
    # Mood label resolution
    # ------------------------------------------------------------------

    def _resolve_mood_label(self, mood: str) -> int:
        """
        Map a free-text mood string to an integer label ID.

        Resolution order:
        1. Exact match against ``mood_names`` (case-insensitive).
        2. Substring match (mood contains a name or name contains mood).
        3. Semantic nearest-neighbour via ``sentence-transformers``
           (only if the library is installed).
        4. Fallback: ``"neutral"`` label, or 0 if not found.

        Args:
            mood: Free-text mood description from the generation request.

        Returns:
            Integer label index for the loaded adapter's embedding table.
        """
        assert self._mood_adapter_config is not None
        mood_names = self._mood_adapter_config.mood_names
        mood_lower = mood.lower().strip()

        # 1. Exact match
        if mood_lower in mood_names:
            return mood_names.index(mood_lower)

        # 2. Substring / partial match
        for i, name in enumerate(mood_names):
            if name in mood_lower or mood_lower in name:
                return i

        # 3. Semantic nearest-neighbour (requires sentence-transformers)
        try:
            from app.mood.embeddings import SENTENCE_TRANSFORMERS_AVAILABLE
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                from app.mood.embeddings import get_embedder
                embedder     = get_embedder()
                query_emb    = embedder.embed(mood_lower)
                cand_embs    = embedder.embed_batch(mood_names)
                sims         = torch.matmul(cand_embs, query_emb)
                best         = int(sims.argmax().item())
                logger.debug(
                    "Mood '%s' → semantic match '%s' (label %d)",
                    mood, mood_names[best], best,
                )
                return best
        except Exception as exc:
            logger.debug("Semantic mood resolution failed: %s", exc)

        # 4. Fallback to "neutral" (or 0)
        try:
            return mood_names.index("neutral")
        except ValueError:
            return 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Full autoregressive generation pipeline.

        Args:
            request: Validated generation parameters.

        Returns:
            ``GenerationResult`` with MIDI bytes, note events, and metadata.

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Generator '{self.name}' is not loaded. Call load() first."
            )

        # ---- Resolve per-request sampling params (override instance defaults) ----
        eff_temperature        = request.temperature        if request.temperature        is not None else self.temperature
        eff_top_k              = request.top_k              if request.top_k              is not None else self.top_k
        eff_top_p              = request.top_p              if request.top_p              is not None else self.top_p
        eff_rep_penalty        = request.repetition_penalty if request.repetition_penalty is not None else self.repetition_penalty
        eff_max_length         = request.max_length         if request.max_length         is not None else self.max_gen_tokens

        # ---- Compute mood vector (None = unconditional) ----
        mood_vector:      Optional[Tensor] = None
        injection_method: str              = "prepend"

        if self._mood_adapter is not None and self._mood_adapter_config is not None:
            label_idx    = self._resolve_mood_label(request.mood)
            label_tensor = torch.tensor(
                [label_idx], dtype=torch.long, device=self._device
            )
            with torch.no_grad():
                mood_vector = self._mood_adapter(label_tensor)   # (1, d_model)
            injection_method = self._mood_adapter_config.injection_method
            logger.debug(
                "[%s] mood='%s' → label=%d ('%s') injection=%s",
                self.name,
                request.mood,
                label_idx,
                self._mood_adapter_config.mood_names[label_idx],
                injection_method,
            )

        # 1. Build structured prompt
        prompt_ids = self._build_prompt(request)

        # 2. Autoregressive generation — with optional alignment-scoring retry loop.
        #
        # Retry semantics:
        #   - ``needs_scoring`` : scorer is loaded → compute score on every attempt.
        #   - ``needs_retry``   : scorer loaded AND threshold > 0 → re-generate on
        #                         low-scoring attempts, up to alignment_max_attempts.
        #   - On each retry the temperature is nudged up by 0.05 (capped at 2.0) so
        #     successive samples differ rather than repeating the same output.
        #   - The attempt with the highest alignment score is kept regardless of
        #     whether any attempt cleared the threshold.
        needs_scoring = self._alignment_scorer is not None
        needs_retry   = needs_scoring and self._alignment_threshold > 0.0
        max_attempts  = self._alignment_max_attempts if needs_retry else 1

        best_token_stream: List[int] = []
        best_score:        float     = -1.0
        alignment_score:   Optional[float] = None

        for attempt in range(max_attempts):
            # Small temperature nudge on retries promotes sample diversity.
            attempt_temp = min(eff_temperature + attempt * 0.05, 2.0)

            token_stream = self._generate_autoregressive(
                prompt_ids, request, mood_vector, injection_method,
                temperature=attempt_temp,
                top_k=eff_top_k,
                top_p=eff_top_p,
                repetition_penalty=eff_rep_penalty,
                token_budget=eff_max_length,
            )

            if needs_scoring:
                score = self._alignment_scorer.score(token_stream, request.mood)
                if score > best_score:
                    best_score        = score
                    best_token_stream = token_stream

                if needs_retry:
                    logger.debug(
                        "[%s] attempt %d/%d alignment=%.3f threshold=%.2f",
                        self.name, attempt + 1, max_attempts,
                        score, self._alignment_threshold,
                    )
                    if score >= self._alignment_threshold:
                        break  # Good enough — stop early.
            else:
                best_token_stream = token_stream
                break

        if needs_scoring:
            alignment_score = best_score
            if needs_retry:
                logger.info(
                    "[%s] Best alignment score=%.3f after %d attempt(s).",
                    self.name, best_score, attempt + 1,
                )

        # 3. Decode token stream → Note objects
        notes = self._decode_token_stream(best_token_stream)

        # 4. Fallback: pad with rule-based arpeggio if too few notes decoded
        if len(notes) < request.note_count:
            logger.info(
                "[%s] Decoded %d/%d notes; padding with rule-based arpeggio.",
                self.name,
                len(notes),
                request.note_count,
            )
            notes = self._pad_with_arpeggio(notes, request)

        # Trim to exact note_count (generation may overshoot by 1 note)
        notes = notes[: request.note_count]

        # 5. Render to MIDI (enforce scale snapping as safety net)
        midi_file  = MIDIRenderer(enforce_scale=True).render_notes(
            notes=notes,
            tempo=request.tempo,
            key=request.key,
            scale=request.scale,
        )
        midi_bytes = midi_to_bytes(midi_file)

        # 6. Build result
        note_results = [
            NoteResult(
                pitch=n.pitch,
                velocity=n.velocity,
                position=n.position,
                duration=n.duration,
            )
            for n in notes
        ]

        seconds_per_beat = 60.0 / max(request.tempo, 1)
        last             = max(notes, key=lambda n: n.position + n.duration)
        duration_seconds = round((last.position + last.duration) * seconds_per_beat, 3)

        result = GenerationResult(
            midi_bytes=midi_bytes,
            notes=note_results,
            note_count=len(note_results),
            duration_seconds=duration_seconds,
            key=request.key,
            scale=request.scale,
            tempo=request.tempo,
            mood=request.mood,
            pattern_used="autoregressive",
            sampling_params={
                "temperature":        eff_temperature,
                "top_k":              eff_top_k,
                "top_p":              eff_top_p,
                "repetition_penalty": eff_rep_penalty,
                "max_length":         max(eff_max_length, request.note_count * 8),
            },
            alignment_score=alignment_score,
        )
        self._log_generation(request, result)
        return result

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, request: GenerationRequest) -> List[int]:
        """
        Build the structured generation prompt.

        Prompt format::

            [BOS] [KEY_X] [SCALE_Y] [TEMPO_Z] [SEP]
            [BAR_0] [POS_0] [PITCH_root] [DUR_0.5] [VEL_80]

        The single seed note:
            - Anchors pitch height to the requested octave.
            - Gives the model one complete (POS, PITCH, DUR, VEL) tuple
              as context before free generation begins.

        Args:
            request: Generation parameters.

        Returns:
            List of token IDs forming the prompt (including BOS).
        """
        vocab = self._vocab
        ids: List[int] = []

        # Header tokens
        ids.append(vocab.bos_id)
        ids.append(vocab.encode(f"KEY_{request.key}"))
        ids.append(vocab.encode(f"SCALE_{request.scale}"))
        ids.append(vocab.encode(f"TEMPO_{quantize_tempo(request.tempo)}"))
        ids.append(vocab.encode("<SEP>"))

        # Open bar 0
        ids.append(vocab.encode("BAR_0"))

        # Seed note: root of the requested key at the requested octave.
        # Clamped to the tokenizer's supported pitch range (36–96).
        root_semitone = _KEY_TO_SEMITONE.get(request.key, 0)
        root_pitch    = max(
            PITCH_MIN,
            min(PITCH_MAX, (request.octave + 1) * 12 + root_semitone),
        )
        ids.append(vocab.encode("POS_0"))
        ids.append(vocab.encode(f"PITCH_{root_pitch}"))
        ids.append(vocab.encode(f"DUR_{quantize_duration(0.5)}"))
        ids.append(vocab.encode(f"VEL_{quantize_velocity(80)}"))

        return ids

    # ------------------------------------------------------------------
    # Autoregressive generation loop
    # ------------------------------------------------------------------

    def _generate_autoregressive(
        self,
        prompt_ids: List[int],
        request: GenerationRequest,
        mood_vector: Optional[Tensor] = None,
        injection_method: str = "prepend",
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        token_budget: int,
    ) -> List[int]:
        """
        Extend ``prompt_ids`` autoregressively until ``note_count`` notes
        are decoded, EOS is generated, or the token budget is exhausted.

        The scale-pitch mask is applied to logits whenever the state
        machine expects a PITCH token next.  Repetition penalty is applied
        at every step to discourage pitch repetition.

        Args:
            prompt_ids:          Structured prompt (including BOS, seed note).
            request:             Generation parameters.
            mood_vector:         ``(1, d_model)`` mood embedding, or ``None``
                                 for unconditional generation.
            injection_method:    ``"prepend"`` or ``"bias"`` — forwarded to
                                 ``_SymbolicMusicTransformer.forward()``.
            temperature:         Sampling temperature for this call.
            top_k:               Top-k filter width for this call (0 = off).
            top_p:               Nucleus threshold for this call (1.0 = off).
            repetition_penalty:  Pitch repetition penalty for this call.
            token_budget:        Maximum new tokens to generate (adaptive
                                 floor: max(budget, note_count × 8)).

        Returns:
            Full token ID list (prompt + generated), BOS stripped.
        """
        assert self._model is not None
        assert self._device is not None

        vocab   = self._vocab
        eos_id  = vocab.eos_id
        pad_id  = vocab.pad_id

        # Pre-build scale-pitch mask for this key/scale pair
        pitch_mask = _build_scale_pitch_mask(
            vocab, request.key, request.scale, self._device
        )

        # State machine — seed note is already in the prompt (counts as 1)
        state = _GenState(
            notes_decoded=1,
            current_bar=0,
            next_expected=_NextExpected.IN_BAR,
            saw_sep=True,
        )

        token_ids: List[int]   = list(prompt_ids)
        max_seq_len: int       = self._model.max_seq_len
        # Adaptive budget: guarantee enough tokens for long sequences
        effective_budget: int  = max(token_budget, request.note_count * 8)

        # When using prepend injection the forward pass is longer by 1, so
        # we read logits from position -1 of the *output* (last position).
        # This is unchanged from the unconditional case — the model always
        # returns logits for every position including the prepended mood slot.

        with torch.no_grad():
            for _ in range(effective_budget):
                if state.notes_decoded >= request.note_count:
                    break

                # Use the most recent `max_seq_len` tokens as context.
                # Reserve 1 slot for prepend so the total stays ≤ max_seq_len.
                reserve  = 1 if (mood_vector is not None and injection_method == "prepend") else 0
                context  = token_ids[-(max_seq_len - reserve):]
                src      = torch.tensor(
                    [context], dtype=torch.long, device=self._device
                )

                # Use the fine-tuned projection head when available;
                # otherwise fall back to the backbone's weight-tied head.
                if self._projection_head is not None:
                    hidden      = self._model.forward_hidden(src, mood_vector, injection_method)
                    logits      = self._projection_head(hidden)
                else:
                    logits      = self._model(src, mood_vector, injection_method)
                next_logits = logits[0, -1, :]   # always take the last position

                # Apply pitch repetition penalty before scale mask so the
                # penalty is not double-applied on top of -inf scale suppression.
                if repetition_penalty > 1.0:
                    next_logits = self._apply_repetition_penalty(
                        next_logits, token_ids, repetition_penalty
                    )

                # Apply scale constraint only when about to sample a pitch
                if state.next_expected == _NextExpected.PITCH:
                    next_logits = next_logits + pitch_mask

                next_id = self._sample(next_logits, temperature, top_k, top_p)

                if next_id in (eos_id, pad_id):
                    break

                token_ids.append(next_id)
                self._advance_state(state, next_id, vocab)

        # Strip leading BOS before returning
        if token_ids and token_ids[0] == vocab.bos_id:
            token_ids = token_ids[1:]
        return token_ids

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Repetition penalty
    # ------------------------------------------------------------------

    def _apply_repetition_penalty(
        self,
        logits: Tensor,
        generated_ids: List[int],
        penalty: float,
        lookback: int = 64,
    ) -> Tensor:
        """
        Apply a pitch-specific repetition penalty (CTRL-style).

        Only ``PITCH_*`` tokens that appear in the last ``lookback`` positions
        are penalised.  Structural tokens (BAR, POS, DUR, VEL) are left
        untouched — they repeat by design and penalising them would break
        the REMI grammar.

        Formula (same as the original CTRL paper):
            - logit > 0  →  logit / penalty   (push toward zero)
            - logit ≤ 0  →  logit × penalty   (push further from zero)

        Args:
            logits:        Raw logit tensor ``(vocab_size,)``; cloned in place.
            generated_ids: All token IDs generated so far (prompt included).
            penalty:       Penalty multiplier.  1.0 = no-op.
            lookback:      How many recent tokens to scan for repeated pitches.

        Returns:
            Logit tensor with penalty applied (new tensor, original unchanged).
        """
        if abs(penalty - 1.0) < 1e-6:
            return logits

        vocab = self._vocab
        recent_pitch_ids: Set[int] = set()
        for tid in generated_ids[-lookback:]:
            if vocab.decode(tid).startswith("PITCH_"):
                recent_pitch_ids.add(tid)

        if not recent_pitch_ids:
            return logits

        logits = logits.clone()
        for tid in recent_pitch_ids:
            if logits[tid] > 0:
                logits[tid] = logits[tid] / penalty
            else:
                logits[tid] = logits[tid] * penalty

        return logits

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(
        self,
        logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> int:
        """
        Sample the next token with temperature, top-k, and nucleus filtering.

        Applied in order:

        1. **Temperature scaling** — divides logits; lower = sharper.
        2. **Top-k filtering** — zeroes all but the k highest-logit tokens.
        3. **Softmax** — converts scaled logits to a probability distribution.
        4. **Top-p (nucleus) filtering** — keeps the minimal set of tokens
           whose cumulative probability ≥ top_p, zeroing the rest.
        5. **Multinomial sample** — draws one token from the distribution.

        Special cases:
        - ``temperature ≤ 0`` → greedy argmax (deterministic).
        - ``top_k = 0``       → no top-k filter (pure temperature / nucleus).
        - ``top_p = 1.0``     → no nucleus filter.
        - Collapse guard: if the distribution collapses to all-zero / NaN
          (e.g. the scale mask wiped out all candidates), fall back to greedy
          argmax on the *original* unmodified logits.

        Args:
            logits:      Raw unnormalised logits ``(vocab_size,)``.
            temperature: Sampling temperature.
            top_k:       Top-k filter width (0 = disabled).
            top_p:       Nucleus threshold (1.0 = disabled).

        Returns:
            Sampled token ID.
        """
        # Greedy shortcut
        if temperature <= 0.0:
            return int(logits.argmax().item())

        scaled = logits / temperature

        # --- Top-k filtering ---
        if top_k > 0:
            k           = min(top_k, scaled.size(-1))
            top_vals, _ = torch.topk(scaled, k)
            threshold   = top_vals[-1]
            scaled      = torch.where(
                scaled < threshold,
                torch.full_like(scaled, float("-inf")),
                scaled,
            )

        probs = F.softmax(scaled, dim=-1)

        # --- Top-p (nucleus) filtering ---
        # Operates in probability space, after top-k has already narrowed
        # the candidates.  Keeps the smallest prefix of the sorted
        # distribution that accounts for at least top_p of the mass.
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative_probs         = torch.cumsum(sorted_probs, dim=-1)
            # Shift by one so we always keep the token that first crosses top_p
            remove                   = (cumulative_probs - sorted_probs) > top_p
            sorted_probs[remove]     = 0.0
            probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)

        # Collapse guard
        if not torch.isfinite(probs).any() or probs.sum() < 1e-9:
            return int(logits.argmax().item())

        return int(torch.multinomial(probs, num_samples=1).item())

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _advance_state(
        self,
        state: _GenState,
        token_id: int,
        vocab: Vocabulary,
    ) -> None:
        """
        Update ``state`` in-place after the model generates ``token_id``.

        Parses the generated token and advances the expected-next-token
        state machine.  Also increments ``notes_decoded`` when a complete
        (POS → PITCH → DUR → VEL) note tuple is recognised.

        Args:
            state:    Mutable generation state (modified in place).
            token_id: Newly generated token ID.
            vocab:    Vocabulary for decoding token IDs to strings.
        """
        tok = vocab.decode(token_id)

        # ---- Header region (before <SEP>) ----
        if not state.saw_sep:
            if tok == "<SEP>":
                state.saw_sep      = True
                state.next_expected = _NextExpected.BAR_OPEN
            return

        # ---- Structural tokens ----
        if tok.startswith("BAR_") and not tok.startswith("<"):
            # "BAR_N" — enter a new bar
            try:
                state.current_bar   = int(tok[4:])
                state.next_expected = _NextExpected.IN_BAR
            except ValueError:
                pass
            return

        if tok == "<BAR_END>":
            state.next_expected = _NextExpected.BAR_OPEN
            return

        # ---- Note tokens ----
        if tok.startswith("POS_") and state.next_expected in (
            _NextExpected.IN_BAR, _NextExpected.BAR_OPEN
        ):
            try:
                state.pending_pos   = int(tok[4:])
                state.next_expected = _NextExpected.PITCH
            except ValueError:
                pass
            return

        if tok.startswith("PITCH_") and state.next_expected == _NextExpected.PITCH:
            try:
                state.pending_pitch  = int(tok[6:])
                state.next_expected  = _NextExpected.DURATION
            except ValueError:
                pass
            return

        if tok.startswith("DUR_") and state.next_expected == _NextExpected.DURATION:
            try:
                state.pending_dur   = float(tok[4:])
                state.next_expected = _NextExpected.VELOCITY
            except ValueError:
                pass
            return

        if tok.startswith("VEL_") and state.next_expected == _NextExpected.VELOCITY:
            try:
                state.pending_vel = int(tok[4:])
                # Complete note — increment counter and reset pending fields
                state.notes_decoded  += 1
                state.pending_pos     = None
                state.pending_pitch   = None
                state.pending_dur     = None
                state.pending_vel     = None
                state.next_expected   = _NextExpected.IN_BAR
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode_token_stream(
        self,
        token_ids: List[int],
        beats_per_bar: float = 4.0,
    ) -> List[Note]:
        """
        Decode the full generated token stream into ``Note`` objects.

        Delegates to ``Tokenizer.detokenize_ids()`` which handles bar
        tracking, position-to-beat conversion, and incomplete trailing
        note groups.

        Args:
            token_ids:    Full token ID list (BOS stripped).
            beats_per_bar: Time signature denominator (default 4/4).

        Returns:
            Sorted list of ``Note`` objects.
        """
        notes, _, _, _ = self._tokenizer.detokenize_ids(token_ids, beats_per_bar)
        notes.sort(key=lambda n: n.position)
        return notes

    # ------------------------------------------------------------------
    # Rule-based arpeggio padding
    # ------------------------------------------------------------------

    def _pad_with_arpeggio(
        self,
        existing: List[Note],
        request: GenerationRequest,
    ) -> List[Note]:
        """
        Extend ``existing`` to ``note_count`` with rule-based notes.

        The padding arpeggio starts immediately after the last generated
        note to maintain temporal continuity.

        Args:
            existing: Notes produced by the model (may be empty).
            request:  Original generation request.

        Returns:
            Extended note list (existing + padding).
        """
        needed = request.note_count - len(existing)
        if needed <= 0:
            return existing

        rng_seed = _random.randint(0, 2_147_483_647)
        pad_arpeggio: Arpeggio = ArpeggioGenerator().generate(
            key=request.key,
            scale=request.scale,
            tempo=request.tempo,
            note_count=needed,
            seed=rng_seed,
            octave=request.octave,
        )

        # Shift padding so it follows existing content without a gap
        offset = 0.0
        if existing:
            last_note = max(existing, key=lambda n: n.position + n.duration)
            offset    = last_note.position + last_note.duration

        padding = [
            Note(
                pitch=n.pitch,
                velocity=n.velocity,
                position=n.position + offset,
                duration=n.duration,
            )
            for n in pad_arpeggio.notes
        ]

        return existing + padding
