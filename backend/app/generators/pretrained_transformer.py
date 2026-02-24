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
- No mood conditioning — planned for a future release.
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

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Full sequence forward pass.

        Args:
            input_ids: ``(batch, seq_len)`` long tensor of token IDs.

        Returns:
            Logits ``(batch, seq_len, vocab_size)``.
        """
        _, seq_len = input_ids.shape

        # Clamp to max_seq_len (graceful handling of over-length inputs)
        seq_len = min(seq_len, self.max_seq_len)
        ids     = input_ids[:, :seq_len]

        positions = torch.arange(seq_len, device=ids.device).unsqueeze(0)  # (1, L)
        x = self.token_embedding(ids) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Causal mask: upper triangle = -inf so each position attends only
        # to itself and previous positions.
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=ids.device),
            diagonal=1,
        )

        x = self.transformer_layers(x, mask=causal_mask)
        x = self.output_norm(x)
        return self.output_projection(x)


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
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        temperature: float = 0.95,
        top_k: int = 50,
        max_gen_tokens: int = 1024,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self.temperature      = float(temperature)
        self.top_k            = int(top_k)
        self.max_gen_tokens   = int(max_gen_tokens)

        self._model:  Optional[_SymbolicMusicTransformer] = None
        self._device: Optional[torch.device]              = None
        self._vocab:  Vocabulary                          = get_vocabulary()
        self._tokenizer: Tokenizer                        = Tokenizer(self._vocab)
        self._lock    = threading.RLock()
        self._loaded: bool = False

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

        # 1. Build structured prompt
        prompt_ids = self._build_prompt(request)

        # 2. Autoregressive generation
        token_stream = self._generate_autoregressive(prompt_ids, request)

        # 3. Decode token stream → Note objects
        notes = self._decode_token_stream(token_stream)

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
    ) -> List[int]:
        """
        Extend ``prompt_ids`` autoregressively until ``note_count`` notes
        are decoded, EOS is generated, or the token budget is exhausted.

        The scale-pitch mask is applied to logits whenever the state
        machine expects a PITCH token next.

        Args:
            prompt_ids: Structured prompt (including BOS, seed note).
            request:    Generation parameters.

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
        token_budget: int      = max(self.max_gen_tokens, request.note_count * 8)

        with torch.no_grad():
            for _ in range(token_budget):
                if state.notes_decoded >= request.note_count:
                    break

                # Use the most recent `max_seq_len` tokens as context
                context = token_ids[-max_seq_len:]
                src     = torch.tensor(
                    [context], dtype=torch.long, device=self._device
                )

                logits     = self._model(src)       # (1, seq_len, vocab_size)
                next_logits = logits[0, -1, :]      # (vocab_size,)

                # Apply scale constraint only when about to sample a pitch
                if state.next_expected == _NextExpected.PITCH:
                    next_logits = next_logits + pitch_mask

                next_id = self._sample(next_logits)

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

    def _sample(self, logits: Tensor) -> int:
        """
        Sample the next token using temperature scaling and top-k filtering.

        - Temperature = 0 → greedy argmax (deterministic).
        - Top-k = 0 → consider all tokens (pure temperature sampling).
        - If the probability distribution collapses (all -inf after masking),
          falls back to greedy argmax on the unmasked logits.

        Args:
            logits: Raw unnormalised logits ``(vocab_size,)``.

        Returns:
            Sampled token ID.
        """
        # Greedy shortcut
        if self.temperature <= 0.0:
            return int(logits.argmax().item())

        scaled = logits / self.temperature

        # Top-k filtering: zero out all but the k highest logit tokens
        if self.top_k > 0:
            k             = min(self.top_k, scaled.size(-1))
            top_vals, _   = torch.topk(scaled, k)
            threshold     = top_vals[-1]
            scaled        = torch.where(
                scaled < threshold,
                torch.full_like(scaled, float("-inf")),
                scaled,
            )

        probs = F.softmax(scaled, dim=-1)

        # Guard: if all probs are zero/nan (e.g. scale mask blocked everything),
        # fall back to greedy argmax on the original logits.
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
