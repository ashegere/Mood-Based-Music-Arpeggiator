"""
Inference engine for the mood-conditioned music transformer.

Provides thread-safe, deterministic inference for transforming base arpeggio
token sequences into mood-conditioned variants using the trained checkpoint.
"""

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Model definition — must exactly match the architecture saved in checkpoints
# (see train_runner.py).  Weights cannot be loaded if this diverges.
# ---------------------------------------------------------------------------

class _MoodConditionedTransformer(nn.Module):
    """Decoder-only transformer with discrete mood conditioning."""

    def __init__(
        self,
        vocab_size: int,
        num_moods: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.mood_embedding = nn.Embedding(num_moods, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: Tensor,                      # (batch, seq_len)
        mood_labels: Tensor,                    # (batch,)
        attention_mask: Optional[Tensor] = None,  # (batch, seq_len) bool
    ) -> Tensor:                                # (batch, seq_len, vocab_size)
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.mood_embedding(mood_labels).unsqueeze(1)
        )
        x = self.dropout(x)

        # MPS-compatible causal mask: large negative instead of -inf
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), -1e9, device=input_ids.device),
            diagonal=1,
        )
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        memory = torch.zeros(batch_size, 1, self.d_model, device=input_ids.device)

        x = self.decoder(
            x, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=key_padding_mask,
        )
        return self.output_projection(x)


# ---------------------------------------------------------------------------
# Mood vocabulary (order must match dataset preparation)
# ---------------------------------------------------------------------------

_VALID_MOODS: Tuple[str, ...] = (
    "melancholic", "dreamy", "energetic", "tense", "happy",
    "sad", "calm", "dark", "joyful", "uplifting",
    "intense", "peaceful", "dramatic", "epic", "mysterious",
    "romantic", "neutral", "flowing", "ominous",
)


def _detect_nhead(d_model: int) -> int:
    """Pick a sensible number of attention heads for a given d_model."""
    for n in (4, 8, 6, 2, 1):
        if d_model % n == 0:
            return n
    return 1  # fallback (always valid)


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Thread-safe inference engine for the mood-conditioned transformer.

    Loads a trained checkpoint once and exposes ``run()`` to transform base
    arpeggio tokens into mood-conditioned tokens.

    Thread safety:
        A reentrant lock guards all model forward passes so multiple threads
        can call ``run()`` concurrently without data races.

    Randomness:
        By default ``run()`` uses greedy argmax — fully deterministic.  Pass
        an integer ``seed`` to enable reproducible multinomial sampling.
    """

    def __init__(self) -> None:
        self._model: Optional[_MoodConditionedTransformer] = None
        self._device: Optional[torch.device] = None
        self._vocab: Optional[Dict] = None
        self._mood_to_id: Dict[str, int] = {m: i for i, m in enumerate(_VALID_MOODS)}

        # Cached reference embeddings for nearest-mood lookup (lazy init)
        self._mood_emb_matrix: Optional[Tensor] = None  # (num_moods, D)

        # Special token IDs; overwritten from dataset metadata when available
        self._pad_id: int = 0
        self._bos_id: int = 1
        self._eos_id: int = 2

        # RLock: safe if the same thread calls run() re-entrantly
        self._lock = threading.RLock()
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        checkpoint_path: Union[str, Path],
        dataset_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Load the trained model from a checkpoint.

        Args:
            checkpoint_path: Path to ``best_model.pt`` or ``final_model.pt``.
            dataset_path: Optional path to ``train_dataset.pt`` for vocab and
                special-token metadata.  Falls back to hard-coded defaults.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the state-dict is incompatible with the model.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with self._lock:
            # Device selection
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")

            # Read vocab metadata from dataset when available
            if dataset_path is not None:
                dataset_path = Path(dataset_path)
                if dataset_path.exists():
                    dataset = torch.load(
                        dataset_path, map_location="cpu", weights_only=False
                    )
                    self._vocab = dataset.get("vocab", {})
                    self._pad_id = self._vocab.get("pad_token_id", self._pad_id)
                    self._bos_id = self._vocab.get("bos_token_id", self._bos_id)
                    self._eos_id = self._vocab.get("eos_token_id", self._eos_id)
                    if "mood_to_id" in self._vocab:
                        self._mood_to_id = self._vocab["mood_to_id"]

            # Load checkpoint
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            state: Dict[str, Tensor] = checkpoint["model_state_dict"]

            # Auto-detect hyperparameters from weight shapes
            vocab_size  = state["token_embedding.weight"].shape[0]
            d_model     = state["token_embedding.weight"].shape[1]
            num_moods   = state["mood_embedding.weight"].shape[0]
            max_seq_len = state["position_embedding.weight"].shape[0]
            num_layers  = sum(
                1 for k in state
                if k.startswith("decoder.layers.") and k.endswith(".norm1.weight")
            )
            nhead = _detect_nhead(d_model)

            self._model = _MoodConditionedTransformer(
                vocab_size=vocab_size,
                num_moods=num_moods,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                dropout=0.0,  # eval mode; no dropout
            )
            self._model.load_state_dict(state)
            self._model.to(self._device)
            self._model.eval()

            self._loaded = True

    # ------------------------------------------------------------------
    # Mood resolution
    # ------------------------------------------------------------------

    def _resolve_mood(self, mood: Union[int, str, Tensor]) -> int:
        """
        Map a mood input to a discrete mood index accepted by the model.

        Args:
            mood: One of:
                - ``int``    – mood index used directly.
                - ``str``    – mood name looked up in the vocabulary.
                - ``Tensor`` – continuous embedding; mapped to the nearest
                               mood via cosine similarity.

        Returns:
            Integer mood index in ``[0, num_moods)``.

        Raises:
            TypeError:    If ``mood`` is none of the accepted types.
            ValueError:   If the string is not in the vocabulary or the index
                          is out of range.
        """
        num_moods = self._model.mood_embedding.num_embeddings  # type: ignore[union-attr]

        if isinstance(mood, int):
            if not 0 <= mood < num_moods:
                raise ValueError(
                    f"Mood index {mood} out of range [0, {num_moods})"
                )
            return mood

        if isinstance(mood, str):
            key = mood.lower().strip()
            if key not in self._mood_to_id:
                raise ValueError(
                    f"Unknown mood '{mood}'. Valid: {list(self._mood_to_id)}"
                )
            return self._mood_to_id[key]

        if isinstance(mood, Tensor):
            vec = mood.float().detach().cpu()
            if vec.dim() == 2 and vec.shape[0] == 1:
                vec = vec.squeeze(0)
            if vec.dim() != 1:
                raise ValueError(
                    f"Mood tensor must be 1-D or (1, D), got shape {tuple(mood.shape)}"
                )
            return self._nearest_mood(vec)

        raise TypeError(
            f"mood must be int, str, or Tensor, got {type(mood).__name__}"
        )

    def _nearest_mood(self, vec: Tensor) -> int:
        """
        Return the index of the closest mood to ``vec`` (cosine similarity).

        Reference embeddings are computed on first call via
        ``app.mood.embeddings`` if available, otherwise a deterministic
        random-projection fallback is used.

        Args:
            vec: 1-D float tensor of arbitrary dimension D.

        Returns:
            Nearest mood index.
        """
        matrix = self._get_mood_emb_matrix(vec.shape[0])
        vec_n = F.normalize(vec.unsqueeze(0), p=2, dim=1)        # (1, D)
        mat_n = F.normalize(matrix.float(), p=2, dim=1)           # (M, D)
        sims = (mat_n @ vec_n.T).squeeze(-1)                      # (M,)
        return int(sims.argmax().item())

    def _get_mood_emb_matrix(self, dim: int) -> Tensor:
        """
        Return (num_moods, dim) reference embeddings for nearest-mood lookup.

        First tries ``app.mood.embeddings.get_mood_embeddings()``.
        Falls back to a seeded random projection when sentence-transformers
        is not installed.

        Args:
            dim: Expected embedding dimension.

        Returns:
            Tensor of shape (num_moods, dim).
        """
        if (
            self._mood_emb_matrix is not None
            and self._mood_emb_matrix.shape[1] == dim
        ):
            return self._mood_emb_matrix

        moods = list(self._mood_to_id.keys())

        try:
            from app.mood.embeddings import get_mood_embeddings  # type: ignore
            matrix = get_mood_embeddings(moods).cpu().float()
            if matrix.shape[1] == dim:
                self._mood_emb_matrix = matrix
                return matrix
        except Exception:
            pass

        # Deterministic fallback: seeded Gaussian projection
        rng = torch.Generator()
        rng.manual_seed(0)
        matrix = torch.randn(len(moods), dim, generator=rng)
        self._mood_emb_matrix = matrix
        return matrix

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(
        self,
        tokens: Union[List[int], Tensor],
        mood_embedding: Union[int, str, Tensor],
        seed: Optional[int] = None,
        temperature: float = 1.0,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Transform base arpeggio tokens under the given mood.

        Teacher-forcing inference: the full input sequence is passed through
        the model in one forward pass.  At each position *i* the model
        predicts what token should follow position *i*, conditioned on the
        mood.  This produces a mood-modified reconstruction of the input.

        Randomness policy:
            - ``seed=None``  → greedy argmax at every position (default).
            - ``seed=<int>`` → multinomial sampling with that seed and the
                               given ``temperature``.  Identical seeds always
                               produce identical outputs.

        Args:
            tokens: 1-D list or 1-D Tensor of integer token IDs (the base
                    arpeggio).  BOS / EOS are added automatically if absent.
            mood_embedding: Mood — one of:
                - ``int``    : mood index in ``[0, num_moods)``
                - ``str``    : mood name, e.g. ``"happy"``
                - ``Tensor`` : continuous embedding (e.g. from
                               sentence-transformer); mapped to nearest mood.
            seed: Optional integer RNG seed.  ``None`` disables sampling.
            temperature: Softmax temperature applied before sampling (ignored
                         when ``seed`` is ``None``).  Lower values make the
                         distribution sharper.
            max_length: Maximum number of tokens to process.  Defaults to the
                        model's ``max_seq_len``.

        Returns:
            List of integer token IDs representing the mood-modified arpeggio
            (BOS and EOS tokens are stripped from the output).

        Raises:
            RuntimeError: If ``load()`` has not been called.
            ValueError:   If the mood specification is invalid.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        mood_idx = self._resolve_mood(mood_embedding)

        # Normalise tokens to a plain Python list
        if isinstance(tokens, Tensor):
            token_list: List[int] = tokens.long().cpu().tolist()
        else:
            token_list = [int(t) for t in tokens]

        # Ensure BOS prefix and EOS suffix
        if not token_list or token_list[0] != self._bos_id:
            token_list = [self._bos_id] + token_list
        if token_list[-1] != self._eos_id:
            token_list = token_list + [self._eos_id]

        # Truncate to model capacity (need at least 2 tokens for a shift)
        cap = max_length or self._model.max_seq_len  # type: ignore[union-attr]
        token_list = token_list[:cap]

        if len(token_list) < 2:
            return []

        # Build device tensors
        src = torch.tensor(token_list, dtype=torch.long, device=self._device).unsqueeze(0)
        mood_t = torch.tensor([mood_idx], dtype=torch.long, device=self._device)

        # Shift: input is tokens[:-1], targets are tokens[1:]
        src_in = src[:, :-1]  # (1, L-1)

        # Optional seeded generator for reproducible sampling
        generator: Optional[torch.Generator] = None
        if seed is not None:
            generator = torch.Generator(device="cpu")  # multinomial needs CPU gen
            generator.manual_seed(seed)

        with self._lock:
            with torch.no_grad():
                logits = self._model(src_in, mood_t)  # (1, L-1, vocab_size)

        logits = logits.squeeze(0)  # (L-1, vocab_size)

        if seed is None:
            # Greedy: no randomness
            predicted: List[int] = logits.argmax(dim=-1).tolist()
        else:
            # Seeded multinomial sampling
            scaled = logits.cpu() / max(float(temperature), 1e-8)
            probs = F.softmax(scaled, dim=-1)
            predicted = torch.multinomial(
                probs, num_samples=1, generator=generator
            ).squeeze(-1).tolist()

        # Strip BOS from the head of the prediction
        if predicted and predicted[0] == self._bos_id:
            predicted = predicted[1:]

        # Truncate at first EOS
        if self._eos_id in predicted:
            predicted = predicted[: predicted.index(self._eos_id)]

        return predicted

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True after a successful ``load()`` call."""
        return self._loaded

    @property
    def device(self) -> Optional[torch.device]:
        """Torch device the model currently runs on."""
        return self._device

    @property
    def valid_moods(self) -> List[str]:
        """Ordered list of accepted mood name strings."""
        return list(self._mood_to_id.keys())

    @property
    def vocab_size(self) -> Optional[int]:
        """Vocabulary size of the loaded model, or None if not loaded."""
        return (
            self._model.token_embedding.num_embeddings
            if self._model is not None
            else None
        )


# ---------------------------------------------------------------------------
# Module-level singleton and convenience API
# ---------------------------------------------------------------------------

_engine: Optional[InferenceEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> InferenceEngine:
    """
    Return the module-level :class:`InferenceEngine` singleton.

    The engine is created on first call and reused across all callers.
    Thread-safe.
    """
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = InferenceEngine()
    return _engine


def load_model(
    checkpoint_path: Union[str, Path],
    dataset_path: Optional[Union[str, Path]] = None,
) -> InferenceEngine:
    """
    Load the trained model into the singleton engine.

    Convenience wrapper around ``get_engine().load(...)``.

    Args:
        checkpoint_path: Path to checkpoint file.
        dataset_path: Optional dataset path for vocab metadata.

    Returns:
        The loaded :class:`InferenceEngine`.
    """
    engine = get_engine()
    engine.load(checkpoint_path, dataset_path)
    return engine


def run_inference(
    tokens: Union[List[int], Tensor],
    mood_embedding: Union[int, str, Tensor],
    seed: Optional[int] = None,
    temperature: float = 1.0,
) -> List[int]:
    """
    Run mood-conditioned inference using the singleton engine.

    The engine must have been loaded via :func:`load_model` before calling
    this function.

    Args:
        tokens: Base arpeggio token IDs.
        mood_embedding: Mood as int index, string name, or embedding tensor.
        seed: Optional RNG seed for reproducible sampling.
        temperature: Sampling temperature (only used when seed is not None).

    Returns:
        Modified token IDs.
    """
    return get_engine().run(tokens, mood_embedding, seed=seed, temperature=temperature)
