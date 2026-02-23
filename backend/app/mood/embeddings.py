"""
Mood embedding module for text-to-vector conversion.

This module converts mood descriptions (keywords or phrases) into
fixed-size embedding vectors suitable for conditioning music generation.

Design Decisions:
-----------------

1. Model Selection: sentence-transformers/all-MiniLM-L6-v2
   - 384-dimensional embeddings (compact but expressive)
   - Fast inference (~14ms per embedding on CPU)
   - Good semantic understanding for short text
   - Small model size (~80MB)
   - Well-suited for mood/emotion phrases

   Alternative considered: all-mpnet-base-v2 (768-dim, better quality)
   Rejected due to larger size and slower inference for minimal gain
   on short mood phrases.

2. Caching Strategy: LRU Cache with configurable size
   - Mood vocabulary is typically limited in practice
   - Same moods are requested repeatedly
   - Memory-bounded to prevent unbounded growth
   - Thread-safe for concurrent requests

3. Lazy Loading: Model loads on first use, not import
   - Prevents slow startup for unused functionality
   - Allows configuration before first call
   - Single instance shared across requests

4. Normalization: L2-normalized embeddings
   - Enables cosine similarity via dot product
   - Consistent magnitude for downstream processing
   - Standard practice for contrastive embeddings

5. Device Handling: Auto-detect GPU, fallback to CPU
   - Transparent to callers
   - Optimizes for available hardware
   - Configurable override for testing

Usage:
------
    from app.mood.embeddings import get_mood_embedding, MoodEmbedder

    # Simple usage with global instance
    embedding = get_mood_embedding("melancholic and reflective")

    # Custom configuration
    embedder = MoodEmbedder(model_name="all-mpnet-base-v2")
    embedding = embedder.embed("energetic")
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
from threading import Lock
import logging
import hashlib

import torch
from torch import Tensor

# Conditional import for sentence-transformers
# Allows graceful degradation if not installed
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore


logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

# Default embedding model
# all-MiniLM-L6-v2: Good balance of speed and quality for short text
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding dimension for the default model
DEFAULT_EMBEDDING_DIM: int = 384

# Maximum cache size (number of unique mood strings)
DEFAULT_CACHE_SIZE: int = 1024

# Maximum input text length (characters)
MAX_TEXT_LENGTH: int = 512

# Predefined mood keywords for validation/suggestion
# These represent common moods in music generation
MOOD_KEYWORDS: Tuple[str, ...] = (
    # Positive/High Energy
    "happy", "joyful", "euphoric", "excited", "energetic",
    "uplifting", "triumphant", "playful", "cheerful", "bright",

    # Calm/Peaceful
    "calm", "peaceful", "serene", "tranquil", "relaxed",
    "gentle", "soothing", "meditative", "dreamy", "floating",

    # Sad/Melancholic
    "sad", "melancholic", "sorrowful", "wistful", "nostalgic",
    "longing", "bittersweet", "mournful", "pensive", "reflective",

    # Dark/Tense
    "dark", "mysterious", "ominous", "tense", "suspenseful",
    "brooding", "intense", "dramatic", "epic", "powerful",

    # Romantic/Emotional
    "romantic", "passionate", "tender", "intimate", "emotional",
    "heartfelt", "loving", "warm", "hopeful", "inspiring",

    # Neutral/Ambient
    "neutral", "ambient", "atmospheric", "minimal", "sparse",
    "subtle", "understated", "steady", "flowing", "continuous",
)


# =============================================================================
# Exceptions
# =============================================================================

class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass


class ModelNotLoadedError(EmbeddingError):
    """Raised when model is accessed before loading."""
    pass


class InvalidMoodTextError(EmbeddingError):
    """Raised when mood text is invalid."""
    pass


class DependencyMissingError(EmbeddingError):
    """Raised when required dependencies are not installed."""
    pass


# =============================================================================
# Embedding Cache
# =============================================================================

class EmbeddingCache:
    """
    Thread-safe LRU cache for mood embeddings.

    Uses string hashing for consistent cache keys and provides
    memory-bounded storage of computed embeddings.
    """

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of cached embeddings.
        """
        self.max_size = max_size
        self._cache: Dict[str, Tensor] = {}
        self._access_order: List[str] = []
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str) -> str:
        """
        Create a cache key from text.

        Uses MD5 hash for consistent, fixed-length keys.
        Collision risk is negligible for this use case.

        Args:
            text: Input text.

        Returns:
            Hash-based cache key.
        """
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, text: str) -> Optional[Tensor]:
        """
        Retrieve cached embedding.

        Args:
            text: Original mood text.

        Returns:
            Cached tensor or None if not found.
        """
        key = self._make_key(text)

        with self._lock:
            if key in self._cache:
                # Move to end of access order (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                self._hits += 1
                # Return a clone to prevent modification
                return self._cache[key].clone()

            self._misses += 1
            return None

    def put(self, text: str, embedding: Tensor) -> None:
        """
        Store embedding in cache.

        Args:
            text: Original mood text.
            embedding: Computed embedding tensor.
        """
        key = self._make_key(text)

        with self._lock:
            # If key exists, update and move to end
            if key in self._cache:
                self._cache[key] = embedding.clone()
                self._access_order.remove(key)
                self._access_order.append(key)
                return

            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]

            # Add new entry
            self._cache[key] = embedding.clone()
            self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        """Current number of cached embeddings."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


# =============================================================================
# Mood Embedder
# =============================================================================

class MoodEmbedder:
    """
    Generates embeddings for mood text using sentence transformers.

    This class handles model loading, caching, and embedding generation.
    It is thread-safe and designed for concurrent use.

    Attributes:
        model_name: Name of the sentence-transformers model.
        embedding_dim: Dimension of output embeddings.
        device: Torch device for computation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_size: int = DEFAULT_CACHE_SIZE,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize the mood embedder.

        Model is loaded lazily on first embed() call.

        Args:
            model_name: HuggingFace model name or path.
            cache_size: Maximum cached embeddings.
            device: Torch device ("cpu", "cuda", "mps"). Auto-detect if None.
            normalize: Whether to L2-normalize embeddings.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise DependencyMissingError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.normalize = normalize
        self._cache = EmbeddingCache(max_size=cache_size)
        self._model: Optional[SentenceTransformer] = None
        self._model_lock = Lock()
        self._embedding_dim: Optional[int] = None

        # Determine device
        if device is not None:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        logger.info(
            f"MoodEmbedder initialized: model={model_name}, device={self._device}"
        )

    def _load_model(self) -> SentenceTransformer:
        """
        Load the sentence transformer model (thread-safe).

        Returns:
            Loaded SentenceTransformer instance.
        """
        with self._model_lock:
            if self._model is None:
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(
                    self.model_name,
                    device=str(self._device),
                )
                # Get embedding dimension from model
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(
                    f"Model loaded: dim={self._embedding_dim}, device={self._device}"
                )
            return self._model

    @property
    def model(self) -> SentenceTransformer:
        """Get the loaded model, loading if necessary."""
        if self._model is None:
            return self._load_model()
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (loads model if needed)."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim  # type: ignore

    @property
    def device(self) -> torch.device:
        """Get the computation device."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None

    def _validate_text(self, text: str) -> str:
        """
        Validate and preprocess mood text.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text.

        Raises:
            InvalidMoodTextError: If text is invalid.
        """
        if not isinstance(text, str):
            raise InvalidMoodTextError(f"Expected string, got {type(text).__name__}")

        # Clean whitespace
        cleaned = " ".join(text.split())

        if not cleaned:
            raise InvalidMoodTextError("Mood text cannot be empty")

        if len(cleaned) > MAX_TEXT_LENGTH:
            raise InvalidMoodTextError(
                f"Mood text too long: {len(cleaned)} > {MAX_TEXT_LENGTH} chars"
            )

        return cleaned

    def embed(
        self,
        text: str,
        use_cache: bool = True,
    ) -> Tensor:
        """
        Generate embedding for mood text.

        Args:
            text: Mood keyword or phrase.
            use_cache: Whether to use/update cache.

        Returns:
            Embedding tensor of shape (embedding_dim,).
        """
        # Validate input
        text = self._validate_text(text)

        # Check cache
        if use_cache:
            cached = self._cache.get(text)
            if cached is not None:
                return cached

        # Generate embedding
        model = self.model

        # encode() returns numpy array, convert to tensor
        with torch.no_grad():
            embedding_np = model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=self.normalize,
            )
            embedding = torch.from_numpy(embedding_np).float()

        # Ensure on correct device
        embedding = embedding.to(self._device)

        # Cache result
        if use_cache:
            self._cache.put(text, embedding)

        return embedding

    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> Tensor:
        """
        Generate embeddings for multiple mood texts.

        Args:
            texts: List of mood keywords or phrases.
            use_cache: Whether to use/update cache.

        Returns:
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        if not texts:
            # Return empty tensor with correct shape
            return torch.empty(0, self.embedding_dim, device=self._device)

        # Validate all texts
        cleaned_texts = [self._validate_text(t) for t in texts]

        # Check cache for each text
        embeddings = []
        texts_to_encode = []
        text_indices = []

        for i, text in enumerate(cleaned_texts):
            if use_cache:
                cached = self._cache.get(text)
                if cached is not None:
                    embeddings.append((i, cached))
                    continue

            texts_to_encode.append(text)
            text_indices.append(i)

        # Encode uncached texts in batch
        if texts_to_encode:
            model = self.model

            with torch.no_grad():
                batch_embeddings_np = model.encode(
                    texts_to_encode,
                    convert_to_tensor=False,
                    normalize_embeddings=self.normalize,
                    batch_size=32,
                )
                batch_embeddings = torch.from_numpy(batch_embeddings_np).float()
                batch_embeddings = batch_embeddings.to(self._device)

            # Cache and collect results
            for j, (idx, text) in enumerate(zip(text_indices, texts_to_encode)):
                emb = batch_embeddings[j]
                if use_cache:
                    self._cache.put(text, emb)
                embeddings.append((idx, emb))

        # Sort by original index and stack
        embeddings.sort(key=lambda x: x[0])
        return torch.stack([e for _, e in embeddings])

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two mood texts.

        Args:
            text1: First mood text.
            text2: Second mood text.

        Returns:
            Cosine similarity (-1 to 1, typically 0 to 1 for moods).
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        # For normalized vectors, dot product = cosine similarity
        if self.normalize:
            return float(torch.dot(emb1, emb2).item())
        else:
            return float(torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            ).item())

    def find_similar_moods(
        self,
        text: str,
        candidates: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find most similar moods from a set of candidates.

        Args:
            text: Query mood text.
            candidates: List of candidate moods. Uses MOOD_KEYWORDS if None.
            top_k: Number of results to return.

        Returns:
            List of (mood, similarity) tuples, sorted by similarity.
        """
        if candidates is None:
            candidates = list(MOOD_KEYWORDS)

        query_emb = self.embed(text)
        candidate_embs = self.embed_batch(candidates)

        # Compute similarities
        if self.normalize:
            similarities = torch.matmul(candidate_embs, query_emb)
        else:
            similarities = torch.nn.functional.cosine_similarity(
                candidate_embs, query_emb.unsqueeze(0).expand(len(candidates), -1)
            )

        # Get top-k
        top_k = min(top_k, len(candidates))
        top_values, top_indices = torch.topk(similarities, top_k)

        results = [
            (candidates[idx], float(val))
            for idx, val in zip(top_indices.tolist(), top_values.tolist())
        ]

        return results

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        return self._cache.stats()

    def unload_model(self) -> None:
        """
        Unload the model to free memory.

        The model will be reloaded on next embed() call.
        """
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
                # Clear CUDA cache if applicable
                if self._device.type == "cuda":
                    torch.cuda.empty_cache()
                logger.info("Model unloaded")


# =============================================================================
# Global Instance and Convenience Functions
# =============================================================================

# Global embedder instance (lazy initialization)
_global_embedder: Optional[MoodEmbedder] = None
_global_lock = Lock()


def get_embedder(
    model_name: str = DEFAULT_MODEL_NAME,
    **kwargs,
) -> MoodEmbedder:
    """
    Get or create the global MoodEmbedder instance.

    Args:
        model_name: Model to use (only used on first call).
        **kwargs: Additional arguments for MoodEmbedder.

    Returns:
        Global MoodEmbedder instance.
    """
    global _global_embedder

    with _global_lock:
        if _global_embedder is None:
            _global_embedder = MoodEmbedder(model_name=model_name, **kwargs)
        return _global_embedder


def get_mood_embedding(text: str) -> Tensor:
    """
    Generate embedding for a mood keyword or phrase.

    This is the main interface for mood embedding. It uses a global
    embedder instance with default configuration.

    Args:
        text: Mood keyword or phrase (e.g., "melancholic and reflective").

    Returns:
        Embedding tensor of shape (384,) for the default model.

    Example:
        >>> embedding = get_mood_embedding("happy and energetic")
        >>> embedding.shape
        torch.Size([384])
    """
    embedder = get_embedder()
    return embedder.embed(text)


def get_mood_embeddings(texts: List[str]) -> Tensor:
    """
    Generate embeddings for multiple mood texts.

    More efficient than calling get_mood_embedding() in a loop.

    Args:
        texts: List of mood keywords or phrases.

    Returns:
        Embedding tensor of shape (batch_size, 384).
    """
    embedder = get_embedder()
    return embedder.embed_batch(texts)


def get_embedding_dim() -> int:
    """
    Get the dimension of mood embeddings.

    Returns:
        Embedding dimension (384 for default model).
    """
    embedder = get_embedder()
    return embedder.embedding_dim


def preload_model() -> None:
    """
    Preload the embedding model.

    Call this during application startup to avoid latency
    on the first embedding request.
    """
    embedder = get_embedder()
    _ = embedder.model  # Triggers loading


def reset_global_embedder() -> None:
    """
    Reset the global embedder (for testing).

    Unloads the model and clears the global instance.
    """
    global _global_embedder

    with _global_lock:
        if _global_embedder is not None:
            _global_embedder.unload_model()
            _global_embedder = None
