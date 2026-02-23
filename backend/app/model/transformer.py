"""
Mood-conditioned music transformer for arpeggio generation.

This module implements an encoder-decoder transformer architecture
designed for symbolic music generation with mood conditioning.

Architecture Overview:
---------------------

    Input Tokens → Encoder → Latent Representation
                                     ↓
    Mood Embedding → Conditioning → Decoder → Output Tokens
                                     ↓
                              Scale-Corrected Output

Design Principles:
-----------------

1. Structure Preservation:
   The encoder captures the musical structure (pitch sequence, rhythm).
   This structure passes through to the decoder relatively unchanged.
   Mood conditioning modifies HOW notes are played, not WHICH notes.

2. Expression Modification:
   FiLM conditioning in decoder layers modulates:
   - Velocity predictions (dynamics)
   - Duration fine-tuning (articulation)
   - Timing micro-adjustments (feel/groove)

3. Small Footprint:
   - 6-12 layers total (3-6 encoder, 3-6 decoder)
   - 256-512 hidden dimension
   - 4-8 attention heads
   Designed for fast inference and reasonable training.

4. Freezing Support:
   Base encoder layers can be frozen to:
   - Preserve learned musical structure
   - Fine-tune only conditioning pathways
   - Reduce training compute

Token Flow:
----------

Input: [BOS] [KEY] [SCALE] [TEMPO] [SEP] [BAR] [POS] [PITCH] [DUR] [VEL] ... [EOS]

The model processes event-based tokens where each note is represented by:
- Position in bar (when)
- Pitch (what)
- Duration (how long)
- Velocity (how loud) ← Primary target for mood modification

Output predicts the same token vocabulary, with mood-adjusted velocities
and potentially modified durations for expressive variation.
"""

import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from app.model.conditioning import (
    FiLM,
    AdaptiveFiLM,
    MoodConditioner,
    MoodProjection,
    CrossAttentionConditioning,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TransformerConfig:
    """
    Configuration for the music transformer.

    This dataclass holds all hyperparameters for model construction.
    Default values are tuned for a small but capable model.

    Attributes:
        vocab_size: Size of token vocabulary.
        d_model: Hidden dimension throughout the model.
        n_encoder_layers: Number of encoder layers.
        n_decoder_layers: Number of decoder layers.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length.
        mood_embedding_dim: Dimension of mood embedding input.
        use_cross_attention: Enable cross-attention conditioning.
        n_conditioning_tokens: Number of conditioning pseudo-tokens.
        adaptive_film: Use adaptive gating in FiLM.
        condition_encoder: Whether to apply conditioning to encoder.
        condition_every_n_layers: Apply FiLM every N decoder layers.
    """
    # Vocabulary
    vocab_size: int = 256

    # Model dimensions
    d_model: int = 384  # Match mood embedding dim for efficiency
    n_encoder_layers: int = 4
    n_decoder_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536  # 4x d_model is standard

    # Regularization
    dropout: float = 0.1

    # Sequence
    max_seq_len: int = 1024

    # Conditioning
    mood_embedding_dim: int = 384  # From sentence-transformers
    use_cross_attention: bool = True
    n_conditioning_tokens: int = 4
    adaptive_film: bool = True
    condition_encoder: bool = False  # Usually only condition decoder
    condition_every_n_layers: int = 1  # 1 = every layer, 2 = every other, etc.

    # Special token IDs (must match tokenizer)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


# =============================================================================
# Positional Encoding
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in "Attention Is All You Need".

    Uses fixed sinusoidal patterns to encode position information.
    This is preferred over learned embeddings for music because:
    - Generalizes to unseen sequence lengths
    - Smooth interpolation between positions
    - No additional parameters

    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Input with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# Encoder Layer
# =============================================================================

class EncoderLayer(nn.Module):
    """
    Single encoder layer with optional mood conditioning.

    Standard transformer encoder layer:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Residual connections and layer normalization

    Optional FiLM conditioning after the FFN for encoder-side
    mood influence (usually disabled).

    Args:
        config: Transformer configuration.
        apply_conditioning: Whether to apply FiLM in this layer.
    """

    def __init__(
        self,
        config: TransformerConfig,
        apply_conditioning: bool = False,
    ):
        super().__init__()

        self.apply_conditioning = apply_conditioning

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        # Optional conditioning
        self.conditioning: Optional[FiLM] = None
        if apply_conditioning:
            if config.adaptive_film:
                self.conditioning = AdaptiveFiLM(config.d_model, config.d_model)
            else:
                self.conditioning = FiLM(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through encoder layer.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            src_mask: Attention mask.
            src_key_padding_mask: Padding mask.
            conditioning: Projected mood embedding (optional).

        Returns:
            Encoded features (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + self.dropout(attn_out)

        # FFN with pre-norm
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        # Optional conditioning
        if self.conditioning is not None and conditioning is not None:
            x = self.conditioning(x, conditioning)

        return x


# =============================================================================
# Decoder Layer
# =============================================================================

class DecoderLayer(nn.Module):
    """
    Single decoder layer with mood conditioning.

    Extended transformer decoder layer:
    1. Masked self-attention (causal)
    2. Cross-attention to encoder output
    3. Feed-forward network
    4. FiLM conditioning (after FFN)
    5. Optional cross-attention to mood

    The conditioning is applied after the main computation to ensure
    it modulates the learned representations rather than interfering
    with the attention mechanisms.

    Args:
        config: Transformer configuration.
        apply_conditioning: Whether to apply FiLM in this layer.
        apply_mood_cross_attention: Whether to apply mood cross-attention.
    """

    def __init__(
        self,
        config: TransformerConfig,
        apply_conditioning: bool = True,
        apply_mood_cross_attention: bool = False,
    ):
        super().__init__()

        self.apply_conditioning = apply_conditioning
        self.apply_mood_cross_attention = apply_mood_cross_attention

        # Masked self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Cross-attention to encoder
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

        # FiLM conditioning
        self.film: Optional[nn.Module] = None
        if apply_conditioning:
            if config.adaptive_film:
                self.film = AdaptiveFiLM(config.d_model, config.d_model)
            else:
                self.film = FiLM(config.d_model, config.d_model)

        # Optional mood cross-attention
        self.mood_cross_attn: Optional[CrossAttentionConditioning] = None
        if apply_mood_cross_attention:
            self.mood_cross_attn = CrossAttentionConditioning(
                d_model=config.d_model,
                n_heads=config.n_heads,
                conditioning_dim=config.d_model,
                dropout=config.dropout,
                n_conditioning_tokens=config.n_conditioning_tokens,
            )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through decoder layer.

        Args:
            x: Input tensor (batch, tgt_len, d_model)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Causal mask for self-attention.
            memory_mask: Mask for cross-attention.
            tgt_key_padding_mask: Target padding mask.
            memory_key_padding_mask: Memory padding mask.
            conditioning: Projected mood embedding.

        Returns:
            Decoded features (batch, tgt_len, d_model)
        """
        # Masked self-attention with pre-norm
        x_norm = self.norm1(x)
        self_attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = x + self.dropout(self_attn_out)

        # Cross-attention to encoder with pre-norm
        x_norm = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(
            x_norm, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        x = x + self.dropout(cross_attn_out)

        # FFN with pre-norm
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        # Apply FiLM conditioning
        # This modulates features AFTER the main transformer computation,
        # allowing mood to influence the learned representations without
        # disrupting attention patterns that encode musical structure.
        if self.film is not None and conditioning is not None:
            x = self.film(x, conditioning)

        # Apply mood cross-attention
        # This provides position-specific mood influence, complementing
        # the global FiLM modulation.
        if self.mood_cross_attn is not None and conditioning is not None:
            x = self.mood_cross_attn(x, conditioning)

        return x


# =============================================================================
# Full Encoder
# =============================================================================

class Encoder(nn.Module):
    """
    Transformer encoder stack.

    Processes input tokens into contextualized representations.
    These representations capture musical structure (pitches, rhythm)
    that should be preserved through mood conditioning.

    Args:
        config: Transformer configuration.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        # Build encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                config,
                apply_conditioning=(
                    config.condition_encoder and
                    (i + 1) % config.condition_every_n_layers == 0
                ),
            )
            for i in range(config.n_encoder_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode input sequence.

        Args:
            x: Embedded input (batch, seq_len, d_model)
            src_mask: Attention mask.
            src_key_padding_mask: Padding mask.
            conditioning: Projected mood (optional, usually None).

        Returns:
            Encoded representations (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, src_mask, src_key_padding_mask, conditioning)

        return self.norm(x)


# =============================================================================
# Full Decoder
# =============================================================================

class Decoder(nn.Module):
    """
    Transformer decoder stack with mood conditioning.

    The decoder generates output tokens while attending to:
    1. Previously generated tokens (causal self-attention)
    2. Encoder representations (cross-attention)
    3. Mood conditioning (FiLM + optional cross-attention)

    Conditioning is applied at regular intervals (controlled by
    condition_every_n_layers) to allow mood influence to accumulate
    through the network depth.

    Args:
        config: Transformer configuration.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        # Build decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                config,
                apply_conditioning=(i + 1) % config.condition_every_n_layers == 0,
                apply_mood_cross_attention=(
                    config.use_cross_attention and
                    (i + 1) % config.condition_every_n_layers == 0
                ),
            )
            for i in range(config.n_decoder_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        conditioning: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode with conditioning.

        Args:
            x: Embedded target (batch, tgt_len, d_model)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Causal mask.
            memory_mask: Cross-attention mask.
            tgt_key_padding_mask: Target padding mask.
            memory_key_padding_mask: Memory padding mask.
            conditioning: Projected mood embedding.

        Returns:
            Decoded representations (batch, tgt_len, d_model)
        """
        for layer in self.layers:
            x = layer(
                x, memory,
                tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask,
                conditioning,
            )

        return self.norm(x)


# =============================================================================
# Complete Model
# =============================================================================

class MoodConditionedMusicTransformer(nn.Module):
    """
    Complete mood-conditioned music transformer.

    This is the main model class combining all components:
    - Token embedding
    - Positional encoding
    - Encoder (structure extraction)
    - Decoder (conditioned generation)
    - Output projection

    The model takes tokenized music sequences and mood embeddings,
    producing modified sequences that express the target mood while
    preserving musical structure.

    Usage:
        config = TransformerConfig(vocab_size=256)
        model = MoodConditionedMusicTransformer(config)

        # Forward pass
        output = model(
            src_tokens,      # Input sequence
            tgt_tokens,      # Target sequence (teacher forcing)
            mood_embedding,  # Mood vector from embedder
        )

        # Generate
        generated = model.generate(
            src_tokens,
            mood_embedding,
            max_length=128,
        )

    Args:
        config: Transformer configuration.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        # Token embeddings (shared between encoder and decoder)
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout,
        )

        # Encoder and decoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Mood projection
        self.mood_projection = MoodProjection(
            input_dim=config.mood_embedding_dim,
            output_dim=config.d_model,
            hidden_dim=config.d_model,
            dropout=config.dropout,
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        # Share weights between embedding and output projection
        # This is a common technique that improves performance
        self.output_projection.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

        # Causal mask cache
        self._causal_mask_cache: Dict[int, Tensor] = {}

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def _get_causal_mask(self, size: int, device: torch.device) -> Tensor:
        """
        Get or create causal attention mask.

        Args:
            size: Sequence length.
            device: Target device.

        Returns:
            Causal mask tensor.
        """
        if size not in self._causal_mask_cache:
            # Create causal mask (True = masked, False = attend)
            mask = torch.triu(
                torch.ones(size, size, dtype=torch.bool, device=device),
                diagonal=1,
            )
            self._causal_mask_cache[size] = mask

        return self._causal_mask_cache[size].to(device)

    def _create_padding_mask(self, tokens: Tensor) -> Tensor:
        """
        Create padding mask from token IDs.

        Args:
            tokens: Token tensor (batch, seq_len)

        Returns:
            Padding mask (batch, seq_len) - True where padded.
        """
        return tokens == self.config.pad_token_id

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        mood_embedding: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for training.

        Args:
            src: Source token IDs (batch, src_len)
            tgt: Target token IDs (batch, tgt_len)
            mood_embedding: Mood vectors (batch, mood_dim)
            src_padding_mask: Source padding mask (optional).
            tgt_padding_mask: Target padding mask (optional).

        Returns:
            Logits over vocabulary (batch, tgt_len, vocab_size)
        """
        # Create padding masks if not provided
        if src_padding_mask is None:
            src_padding_mask = self._create_padding_mask(src)
        if tgt_padding_mask is None:
            tgt_padding_mask = self._create_padding_mask(tgt)

        # Get causal mask for decoder
        tgt_len = tgt.size(1)
        tgt_mask = self._get_causal_mask(tgt_len, tgt.device)

        # Project mood embedding
        conditioning = self.mood_projection(mood_embedding)

        # Embed and encode source
        src_embedded = self.token_embedding(src)
        src_embedded = self.pos_encoding(src_embedded)
        memory = self.encoder(
            src_embedded,
            src_key_padding_mask=src_padding_mask,
            conditioning=conditioning if self.config.condition_encoder else None,
        )

        # Embed and decode target
        tgt_embedded = self.token_embedding(tgt)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        decoded = self.decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            conditioning=conditioning,
        )

        # Project to vocabulary
        logits = self.output_projection(decoded)

        return logits

    def encode(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        mood_embedding: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Encode source sequence.

        Separate encode step for generation.

        Args:
            src: Source token IDs (batch, src_len)
            src_padding_mask: Padding mask.
            mood_embedding: Optional mood for encoder conditioning.

        Returns:
            Tuple of (memory, projected_conditioning)
        """
        if src_padding_mask is None:
            src_padding_mask = self._create_padding_mask(src)

        # Project mood if provided
        conditioning = None
        if mood_embedding is not None:
            conditioning = self.mood_projection(mood_embedding)

        # Embed and encode
        src_embedded = self.token_embedding(src)
        src_embedded = self.pos_encoding(src_embedded)
        memory = self.encoder(
            src_embedded,
            src_key_padding_mask=src_padding_mask,
            conditioning=conditioning if self.config.condition_encoder else None,
        )

        return memory, conditioning

    def decode_step(
        self,
        tgt: Tensor,
        memory: Tensor,
        conditioning: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Single decoding step for generation.

        Args:
            tgt: Current target sequence (batch, tgt_len)
            memory: Encoder output.
            conditioning: Projected mood.
            memory_padding_mask: Memory padding mask.

        Returns:
            Logits for next token (batch, tgt_len, vocab_size)
        """
        tgt_len = tgt.size(1)
        tgt_mask = self._get_causal_mask(tgt_len, tgt.device)

        # Embed target
        tgt_embedded = self.token_embedding(tgt)
        tgt_embedded = self.pos_encoding(tgt_embedded)

        # Decode
        decoded = self.decoder(
            tgt_embedded,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_padding_mask,
            conditioning=conditioning,
        )

        # Project to vocabulary
        return self.output_projection(decoded)

    @torch.no_grad()
    def generate(
        self,
        src: Tensor,
        mood_embedding: Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        src_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Generate mood-conditioned sequence.

        Uses autoregressive decoding with optional temperature,
        top-k, and top-p (nucleus) sampling.

        Args:
            src: Source token IDs (batch, src_len)
            mood_embedding: Mood vectors (batch, mood_dim)
            max_length: Maximum generation length.
            temperature: Sampling temperature (1.0 = neutral).
            top_k: Keep only top-k tokens (optional).
            top_p: Nucleus sampling threshold (optional).
            src_padding_mask: Source padding mask.

        Returns:
            Generated token IDs (batch, gen_len)
        """
        batch_size = src.size(0)
        device = src.device

        # Encode source
        memory, conditioning = self.encode(src, src_padding_mask, mood_embedding)

        # Initialize with BOS token
        generated = torch.full(
            (batch_size, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=device,
        )

        # Track which sequences are done
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            # Get logits for next token
            logits = self.decode_step(
                generated,
                memory,
                conditioning,
                src_padding_mask,
            )

            # Get logits for the last position
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(probs, dim=-1)

                # Remove tokens with cumulative prob above threshold
                remove_mask = cumsum_probs > top_p
                remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                remove_mask[:, 0] = False

                sorted_logits[remove_mask] = float("-inf")

                # Restore original order
                next_logits = sorted_logits.gather(-1, sorted_indices.argsort(-1))

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Don't update done sequences
            next_token = torch.where(
                done.unsqueeze(-1),
                torch.full_like(next_token, self.config.pad_token_id),
                next_token,
            )

            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)

            # Update done status
            done = done | (next_token.squeeze(-1) == self.config.eos_token_id)

            # Stop if all sequences are done
            if done.all():
                break

        return generated


# =============================================================================
# Layer Freezing Utilities
# =============================================================================

def freeze_encoder(model: MoodConditionedMusicTransformer) -> None:
    """
    Freeze all encoder parameters.

    Useful for fine-tuning only the decoder and conditioning
    while preserving learned structure representations.

    Args:
        model: Model to modify.
    """
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.token_embedding.parameters():
        param.requires_grad = False


def freeze_encoder_layers(
    model: MoodConditionedMusicTransformer,
    n_layers: int,
) -> None:
    """
    Freeze the first N encoder layers.

    Allows later layers to adapt while preserving
    low-level representations.

    Args:
        model: Model to modify.
        n_layers: Number of layers to freeze (from the bottom).
    """
    for i, layer in enumerate(model.encoder.layers):
        if i < n_layers:
            for param in layer.parameters():
                param.requires_grad = False


def freeze_decoder_base(model: MoodConditionedMusicTransformer) -> None:
    """
    Freeze decoder except conditioning modules.

    Keeps FiLM and cross-attention trainable while freezing
    the core transformer layers.

    Args:
        model: Model to modify.
    """
    for layer in model.decoder.layers:
        # Freeze attention and FFN
        for param in layer.self_attn.parameters():
            param.requires_grad = False
        for param in layer.cross_attn.parameters():
            param.requires_grad = False
        for param in layer.ffn.parameters():
            param.requires_grad = False
        for param in layer.norm1.parameters():
            param.requires_grad = False
        for param in layer.norm2.parameters():
            param.requires_grad = False
        for param in layer.norm3.parameters():
            param.requires_grad = False

        # Keep conditioning trainable
        if layer.film is not None:
            for param in layer.film.parameters():
                param.requires_grad = True
        if layer.mood_cross_attn is not None:
            for param in layer.mood_cross_attn.parameters():
                param.requires_grad = True


def unfreeze_all(model: MoodConditionedMusicTransformer) -> None:
    """
    Unfreeze all model parameters.

    Args:
        model: Model to modify.
    """
    for param in model.parameters():
        param.requires_grad = True


def get_trainable_parameters(
    model: MoodConditionedMusicTransformer,
) -> List[nn.Parameter]:
    """
    Get list of trainable parameters.

    Args:
        model: Model to inspect.

    Returns:
        List of parameters with requires_grad=True.
    """
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(
    model: MoodConditionedMusicTransformer,
) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: Model to analyze.

    Returns:
        Dictionary with parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


# =============================================================================
# Factory Functions
# =============================================================================

def create_small_model(vocab_size: int = 256) -> MoodConditionedMusicTransformer:
    """
    Create a small model for fast experimentation.

    ~2M parameters, suitable for CPU training.

    Args:
        vocab_size: Vocabulary size.

    Returns:
        Configured model.
    """
    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=256,
        n_encoder_layers=3,
        n_decoder_layers=4,
        n_heads=4,
        d_ff=1024,
        mood_embedding_dim=384,
        use_cross_attention=False,
        adaptive_film=True,
    )
    return MoodConditionedMusicTransformer(config)


def create_medium_model(vocab_size: int = 256) -> MoodConditionedMusicTransformer:
    """
    Create a medium model for balanced performance.

    ~8M parameters, suitable for GPU training.

    Args:
        vocab_size: Vocabulary size.

    Returns:
        Configured model.
    """
    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=384,
        n_encoder_layers=4,
        n_decoder_layers=6,
        n_heads=6,
        d_ff=1536,
        mood_embedding_dim=384,
        use_cross_attention=True,
        adaptive_film=True,
    )
    return MoodConditionedMusicTransformer(config)


def create_large_model(vocab_size: int = 256) -> MoodConditionedMusicTransformer:
    """
    Create a larger model for best quality.

    ~20M parameters, requires GPU.

    Args:
        vocab_size: Vocabulary size.

    Returns:
        Configured model.
    """
    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=512,
        n_encoder_layers=6,
        n_decoder_layers=8,
        n_heads=8,
        d_ff=2048,
        mood_embedding_dim=384,
        use_cross_attention=True,
        adaptive_film=True,
        n_conditioning_tokens=8,
    )
    return MoodConditionedMusicTransformer(config)
