"""
Mood conditioning modules for the music transformer.

This module implements Feature-wise Linear Modulation (FiLM) and related
conditioning mechanisms to inject mood information into the transformer.

Conditioning Philosophy:
------------------------

The goal is to modify musical EXPRESSION while preserving STRUCTURE.

Structure (should NOT change):
- Note pitches (which notes are played)
- Basic rhythm (when notes occur)
- Harmonic progression (chord sequence)
- Overall form (intro, verse, chorus, etc.)

Expression (CAN change):
- Velocity/dynamics (how loud/soft)
- Articulation (staccato vs legato feel)
- Timing micro-variations (slight rushes/drags)
- Note density (within rhythmic framework)

FiLM Conditioning:
-----------------

FiLM (Feature-wise Linear Modulation) modulates intermediate features
using learned scale (γ) and shift (β) parameters derived from the
conditioning signal (mood embedding).

    output = γ * input + β

Where:
- γ (gamma): Learned scaling, allows amplifying/dampening features
- β (beta): Learned bias, allows shifting feature distributions
- Both are derived from the mood embedding via linear projection

Why FiLM for Music:
- Non-destructive: Original information preserved, just modulated
- Expressive: Can emphasize or suppress different musical features
- Learnable: Network learns which features to modulate for each mood
- Efficient: Minimal parameter overhead

Cross-Attention Conditioning:
----------------------------

Optional cross-attention allows the decoder to directly attend to
mood information, providing a complementary conditioning path.

- FiLM: Global modulation of all features
- Cross-Attention: Selective attention to mood aspects

Combined, these allow nuanced mood expression.

References:
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer"
- Dhariwal et al., "Jukebox: A Generative Model for Music"
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# FiLM Layer
# =============================================================================

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Applies affine transformation to input features based on conditioning:
        output = gamma * input + beta

    The gamma and beta parameters are computed from the conditioning signal
    (mood embedding) via learned linear projections.

    This is the core mechanism for mood conditioning - it allows the mood
    to scale and shift feature activations without destroying the underlying
    musical structure encoded in those features.

    Args:
        feature_dim: Dimension of input features to modulate.
        conditioning_dim: Dimension of conditioning signal (mood embedding).

    Shape:
        - Input: (batch, seq_len, feature_dim)
        - Conditioning: (batch, conditioning_dim)
        - Output: (batch, seq_len, feature_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        conditioning_dim: int,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.conditioning_dim = conditioning_dim

        # Project conditioning to gamma (scale) parameters
        # Initialize to output ~1.0 (identity scaling)
        self.gamma_proj = nn.Linear(conditioning_dim, feature_dim)

        # Project conditioning to beta (shift) parameters
        # Initialize to output ~0.0 (no shift)
        self.beta_proj = nn.Linear(conditioning_dim, feature_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights for stable training.

        Gamma initialized to produce ~1.0 (identity)
        Beta initialized to produce ~0.0 (no shift)
        """
        # Small weights, zero bias for gamma → outputs ~0, then +1 = 1
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)

        # Small weights, zero bias for beta → outputs ~0
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
    ) -> Tensor:
        """
        Apply FiLM conditioning to input features.

        Args:
            x: Input features (batch, seq_len, feature_dim)
            conditioning: Mood embedding (batch, conditioning_dim)

        Returns:
            Modulated features (batch, seq_len, feature_dim)
        """
        # Compute scale and shift from conditioning
        # gamma: add 1 so default is identity (1 * x + 0 = x)
        gamma = self.gamma_proj(conditioning) + 1.0  # (batch, feature_dim)
        beta = self.beta_proj(conditioning)  # (batch, feature_dim)

        # Expand for broadcasting over sequence dimension
        # (batch, feature_dim) -> (batch, 1, feature_dim)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # Apply FiLM: element-wise scale and shift
        return gamma * x + beta


class AdaptiveFiLM(nn.Module):
    """
    Adaptive FiLM with learned gating.

    Extends FiLM with a learnable gate that controls how much
    conditioning affects the output. This allows the network to
    learn when conditioning should have strong vs. weak influence.

    gate = sigmoid(linear(conditioning))
    output = gate * (gamma * x + beta) + (1 - gate) * x

    When gate → 0: output ≈ x (no conditioning effect)
    When gate → 1: output = gamma * x + beta (full conditioning)

    This is useful for:
    - Allowing some layers to be unaffected by mood
    - Learning which features are mood-dependent vs. mood-invariant
    - Smoother interpolation between conditioned/unconditioned

    Args:
        feature_dim: Dimension of input features.
        conditioning_dim: Dimension of conditioning signal.
        init_gate_bias: Initial bias for gate (negative = less conditioning)
    """

    def __init__(
        self,
        feature_dim: int,
        conditioning_dim: int,
        init_gate_bias: float = 0.0,
    ):
        super().__init__()

        self.film = FiLM(feature_dim, conditioning_dim)

        # Gate projection: determines how much conditioning to apply
        self.gate_proj = nn.Linear(conditioning_dim, 1)

        # Initialize gate bias
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_gate_bias)

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
    ) -> Tensor:
        """
        Apply gated FiLM conditioning.

        Args:
            x: Input features (batch, seq_len, feature_dim)
            conditioning: Mood embedding (batch, conditioning_dim)

        Returns:
            Conditionally modulated features.
        """
        # Compute gate value
        gate = torch.sigmoid(self.gate_proj(conditioning))  # (batch, 1)
        gate = gate.unsqueeze(1)  # (batch, 1, 1)

        # Apply FiLM
        conditioned = self.film(x, conditioning)

        # Blend based on gate
        return gate * conditioned + (1 - gate) * x


# =============================================================================
# Mood Projection
# =============================================================================

class MoodProjection(nn.Module):
    """
    Projects mood embedding to transformer-compatible conditioning.

    The raw mood embedding (e.g., 384-dim from sentence transformer)
    may not match the transformer's hidden dimension. This module:
    1. Projects to the required dimension
    2. Optionally applies normalization
    3. Optionally applies non-linearity for richer representations

    Args:
        input_dim: Dimension of mood embedding (e.g., 384).
        output_dim: Dimension required by transformer (e.g., 512).
        hidden_dim: Optional hidden layer dimension for MLP.
        dropout: Dropout probability.
        normalize: Whether to L2-normalize output.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        normalize: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize

        if hidden_dim is None:
            # Simple linear projection
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            # MLP projection for richer transformation
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, mood_embedding: Tensor) -> Tensor:
        """
        Project mood embedding to conditioning vector.

        Args:
            mood_embedding: Raw mood embedding (batch, input_dim)

        Returns:
            Projected conditioning (batch, output_dim)
        """
        x = self.projection(mood_embedding)
        x = self.dropout(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x


# =============================================================================
# Cross-Attention Conditioning
# =============================================================================

class CrossAttentionConditioning(nn.Module):
    """
    Cross-attention module for decoder-side mood conditioning.

    Allows decoder tokens to attend to mood information, providing
    a complementary conditioning path to FiLM.

    How it works:
    - Query: Decoder hidden states (what we're generating)
    - Key/Value: Mood embedding (expanded to pseudo-sequence)

    This lets each position in the output sequence selectively
    attend to the mood conditioning, potentially allowing different
    parts of the music to emphasize different mood aspects.

    Architecture:
    - Multi-head attention with decoder states as Q, mood as K/V
    - Optional learned positional embedding for mood
    - Residual connection and layer norm

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        conditioning_dim: Dimension of conditioning signal.
        dropout: Dropout probability.
        n_conditioning_tokens: Number of pseudo-tokens for mood.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        conditioning_dim: int,
        dropout: float = 0.1,
        n_conditioning_tokens: int = 4,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_conditioning_tokens = n_conditioning_tokens

        # Project conditioning to multiple tokens
        # This creates a "pseudo-sequence" for the mood to attend to
        self.conditioning_expand = nn.Linear(
            conditioning_dim,
            d_model * n_conditioning_tokens,
        )

        # Learned positional embeddings for conditioning tokens
        self.conditioning_pos = nn.Parameter(
            torch.randn(1, n_conditioning_tokens, d_model) * 0.02
        )

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply cross-attention conditioning.

        Args:
            x: Decoder hidden states (batch, seq_len, d_model)
            conditioning: Mood conditioning (batch, conditioning_dim)
            key_padding_mask: Optional mask for conditioning tokens.

        Returns:
            Conditioned hidden states (batch, seq_len, d_model)
        """
        batch_size = x.size(0)

        # Expand conditioning to multiple tokens
        # (batch, conditioning_dim) -> (batch, n_tokens * d_model)
        cond_expanded = self.conditioning_expand(conditioning)

        # Reshape to (batch, n_tokens, d_model)
        cond_tokens = cond_expanded.view(
            batch_size,
            self.n_conditioning_tokens,
            self.d_model,
        )

        # Add positional embeddings
        cond_tokens = cond_tokens + self.conditioning_pos

        # Cross-attention: decoder attends to conditioning
        attended, _ = self.cross_attention(
            query=x,
            key=cond_tokens,
            value=cond_tokens,
            key_padding_mask=key_padding_mask,
        )

        # Residual connection and norm
        x = self.norm(x + self.dropout(attended))

        return x


# =============================================================================
# Combined Conditioning Module
# =============================================================================

class MoodConditioner(nn.Module):
    """
    Combined mood conditioning module.

    Integrates FiLM and optional cross-attention conditioning into
    a single module that can be used within transformer layers.

    This module:
    1. Projects raw mood embedding to model dimension
    2. Applies FiLM modulation to features
    3. Optionally applies cross-attention conditioning

    The combination allows for both:
    - Global feature modulation (FiLM)
    - Position-specific mood attention (cross-attention)

    Args:
        d_model: Model dimension.
        mood_embedding_dim: Dimension of mood embedding.
        use_cross_attention: Whether to include cross-attention.
        n_heads: Number of attention heads (if using cross-attention).
        n_conditioning_tokens: Conditioning tokens for cross-attention.
        adaptive_gate: Whether to use adaptive gating in FiLM.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        mood_embedding_dim: int,
        use_cross_attention: bool = False,
        n_heads: int = 4,
        n_conditioning_tokens: int = 4,
        adaptive_gate: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.mood_embedding_dim = mood_embedding_dim
        self.use_cross_attention = use_cross_attention

        # Project mood embedding to model dimension
        self.mood_projection = MoodProjection(
            input_dim=mood_embedding_dim,
            output_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout,
        )

        # FiLM conditioning (always used)
        if adaptive_gate:
            self.film = AdaptiveFiLM(d_model, d_model)
        else:
            self.film = FiLM(d_model, d_model)

        # Optional cross-attention
        self.cross_attention: Optional[CrossAttentionConditioning] = None
        if use_cross_attention:
            self.cross_attention = CrossAttentionConditioning(
                d_model=d_model,
                n_heads=n_heads,
                conditioning_dim=d_model,
                dropout=dropout,
                n_conditioning_tokens=n_conditioning_tokens,
            )

    def forward(
        self,
        x: Tensor,
        mood_embedding: Tensor,
        projected_mood: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply mood conditioning to features.

        Args:
            x: Input features (batch, seq_len, d_model)
            mood_embedding: Raw mood embedding (batch, mood_embedding_dim)
            projected_mood: Pre-projected mood (optional, for efficiency)

        Returns:
            Tuple of (conditioned features, projected mood for reuse)
        """
        # Project mood if not already done
        if projected_mood is None:
            projected_mood = self.mood_projection(mood_embedding)

        # Apply FiLM modulation
        x = self.film(x, projected_mood)

        # Apply cross-attention if enabled
        if self.cross_attention is not None:
            x = self.cross_attention(x, projected_mood)

        return x, projected_mood


# =============================================================================
# Utility Functions
# =============================================================================

def interpolate_conditioning(
    cond1: Tensor,
    cond2: Tensor,
    alpha: float,
) -> Tensor:
    """
    Linearly interpolate between two conditioning vectors.

    Useful for smooth mood transitions.

    Args:
        cond1: First conditioning vector.
        cond2: Second conditioning vector.
        alpha: Interpolation factor (0 = cond1, 1 = cond2).

    Returns:
        Interpolated conditioning.
    """
    return (1 - alpha) * cond1 + alpha * cond2


def blend_multiple_moods(
    conditions: list[Tensor],
    weights: list[float],
) -> Tensor:
    """
    Blend multiple mood conditioning vectors.

    Allows for complex mood combinations (e.g., "70% melancholic, 30% hopeful").

    Args:
        conditions: List of conditioning vectors.
        weights: List of blend weights (should sum to 1).

    Returns:
        Blended conditioning vector.
    """
    if len(conditions) != len(weights):
        raise ValueError("Number of conditions must match number of weights")

    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    # Weighted sum
    result = torch.zeros_like(conditions[0])
    for cond, weight in zip(conditions, weights):
        result = result + weight * cond

    return result
