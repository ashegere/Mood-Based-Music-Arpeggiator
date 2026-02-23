"""
Loss functions for mood-conditioned music transformer training.

This module implements specialized losses for music generation:

1. Token Classification Loss (Cross-Entropy)
   - Standard next-token prediction loss
   - Handles padding masking

2. Pitch Classification Loss
   - Focused loss on pitch tokens
   - Higher weight for pitch accuracy

3. Velocity Regression Loss
   - Treats velocity as continuous value
   - Smoother gradients than classification

4. Smoothness Regularization
   - Penalizes abrupt changes in velocity/dynamics
   - Encourages musical phrasing

5. Expression Preservation Loss
   - Ensures expressive outputs maintain structure
   - Pitch tokens should match input

Design Philosophy:
-----------------

The loss function balances multiple objectives:

1. Reconstruction Accuracy
   - Model must predict correct tokens
   - Pitch accuracy is critical (wrong notes sound bad)

2. Expression Quality
   - Velocity predictions should be smooth
   - Dynamics should have musical phrasing

3. Structure Preservation
   - Pitch sequence should not change with mood
   - Only expressive attributes should vary
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Token Type Ranges (must match tokenization.py)
# =============================================================================

# These ranges define which token IDs correspond to which musical attributes
# Adjust based on your actual vocabulary structure

@dataclass
class TokenRanges:
    """Token ID ranges for different token types."""
    # Special tokens
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    sep_id: int = 3
    unk_id: int = 4

    # Pitch tokens: PITCH_36 to PITCH_96 (61 tokens)
    pitch_start: int = 123  # Adjust based on vocab
    pitch_end: int = 184

    # Velocity tokens: VEL_16 to VEL_127 (8 tokens)
    velocity_start: int = 195
    velocity_end: int = 203

    # Duration tokens: DUR_* (10 tokens)
    duration_start: int = 185
    duration_end: int = 195

    # Position tokens: POS_0 to POS_15 (16 tokens)
    position_start: int = 203
    position_end: int = 219


# Default token ranges
DEFAULT_RANGES = TokenRanges()


# =============================================================================
# Basic Classification Loss
# =============================================================================

class TokenClassificationLoss(nn.Module):
    """
    Standard cross-entropy loss for token classification.

    Handles padding masking and optional label smoothing.

    Args:
        pad_token_id: Token ID for padding (ignored in loss).
        label_smoothing: Label smoothing factor (0 = none).
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute classification loss.

        Args:
            logits: Model output (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)

        Returns:
            Scalar loss tensor.
        """
        # Reshape for cross-entropy
        # (batch * seq_len, vocab_size) vs (batch * seq_len)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        return self.loss_fn(logits_flat, targets_flat)


# =============================================================================
# Pitch Classification Loss
# =============================================================================

class PitchClassificationLoss(nn.Module):
    """
    Focused loss on pitch token predictions.

    Applies higher weight to pitch tokens to ensure
    melodic accuracy.

    Args:
        token_ranges: Token range configuration.
        pitch_weight: Weight multiplier for pitch tokens.
        pad_token_id: Token ID for padding.
    """

    def __init__(
        self,
        token_ranges: TokenRanges = DEFAULT_RANGES,
        pitch_weight: float = 2.0,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.ranges = token_ranges
        self.pitch_weight = pitch_weight
        self.pad_token_id = pad_token_id

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute pitch-weighted classification loss.

        Args:
            logits: Model output (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)

        Returns:
            Scalar loss tensor.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Compute per-token cross-entropy (no reduction)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # Compute loss per position
        loss_per_token = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="none",
            ignore_index=self.pad_token_id,
        )

        # Identify pitch tokens in targets
        is_pitch = (targets_flat >= self.ranges.pitch_start) & \
                   (targets_flat < self.ranges.pitch_end)

        # Apply weight to pitch tokens
        weights = torch.ones_like(loss_per_token)
        weights[is_pitch] = self.pitch_weight

        # Weighted mean (excluding padding)
        valid_mask = targets_flat != self.pad_token_id
        weighted_loss = loss_per_token * weights
        loss = weighted_loss[valid_mask].mean()

        return loss


# =============================================================================
# Velocity Regression Loss
# =============================================================================

class VelocityRegressionLoss(nn.Module):
    """
    Regression loss for velocity prediction.

    Treats velocity as a continuous value rather than
    discrete classes, providing smoother gradients.

    Two modes:
    1. Soft targets: Use softmax probabilities
    2. Direct regression: Map logits to velocity value

    Args:
        token_ranges: Token range configuration.
        velocity_values: Actual velocity values for each token.
        regression_weight: Weight for regression component.
    """

    def __init__(
        self,
        token_ranges: TokenRanges = DEFAULT_RANGES,
        velocity_values: Optional[List[int]] = None,
        regression_weight: float = 1.0,
    ):
        super().__init__()
        self.ranges = token_ranges
        self.regression_weight = regression_weight

        # Default velocity bin values (from tokenization.py)
        if velocity_values is None:
            velocity_values = [16, 32, 48, 64, 80, 96, 112, 127]

        # Register as buffer for device handling
        self.register_buffer(
            "velocity_values",
            torch.tensor(velocity_values, dtype=torch.float),
        )
        self.n_velocity_tokens = len(velocity_values)

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute velocity regression loss.

        Uses expected value of velocity distribution.

        Args:
            logits: Model output (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)

        Returns:
            Scalar loss tensor.
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # Find positions where target is a velocity token
        vel_start = self.ranges.velocity_start
        vel_end = self.ranges.velocity_end

        is_velocity = (targets >= vel_start) & (targets < vel_end)

        if not is_velocity.any():
            return torch.tensor(0.0, device=device)

        # Extract velocity logits (only velocity token columns)
        velocity_logits = logits[:, :, vel_start:vel_end]  # (batch, seq, n_vel)

        # Compute expected velocity from logits
        velocity_probs = F.softmax(velocity_logits, dim=-1)
        predicted_velocity = torch.sum(
            velocity_probs * self.velocity_values.view(1, 1, -1),
            dim=-1,
        )  # (batch, seq)

        # Get target velocity values
        target_velocity_idx = targets - vel_start  # Offset to 0-based index
        target_velocity_idx = target_velocity_idx.clamp(0, self.n_velocity_tokens - 1)
        target_velocity = self.velocity_values[target_velocity_idx.long()]

        # Compute MSE on velocity positions only
        mse = F.mse_loss(
            predicted_velocity[is_velocity],
            target_velocity[is_velocity],
        )

        # Normalize by velocity range for consistent scale
        normalized_mse = mse / (127.0 ** 2)

        return self.regression_weight * normalized_mse


# =============================================================================
# Smoothness Regularization
# =============================================================================

class SmoothnessRegularization(nn.Module):
    """
    Regularization for smooth velocity/expression contours.

    Penalizes large differences between consecutive velocity
    predictions, encouraging musical phrasing.

    The smoothness is computed on the expected velocity values
    (not raw logits) to be musically meaningful.

    Args:
        token_ranges: Token range configuration.
        velocity_values: Velocity bin values.
        smoothness_weight: Weight for smoothness penalty.
        max_jump: Maximum allowed velocity jump (no penalty below this).
    """

    def __init__(
        self,
        token_ranges: TokenRanges = DEFAULT_RANGES,
        velocity_values: Optional[List[int]] = None,
        smoothness_weight: float = 0.1,
        max_jump: float = 20.0,
    ):
        super().__init__()
        self.ranges = token_ranges
        self.smoothness_weight = smoothness_weight
        self.max_jump = max_jump

        if velocity_values is None:
            velocity_values = [16, 32, 48, 64, 80, 96, 112, 127]

        self.register_buffer(
            "velocity_values",
            torch.tensor(velocity_values, dtype=torch.float),
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute smoothness regularization.

        Args:
            logits: Model output (batch, seq_len, vocab_size)
            targets: Target token IDs (for identifying velocity positions)

        Returns:
            Scalar smoothness penalty.
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        if seq_len < 2:
            return torch.tensor(0.0, device=device)

        vel_start = self.ranges.velocity_start
        vel_end = self.ranges.velocity_end

        # Extract velocity logits and compute expected values
        velocity_logits = logits[:, :, vel_start:vel_end]
        velocity_probs = F.softmax(velocity_logits, dim=-1)
        predicted_velocity = torch.sum(
            velocity_probs * self.velocity_values.view(1, 1, -1),
            dim=-1,
        )  # (batch, seq)

        # Identify consecutive velocity token positions
        is_velocity = (targets >= vel_start) & (targets < vel_end)

        # Compute differences between consecutive positions
        velocity_diff = torch.abs(predicted_velocity[:, 1:] - predicted_velocity[:, :-1])

        # Only penalize jumps at consecutive velocity positions
        # Both current and previous should be velocity tokens
        consecutive_vel = is_velocity[:, 1:] & is_velocity[:, :-1]

        if not consecutive_vel.any():
            return torch.tensor(0.0, device=device)

        # Apply soft penalty (ReLU to allow some variation)
        penalty = F.relu(velocity_diff - self.max_jump)
        smoothness_loss = penalty[consecutive_vel].mean()

        return self.smoothness_weight * smoothness_loss


# =============================================================================
# Expression Preservation Loss
# =============================================================================

class ExpressionPreservationLoss(nn.Module):
    """
    Loss to ensure pitch structure is preserved.

    During mood conditioning (Phase 2), we want:
    - Pitch tokens to remain the same as input
    - Only velocity/dynamics to change

    This loss penalizes deviations in pitch predictions
    from the source sequence.

    Args:
        token_ranges: Token range configuration.
        preservation_weight: Weight for preservation penalty.
    """

    def __init__(
        self,
        token_ranges: TokenRanges = DEFAULT_RANGES,
        preservation_weight: float = 1.0,
    ):
        super().__init__()
        self.ranges = token_ranges
        self.preservation_weight = preservation_weight

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        source: Tensor,
    ) -> Tensor:
        """
        Compute structure preservation loss.

        Ensures pitch predictions match source pitches.

        Args:
            logits: Model output (batch, seq_len, vocab_size)
            targets: Target token IDs
            source: Source token IDs (neutral input)

        Returns:
            Scalar preservation loss.
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        pitch_start = self.ranges.pitch_start
        pitch_end = self.ranges.pitch_end

        # Find pitch positions in source
        is_pitch_in_source = (source >= pitch_start) & (source < pitch_end)

        if not is_pitch_in_source.any():
            return torch.tensor(0.0, device=device)

        # At pitch positions, the prediction should match source
        logits_flat = logits.view(-1, vocab_size)
        source_flat = source.view(-1)
        is_pitch_flat = is_pitch_in_source.view(-1)

        # Compute cross-entropy against SOURCE pitches (not target)
        # This ensures the model preserves input structure
        loss_per_token = F.cross_entropy(
            logits_flat,
            source_flat,
            reduction="none",
        )

        preservation_loss = loss_per_token[is_pitch_flat].mean()

        return self.preservation_weight * preservation_loss


# =============================================================================
# Combined Loss
# =============================================================================

class MoodConditionedLoss(nn.Module):
    """
    Combined loss for mood-conditioned music generation.

    Aggregates multiple loss components:
    1. Token classification (main loss)
    2. Pitch accuracy (weighted classification)
    3. Velocity regression (smooth dynamics)
    4. Smoothness regularization (phrasing)
    5. Expression preservation (structure integrity)

    Different phases use different loss combinations:
    - Phase 1: Classification only (reconstruction)
    - Phase 2: All losses (conditioning)

    Args:
        token_ranges: Token range configuration.
        classification_weight: Weight for token classification.
        pitch_weight: Extra weight for pitch tokens.
        velocity_weight: Weight for velocity regression.
        smoothness_weight: Weight for smoothness regularization.
        preservation_weight: Weight for structure preservation.
        label_smoothing: Label smoothing factor.
    """

    def __init__(
        self,
        token_ranges: TokenRanges = DEFAULT_RANGES,
        classification_weight: float = 1.0,
        pitch_weight: float = 2.0,
        velocity_weight: float = 0.5,
        smoothness_weight: float = 0.1,
        preservation_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.classification_weight = classification_weight
        self.velocity_weight = velocity_weight
        self.smoothness_weight = smoothness_weight
        self.preservation_weight = preservation_weight

        # Component losses
        self.token_loss = TokenClassificationLoss(
            pad_token_id=token_ranges.pad_id,
            label_smoothing=label_smoothing,
        )

        self.pitch_loss = PitchClassificationLoss(
            token_ranges=token_ranges,
            pitch_weight=pitch_weight,
        )

        self.velocity_loss = VelocityRegressionLoss(
            token_ranges=token_ranges,
            regression_weight=velocity_weight,
        )

        self.smoothness_loss = SmoothnessRegularization(
            token_ranges=token_ranges,
            smoothness_weight=smoothness_weight,
        )

        self.preservation_loss = ExpressionPreservationLoss(
            token_ranges=token_ranges,
            preservation_weight=preservation_weight,
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        source: Optional[Tensor] = None,
        phase: int = 2,
        return_components: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Compute combined loss.

        Args:
            logits: Model output (batch, seq_len, vocab_size)
            targets: Target token IDs (batch, seq_len)
            source: Source token IDs (for preservation loss)
            phase: Training phase (1 or 2)
            return_components: Whether to return individual losses.

        Returns:
            Tuple of (total_loss, component_dict or None)
        """
        components = {}

        # Always compute classification loss
        cls_loss = self.token_loss(logits, targets)
        components["classification"] = cls_loss

        # Phase 1: Only classification (reconstruction)
        if phase == 1:
            total = self.classification_weight * cls_loss
            if return_components:
                return total, components
            return total, None

        # Phase 2: Full loss suite
        # Pitch accuracy
        pitch_loss = self.pitch_loss(logits, targets)
        components["pitch"] = pitch_loss

        # Velocity regression
        vel_loss = self.velocity_loss(logits, targets)
        components["velocity"] = vel_loss

        # Smoothness
        smooth_loss = self.smoothness_loss(logits, targets)
        components["smoothness"] = smooth_loss

        # Structure preservation (if source provided)
        if source is not None:
            preserve_loss = self.preservation_loss(logits, targets, source)
            components["preservation"] = preserve_loss
        else:
            preserve_loss = torch.tensor(0.0, device=logits.device)
            components["preservation"] = preserve_loss

        # Combine all losses
        total = (
            self.classification_weight * cls_loss +
            pitch_loss +  # Already weighted internally
            vel_loss +    # Already weighted internally
            smooth_loss + # Already weighted internally
            preserve_loss # Already weighted internally
        )

        components["total"] = total

        if return_components:
            return total, components

        return total, None


# =============================================================================
# Utility Functions
# =============================================================================

def compute_accuracy(
    logits: Tensor,
    targets: Tensor,
    pad_token_id: int = 0,
) -> Tensor:
    """
    Compute token prediction accuracy.

    Args:
        logits: Model output (batch, seq_len, vocab_size)
        targets: Target token IDs (batch, seq_len)
        pad_token_id: Token ID to ignore.

    Returns:
        Accuracy as a scalar tensor.
    """
    predictions = logits.argmax(dim=-1)
    mask = targets != pad_token_id
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy


def compute_pitch_accuracy(
    logits: Tensor,
    targets: Tensor,
    token_ranges: TokenRanges = DEFAULT_RANGES,
) -> Tensor:
    """
    Compute accuracy specifically for pitch tokens.

    Args:
        logits: Model output.
        targets: Target token IDs.
        token_ranges: Token range configuration.

    Returns:
        Pitch accuracy as a scalar tensor.
    """
    predictions = logits.argmax(dim=-1)

    is_pitch = (targets >= token_ranges.pitch_start) & \
               (targets < token_ranges.pitch_end)

    if not is_pitch.any():
        return torch.tensor(1.0, device=logits.device)

    correct = (predictions == targets) & is_pitch
    accuracy = correct.sum().float() / is_pitch.sum().float()
    return accuracy


def compute_velocity_mae(
    logits: Tensor,
    targets: Tensor,
    token_ranges: TokenRanges = DEFAULT_RANGES,
    velocity_values: Optional[List[int]] = None,
) -> Tensor:
    """
    Compute mean absolute error for velocity predictions.

    Args:
        logits: Model output.
        targets: Target token IDs.
        token_ranges: Token range configuration.
        velocity_values: Velocity bin values.

    Returns:
        MAE as a scalar tensor.
    """
    if velocity_values is None:
        velocity_values = [16, 32, 48, 64, 80, 96, 112, 127]

    device = logits.device
    vel_values = torch.tensor(velocity_values, dtype=torch.float, device=device)

    vel_start = token_ranges.velocity_start
    vel_end = token_ranges.velocity_end

    is_velocity = (targets >= vel_start) & (targets < vel_end)

    if not is_velocity.any():
        return torch.tensor(0.0, device=device)

    # Predicted velocity (expected value)
    velocity_logits = logits[:, :, vel_start:vel_end]
    velocity_probs = F.softmax(velocity_logits, dim=-1)
    predicted = torch.sum(velocity_probs * vel_values.view(1, 1, -1), dim=-1)

    # Target velocity
    target_idx = (targets - vel_start).clamp(0, len(velocity_values) - 1)
    target_vel = vel_values[target_idx.long()]

    mae = torch.abs(predicted[is_velocity] - target_vel[is_velocity]).mean()
    return mae
