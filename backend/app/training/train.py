"""
Training script for mood-conditioned music transformer.

Two-phase training approach:

Phase 1: Reconstruction
-----------------------
- Input: neutral sequence
- Output: neutral sequence (same as input)
- Objective: Learn musical structure and token relationships
- No mood conditioning (zero embedding)
- Train full model

Phase 2: Conditioning
---------------------
- Input: neutral sequence + mood embedding
- Output: expressive sequence
- Objective: Learn mood → expression mapping
- Freeze encoder (optionally)
- Train decoder conditioning pathways

Usage:
------
    # Phase 1: Reconstruction training
    python -m app.training.train --phase 1 --data-dir data/training --epochs 50

    # Phase 2: Conditioning training (freeze encoder)
    python -m app.training.train --phase 2 --data-dir data/training --epochs 100 \\
        --checkpoint checkpoints/phase1_best.pt --freeze-encoder

    # Full training pipeline
    python -m app.training.train --full-pipeline --data-dir data/training
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.model.transformer import (
    MoodConditionedMusicTransformer,
    TransformerConfig,
    freeze_encoder,
    freeze_decoder_base,
    unfreeze_all,
    count_parameters,
    create_small_model,
    create_medium_model,
)
from app.training.dataset import (
    MoodArpeggioDataset,
    ReconstructionDataset,
    create_phase1_dataloader,
    create_phase2_dataloader,
    split_dataset,
    create_dataloader,
    BatchedSample,
)
from app.training.losses import (
    MoodConditionedLoss,
    TokenRanges,
    compute_accuracy,
    compute_pitch_accuracy,
    compute_velocity_mae,
)
from app.music.tokenization import get_vocabulary


# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training run."""
    # Data
    data_dir: str = "data/training"
    max_seq_len: int = 512
    batch_size: int = 32
    val_ratio: float = 0.1

    # Training
    phase: int = 2
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # Model
    model_size: str = "medium"  # small, medium, large
    vocab_size: int = 256

    # Freezing
    freeze_encoder: bool = False
    freeze_decoder_base: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    checkpoint_path: Optional[str] = None  # Resume from

    # Logging
    log_every: int = 50
    eval_every: int = 500

    # Hardware
    device: str = "auto"
    mixed_precision: bool = True
    num_workers: int = 0

    # Loss weights
    classification_weight: float = 1.0
    pitch_weight: float = 2.0
    velocity_weight: float = 0.5
    smoothness_weight: float = 0.1
    preservation_weight: float = 0.5


# =============================================================================
# Training State
# =============================================================================

@dataclass
class TrainingState:
    """Tracks training progress."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    train_losses: List[float] = None
    val_losses: List[float] = None

    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []
        if self.val_losses is None:
            self.val_losses = []


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as ratio of initial LR.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> float:
        """Update learning rate and return current LR."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            scale = self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / \
                       max(1, self.total_steps - self.warmup_steps)
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * \
                    0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * scale

        return self.optimizer.param_groups[0]["lr"]

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


# =============================================================================
# Trainer Class
# =============================================================================

class Trainer:
    """
    Trainer for mood-conditioned music transformer.

    Handles:
    - Model setup and initialization
    - Training loop with gradient accumulation
    - Validation
    - Checkpointing
    - Logging
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainingState()

        # Setup device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        logger.info(f"Using device: {self.device}")

        # Get vocabulary size
        vocab = get_vocabulary()
        self.vocab_size = vocab.size
        logger.info(f"Vocabulary size: {self.vocab_size}")

        # Setup model
        self.model = self._create_model()
        self.model.to(self.device)

        # Log parameter counts
        param_counts = count_parameters(self.model)
        logger.info(f"Model parameters: {param_counts}")

        # Setup loss function
        self.loss_fn = MoodConditionedLoss(
            classification_weight=config.classification_weight,
            pitch_weight=config.pitch_weight,
            velocity_weight=config.velocity_weight,
            smoothness_weight=config.smoothness_weight,
            preservation_weight=config.preservation_weight,
        )

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup gradient scaler for mixed precision
        self.scaler = GradScaler() if config.mixed_precision and self.device.type == "cuda" else None

        # Scheduler will be set up after dataloader is created
        self.scheduler = None

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load checkpoint if specified
        if config.checkpoint_path:
            self._load_checkpoint(config.checkpoint_path)

        # Apply freezing if configured
        self._apply_freezing()

    def _create_model(self) -> MoodConditionedMusicTransformer:
        """Create model based on config."""
        if self.config.model_size == "small":
            model = create_small_model(self.vocab_size)
        elif self.config.model_size == "medium":
            model = create_medium_model(self.vocab_size)
        else:
            # Default to medium
            model = create_medium_model(self.vocab_size)

        logger.info(f"Created {self.config.model_size} model")
        return model

    def _apply_freezing(self) -> None:
        """Apply layer freezing based on config."""
        if self.config.freeze_encoder:
            freeze_encoder(self.model)
            logger.info("Froze encoder layers")

        if self.config.freeze_decoder_base:
            freeze_decoder_base(self.model)
            logger.info("Froze decoder base (keeping conditioning trainable)")

        # Update optimizer to only include trainable params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        param_counts = count_parameters(self.model)
        logger.info(f"Trainable parameters: {param_counts['trainable']}")

    def _save_checkpoint(
        self,
        filename: str,
        is_best: bool = False,
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.state.best_val_loss,
            "best_val_accuracy": self.state.best_val_accuracy,
            "config": asdict(self.config),
            "train_losses": self.state.train_losses,
            "val_losses": self.state.val_losses,
        }

        if self.scheduler:
            checkpoint["scheduler_step"] = self.scheduler.current_step

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.checkpoint_dir / f"phase{self.config.phase}_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

    def _load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.global_step = checkpoint.get("global_step", 0)
        self.state.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.state.best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
        self.state.train_losses = checkpoint.get("train_losses", [])
        self.state.val_losses = checkpoint.get("val_losses", [])

        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"Resuming from epoch {self.state.epoch}, step {self.state.global_step}")

    def _train_step(
        self,
        batch: BatchedSample,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Single training step.

        Returns:
            Tuple of (loss value, metrics dict)
        """
        self.model.train()

        # Move batch to device
        src = batch.src.to(self.device)
        tgt_input = batch.tgt_input.to(self.device)
        tgt_output = batch.tgt_output.to(self.device)
        mood_embedding = batch.mood_embedding.to(self.device)
        src_padding_mask = batch.src_padding_mask.to(self.device)
        tgt_padding_mask = batch.tgt_padding_mask.to(self.device)

        # Forward pass with optional mixed precision
        use_amp = self.scaler is not None

        with autocast(enabled=use_amp):
            logits = self.model(
                src=src,
                tgt=tgt_input,
                mood_embedding=mood_embedding,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )

            # Compute loss
            loss, components = self.loss_fn(
                logits=logits,
                targets=tgt_output,
                source=src,
                phase=self.config.phase,
                return_components=True,
            )

        # Backward pass
        self.optimizer.zero_grad()

        if use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

        # Update scheduler
        if self.scheduler:
            self.scheduler.step()

        # Compute metrics
        with torch.no_grad():
            accuracy = compute_accuracy(logits, tgt_output).item()
            pitch_acc = compute_pitch_accuracy(logits, tgt_output).item()
            vel_mae = compute_velocity_mae(logits, tgt_output).item()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "pitch_accuracy": pitch_acc,
            "velocity_mae": vel_mae,
            "lr": self.scheduler.get_lr() if self.scheduler else self.config.learning_rate,
        }

        # Add component losses
        if components:
            for name, value in components.items():
                if isinstance(value, Tensor):
                    metrics[f"loss_{name}"] = value.item()

        return loss.item(), metrics

    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dict of averaged metrics.
        """
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        total_pitch_acc = 0.0
        total_vel_mae = 0.0
        n_batches = 0

        for batch in val_loader:
            src = batch.src.to(self.device)
            tgt_input = batch.tgt_input.to(self.device)
            tgt_output = batch.tgt_output.to(self.device)
            mood_embedding = batch.mood_embedding.to(self.device)
            src_padding_mask = batch.src_padding_mask.to(self.device)
            tgt_padding_mask = batch.tgt_padding_mask.to(self.device)

            logits = self.model(
                src=src,
                tgt=tgt_input,
                mood_embedding=mood_embedding,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )

            loss, _ = self.loss_fn(
                logits=logits,
                targets=tgt_output,
                source=src,
                phase=self.config.phase,
            )

            total_loss += loss.item()
            total_accuracy += compute_accuracy(logits, tgt_output).item()
            total_pitch_acc += compute_pitch_accuracy(logits, tgt_output).item()
            total_vel_mae += compute_velocity_mae(logits, tgt_output).item()
            n_batches += 1

        return {
            "val_loss": total_loss / n_batches,
            "val_accuracy": total_accuracy / n_batches,
            "val_pitch_accuracy": total_pitch_acc / n_batches,
            "val_velocity_mae": total_vel_mae / n_batches,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Main training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
        """
        # Calculate total steps for scheduler
        total_steps = len(train_loader) * self.config.epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
        )

        # Skip steps if resuming
        for _ in range(self.state.global_step):
            self.scheduler.step()

        logger.info(f"Starting Phase {self.config.phase} training")
        logger.info(f"Total epochs: {self.config.epochs}")
        logger.info(f"Steps per epoch: {len(train_loader)}")
        logger.info(f"Total steps: {total_steps}")

        start_epoch = self.state.epoch

        for epoch in range(start_epoch, self.config.epochs):
            self.state.epoch = epoch
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            n_steps = 0

            for batch_idx, batch in enumerate(train_loader):
                loss, metrics = self._train_step(batch)
                self.state.global_step += 1
                epoch_loss += loss
                epoch_accuracy += metrics["accuracy"]
                n_steps += 1

                # Logging
                if self.state.global_step % self.config.log_every == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.config.epochs} | "
                        f"Step {self.state.global_step} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Acc: {metrics['accuracy']:.4f} | "
                        f"Pitch: {metrics['pitch_accuracy']:.4f} | "
                        f"LR: {metrics['lr']:.2e}"
                    )

                # Validation
                if val_loader and self.state.global_step % self.config.eval_every == 0:
                    val_metrics = self._validate(val_loader)
                    logger.info(
                        f"Validation | "
                        f"Loss: {val_metrics['val_loss']:.4f} | "
                        f"Acc: {val_metrics['val_accuracy']:.4f} | "
                        f"Pitch: {val_metrics['val_pitch_accuracy']:.4f}"
                    )

                    # Check for best model
                    if val_metrics["val_loss"] < self.state.best_val_loss:
                        self.state.best_val_loss = val_metrics["val_loss"]
                        self.state.best_val_accuracy = val_metrics["val_accuracy"]
                        self._save_checkpoint(
                            f"phase{self.config.phase}_step{self.state.global_step}.pt",
                            is_best=True,
                        )

                    self.state.val_losses.append(val_metrics["val_loss"])

            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / n_steps
            avg_accuracy = epoch_accuracy / n_steps
            self.state.train_losses.append(avg_loss)

            logger.info(
                f"Epoch {epoch+1} complete | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Avg Acc: {avg_accuracy:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(f"phase{self.config.phase}_epoch{epoch+1}.pt")

        # Final checkpoint
        self._save_checkpoint(f"phase{self.config.phase}_final.pt")
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.state.best_val_loss:.4f}")
        logger.info(f"Best validation accuracy: {self.state.best_val_accuracy:.4f}")


# =============================================================================
# Training Functions
# =============================================================================

def train_phase1(config: TrainingConfig) -> str:
    """
    Run Phase 1 (reconstruction) training.

    Returns:
        Path to best checkpoint.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Reconstruction Training")
    logger.info("=" * 60)

    config.phase = 1
    config.freeze_encoder = False
    config.freeze_decoder_base = False

    # Create datasets
    dataset = ReconstructionDataset(
        data_path=config.data_dir,
        max_seq_len=config.max_seq_len,
    )

    train_dataset, val_dataset = split_dataset(dataset, config.val_ratio)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Train
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

    return str(trainer.checkpoint_dir / "phase1_best.pt")


def train_phase2(
    config: TrainingConfig,
    phase1_checkpoint: Optional[str] = None,
) -> str:
    """
    Run Phase 2 (conditioning) training.

    Args:
        config: Training configuration.
        phase1_checkpoint: Path to Phase 1 checkpoint to load.

    Returns:
        Path to best checkpoint.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Conditioning Training")
    logger.info("=" * 60)

    config.phase = 2

    if phase1_checkpoint:
        config.checkpoint_path = phase1_checkpoint
        logger.info(f"Loading Phase 1 checkpoint: {phase1_checkpoint}")

    # Create datasets
    dataset = MoodArpeggioDataset(
        data_path=config.data_dir,
        phase=2,
        max_seq_len=config.max_seq_len,
        cache_embeddings=True,
    )

    train_dataset, val_dataset = split_dataset(dataset, config.val_ratio)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Train
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

    return str(trainer.checkpoint_dir / "phase2_best.pt")


def run_full_pipeline(config: TrainingConfig) -> None:
    """
    Run complete two-phase training pipeline.

    Phase 1: Reconstruction (learn structure)
    Phase 2: Conditioning (learn expression)
    """
    logger.info("=" * 60)
    logger.info("FULL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Phase 1
    phase1_config = TrainingConfig(**asdict(config))
    phase1_config.epochs = config.epochs // 2  # Half epochs for phase 1
    phase1_checkpoint = train_phase1(phase1_config)

    # Phase 2
    phase2_config = TrainingConfig(**asdict(config))
    phase2_config.epochs = config.epochs
    phase2_config.freeze_encoder = True
    train_phase2(phase2_config, phase1_checkpoint)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train mood-conditioned music transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/training",
                        help="Path to training data directory")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2],
                        help="Training phase (1=reconstruction, 2=conditioning)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="Learning rate warmup steps")

    # Model arguments
    parser.add_argument("--model-size", type=str, default="medium",
                        choices=["small", "medium", "large"],
                        help="Model size")

    # Freezing arguments
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder layers")
    parser.add_argument("--freeze-decoder-base", action="store_true",
                        help="Freeze decoder (except conditioning)")

    # Checkpoint arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")

    # Pipeline arguments
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run full two-phase training")

    # Hardware arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--no-mixed-precision", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers")

    args = parser.parse_args()

    # Build config
    config = TrainingConfig(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        phase=args.phase,
        epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        model_size=args.model_size,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder_base=args.freeze_decoder_base,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint,
        save_every=args.save_every,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        num_workers=args.num_workers,
    )

    # Run training
    if args.full_pipeline:
        run_full_pipeline(config)
    elif args.phase == 1:
        train_phase1(config)
    else:
        train_phase2(config, args.checkpoint)


if __name__ == "__main__":
    main()
