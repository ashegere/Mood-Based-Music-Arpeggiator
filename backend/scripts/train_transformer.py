#!/usr/bin/env python3
"""
Transformer Fine-Tuning Training Script

Trains a mood-conditioned music transformer using decoder-only architecture.

Features:
- Transformer decoder architecture with mood conditioning
- Cross-entropy loss with label smoothing
- AdamW optimizer with weight decay
- Learning rate scheduler (cosine with warmup)
- Gradient clipping
- Mixed precision training (AMP)
- Checkpoint saving every N epochs
- Validation loop with metrics
- TensorBoard logging support

Usage:
    # Basic training
    python scripts/train_transformer.py \
        --train-data data/training/train_dataset.pt \
        --val-data data/training/val_dataset.pt

    # Custom configuration
    python scripts/train_transformer.py \
        --train-data data/training/train_dataset.pt \
        --val-data data/training/val_dataset.pt \
        --batch-size 32 \
        --lr 1e-4 \
        --epochs 100 \
        --checkpoint-every 10

    # Resume training
    python scripts/train_transformer.py \
        --train-data data/training/train_dataset.pt \
        --val-data data/training/val_dataset.pt \
        --resume checkpoints/checkpoint_epoch_50.pt
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Transformer model configuration."""
    vocab_size: int = 512
    max_seq_length: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    num_moods: int = 19
    mood_embedding_dim: int = 64
    pad_token_id: int = 0


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    batch_size: int = 32
    num_workers: int = 0

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9

    # LR Schedule
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1

    # Training
    epochs: int = 100
    grad_clip: float = 1.0
    label_smoothing: float = 0.1

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 10
    save_best: bool = True

    # Logging
    log_every: int = 50
    eval_every: int = 500


# =============================================================================
# Dataset
# =============================================================================

class MusicDataset(Dataset):
    """PyTorch Dataset for music sequences."""

    def __init__(self, path: str):
        data = torch.load(path)
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.mood_labels = data["mood_labels"]
        self.vocab = data["vocab"]
        self.config = data.get("config", {})

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "mood_label": self.mood_labels[idx],
        }

    @property
    def vocab_size(self) -> int:
        return self.vocab.get("vocab_size", 512)

    @property
    def pad_token_id(self) -> int:
        return self.vocab.get("pad_token_id", 0)


# =============================================================================
# Model Components
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MoodConditionedTransformerDecoder(nn.Module):
    """
    Transformer decoder with mood conditioning.

    Architecture:
    - Token embedding + positional encoding
    - Mood embedding (added to sequence)
    - N transformer decoder layers
    - Output projection to vocabulary

    The mood embedding is added as a learned bias to the token embeddings,
    allowing the model to condition generation on the target mood.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model,
            config.max_seq_length,
            config.dropout,
        )

        # Mood embedding
        self.mood_embedding = nn.Embedding(config.num_moods, config.mood_embedding_dim)
        self.mood_projection = nn.Linear(config.mood_embedding_dim, config.d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.n_layers)

        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        # Tie weights between embedding and output
        self.output_projection.weight = self.token_embedding.weight

        # Layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Initialize weights
        self._init_weights()

        # Cache for causal mask
        self._causal_mask_cache: Dict[int, Tensor] = {}

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _get_causal_mask(self, size: int, device: torch.device) -> Tensor:
        """Get or create causal attention mask."""
        if size not in self._causal_mask_cache:
            mask = torch.triu(
                torch.ones(size, size, dtype=torch.bool, device=device),
                diagonal=1,
            )
            self._causal_mask_cache[size] = mask
        return self._causal_mask_cache[size].to(device)

    def forward(
        self,
        input_ids: Tensor,
        mood_labels: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            mood_labels: Mood label indices (batch,)
            attention_mask: Attention mask (batch, seq_len), 1=attend, 0=ignore

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Add mood conditioning
        mood_emb = self.mood_embedding(mood_labels)  # (batch, mood_dim)
        mood_proj = self.mood_projection(mood_emb)  # (batch, d_model)
        x = x + mood_proj.unsqueeze(1)  # Broadcast to all positions

        # Create causal mask
        causal_mask = self._get_causal_mask(seq_len, device)

        # Create padding mask (True = ignore)
        if attention_mask is not None:
            padding_mask = attention_mask == 0
        else:
            padding_mask = None

        # Create dummy memory (decoder-only, so we use self as memory)
        memory = torch.zeros(batch_size, 1, self.config.d_model, device=device)

        # Decoder forward
        x = self.decoder(
            x,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.output_projection(x)

        return logits

    def generate(
        self,
        prompt: Tensor,
        mood_label: Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: int = 2,
    ) -> Tensor:
        """
        Generate sequence autoregressively.

        Args:
            prompt: Initial tokens (batch, prompt_len)
            mood_label: Mood labels (batch,)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            eos_token_id: EOS token ID

        Returns:
            Generated sequence (batch, gen_len)
        """
        self.eval()
        device = prompt.device
        batch_size = prompt.size(0)

        generated = prompt.clone()
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(max_length - prompt.size(1)):
                # Forward pass
                logits = self.forward(generated, mood_label)
                next_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")

                # Top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumsum > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    sorted_logits[remove] = float("-inf")
                    next_logits = sorted_logits.gather(-1, sorted_idx.argsort(-1))

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update done
                done = done | (next_token.squeeze(-1) == eos_token_id)

                # Append
                generated = torch.cat([generated, next_token], dim=1)

                if done.all():
                    break

        return generated


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create cosine learning rate schedule with linear warmup.

    Args:
        optimizer: The optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial LR

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Transformer trainer with all training features.
    """

    def __init__(
        self,
        model: MoodConditionedTransformerDecoder,
        train_dataset: MusicDataset,
        val_dataset: MusicDataset,
        config: TrainingConfig,
        model_config: ModelConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.model_config = model_config
        self.device = device

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )

        # LR Scheduler
        total_steps = len(self.train_loader) * config.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            config.warmup_steps,
            total_steps,
            config.min_lr_ratio,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=model_config.pad_token_id,
            label_smoothing=config.label_smoothing,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and device.type == "cuda" else None

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Metrics history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            mood_labels = batch["mood_label"].to(self.device)

            # Create targets (shifted input)
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()

            # Forward pass with mixed precision
            with autocast(enabled=self.scaler is not None):
                logits = self.model(input_ids, mood_labels, attention_mask)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            num_batches += 1

            # Logging
            if self.global_step % self.config.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                avg_loss = total_loss / num_batches
                print(
                    f"  Step {self.global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e}"
                )

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            mood_labels = batch["mood_label"].to(self.device)

            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()

            logits = self.model(input_ids, mood_labels, attention_mask)

            # Loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            total_loss += loss.item()

            # Accuracy (non-padding tokens only)
            predictions = logits.argmax(dim=-1)
            mask = targets != self.model_config.pad_token_id
            correct = (predictions == targets) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            num_batches += 1

        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        return avg_loss, accuracy

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "model_config": asdict(self.model_config),
            "history": self.history,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint.get("history", self.history)

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Loaded checkpoint from {path}")
        print(f"Resuming from epoch {self.epoch + 1}, step {self.global_step}")

    def train(self) -> None:
        """Main training loop."""
        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Total steps: {len(self.train_loader) * self.config.epochs}")
        print("=" * 60 + "\n")

        start_epoch = self.epoch
        start_time = time.time()

        for epoch in range(start_epoch, self.config.epochs):
            self.epoch = epoch
            epoch_start = time.time()

            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print("-" * 40)

            # Train
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss, val_acc = self.validate()
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])

            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_acc:.4f}")
            print(f"  Time:       {epoch_time:.1f}s")

            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  New best validation loss!")

            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", is_best=False)

            if is_best and self.config.save_best:
                self.save_checkpoint("best_model.pt", is_best=True)

        # Final checkpoint
        self.save_checkpoint("final_model.pt")

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print("=" * 60)

        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history: {history_path}")


# =============================================================================
# Main
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(
        description="Train mood-conditioned music transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--train-data", type=str, required=True,
                        help="Path to train_dataset.pt")
    parser.add_argument("--val-data", type=str, required=True,
                        help="Path to val_dataset.pt")

    # Model
    parser.add_argument("--d-model", type=int, default=256,
                        help="Model dimension (default: 256)")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--n-layers", type=int, default=6,
                        help="Number of layers (default: 6)")
    parser.add_argument("--d-ff", type=int, default=1024,
                        help="Feed-forward dimension (default: 1024)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (default: 0.1)")

    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs (default: 100)")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Warmup steps (default: 500)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping (default: 1.0)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing (default: 0.1)")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory (default: checkpoints)")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (default: 0)")

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load datasets
    print(f"Loading train dataset: {args.train_data}")
    train_dataset = MusicDataset(args.train_data)

    print(f"Loading val dataset: {args.val_data}")
    val_dataset = MusicDataset(args.val_data)

    # Get vocab info from dataset
    vocab_size = train_dataset.vocab_size
    pad_token_id = train_dataset.pad_token_id
    max_seq_length = train_dataset.input_ids.shape[1]
    num_moods = len(train_dataset.vocab.get("mood_to_id", {})) or 19

    print(f"Vocab size: {vocab_size}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Number of moods: {num_moods}")

    # Model config
    model_config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        num_moods=num_moods,
        pad_token_id=pad_token_id,
    )

    # Training config
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        use_amp=not args.no_amp and device.type == "cuda",
        num_workers=args.num_workers,
    )

    # Create model
    model = MoodConditionedTransformerDecoder(model_config)
    print(f"Model parameters: {count_parameters(model):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        model_config=model_config,
        device=device,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()

    return 0


if __name__ == "__main__":
    sys.exit(main())
