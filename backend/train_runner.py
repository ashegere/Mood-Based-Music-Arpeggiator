#!/usr/bin/env python3
"""CLI wrapper for mood-conditioned music transformer training."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MoodConditionedTransformer(nn.Module):
    """Transformer decoder with mood conditioning."""

    def __init__(self, vocab_size: int, num_moods: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2, max_seq_len: int = 512,
                 dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.mood_embedding = nn.Embedding(num_moods, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, mood_labels: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = x + self.mood_embedding(mood_labels).unsqueeze(1)
        x = self.dropout(x)

        # MPS-compatible causal mask (use large negative instead of -inf)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), -1e9, device=input_ids.device),
            diagonal=1
        )

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        memory = torch.zeros(batch_size, 1, self.d_model, device=input_ids.device)
        x = self.decoder(x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=key_padding_mask)

        return self.output_projection(x)


class Trainer:
    """Training loop with logging and checkpointing."""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, lr: float, device: torch.device,
                 checkpoint_dir: str = "checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            mood_labels = batch['mood_label'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            mask = attention_mask[:, :-1].contiguous()

            self.optimizer.zero_grad()
            logits = self.model(inputs, mood_labels, mask)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            mood_labels = batch['mood_label'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            mask = attention_mask[:, :-1].contiguous()

            logits = self.model(inputs, mood_labels, mask)
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, filename: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, self.checkpoint_dir / filename)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint.get('history', self.history)
        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"Best val loss so far: {self.best_val_loss:.4f}")
        print(f"Previous epochs: {len(self.history['train_loss'])}")

    def train(self, epochs: int):
        print(f"\n{'='*60}")
        print("TRAINING STARTED")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            epoch_time = time.time() - epoch_start
            is_best = val_loss < self.best_val_loss

            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s", end="")

            if is_best:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                print(" *")
            else:
                print()

        total_time = time.time() - start_time
        self.save_checkpoint("final_model.pt")

        with open(self.checkpoint_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*60}\n")


class TensorDataset:
    """Simple dataset wrapper for tensor data."""

    def __init__(self, input_ids, attention_mask, mood_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.mood_labels = mood_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'mood_label': self.mood_labels[idx]
        }


def collate_fn(batch):
    """Collate batch of samples."""
    input_ids = torch.stack([s['input_ids'] for s in batch])
    attention_mask = torch.stack([s['attention_mask'] for s in batch])
    mood_labels = torch.stack([s['mood_label'] for s in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'mood_label': mood_labels}


def main():
    parser = argparse.ArgumentParser(description="Train mood-conditioned music transformer")
    parser.add_argument("--train", required=True, help="Path to train_dataset.pt")
    parser.add_argument("--val", required=True, help="Path to val_dataset.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load datasets
    print(f"Loading train dataset: {args.train}")
    train_data = torch.load(args.train, weights_only=False)
    print(f"Loading val dataset: {args.val}")
    val_data = torch.load(args.val, weights_only=False)

    # Extract metadata
    vocab_size = train_data['vocab']['vocab_size']
    num_moods = len(train_data['config']['valid_moods'])
    max_seq_len = train_data['config']['max_length']

    print(f"Vocab size: {vocab_size}")
    print(f"Num moods: {num_moods}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Train samples: {len(train_data['input_ids'])}")
    print(f"Val samples: {len(val_data['input_ids'])}")

    # Create datasets
    train_dataset = TensorDataset(
        train_data['input_ids'],
        train_data['attention_mask'],
        train_data['mood_labels']
    )
    val_dataset = TensorDataset(
        val_data['input_ids'],
        val_data['attention_mask'],
        val_data['mood_labels']
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # Create model
    model = MoodConditionedTransformer(
        vocab_size=vocab_size,
        num_moods=num_moods,
        max_seq_len=max_seq_len
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, args.lr, device, args.checkpoint_dir)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
