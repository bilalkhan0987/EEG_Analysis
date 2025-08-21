"""
train.py — Training utilities (loops, evaluation, checkpointing).
"""
from typing import Dict, Tuple
import os
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RandomSequenceDataset(Dataset):
    """Fallback synthetic dataset for quick smoke tests.

    Each sample: Tensor shape (C=1, L=seq_len), label in [0, num_classes-1].
    """

    def __init__(self, length: int, seq_len: int, num_classes: int, in_channels: int = 1):
        self.length = length
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.in_channels = in_channels

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        x = torch.randn(self.in_channels, self.seq_len)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y


def build_loaders(
    train_size: int,
    val_size: int,
    seq_len: int,
    num_classes: int,
    in_channels: int,
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = RandomSequenceDataset(train_size, seq_len, num_classes, in_channels)
    val_ds = RandomSequenceDataset(val_size, seq_len, num_classes, in_channels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return {"loss": total_loss / total, "acc": correct / total if total > 0 else 0.0}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return {"loss": total_loss / total, "acc": correct / total if total > 0 else 0.0}


def save_checkpoint(state: dict, save_dir: str, filename: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
