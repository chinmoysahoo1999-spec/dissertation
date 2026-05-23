"""
train.py — Block 9: training loop for the MIND classifier.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .classifier import MINDClassifier, MINDDataset
from .config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    EPOCHS,
    LEARNING_RATE,
    TEST_PATH,
    TRAIN_PATH,
    WEIGHT_DECAY,
)

log = logging.getLogger(__name__)


def load_data(
    train_path: str = TRAIN_PATH, test_path: str = TEST_PATH
) -> Tuple[List[Dict], List[Dict]]:
    """Read the JSONs written by :func:`dataset_gen.split_and_save`."""
    with open(train_path, "r") as f:
        train_data = json.load(f)
    with open(test_path, "r") as f:
        test_data = json.load(f)
    return train_data, test_data


def train(
    train_data: List[Dict],
    test_data: List[Dict],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    save_path: str = BEST_MODEL_PATH,
    device: str = None,
) -> Tuple[MINDClassifier, float]:
    """
    Train the MINDClassifier and save the best checkpoint by test accuracy.

    Returns
    -------
    (best_model, best_test_acc)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    train_dataset = MINDDataset(train_data)
    test_dataset = MINDDataset(test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = len(train_data[0]["embedding"])
    model = MINDClassifier(input_dim).to(device)

    log.info(
        "Training | input_dim=%d | params=%s | device=%s",
        input_dim,
        f"{sum(p.numel() for p in model.parameters()):,}",
        device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):
        # --- train -------------------------------------------------------
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for emb, y in train_loader:
            emb, y = emb.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(emb)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.size(0)
        tr_loss /= max(len(train_loader), 1)
        tr_acc = tr_correct / max(tr_total, 1)

        # --- eval --------------------------------------------------------
        model.eval()
        te_loss = 0.0
        te_correct = 0
        te_total = 0
        with torch.no_grad():
            for emb, y in test_loader:
                emb, y = emb.to(device), y.to(device)
                logits = model(emb)
                te_loss += criterion(logits, y).item()
                te_correct += (logits.argmax(1) == y).sum().item()
                te_total += y.size(0)
        te_loss /= max(len(test_loader), 1)
        te_acc = te_correct / max(te_total, 1)

        marker = ""
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), save_path)
            marker = " *"

        print(
            f"Epoch {epoch + 1:2d}/{epochs}: "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"test loss={te_loss:.4f} acc={te_acc:.4f}{marker}"
        )

    log.info("Best test accuracy: %.4f (saved to %s)", best_acc, save_path)
    return model, best_acc
