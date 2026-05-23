"""
classifier.py — Block 8: MIND-paper 5-layer MLP and its Dataset wrapper.

    P = MLP(ReLU(W · H + b))      (MIND paper, Eq. 1)
    L = BCE(P, y)                  (Eq. 2)

The architecture is parameterised through ``config.HIDDEN_LAYERS`` so we can
ablate widths without editing this file.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .config import DROPOUT_RATE, HIDDEN_LAYERS, NUM_CLASSES


class MINDClassifier(nn.Module):
    """
    Simple feed-forward classifier over an embedding vector.

    Architecture by default: input_dim → 512 → 256 → 128 → 64 → 2
    with ReLU activations and dropout after the first hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int] = HIDDEN_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT_RATE,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev = input_dim
        for i, width in enumerate(hidden_layers):
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            if i == 0 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """``H``: [batch_size, input_dim] → logits [batch_size, num_classes]"""
        return self.layers(H)


class MINDDataset(Dataset):
    """Wraps a list of ``{embedding, label}`` dicts into a torch Dataset."""

    def __init__(self, data: List[dict]) -> None:
        self.embeddings = torch.FloatTensor([d["embedding"] for d in data])
        self.labels = torch.LongTensor([d["label"] for d in data])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
