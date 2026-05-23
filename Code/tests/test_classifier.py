"""
Tests for MINDClassifier and MINDDataset — CPU only.

We don't need GPU or the LLM for these; the classifier is a tiny MLP.
The whole module is skipped if torch isn't installed.
"""

import pytest

torch = pytest.importorskip("torch")

from src.classifier import MINDClassifier, MINDDataset  # noqa: E402
from src.config import HIDDEN_LAYERS, NUM_CLASSES        # noqa: E402


def test_classifier_output_shape():
    model = MINDClassifier(input_dim=128)
    x = torch.randn(4, 128)
    y = model(x)
    assert y.shape == (4, NUM_CLASSES)


def test_classifier_default_layer_count():
    """One Linear per HIDDEN_LAYERS entry, plus one final Linear."""
    model = MINDClassifier(input_dim=64)
    linear_layers = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == len(HIDDEN_LAYERS) + 1


def test_classifier_custom_architecture():
    model = MINDClassifier(input_dim=64, hidden_layers=(32, 16), num_classes=3, dropout=0.0)
    out = model(torch.randn(2, 64))
    assert out.shape == (2, 3)
    linear_layers = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3  # 2 hidden + 1 final


def test_classifier_backward_pass():
    """Gradients should flow end-to-end."""
    model = MINDClassifier(input_dim=32)
    x = torch.randn(8, 32, requires_grad=False)
    y = torch.randint(0, NUM_CLASSES, (8,))
    logits = model(x)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_mind_dataset_basic():
    data = [
        {"embedding": [0.1] * 10, "label": 0},
        {"embedding": [0.2] * 10, "label": 1},
        {"embedding": [0.3] * 10, "label": 1},
    ]
    ds = MINDDataset(data)
    assert len(ds) == 3
    emb, label = ds[1]
    assert emb.shape == (10,)
    assert label.item() == 1
    assert emb.dtype == torch.float32
    assert label.dtype == torch.long


def test_mind_dataset_iterable_in_dataloader():
    from torch.utils.data import DataLoader

    data = [{"embedding": [float(i)] * 4, "label": i % 2} for i in range(10)]
    loader = DataLoader(MINDDataset(data), batch_size=4, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    emb, labels = batches[0]
    assert emb.shape == (4, 4)
    assert labels.shape == (4,)
