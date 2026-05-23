"""
embeddings.py — Block 4: hidden-state extraction (the ``get_hd`` function).

Returns three views of the hidden states for a given ``text``:

1. ``hds``       — average over all layers of the *last token* embedding.
2. ``mean1``     — mean over tokens of the *first* layer (from EMBED_START_AT on).
3. ``mean2``     — mean over tokens of the *last* layer (from EMBED_START_AT on).
                   This is the MIND paper default and goes into the classifier.
"""

from __future__ import annotations

from typing import List, Tuple

from .config import EMBED_START_AT


def get_hd(
    text: str, tokenizer, model, start_at: int = EMBED_START_AT
) -> Tuple[List[float], List[float], List[float]]:
    """
    Extract the three embeddings as defined in the MIND-style original notebook.

    Parameters
    ----------
    text : str
    tokenizer, model : transformers tokenizer + causal LM (must be in eval mode)
    start_at : int
        First token position used in the "mean" embeddings. The notebook uses 2.

    Returns
    -------
    (hds, mean1, mean2) — each a flat Python list of floats.
    """
    import torch

    ids = tokenizer(text.strip(), return_tensors="pt")["input_ids"].tolist()

    with torch.no_grad():
        out = model(torch.tensor(ids).to(model.device), output_hidden_states=True)

    hd = out.hidden_states  # tuple of length n_layers+1

    # Method 1: average of the last token across all layers (skip embedding layer at index 0)
    hds = hd[1][0][-1].clone().detach()
    for i in range(2, len(hd)):
        hds += hd[i][0][-1].clone().detach()
    hds = hds / (len(hd) - 1)

    # Method 2: mean of *first* layer from start_at-1 onwards
    mean1 = torch.mean(hd[1][0][start_at - 1 :], dim=0)

    # Method 3: mean of *last* layer from start_at-1 onwards  (MIND default)
    mean2 = torch.mean(hd[-1][0][start_at - 1 :], dim=0)

    return hds.tolist(), mean1.tolist(), mean2.tolist()
