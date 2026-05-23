"""
evaluate.py — Block 10: comprehensive evaluation of a trained classifier.
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .classifier import MINDClassifier, MINDDataset
from .config import BATCH_SIZE, BEST_MODEL_PATH


def evaluate(
    model: MINDClassifier,
    test_data: List[Dict],
    batch_size: int = BATCH_SIZE,
    device: str = None,
    weights_path: str = BEST_MODEL_PATH,
    load_best: bool = True,
    n_examples: int = 20,
) -> Dict:
    """
    Compute classification metrics and a few qualitative predictions.

    Returns a dict with: accuracy, precision, recall, f1, auc, confusion_matrix,
    report, examples (list of dicts).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if load_best:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()

    test_loader = DataLoader(MINDDataset(test_data), batch_size=batch_size, shuffle=False)

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for emb, y in test_loader:
            emb = emb.to(device)
            logits = model(emb)
            probs = F.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
            all_labels.extend(y.numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)
    all_probs_arr = np.array(all_probs)

    acc = float(accuracy_score(all_labels_arr, all_preds_arr))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_arr, all_preds_arr, average="binary"
    )
    auc = float(roc_auc_score(all_labels_arr, all_probs_arr))
    cm = confusion_matrix(all_labels_arr, all_preds_arr).tolist()
    report = classification_report(
        all_labels_arr,
        all_preds_arr,
        target_names=["Non-hallucination (0)", "Hallucination (1)"],
        digits=4,
        output_dict=False,
    )

    # qualitative examples
    examples: List[Dict] = []
    n_examples = min(n_examples, len(test_data))
    if n_examples:
        sample_idx = random.sample(range(len(test_data)), n_examples)
        for idx in sample_idx:
            sample = test_data[idx]
            emb = torch.FloatTensor(sample["embedding"]).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(emb)
                probs = F.softmax(logits, dim=1)
                pred = int(logits.argmax(1).item())
                conf = float(probs[0, pred].item())
            examples.append(
                {
                    "true": sample["label"],
                    "pred": pred,
                    "confidence": conf,
                    "entity": sample.get("entity"),
                    "text": sample.get("text", "")[:100],
                    "correct": pred == sample["label"],
                }
            )

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "confusion_matrix": cm,
        "report": report,
        "examples": examples,
    }


def print_report(metrics: Dict, n_total: int) -> None:
    """Pretty-print the dict returned by :func:`evaluate`."""
    acc = metrics["accuracy"]
    cm = metrics["confusion_matrix"]

    print("=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"  Accuracy : {acc:.4f}   ({int(acc * n_total)} of {n_total} correct)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc']:.4f}")
    print()
    print("Confusion Matrix:")
    print("              Predicted")
    print("              Non-H  Hall")
    print(f"  Actual Non-H {cm[0][0]:5d} {cm[0][1]:5d}")
    print(f"         Hall  {cm[1][0]:5d} {cm[1][1]:5d}")
    print()
    print(metrics["report"])

    for i, ex in enumerate(metrics["examples"], 1):
        mark = "OK" if ex["correct"] else "X "
        true_lbl = ["Non-hallucination", "Hallucination"][ex["true"]]
        pred_lbl = ["Non-hallucination", "Hallucination"][ex["pred"]]
        print(f"[{i:2d}] {mark}  true={true_lbl}  pred={pred_lbl} ({ex['confidence']:.1%})")
        print(f"     entity: {ex['entity']}")
        print(f"     text:   {ex['text']}...")
