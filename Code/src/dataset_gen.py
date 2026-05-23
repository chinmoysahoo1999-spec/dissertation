"""
dataset_gen.py — Block 6 + Block 7: build the (hallucinated, non-hallucinated)
dataset from streaming Wikipedia and split it train/test.

The heart of the module is :func:`generate_sample`, which mirrors the original
notebook's per-article logic. :func:`build_dataset` is the streaming loop.
:func:`split_and_save` saves train/test JSONs to disk.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Dict, List, Optional, Tuple

from .config import (
    CACHE_CLEAR_EVERY,
    CHECKPOINT_EVERY,
    CHECKPOINT_PATH,
    EMBEDDING_KEY,
    SENTENCES_PER_DOC,
    TARGET_SAMPLES,
    TEST_PATH,
    TOPK_FIRST_TOKEN,
    TRAIN_FRACTION,
    TRAIN_PATH,
    WIKI_CONFIG,
    WIKI_DATASET,
    WINDOWS,
)
from .embeddings import get_hd
from .entities import filter_valid_entities, get_entities
from .tokens import find_first_and_next_token

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-sample generation
# ---------------------------------------------------------------------------
def generate_sample(
    text: str, entity: str, idx: int, title: str, tokenizer, model
) -> Optional[Dict]:
    """
    Try to generate one (hallucinated, non-hallucinated) pair from a single text.

    The procedure follows the original notebook:

    1. Tokenise the prompt = ``text[:idx]``.
    2. Use the ``@``-marker trick to find ``first_token`` (the model's expected
       continuation when the entity is the right answer) and ``next_token``
       (the token that should follow the entity).
    3. Ask the model what it predicts. If ``first_token`` is in the top-k, the
       model already "knows" the entity — discard the sample.
    4. Otherwise, let the model freely generate up to ``entity_len + WINDOWS``
       tokens, watching for ``next_token``. If we find it, the produced span is
       a *hallucinated* entity. Splice it back into the text.
    5. Quality-filter the new entity. Extract embeddings for both texts.

    Returns ``None`` on any failure / filter rejection.
    """
    import torch

    input_id = tokenizer(text[:idx].strip(), return_tensors="pt")["input_ids"].tolist()

    tokens = find_first_and_next_token(text, entity, idx, input_id, tokenizer)
    if not tokens:
        return None
    first_, next_, entity_len, last_id = tokens

    # Stage A: one-shot next-token prediction
    out = model.generate(
        torch.tensor(input_id).to(model.device),
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    _, indices = torch.topk(out.scores[0], k=TOPK_FIRST_TOKEN)
    if first_ in indices[0].tolist():
        return None  # model would have got it right; not a hallucination case

    # Stage C: keep generating until we see next_token (or give up)
    sequences = out.sequences
    found = False
    for _ in range(entity_len + WINDOWS):
        out = model.generate(
            sequences,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        _, indices = torch.topk(out.scores[0], k=TOPK_FIRST_TOKEN)
        if next_ in indices[0].tolist():
            found = True
            break
        sequences = out.sequences

    if not found:
        return None

    # Splice the freshly-generated entity back into the surrounding text.
    new_sequence = sequences[0].tolist()
    new_entity_id = new_sequence[len(input_id[0]) :]
    all_new_text_id = input_id[0] + new_entity_id + last_id
    hallucinated_text = tokenizer.decode(all_new_text_id, skip_special_tokens=True)
    new_entity = tokenizer.decode(new_entity_id, skip_special_tokens=True).strip().lower()

    # Quality filter: reject empty entities, entities that contain the original
    # entity, or entities whose tokens already appear in the source text
    # (likely just copying).
    if (
        not new_entity
        or entity.lower() in new_entity
        or any(
            piece.strip() in text.lower()
            for piece in new_entity.split(" ")
            if piece.strip()
        )
    ):
        return None

    hds_hall, mean1_hall, mean2_hall = get_hd(hallucinated_text, tokenizer, model)
    hds_orig, mean1_orig, mean2_orig = get_hd(text, tokenizer, model)

    return {
        "text_hall": hallucinated_text,
        "entity_hall": new_entity,
        "hds_hall": hds_hall,
        "mean1_hall": mean1_hall,
        "mean2_hall": mean2_hall,
        "label_hall": 1,
        "text_orig": text,
        "entity_orig": entity,
        "hds_orig": hds_orig,
        "mean1_orig": mean1_orig,
        "mean2_orig": mean2_orig,
        "label_orig": 0,
        "title": title,
    }


# ---------------------------------------------------------------------------
# Embedding selector — picks which of the three embeddings goes into the
# saved dataset, controlled by ``EMBEDDING_KEY`` in config.
# ---------------------------------------------------------------------------
def _embedding_pair(result: Dict) -> Tuple[List[float], List[float]]:
    """Return (hall_embedding, orig_embedding) according to EMBEDDING_KEY."""
    key = EMBEDDING_KEY
    if key not in {"mean2", "mean1", "hds"}:
        raise ValueError(f"Unknown EMBEDDING_KEY {key!r}")
    return result[f"{key}_hall"], result[f"{key}_orig"]


# ---------------------------------------------------------------------------
# Streaming loop over Wikipedia
# ---------------------------------------------------------------------------
def build_dataset(
    tokenizer,
    model,
    nlp,
    target_samples: int = TARGET_SAMPLES,
    show_progress: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stream Wikipedia until ``target_samples`` per class are collected.

    Returns
    -------
    (dataset_hall, dataset_non_hall) — each a list of dicts with keys
    ``label``, ``text``, ``entity``, ``embedding``, ``title``.
    """
    import torch
    from datasets import load_dataset
    from nltk.tokenize import sent_tokenize

    wiki = load_dataset(WIKI_DATASET, WIKI_CONFIG, split="train", streaming=True)

    dataset_hall: List[Dict] = []
    dataset_non_hall: List[Dict] = []
    processed = 0

    pbar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=target_samples * 2, desc="Samples")
        except ImportError:
            pbar = None

    for row in wiki:
        if (
            len(dataset_hall) >= target_samples
            and len(dataset_non_hall) >= target_samples
        ):
            break

        processed += 1

        try:
            sents = sent_tokenize(row["text"])
            if len(sents) < SENTENCES_PER_DOC:
                continue
            text = " ".join(sents[:SENTENCES_PER_DOC])
            title = row.get("title", "")

            entities = get_entities(text, nlp)
            if not entities:
                continue

            entities_filtered = filter_valid_entities(entities, title)
            if not entities_filtered:
                continue

            entity, char_idx = random.choice(entities_filtered)
            result = generate_sample(text, entity, char_idx, title, tokenizer, model)
            if result is None:
                continue

            hall_emb, orig_emb = _embedding_pair(result)

            if len(dataset_hall) < target_samples:
                dataset_hall.append(
                    {
                        "label": result["label_hall"],
                        "text": result["text_hall"],
                        "entity": result["entity_hall"],
                        "embedding": hall_emb,
                        "title": result["title"],
                    }
                )
                if pbar:
                    pbar.update(1)

            if len(dataset_non_hall) < target_samples:
                dataset_non_hall.append(
                    {
                        "label": result["label_orig"],
                        "text": result["text_orig"],
                        "entity": result["entity_orig"],
                        "embedding": orig_emb,
                        "title": result["title"],
                    }
                )
                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.set_postfix(
                    {
                        "H=1": len(dataset_hall),
                        "H=0": len(dataset_non_hall),
                        "Proc": processed,
                    }
                )

            total = len(dataset_hall) + len(dataset_non_hall)
            if total and total % CHECKPOINT_EVERY == 0:
                with open(CHECKPOINT_PATH, "w") as f:
                    json.dump(dataset_hall + dataset_non_hall, f)

        except Exception as exc:  # noqa: BLE001 - mirror original notebook
            if processed % 1000 == 0:
                log.warning("Error at processed=%d: %s", processed, exc)
            continue

        if processed % CACHE_CLEAR_EVERY == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if pbar:
        pbar.close()

    log.info(
        "Generation complete: H=1 %d, H=0 %d, processed %d",
        len(dataset_hall),
        len(dataset_non_hall),
        processed,
    )

    return dataset_hall, dataset_non_hall


# ---------------------------------------------------------------------------
# Train / test split + save
# ---------------------------------------------------------------------------
def split_and_save(
    dataset_hall: List[Dict],
    dataset_non_hall: List[Dict],
    train_path: str = TRAIN_PATH,
    test_path: str = TEST_PATH,
    train_fraction: float = TRAIN_FRACTION,
) -> Tuple[List[Dict], List[Dict]]:
    """Shuffle, 80/20 split, save JSON, return (train, test)."""
    dataset = dataset_hall + dataset_non_hall
    random.shuffle(dataset)

    split_idx = int(train_fraction * len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)

    log.info("Saved %d train / %d test samples", len(train_data), len(test_data))
    return train_data, test_data
