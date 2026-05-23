"""
entities.py — Block 2: entity extraction utilities.

Pure functions over text (no model dependency), plus the spaCy-backed
``get_entities`` which takes the nlp pipeline as an argument so it stays
testable.
"""

from __future__ import annotations

from typing import List, Tuple


def delete_substrings(lst: List[str]) -> List[str]:
    """
    Remove any string in ``lst`` that is a substring of another string in ``lst``.

    >>> sorted(delete_substrings(["Einstein", "Albert Einstein", "Ulm"]))
    ['Albert Einstein', 'Ulm']
    """
    lst = list(set(lst))
    substrings = [s for s in lst if any(s in o for o in lst if o != s)]
    for s in substrings:
        lst.remove(s)
    return lst


def find_boundaries(text: str, words: List[str]) -> List[str]:
    """
    Expand each ``word`` to its surrounding word boundaries inside ``text``.

    A "word boundary" here means: keep expanding the match left/right while
    the neighbouring character is not a space. The original notebook logic is
    preserved verbatim — this is mainly a tokenisation safety net for spaCy
    entity spans that don't sit cleanly on whitespace.
    """
    boundaries: List[str] = []
    for word in words:
        ntext = text
        while True:
            start = ntext.find(word)
            if start == -1:
                break
            end = start + len(word) - 1
            while start > 0 and ntext[start - 1] != " ":
                start -= 1
            while end < len(ntext) - 1 and ntext[end + 1] != " ":
                end += 1
            boundaries.append("".join(ntext[i] for i in range(start, end + 1)))
            ntext = ntext[end + 1 :]
    return boundaries


def get_entities(text: str, nlp) -> List[Tuple[str, int]]:
    """
    Return every occurrence of every named entity in ``text``.

    Parameters
    ----------
    text : str
        Source text.
    nlp : spacy.Language
        Loaded spaCy pipeline (e.g. from :func:`model_loader.load_nlp`).

    Returns
    -------
    list of (entity_string, char_index) tuples.
    """
    entities_ = list({str(e) for e in nlp(text).ents})
    entities_ = find_boundaries(text, entities_)
    entities = delete_substrings(entities_)

    occurrences: List[Tuple[str, int]] = []
    for i in range(len(text)):
        for e in entities:
            if text[i:].startswith(e):
                occurrences.append((e, i))
    return occurrences


def filter_valid_entities(
    entities: List[Tuple[str, int]], title: str
) -> List[Tuple[str, int]]:
    """
    Apply the notebook's filter: drop entities at index 0 (sentence start)
    or that appear inside the article ``title``. Deduplicate by char index.
    """
    seen: List[int] = []
    out: List[Tuple[str, int]] = []
    for e, idx in entities:
        if idx == 0 or e in title:
            continue
        if idx not in seen:
            seen.append(idx)
            out.append((e, idx))
    return out
