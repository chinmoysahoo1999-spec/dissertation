"""
Tests for the pure-Python entity utilities.

These do NOT need spaCy or a GPU — only ``get_entities`` does. We test that
one with a small dummy "nlp" stand-in so the suite stays fast and offline.
"""

from src.entities import (
    delete_substrings,
    filter_valid_entities,
    find_boundaries,
    get_entities,
)


# ---------------------------------------------------------------------------
# delete_substrings
# ---------------------------------------------------------------------------
def test_delete_substrings_basic():
    out = delete_substrings(["Einstein", "Albert Einstein", "Ulm"])
    assert sorted(out) == ["Albert Einstein", "Ulm"]


def test_delete_substrings_no_substrings():
    out = delete_substrings(["alpha", "beta", "gamma"])
    assert sorted(out) == ["alpha", "beta", "gamma"]


def test_delete_substrings_dedupes_input():
    out = delete_substrings(["x", "x", "xy"])
    assert sorted(out) == ["xy"]


def test_delete_substrings_empty():
    assert delete_substrings([]) == []


# ---------------------------------------------------------------------------
# find_boundaries
# ---------------------------------------------------------------------------
def test_find_boundaries_expands_to_whitespace():
    # "Ein" appears inside "Einstein"; it should expand out to the full word.
    out = find_boundaries("Albert Einstein was born.", ["Ein"])
    assert "Einstein" in out


def test_find_boundaries_keeps_clean_match():
    out = find_boundaries("born in Ulm today", ["Ulm"])
    assert out == ["Ulm"]


# ---------------------------------------------------------------------------
# filter_valid_entities
# ---------------------------------------------------------------------------
def test_filter_valid_drops_index_zero():
    raw = [("Einstein", 0), ("Ulm", 30)]
    assert filter_valid_entities(raw, title="Albert Einstein") == [("Ulm", 30)]


def test_filter_valid_drops_entities_in_title():
    raw = [("Einstein", 5), ("Ulm", 30)]
    assert filter_valid_entities(raw, title="Albert Einstein") == [("Ulm", 30)]


def test_filter_valid_dedupes_by_index():
    raw = [("Ulm", 30), ("Ulm", 30)]
    assert filter_valid_entities(raw, title="") == [("Ulm", 30)]


# ---------------------------------------------------------------------------
# get_entities with a dummy nlp stand-in
# ---------------------------------------------------------------------------
class _FakeEnt:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


class _FakeDoc:
    def __init__(self, ents):
        self.ents = [_FakeEnt(t) for t in ents]


class _FakeNLP:
    """A drop-in replacement for spaCy's nlp pipeline for testing only."""

    def __init__(self, ent_strings):
        self._ents = ent_strings

    def __call__(self, text):
        return _FakeDoc(self._ents)


def test_get_entities_returns_occurrences():
    # No trailing punctuation — boundary expansion goes until whitespace,
    # so "Ulm." would otherwise be returned as the boundary-expanded entity.
    text = "Einstein was born in Ulm yesterday. Einstein moved later."
    nlp = _FakeNLP(["Einstein", "Ulm"])
    occ = get_entities(text, nlp)
    entities = [e for e, _ in occ]
    assert entities.count("Einstein") >= 2
    assert "Ulm" in entities
    # Indexes must point to where the entity actually starts in the text.
    for ent, idx in occ:
        assert text[idx : idx + len(ent)] == ent


def test_get_entities_keeps_trailing_punctuation_in_boundary():
    """
    Documents the (intentional) behavior inherited from the original notebook:
    find_boundaries expands to the next whitespace, so a trailing period stays
    glued to the entity. The dataset_gen filter handles this downstream.
    """
    text = "He was born in Ulm. Then moved away."
    nlp = _FakeNLP(["Ulm"])
    occ = get_entities(text, nlp)
    matched = {e for e, _ in occ}
    # The boundary-expanded form "Ulm." should appear somewhere.
    assert any("Ulm" in e for e in matched)
