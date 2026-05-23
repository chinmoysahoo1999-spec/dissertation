"""Sanity checks on config.py — catches accidental edits that break invariants."""

from src import config


def test_model_name_is_llama32():
    assert config.MODEL_NAME == "meta-llama/Llama-3.2-1B"


def test_hyperparam_types_and_ranges():
    assert isinstance(config.TOPK_FIRST_TOKEN, int) and config.TOPK_FIRST_TOKEN > 0
    assert isinstance(config.WINDOWS, int) and config.WINDOWS > 0
    assert isinstance(config.TARGET_SAMPLES, int) and config.TARGET_SAMPLES > 0
    assert isinstance(config.CHECKPOINT_EVERY, int) and config.CHECKPOINT_EVERY > 0
    assert isinstance(config.SENTENCES_PER_DOC, int) and config.SENTENCES_PER_DOC >= 1
    assert isinstance(config.EMBED_START_AT, int) and config.EMBED_START_AT >= 1


def test_train_fraction_is_a_valid_probability():
    assert 0.0 < config.TRAIN_FRACTION < 1.0


def test_hidden_layers_shape():
    assert isinstance(config.HIDDEN_LAYERS, tuple)
    assert len(config.HIDDEN_LAYERS) >= 1
    assert all(isinstance(w, int) and w > 0 for w in config.HIDDEN_LAYERS)


def test_dropout_in_unit_interval():
    assert 0.0 <= config.DROPOUT_RATE < 1.0


def test_embedding_key_is_valid():
    assert config.EMBEDDING_KEY in {"mean2", "mean1", "hds"}


def test_paths_are_strings():
    for path in (
        config.TRAIN_PATH,
        config.TEST_PATH,
        config.CHECKPOINT_PATH,
        config.BEST_MODEL_PATH,
    ):
        assert isinstance(path, str) and path
