"""
Smoke tests — does every module in src/ at least import cleanly?

Modules that hard-depend on torch are gated behind ``pytest.importorskip``
so the suite still gives a useful signal on a CPU-only / torch-less machine.
"""

import importlib

import pytest


def test_src_package_imports():
    pkg = importlib.import_module("src")
    assert hasattr(pkg, "__version__")


def test_config_imports():
    importlib.import_module("src.config")


def test_entities_imports():
    importlib.import_module("src.entities")


def test_tokens_imports():
    importlib.import_module("src.tokens")


def test_embeddings_imports():
    importlib.import_module("src.embeddings")


def test_model_loader_imports():
    importlib.import_module("src.model_loader")


def test_classifier_imports():
    pytest.importorskip("torch")
    importlib.import_module("src.classifier")


def test_train_imports():
    pytest.importorskip("torch")
    importlib.import_module("src.train")


def test_evaluate_imports():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    importlib.import_module("src.evaluate")


def test_dataset_gen_imports():
    importlib.import_module("src.dataset_gen")
